"""
Brand Comment Intelligence & Ad Optimizer (Streamlit)

Enter a brand -> the app finds YouTube videos, reads their comments, saves ONE CSV, shows what kinds of
comments exist and what people like, and two Gemini agents (run together) recommend where to invest.

Run:   python -m streamlit run app.py
Keys: put them in a .env file next to this file (see .env.example). They are never shown in the app.
Never upload .env to GitHub.
"""

import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from agents import DEFAULT_MODEL, build_context, run_agents, run_source_agent
from brand_insights import coverage, discovery_summary, language_summary, recommend_budget, summarize_group
from brand_pipeline import collect_brand_data, save_csv
from config import get_key, scrub
from topics import top_words, topic_summary, type_summary

st.set_page_config(page_title="Brand Comment Intelligence", page_icon="🎯", layout="wide",
                   initial_sidebar_state="collapsed")

# The app has no sidebar at all: hide the sidebar and every button that could open it.
st.markdown(
    """
    <style>
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Helpers ----------
def show_table(data, **kwargs):
    """Display a dataframe full-width; works on both old and new Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def bar_chart(labels, values, title, xlabel, colors="#2196F3", fmt="{:,.0f}"):
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.45 * len(labels) + 1)))
    ax.barh([str(x) for x in labels][::-1], list(values)[::-1],
            color=colors[::-1] if isinstance(colors, list) else colors)
    top = max([abs(v) for v in values] + [1])
    for y, v in enumerate(list(values)[::-1]):
        ax.text(v, y, "  " + fmt.format(v), va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_xlim(min(0, min(values)) * 1.4 if min(values) < 0 else 0, top * 1.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def budget_chart(table, name_col):
    data = table.sort_values("budget_rs")
    fig, ax = plt.subplots(figsize=(5, max(2.5, 0.5 * len(data) + 1)))
    ax.barh(data[name_col].astype(str), data["budget_rs"], color="#2196F3")
    for y, (val, pct) in enumerate(zip(data["budget_rs"], data["share_%"])):
        ax.text(val, y, f"  Rs.{val:,.0f} ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, data["budget_rs"].max() * 1.5)
    ax.set_xlabel("Suggested budget (Rs.)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if data is not None and not data.empty:
                data.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def analysis_tables(df: pd.DataFrame, videos: pd.DataFrame) -> dict:
    titles = videos.set_index("video_id")["title"].to_dict() if not videos.empty else {}
    video_table = summarize_group(df, "video_id", skip=())
    if not video_table.empty:
        video_table.insert(1, "title", video_table["video_id"].map(titles).fillna(""))
    return {
        "types": type_summary(df),
        "topics": topic_summary(df),
        "source": summarize_group(df, "source"),
        "video": video_table,
        "location": summarize_group(df, "location"),
        "discovery": discovery_summary(df),
        "bought": summarize_group(df, "bought_from"),
        "language": language_summary(df),
        "coverage": coverage(df),
    }


def bullets(items):
    for x in items or []:
        st.markdown(f"- {x if isinstance(x, str) else ' '.join(str(v) for v in x.values())}")


# ---------- State ----------
st.session_state.setdefault("brands", {})
if "pending_select" in st.session_state:
    st.session_state["selected_brand"] = st.session_state.pop("pending_select")

# ---------- Settings (read silently from the .env file; nothing about them is shown on the page) ----------
yt_key = get_key("YOUTUBE_API_KEY")
gem_key = get_key("GEMINI_API_KEY")
model = get_key("GEMINI_MODEL") or DEFAULT_MODEL
use_search = get_key("USE_WEB_SEARCH").strip().lower() not in ("0", "false", "no", "off")

# ---------- Page ----------
st.title("🎯 Brand Comment Intelligence & Ad Optimizer")
flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

brand_names = list(st.session_state["brands"])
if brand_names:
    st.selectbox("Brand (each saved brand has its own comments and budget)", brand_names, key="selected_brand")
name = st.session_state.get("selected_brand") if brand_names else None
data = st.session_state["brands"].get(name) if name else None

tab_collect, tab_insights, tab_ads, tab_data = st.tabs(
    ["1️⃣ Collect comments", "2️⃣ Comments & insights", "3️⃣ Ad optimizer", "4️⃣ Data & download"]
)


def run_agents_now(name, data, tables):
    if not gem_key:
        st.warning("The Gemini key is missing. Add GEMINI_API_KEY to your .env file, save it, and restart the app.")
        return
    ctx = build_context(name, data["niche"], data["budget"], data["comments"], data["videos"],
                        tables["types"], tables["topics"], tables["source"], tables["video"], tables["location"],
                        tables["discovery"], tables["bought"], tables["coverage"], tables["language"],
                        (data.get("source_ai") or {}).get("mentions"))
    with st.spinner("Both AI agents are working at the same time (can take up to a minute)..."):
        data["agents"] = run_agents(ctx, gem_key, model, use_search)
    st.rerun()


# ===== Tab 1: collect =====
with tab_collect:
    st.subheader("Tell me about the brand")
    c1, c2 = st.columns(2)
    brand = c1.text_input("Brand name", placeholder="e.g. Khaadi, Nike, Samsung")
    niche = c2.text_input("Niche (optional)", placeholder="e.g. fashion, sportswear, smartphones")
    c3, c4, c5 = st.columns(3)
    max_videos = c3.number_input("Max videos", 1, 50, 10)
    per_video = c4.number_input("Max comments per video", 20, 3000, 300, step=20)
    total_cap = c5.number_input("Total comments (all videos)", 50, 50000, 2000, step=100)
    strict = st.checkbox("Only use videos that mention the brand name (keeps the analysis about this brand only)", True)
    recent = st.radio("Comment order", ["Most popular", "Newest"], horizontal=True) == "Newest"
    extra = st.text_area("Optional: add specific video links (one per line)", height=70)

    if st.button("Collect comments", type="primary"):
        brand = brand.strip()
        if not brand:
            st.warning("Please enter a brand name.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                comments_df, videos_df, notes = collect_brand_data(
                    brand, niche.strip(), int(max_videos), int(per_video), int(total_cap), yt_key, strict, recent,
                    [x for x in extra.split() if x],
                    progress=lambda frac, text: bar.progress(min(max(frac, 0.0), 1.0), text=text),
                )
            except Exception as e:
                bar.empty()
                st.error(f"Could not collect data: {scrub(e, yt_key, gem_key)}")
            else:
                bar.empty()
                if comments_df.empty:
                    st.error("No comments could be read. " + scrub(" ".join(notes), yt_key, gem_key))
                else:
                    path = save_csv(comments_df, brand)
                    old = st.session_state["brands"].get(brand, {})
                    st.session_state["brands"][brand] = {
                        "niche": niche.strip(), "comments": comments_df, "videos": videos_df,
                        "notes": notes, "budget": old.get("budget", 1_000_000), "agents": None, "csv_path": path,
                    }
                    st.session_state["pending_select"] = brand
                    st.session_state["flash"] = (
                        f"Collected {len(comments_df):,} comments from {comments_df['video_id'].nunique()} "
                        f"video(s) for {brand}. Saved as {path}. Open the next tabs to see the results."
                    )
                    st.rerun()

    if data:
        st.divider()
        st.write(f"**Current brand: {name}**  ·  {len(data['comments']):,} comments  ·  "
                 f"{data['comments']['video_id'].nunique()} videos  ·  saved to `{data['csv_path']}`")
        for note in data["notes"]:
            st.caption("• " + scrub(note, yt_key, gem_key))
        show_table(data["videos"][["title", "channel", "views", "likes", "comment_count", "comments_scraped"]])

# ===== Tab 2: comments & insights =====
with tab_insights:
    if not data:
        st.info("Collect comments for a brand first (tab 1).")
    else:
        df, videos = data["comments"], data["videos"]
        tables = analysis_tables(df, videos)
        st.subheader(f"{name}: what people are saying")
        total = len(df)
        m = st.columns(5)
        m[0].metric("Comments", f"{total:,}")
        m[1].metric("😊 Positive", f"{(df['compound'] >= 0.05).mean():.0%}")
        m[2].metric("😠 Negative", f"{(df['compound'] <= -0.05).mean():.0%}")
        m[3].metric("Say they bought it", f"{int(df['purchased'].sum()):,}")
        m[4].metric("Want to buy", f"{int(df['intent'].sum()):,}")

        left, right = st.columns(2)
        with left:
            st.markdown("**What kinds of comments are there?**")
            t = tables["types"]
            st.pyplot(bar_chart(t["comment_type"], t["comments"], "Comment types", "comments"))
        with right:
            st.markdown("**What do people like or dislike?**")
            tp = tables["topics"]
            if tp.empty:
                st.info("No common topics (price, quality, delivery...) found in these comments.")
            else:
                colors = ["#4CAF50" if v >= 0.2 else "#F44336" if v <= -0.1 else "#9E9E9E" for v in tp["avg_sentiment"]]
                st.pyplot(bar_chart(tp["topic"], tp["avg_sentiment"], "Average sentiment per topic",
                                    "green = liked, red = disliked", colors, "{:+.2f}"))
        if not tables["topics"].empty:
            show_table(tables["topics"])

        w1, w2 = st.columns(2)
        with w1:
            st.markdown("**Words in positive comments**")
            show_table(top_words(df[df["compound"] >= 0.05]))
        with w2:
            st.markdown("**Words in negative comments**")
            show_table(top_words(df[df["compound"] <= -0.05]))

        p, q = st.columns(2)
        with p:
            st.markdown("**Most-liked positive comments**")
            for _, r in df[df["compound"] >= 0.05].nlargest(5, "likes").iterrows():
                st.success(f"👍 {r['likes']:,}  {r['comment'][:240]}")
        with q:
            st.markdown("**Most-liked negative comments**")
            for _, r in df[df["compound"] <= -0.05].nlargest(5, "likes").iterrows():
                st.error(f"👍 {r['likes']:,}  {r['comment'][:240]}")

        st.divider()
        st.subheader("🤖 AI insights agent")
        if st.button("Run AI agents (insights + marketing together)", key="run_agents_insights", type="primary"):
            run_agents_now(name, data, tables)
        res = (data.get("agents") or {}).get("insights")
        if not res:
            st.info("Click the button to let the AI agents read the evidence and write recommendations.")
        elif "error" in res:
            st.error(f"Insights agent failed: {res['error']}")
        else:
            st.write(res.get("summary", ""))
            a, b = st.columns(2)
            with a:
                st.markdown("**What people like**")
                bullets(res.get("what_people_like"))
            with b:
                st.markdown("**What people dislike**")
                bullets(res.get("what_people_dislike"))
            st.markdown("**Questions the audience keeps asking**")
            bullets(res.get("audience_questions"))
            st.markdown("**Recommendations**")
            for rec in res.get("recommendations", []) or []:
                if isinstance(rec, dict):
                    st.markdown(f"- **{rec.get('action', '')}**: {rec.get('why', '')}")
                else:
                    st.markdown(f"- {rec}")
            st.caption(f"Model: {res.get('_model', '')} · {res.get('_backend', '')}")

# ===== Tab 3: ad optimizer =====
with tab_ads:
    if not data:
        st.info("Collect comments for a brand first (tab 1).")
    else:
        df, videos = data["comments"], data["videos"]
        tables = analysis_tables(df, videos)
        st.subheader(f"Ad optimizer for {name}")
        st.caption("Only this brand's comments are used. Every brand keeps its own budget.")
        b1, b2 = st.columns(2)
        budget = b1.number_input(f"Budget for {name} (Rs.)", min_value=10_000, value=int(data["budget"]),
                                 step=50_000, key=f"budget_{name}")
        data["budget"] = budget
        min_share = b2.slider("Minimum share for every option (%)", 0, 15, 5, key=f"min_{name}") / 100

        st.markdown("### 🧭 Where is the audience coming from?")
        st.caption("This is what commenters SAY. It is not real traffic data (that is in YouTube Studio analytics, "
                   "which only the channel owner can see), and most comments name no source at all.")
        cov = tables["coverage"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Comments that name a source", f"{cov['named_any']:,}",
                  f"{cov['named_any_%']}% of {cov['comments']:,}", delta_color="off")
        c2.metric("Say where they found it", f"{cov['found_on_platform']:,}", "platform they saw it on", delta_color="off")
        c3.metric("Say where they bought it", f"{cov['bought_from_named']:,}", f"of {cov['say_bought']:,} who bought", delta_color="off")
        c4.metric("Friend, advert or influencer", f"{cov['friend_ad_influencer']:,}", "how they heard of it", delta_color="off")
        if cov["named_any_%"] < 5:
            st.warning("Fewer than 5% of the comments name a source, so treat these tables as weak evidence.")

        d_left, d_right = st.columns(2)
        with d_left:
            st.markdown("**How people found the brand**")
            if tables["discovery"].empty:
                st.info("No comment says how the person found the brand.")
            else:
                show_table(tables["discovery"])
        with d_right:
            st.markdown("**Where people bought it**")
            if tables["bought"].empty:
                st.info("No buyer named where they bought it.")
            else:
                show_table(tables["bought"][["bought_from", "mentions", "purchases", "avg_sentiment"]])
        st.markdown("**Language of the comments** (a rough hint about the audience)")
        show_table(tables["language"])

        if st.button("🤖 Ask AI to find more source mentions (reads up to 150 comments)", key="run_source_agent"):
            if not gem_key:
                st.warning("The Gemini key is missing. Add GEMINI_API_KEY to your .env file, save it, and restart the app.")
            else:
                with st.spinner("The AI is reading the comments..."):
                    data["source_ai"] = run_source_agent(name, df, gem_key, model)
                st.rerun()
        found = data.get("source_ai")
        if found and "error" in found:
            st.error(f"Source reader failed: {found['error']}")
        elif found:
            st.caption(f"The AI read {found['read']} comments and found {len(found['mentions'])} that say where the "
                       f"person found or bought the brand · {found.get('_model', '')} · {found.get('_backend', '')}")
            if found["mentions"]:
                show_table(pd.DataFrame(found["mentions"])[["comment", "found_on", "bought_from", "how_found", "evidence", "likes"]])

        st.markdown("### Evidence from the comments")
        if tables["source"].empty:
            st.info("No comment named a place where people bought or found the brand (Daraz, Instagram, a shop...). "
                    "The video table below still shows which videos reach the most buyers.")
        else:
            st.markdown("**Where people buy / find the brand**")
            src = recommend_budget(tables["source"], budget, min_share)
            l, r = st.columns([3, 2])
            with l:
                show_table(src[["source", "mentions", "purchases", "intents", "show_offs",
                                "avg_sentiment", "share_%", "budget_rs"]])
            with r:
                st.pyplot(budget_chart(src, "source"))

        st.markdown("**Which videos reach the most buyers**")
        vid = tables["video"]
        if not vid.empty:
            merged = vid.merge(videos[["video_id", "channel", "views", "likes", "comment_count"]],
                               on="video_id", how="left")
            merged = recommend_budget(merged, budget, min_share)
            l, r = st.columns([3, 2])
            with l:
                show_table(merged[["title", "channel", "views", "likes", "purchases", "intents",
                                   "avg_sentiment", "share_%", "budget_rs"]])
            with r:
                chart_tbl = merged.assign(label=merged["title"].str.slice(0, 28))
                st.pyplot(budget_chart(chart_tbl, "label"))
        if not tables["location"].empty:
            st.markdown("**Cities / countries mentioned**")
            show_table(tables["location"][["location", "mentions", "purchases", "intents", "avg_sentiment"]])

        st.markdown("### 🤖 AI marketing agent: where should the money go?")
        if st.button("Run AI agents (insights + marketing together)", key="run_agents_ads", type="primary"):
            run_agents_now(name, data, tables)
        res = (data.get("agents") or {}).get("marketing")
        if not res:
            st.info("Click the button. The agent studies the comments (and, if enabled, searches the web for how the "
                    "brand markets itself) and proposes a platform split.")
        elif "error" in res:
            st.error(f"Marketing agent failed: {res['error']}")
        else:
            st.write(res.get("summary", ""))
            alloc = pd.DataFrame(res.get("allocation", []))
            if not alloc.empty:
                alloc["share_%"] = alloc["percent"]
                alloc["budget_rs"] = (alloc["percent"] / 100 * budget).round(0)
                l, r = st.columns([3, 2])
                with l:
                    show_table(alloc[["platform", "share_%", "budget_rs", "reason"]])
                with r:
                    st.pyplot(budget_chart(alloc, "platform"))
            x, y, z = st.columns(3)
            with x:
                st.markdown("**How the brand markets itself now**")
                bullets(res.get("brand_marketing_observations"))
            with y:
                st.markdown("**Risks / what the data can't tell us**")
                bullets(res.get("risks"))
            with z:
                st.markdown("**Next steps**")
                bullets(res.get("next_steps"))
            web = "used web search" if res.get("_grounded") else "no web search"
            st.caption(f"Confidence: {res.get('confidence', '?')} · Model: {res.get('_model', '')} · {res.get('_backend', '')} · {web}")

# ===== Tab 4: data & download =====
with tab_data:
    if not data:
        st.info("Collect comments for a brand first (tab 1).")
    else:
        df, videos = data["comments"], data["videos"]
        tables = analysis_tables(df, videos)
        st.subheader(f"{name}: all collected comments")
        f1, f2, f3 = st.columns(3)
        sent = f1.multiselect("Sentiment", ["Positive", "Neutral", "Negative"], default=["Positive", "Neutral", "Negative"])
        ctype = f2.multiselect("Comment type", sorted(df["comment_type"].unique()), default=sorted(df["comment_type"].unique()))
        word = f3.text_input("Search word")
        view = df[df["sentiment"].isin(sent) & df["comment_type"].isin(ctype)]
        if word:
            view = view[view["comment"].str.contains(word, case=False, na=False, regex=False)]
        st.caption(f"Showing {len(view):,} of {len(df):,} comments")
        show_table(view[["video_title", "comment", "likes", "sentiment", "comment_type", "topics", "source",
                         "discovered_on", "bought_from", "language", "location"]], height=380)

        agent_sheet = None
        mk = (data.get("agents") or {}).get("marketing")
        if mk and "allocation" in mk and mk["allocation"]:
            agent_sheet = pd.DataFrame(mk["allocation"])
            agent_sheet["budget_rs"] = (agent_sheet["percent"] / 100 * data["budget"]).round(0)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download all comments (CSV)", df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                           file_name=f"{name}_comments.csv", mime="text/csv")
        try:
            d2.download_button(
                "⬇️ Download full analysis (Excel)",
                to_excel_bytes({"Comments": df, "Videos": videos, "CommentTypes": tables["types"],
                                "Topics": tables["topics"], "BySource": tables["source"], "ByVideo": tables["video"],
                                "HowFound": tables["discovery"], "BoughtFrom": tables["bought"],
                                "Languages": tables["language"],
                                "AISourceMentions": pd.DataFrame((data.get("source_ai") or {}).get("mentions") or []),
                                "AIPlatformPlan": agent_sheet}),
                file_name=f"{name}_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except ImportError:
            d2.caption("Install openpyxl to enable the Excel download.")

        st.divider()
        st.subheader("All saved brands (each has its own budget)")
        overview = pd.DataFrame([
            {"brand": b, "comments": len(v["comments"]), "videos": v["comments"]["video_id"].nunique(),
             "budget_rs": v["budget"], "agents_run": bool(v.get("agents"))}
            for b, v in st.session_state["brands"].items()
        ])
        show_table(overview)