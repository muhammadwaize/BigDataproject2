"""
YouTube Sentiment Analyzer - Web Interface (Streamlit)

Keep this file in the SAME folder as youtube_sentiment.py (it reuses its functions).

Install:
    pip install streamlit youtube-comment-downloader vaderSentiment pandas matplotlib

Run:
    streamlit run app.py
"""

import io
import re
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from youtube_comment_downloader import (
    SORT_BY_POPULAR,
    SORT_BY_RECENT,
    YoutubeCommentDownloader,
)

from brand_insights import recommend_budget, summarize_group
from youtube_sentiment import analyze, extract_video_id

ORDER = ["Positive", "Neutral", "Negative"]
COLORS = {"Positive": "#4CAF50", "Neutral": "#9E9E9E", "Negative": "#F44336"}

st.set_page_config(page_title="YouTube Sentiment Analyzer", page_icon="💬", layout="wide")


# ---------- Helpers ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_comments(video_id: str, limit: int, sort_recent: bool) -> list[dict]:
    downloader = YoutubeCommentDownloader()
    sort = SORT_BY_RECENT if sort_recent else SORT_BY_POPULAR
    comments = downloader.get_comments(video_id, sort_by=sort)
    return [
        {"text": c["text"], "votes": c.get("votes", "0"), "replies": c.get("replies", 0)}
        for c in islice(comments, limit)
    ]


def show_table(data, **kwargs):
    """Display a dataframe full-width; works on both old and new Streamlit versions."""
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def parse_links(raw: str) -> list[str]:
    return [t for t in re.split(r"[\s,]+", raw.strip()) if t]


def run_analysis(links: list[str], limit: int, sort_recent: bool):
    frames, problems, seen = [], [], set()
    progress = st.progress(0.0, text="Starting...")

    for i, link in enumerate(links):
        video_id = extract_video_id(link)
        if not video_id:
            problems.append(f"Not a valid YouTube link: {link}")
        elif video_id not in seen:
            seen.add(video_id)
            progress.progress(i / len(links), text=f"Scraping comments from {video_id}...")
            try:
                comments = fetch_comments(video_id, limit, sort_recent)
                if not comments:
                    problems.append(f"{video_id}: no comments found (they may be disabled).")
                else:
                    df = analyze(comments, video_id)
                    if not df.empty:
                        frames.append(df)
            except Exception as e:
                problems.append(f"{video_id}: failed to scrape ({e})")

    progress.progress(1.0, text="Done")
    progress.empty()
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, problems


def pie_chart(df: pd.DataFrame):
    counts = df["sentiment"].value_counts().reindex(ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.pie(
        counts,
        labels=ORDER,
        colors=[COLORS[k] for k in ORDER],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title("Overall Sentiment")
    return fig


def per_video_chart(df: pd.DataFrame):
    pct = (
        df.groupby("video_id")["sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=ORDER, fill_value=0)
        * 100
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    pct.plot(kind="bar", stacked=True, color=[COLORS[k] for k in ORDER], ax=ax, width=0.7)
    ax.set_ylabel("% of comments")
    ax.set_xlabel("Video ID")
    ax.set_title("Sentiment per Video")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    return fig


def histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(df["compound"], bins=20, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Compound score (-1 very negative, +1 very positive)")
    ax.set_ylabel("Comments")
    ax.set_title("Score Distribution")
    plt.tight_layout()
    return fig


def budget_chart(table: pd.DataFrame, name_col: str):
    data = table.sort_values("budget_rs")
    fig, ax = plt.subplots(figsize=(5, max(2.5, 0.5 * len(data) + 1)))
    ax.barh(data[name_col].astype(str), data["budget_rs"], color="#2196F3")
    for y, (val, pct) in enumerate(zip(data["budget_rs"], data["share_%"])):
        ax.text(val, y, f"  Rs.{val:,.0f} ({pct:.0f}%)", va="center", fontsize=8)
    ax.set_xlim(0, data["budget_rs"].max() * 1.45)
    ax.set_xlabel("Suggested budget (Rs.)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def budget_view(title: str, table: pd.DataFrame, name_col: str, empty_msg: str, budget: float, min_share: float):
    st.subheader(title)
    if table.empty:
        st.info(empty_msg)
        return table
    table = recommend_budget(table, budget, min_share)
    left, right = st.columns([3, 2])
    with left:
        show_table(
            table[[name_col, "mentions", "purchases", "intents", "show_offs",
                   "avg_sentiment", "share_%", "budget_rs"]].rename(columns={
                "mentions": "comments", "intents": "want to buy", "show_offs": "show-off",
                "avg_sentiment": "avg sentiment", "budget_rs": "budget (Rs.)"}),
        )
    with right:
        st.pyplot(budget_chart(table, name_col))
    return table


def to_excel_bytes(sheets: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, data in sheets.items():
            if not data.empty:
                data.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    limit = st.slider("Max comments per video", 50, 2000, 1000, step=50)
    sort_recent = st.radio("Comment order", ["Most popular", "Newest"]) == "Newest"
    st.divider()
    st.subheader("Ad budget")
    budget = st.number_input("Total budget (Rs.)", min_value=10_000, value=1_000_000, step=50_000)
    min_share = st.slider("Minimum share for every option (%)", 0, 15, 5) / 100
    st.caption(
        "More comments = more accurate results, but slower scraping. "
        "Sentiment is scored with VADER (best on English text)."
    )

# ---------- Main page ----------
st.title("💬 YouTube Comment Sentiment Analyzer")
st.write("Paste one or more YouTube links (one per line) and see what viewers feel.")

raw_links = st.text_area(
    "YouTube links",
    height=130,
    placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...\nhttps://www.youtube.com/shorts/...",
)

if st.button("Analyze", type="primary"):
    links = parse_links(raw_links)
    if not links:
        st.warning("Please paste at least one YouTube link.")
    else:
        df, problems = run_analysis(links, limit, sort_recent)
        st.session_state["results"] = df
        st.session_state["problems"] = problems

df = st.session_state.get("results")
for msg in st.session_state.get("problems", []):
    st.warning(msg)

if df is not None:
    if df.empty:
        st.error("No comments could be analyzed. Check your links and try again.")
        st.stop()

    total = len(df)
    pct = df["sentiment"].value_counts(normalize=True).reindex(ORDER, fill_value=0) * 100

    st.subheader("Results")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Comments analyzed", f"{total:,}")
    c2.metric("😊 Positive", f"{pct['Positive']:.1f}%")
    c3.metric("😐 Neutral", f"{pct['Neutral']:.1f}%")
    c4.metric("😠 Negative", f"{pct['Negative']:.1f}%")
    c5.metric("Average score", f"{df['compound'].mean():+.3f}")

    n_videos = df["video_id"].nunique()
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(pie_chart(df))
    with col_b:
        st.pyplot(per_video_chart(df) if n_videos > 1 else histogram(df))
    if n_videos > 1:
        st.pyplot(histogram(df))

    if n_videos > 1:
        st.subheader("Per-video summary")
        summary = (
            df.groupby("video_id")
            .agg(comments=("comment", "count"), avg_score=("compound", "mean"))
            .round(3)
        )
        share = (
            df.groupby("video_id")["sentiment"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .reindex(columns=ORDER, fill_value=0)
            * 100
        ).round(1)
        show_table(summary.join(share))

    left, right = st.columns(2)
    with left:
        st.subheader("Most positive comments")
        for _, r in df.nlargest(5, "compound").iterrows():
            st.success(f"**{r['compound']:+.2f}**  {r['comment'][:250]}")
    with right:
        st.subheader("Most negative comments")
        for _, r in df.nsmallest(5, "compound").iterrows():
            st.error(f"**{r['compound']:+.2f}**  {r['comment'][:250]}")

    st.divider()
    st.header("🎯 Brand reach and ad budget optimizer")
    st.caption(
        "Comments are scanned for people who say they bought the product (even from another shop, they still "
        "show the brand's reach), people who want to buy, where they bought or found it, and how happy they are."
    )

    n_buy, n_int, n_show = int(df["purchased"].sum()), int(df["intent"].sum()), int(df["show_off"].sum())
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Say they bought it", f"{n_buy:,}", f"{n_buy / total:.1%} of comments", delta_color="off")
    k2.metric("Want to buy / asking", f"{n_int:,}", f"{n_int / total:.1%} of comments", delta_color="off")
    k3.metric("Show-off comments", f"{n_show:,}", "bought it and happy", delta_color="off")
    k4.metric("Mention a place / shop", f"{int((df['source'] != 'Not mentioned').sum()):,}", "where they bought or found it", delta_color="off")

    by_source = budget_view(
        "Where are people buying / finding the brand?",
        summarize_group(df, "source"), "source",
        "No comment named a place (Daraz, Instagram, a shop, the website...). This is common, so use the "
        "video table below, or add more places in the SOURCES list inside brand_insights.py.",
        budget, min_share,
    )
    by_video = budget_view(
        "Which video reaches the most buyers?",
        summarize_group(df, "video_id", skip=()), "video_id",
        "No videos to compare.", budget, min_share,
    )
    by_location = summarize_group(df, "location")
    if not by_location.empty:
        st.subheader("Cities / countries mentioned")
        show_table(by_location[["location", "mentions", "purchases", "intents", "avg_sentiment"]])

    with st.expander("See the 'bought it' and show-off comments"):
        bought = df[df["purchased"]].sort_values("likes", ascending=False)
        if bought.empty:
            st.write("No comments said they bought it.")
        else:
            show_table(bought[["video_id", "comment", "likes", "source", "sentiment", "show_off"]].head(200), height=300)

    try:
        st.download_button(
            "⬇️ Download brand analysis as Excel",
            to_excel_bytes({"Comments": df, "BySource": by_source, "ByVideo": by_video, "ByLocation": by_location}),
            file_name="brand_reach_budget.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except ImportError:
        st.caption("Install openpyxl (python -m pip install openpyxl) to enable the Excel download.")

    st.divider()
    st.subheader("Browse all comments")
    f1, f2 = st.columns([1, 2])
    chosen = f1.multiselect("Sentiment", ORDER, default=ORDER)
    keyword = f2.text_input("Search keyword")
    view = df[df["sentiment"].isin(chosen)]
    if keyword:
        view = view[view["comment"].str.contains(keyword, case=False, na=False, regex=False)]
    st.caption(f"Showing {len(view):,} of {total:,} comments")
    show_table(view[["video_id", "comment", "likes", "compound", "sentiment"]], height=350)

    st.download_button(
        "⬇️ Download results as CSV",
        df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="comments_sentiment.csv",
        mime="text/csv",
    )