"""
YouTube Comment Sentiment Analyzer (multi-video)
------------------------------------------------
Works with ANY YouTube link(s): watch links, youtu.be short links, Shorts,
embed links, or a bare video ID. Analyze one video or many at once.

Install:
    pip install youtube-comment-downloader vaderSentiment pandas matplotlib

Usage (pick any):
    python youtube_sentiment.py                       # asks you to paste links
    python youtube_sentiment.py LINK1 LINK2 LINK3     # links as arguments
    python youtube_sentiment.py --file links.txt      # one link per line
    python youtube_sentiment.py LINK1 --limit 1000    # more comments per video
"""

import argparse
import re
from itertools import islice

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR

from brand_insights import (
    parse_count,
    recommend_budget,
    summarize_group,
    tag_comments,
)

VIDEO_ID_PATTERNS = [
    r"(?:v=|vi=)([A-Za-z0-9_-]{11})",                 # watch?v=ID
    r"youtu\.be/([A-Za-z0-9_-]{11})",                 # youtu.be/ID
    r"youtube\.com/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})",
    r"^([A-Za-z0-9_-]{11})$",                         # bare ID
]


def extract_video_id(link: str) -> str | None:
    """Return the 11-character video ID from any common YouTube link format."""
    link = link.strip()
    for pattern in VIDEO_ID_PATTERNS:
        match = re.search(pattern, link)
        if match:
            return match.group(1)
    return None


def get_links(args) -> list[str]:
    """Collect links from arguments, a text file, or interactive input."""
    links = list(args.urls)

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            links += [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not links:
        print("Paste YouTube links (separated by spaces or one per line).")
        print("Press Enter on an empty line when done:")
        while True:
            line = input("> ").strip()
            if not line:
                break
            links += line.split()

    return links


def scrape_comments(video_id: str, limit: int) -> list[dict]:
    """Return comments with their text, likes (votes) and number of replies."""
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
    return [
        {"text": c["text"], "votes": c.get("votes", "0"), "replies": c.get("replies", 0)}
        for c in islice(comments, limit)
    ]


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def label_from_score(compound: float) -> str:
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze(comments: list, video_id: str) -> pd.DataFrame:
    """comments can be plain strings or dicts with text / votes / replies."""
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for item in comments:
        raw = item["text"] if isinstance(item, dict) else item
        likes = parse_count(item.get("votes", 0)) if isinstance(item, dict) else 0
        replies = parse_count(item.get("replies", 0)) if isinstance(item, dict) else 0
        text = clean_text(raw)
        if not text:
            continue
        scores = analyzer.polarity_scores(text)
        rows.append(
            {
                "video_id": video_id,
                "comment": text,
                "likes": likes,
                "replies": replies,
                "compound": scores["compound"],
                "positive": scores["pos"],
                "neutral": scores["neu"],
                "negative": scores["neg"],
                "sentiment": label_from_score(scores["compound"]),
            }
        )
    df = pd.DataFrame(rows)
    return tag_comments(df) if not df.empty else df


def print_report(df: pd.DataFrame, title: str) -> None:
    total = len(df)
    counts = df["sentiment"].value_counts()
    print(f"\n===== {title} =====")
    print(f"Comments analyzed: {total}")
    for label in ["Positive", "Neutral", "Negative"]:
        n = counts.get(label, 0)
        print(f"  {label:<9}: {n:>5}  ({n / total:.1%})")
    print(f"  Average compound score: {df['compound'].mean():+.3f}")

    print("  Most positive:")
    for _, r in df.nlargest(2, "compound").iterrows():
        print(f"    [{r['compound']:+.2f}] {r['comment'][:100]}")
    print("  Most negative:")
    for _, r in df.nsmallest(2, "compound").iterrows():
        print(f"    [{r['compound']:+.2f}] {r['comment'][:100]}")


def print_brand_report(df: pd.DataFrame, budget: float) -> None:
    """Buying signals, where people bought/found the brand, and a suggested budget split."""
    total = len(df)
    print("\n===== BRAND REACH & BUYING SIGNALS =====")
    print(f"Say they bought it : {int(df['purchased'].sum()):>5}  ({df['purchased'].mean():.1%})")
    print(f"Want to buy / ask  : {int(df['intent'].sum()):>5}  ({df['intent'].mean():.1%})")
    print(f"Show-off comments  : {int(df['show_off'].sum()):>5}  (bought it and happy about it)")

    for title, col, skip in [("SOURCE (where they bought / found the brand)", "source", ("Not mentioned",)),
                             ("VIDEO", "video_id", ()),
                             ("LOCATION mentioned", "location", ("Not mentioned",))]:
        table = summarize_group(df, col, skip=skip)
        if table.empty:
            print(f"\n--- {title}: nothing mentioned in these comments ---")
            continue
        if col != "location":
            table = recommend_budget(table, budget)
        print(f"\n--- {title} ---")
        print(table.to_string(index=False))


def plot_results(df: pd.DataFrame, out_file: str = "sentiment_chart.png") -> None:
    order = ["Positive", "Neutral", "Negative"]
    colors = ["#4CAF50", "#9E9E9E", "#F44336"]

    # Percentage of each sentiment per video
    pct = (
        df.groupby("video_id")["sentiment"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
        * 100
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    overall = df["sentiment"].value_counts().reindex(order, fill_value=0)
    axes[0].pie(overall, labels=order, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Overall Sentiment (all videos)")

    pct.plot(kind="bar", stacked=True, color=colors, ax=axes[1], width=0.7)
    axes[1].set_title("Sentiment per Video")
    axes[1].set_ylabel("% of comments")
    axes[1].set_xlabel("Video ID")
    axes[1].legend(loc="upper right")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"\nChart saved to {out_file}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube comment sentiment analysis (any links)")
    parser.add_argument("urls", nargs="*", help="One or more YouTube links or video IDs")
    parser.add_argument("--file", help="Text file with one link per line")
    parser.add_argument("--limit", type=int, default=1000, help="Max comments per video (default 1000)")
    parser.add_argument("--csv", default="comments_sentiment.csv", help="Output CSV file")
    parser.add_argument("--budget", type=float, default=1_000_000, help="Ad budget in Rs. to split (default 1,000,000)")
    args = parser.parse_args()

    links = get_links(args)
    if not links:
        print("No links provided.")
        return

    all_results = []
    seen = set()
    for link in links:
        video_id = extract_video_id(link)
        if not video_id:
            print(f"Skipping (not a valid YouTube link): {link}")
            continue
        if video_id in seen:
            continue
        seen.add(video_id)

        print(f"\nScraping up to {args.limit} comments from {video_id}...")
        try:
            comments = scrape_comments(video_id, args.limit)
        except Exception as e:
            print(f"  Failed to scrape {video_id}: {e}")
            continue

        if not comments:
            print("  No comments found (comments may be disabled).")
            continue

        df = analyze(comments, video_id)
        if df.empty:
            continue
        all_results.append(df)
        print_report(df, f"VIDEO {video_id}")

    if not all_results:
        print("\nNothing to analyze.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(args.csv, index=False, encoding="utf-8-sig")
    print(f"\nAll results saved to {args.csv}")

    if len(all_results) > 1:
        print_report(combined, "OVERALL (ALL VIDEOS)")

    print_brand_report(combined, args.budget)

    plot_results(combined)


if __name__ == "__main__":
    main()