"""
Brand pipeline: brand name -> videos -> comments -> sentiment + buying signals + topics -> ONE CSV.
"""

import os
import re
from itertools import islice

import pandas as pd
from youtube_comment_downloader import SORT_BY_POPULAR, SORT_BY_RECENT, YoutubeCommentDownloader

from topics import add_comment_types
from youtube_search import search_videos
from youtube_sentiment import analyze, extract_video_id


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "brand"


def fetch_video_comments(video_id: str, limit: int, recent: bool = False) -> list[dict]:
    downloader = YoutubeCommentDownloader()
    sort = SORT_BY_RECENT if recent else SORT_BY_POPULAR
    comments = downloader.get_comments(video_id, sort_by=sort)
    return [
        {"text": c["text"], "votes": c.get("votes", "0"), "replies": c.get("replies", 0)}
        for c in islice(comments, limit)
    ]


def collect_brand_data(brand: str, niche: str = "", max_videos: int = 10, per_video: int = 300,
                       total_comments: int = 2000, api_key: str = "", strict: bool = True,
                       recent: bool = False, extra_links: list[str] | None = None, progress=None):
    """Returns (comments_df, videos_df, notes). `progress(fraction, text)` is optional."""
    say = progress or (lambda frac, text: None)
    notes = []

    say(0.0, f"Searching YouTube for {brand}...")
    videos, mode, search_notes = search_videos(brand, niche, max_videos, api_key, strict)
    notes += search_notes
    notes.append(f"Found {len(videos)} video(s) using {mode}.")

    # Videos the user added by hand
    known = {v["video_id"] for v in videos}
    for link in extra_links or []:
        vid = extract_video_id(link)
        if vid and vid not in known:
            videos.append({"video_id": vid, "title": f"(added by you) {vid}", "channel": "", "published": "",
                           "views": 0, "likes": 0, "comment_count": 0})
            known.add(vid)
        elif not vid:
            notes.append(f"Skipped invalid link: {link}")

    if not videos:
        raise RuntimeError(
            "No videos found. Try turning off 'only videos that mention the brand', "
            "check the spelling, or add video links by hand."
        )

    frames, kept, used = [], [], 0
    for i, video in enumerate(videos):
        remaining = total_comments - used
        if remaining <= 0:
            notes.append("Reached the total comment limit.")
            break
        say(0.05 + 0.9 * i / len(videos), f"Reading comments: {video['title'][:55]}")
        try:
            comments = fetch_video_comments(video["video_id"], min(per_video, remaining), recent)
        except Exception as e:
            notes.append(f"Could not read {video['video_id']}: {e}")
            continue
        if not comments:
            notes.append(f"No comments on '{video['title'][:50]}' (comments may be off).")
            continue
        df = analyze(comments, video["video_id"])
        if df.empty:
            continue
        df["video_title"] = video["title"]
        df["channel"] = video["channel"]
        df["video_views"] = video.get("views", 0)
        frames.append(df)
        used += len(df)
        kept.append({**{k: video.get(k, "") for k in
                        ("video_id", "title", "channel", "published", "views", "likes", "comment_count")},
                     "comments_scraped": len(df)})

    say(1.0, "Done")
    if not frames:
        return pd.DataFrame(), pd.DataFrame(), notes

    comments_df = add_comment_types(pd.concat(frames, ignore_index=True))
    comments_df.insert(0, "niche", niche)
    comments_df.insert(0, "brand", brand)
    return comments_df, pd.DataFrame(kept), notes


def save_csv(df: pd.DataFrame, brand: str, folder: str = "data") -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{slugify(brand)}_comments.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path