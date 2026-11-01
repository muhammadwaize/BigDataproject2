"""
What kinds of comments are there, and what do people like or dislike?

* comment_type : Praise, Complaint, Question, Suggestion, Bought it, Wants to buy, Other
* topics       : price, quality, design, performance, delivery, service/trust, durability/comfort, availability
Keyword based, works with English and Roman Urdu. Add or change words in TOPICS below.
"""

import re
from collections import Counter

import pandas as pd

TYPE_ORDER = ["Praise", "Complaint", "Question", "Suggestion", "Bought it", "Wants to buy", "Other"]

_question_re = re.compile(
    r"\?|^\s*(?:how|what|why|when|where|which|who|can|could|does|do|is|are|kya|kaise|kab|kahan|kyun|kyu|kis)\b",
    re.IGNORECASE,
)
_complaint_re = re.compile(
    r"\b(?:problem|issue|worst|scam|fake|fraud|broken|waste|disappoint\w*|poor|not\s+working|refund|"
    r"dhoka|bekar|kharab|ghatiya|overpriced)\b",
    re.IGNORECASE,
)
_suggestion_re = re.compile(
    r"\b(?:should|please\s+(?:add|make|review|do)|plz|kindly|suggest\w*|chahiye|make\s+a\s+video)\b",
    re.IGNORECASE,
)

TOPICS = {
    "Price & value": r"\b(?:price|cost|expensive|cheap|afford\w*|worth|discount|value\s+for\s+money|mehng\w*|mehenga|sasta|qeemat|paisa|paisay)\b",
    "Quality": r"\b(?:quality|build|material|premium|original|genuine|asli|authentic)\b",
    "Design & looks": r"\b(?:design|looks?|looking|style|stylish|colou?rs?|beautiful|sleek|khoobsurat|pyara)\b",
    "Performance & features": r"\b(?:performance|speed|fast|feature\w*|battery|camera|sound|taste|flavou?r|smell|power\w*|works?|working)\b",
    "Delivery & packaging": r"\b(?:delivery|deliver\w*|shipping|shipped|packag\w*|parcel|courier|arrived|late)\b",
    "Service & trust": r"\b(?:customer\s+service|support|refund|return|warranty|scam|fake|trust\w*|service)\b",
    "Durability & comfort": r"\b(?:durab\w*|lasts?|lasting|comfort\w*|fit|fits|size|soft)\b",
    "Availability": r"\b(?:available|availability|in\s+stock|out\s+of\s+stock|stock|milta|milega)\b",
}
_topic_res = {name: re.compile(p, re.IGNORECASE) for name, p in TOPICS.items()}

_STOPWORDS = set("""
the and for that this with you your are was were have has had but not all can will just from they them their
there what when who how why which about like very more much really also been than then them its out one get got
too our his her she him did does done into over only any some such even still ever make made see saw video videos
bhai yaar hai hain kya hai ke ki ka ko se ne mein me aur ya bhi tha thi wala wali hum aap main mai tum nahi nahin
""".split())


def comment_type(text: str, compound: float, purchased: bool, intent: bool) -> str:
    if purchased:
        return "Bought it"
    if intent:
        return "Wants to buy"
    if _question_re.search(text):
        return "Question"
    if _complaint_re.search(text) or compound <= -0.3:
        return "Complaint"
    if _suggestion_re.search(text):
        return "Suggestion"
    if compound >= 0.4:
        return "Praise"
    return "Other"


def topic_list(text: str) -> str:
    return "; ".join(name for name, rx in _topic_res.items() if rx.search(text))


def add_comment_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["comment_type"] = [
        comment_type(t, c, p, i)
        for t, c, p, i in zip(df["comment"].astype(str), df["compound"], df["purchased"], df["intent"])
    ]
    df["topics"] = df["comment"].astype(str).apply(topic_list).fillna("")
    return df


def type_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["comment_type"].value_counts().reindex(TYPE_ORDER, fill_value=0)
    out = counts.rename_axis("comment_type").reset_index(name="comments")
    out["share_%"] = (out["comments"] / max(len(df), 1) * 100).round(1)
    return out


def topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["topic", "comments", "positive_%", "negative_%", "avg_sentiment", "verdict"]
    rows = []
    for name in TOPICS:
        sub = df[df["topics"].str.contains(name, regex=False)]
        if sub.empty:
            continue
        avg = sub["compound"].mean()
        rows.append({
            "topic": name,
            "comments": len(sub),
            "positive_%": round((sub["compound"] >= 0.05).mean() * 100, 1),
            "negative_%": round((sub["compound"] <= -0.05).mean() * 100, 1),
            "avg_sentiment": round(avg, 3),
            "verdict": "Liked" if avg >= 0.2 else ("Disliked" if avg <= -0.1 else "Mixed"),
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values("comments", ascending=False).reset_index(drop=True)


def top_words(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    counter = Counter()
    for text in df["comment"].astype(str):
        for w in re.findall(r"[a-zA-Z]{3,}", text.lower()):
            if w not in _STOPWORDS:
                counter[w] += 1
    return pd.DataFrame(counter.most_common(n), columns=["word", "count"])