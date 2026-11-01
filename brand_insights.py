"""
Brand reach + ad budget optimizer (works on the comments scraped in Project 1)
------------------------------------------------------------------------------
Scans every comment for:
  * BUYING signals    - "bought it", "ordered", "le liya", "mil gaya" ...
  * BUYING INTENT     - "where to buy?", "price?", "kahan se milega" ...
  * SHOW-OFF comments - people who say they bought it AND are happy about it
                        (even if they bought it somewhere else, they still prove the
                        brand's reach and word-of-mouth value)
  * SOURCE mentions   - platforms / shops named in the comment (Daraz, Instagram, store ...)
  * DISCOVERED ON     - where they SAW or FOUND the brand ("saw it on Instagram", "TikTok se aya")
  * BOUGHT FROM       - where they BOUGHT it ("ordered from Daraz")
  * HOW THEY FOUND IT - friend / family, an advert, an influencer or reviewer
  * LANGUAGE          - English, Roman Urdu or Urdu script (a rough hint about the audience)
  * LOCATION mentions - city or country named in the comment

IMPORTANT: this is what commenters SAY. It is not real traffic data (that is in YouTube Studio
analytics, which only the channel owner can see), and most comments name no source at all.

To track more places, add a line to SOURCES below (name -> regex of words).
Works with English and Roman Urdu comments. It is a keyword scanner, so it can
miss or misread some comments (for example sarcasm or "I never bought it").
"""

import re

import numpy as np
import pandas as pd

# ---------- Keyword lists (edit freely) ----------
PURCHASE_PATTERNS = [
    r"\b(?:bought|purchased|ordered)\b",
    r"\b(?:got\s+mine|just\s+got|finally\s+got|mine\s+(?:arrived|came|is\s+here))\b",
    r"\b(?:received|got)\s+my\s+(?:order|parcel|package)\b",
    r"\b(?:le\s+li(?:ya|a)?|kharid(?:a|i|e|li|liya)?|khareed(?:a|i|e)?|order\s+ki(?:ya|a)|"
    r"mangwa(?:ya|\s*liya|\s*li)|mil\s+gay(?:a|i))\b",
]

INTENT_PATTERNS = [
    r"\bwhere\s+(?:can\s+i|can\s+we|to|do\s+i)?\s*(?:buy|get|order|find)\b",
    r"\bhow\s+much\b",
    r"\bprice\b",
    r"\bcost\b",
    r"\b(?:want(?:\s+to)?|wanna|going\s+to|gonna|planning\s+to|will|should\s+i|can\s+i|need\s+to)\s+(?:buy|order|get)\b",
    r"\b(?:kahan|kidhar)\s+se\s+(?:mil|le|kharid|mangwa)\w*",
    r"\bkitne\s+ka\b",
    r"\b(?:order|mangwa\w*)\s+kaise\b",
    r"\bkaise\s+(?:milega|mangwa\w*|order|le)\b",
]

# Platforms and shops people mention: name -> regex
SOURCES = {
    "Daraz": r"\bdaraz\b",
    "Amazon": r"\bamazon\b",
    "AliExpress": r"\bali\s?express\b",
    "OLX": r"\bolx\b",
    "Instagram": r"\b(?:insta\s?gram|insta)\b",
    "Facebook": r"\b(?:facebook|fb)\b",
    "TikTok": r"\btik\s?tok\b",
    "WhatsApp": r"\bwhats\s?app\b",
    "Telegram": r"\btelegram\b",
    "Snapchat": r"\bsnap\s?chat\b",
    "Twitter/X": r"\btwitter\b",
    "LinkedIn": r"\blinked\s?in\b",
    "Official website/app": r"\b(?:website|web\s?site|official\s+(?:site|store|app))\b",
    "Physical store": r"\b(?:store|shop|outlet|mall|market|showroom|branch|franchise|dukaan|dukan|bazaar|bazar)\b",
    "Google search": r"\bgoogl(?:e|ed)\b",
}

# Words that say "I SAW / FOUND it here" and "I BOUGHT it here" (English + Roman Urdu)
DISCOVER_PATTERNS = [
    r"\b(?:saw|seen|watched|found|discovered|heard|noticed|came\s+across|came\s+from|coming\s+from|here\s+from|followed|follow)\b",
    r"\blink\s+in\s+(?:bio|description|profile)\b",
    r"\b(?:dekh[aei]?|dekh\s+kar|suna|suni|pata\s+chala|dhoond\w*|se\s+aa?y[aei]|se\s+aa?ye|aa?y[aei]\s+hun)\b",
]
BUY_CUE_PATTERNS = [
    r"\b(?:order(?:ed)?|ordering|buy(?:ing)?|bought|purchase\w*|got|khareed\w*|kharid\w*|mangw\w*|le\s+li\w*)\b",
]

# How people found the brand without naming a platform
WORD_OF_MOUTH_PATTERNS = [
    r"\b(?:friend|friends|cousin|brother|sister|colleague|relative|family|neighbou?r)\b.{0,30}\b(?:told|recommend\w*|suggest\w*|showed|gave|bought|use[sd]?)\b",
    r"\b(?:dost|doston|bhai|behan|cousin|rishtedar)\b.{0,30}\b(?:ne\s+(?:bataya|btaya|suggest|dikhaya|diya)|recommend\w*|use)\b",
    r"\b(?:recommended|suggested)\s+(?:to\s+me\s+)?by\s+(?:a\s+)?(?:friend|family|cousin|brother|sister)\b",
]
AD_PATTERNS = [
    r"\b(?:saw|seen|noticed|came\s+across|found)\b.{0,25}\b(?:ad|ads|advert\w*|advertisement|commercial)\b",
    r"\b(?:ad|ads|advert\w*|advertisement|commercial)\b.{0,25}\b(?:dekh\w*|saw|seen|aya|aaya|aayi)\b",
    r"\b(?:your|their|the)\s+(?:ad|ads|advert\w*|advertisement|commercial)\b",
]
INFLUENCER_PATTERNS = [
    r"\b(?:influencer|youtuber|vlogger|blogger|reviewer|content\s+creator)\b",
]

# Words that show a comment is written in Roman Urdu (Urdu typed in English letters)
ROMAN_URDU_STRONG = {
    "hai", "hain", "kya", "nahi", "nahin", "bhai", "yaar", "kahan", "kaise", "mujhe", "bohat", "bahut",
    "zabardast", "acha", "achha", "achi", "mera", "meri", "mere", "kitne", "kitna", "milega", "dekha",
    "bataya", "liya", "kiya", "hoga", "tha", "thi", "karo", "karen", "mein", "aap", "wala", "wali",
}

LOCATIONS = [
    "Lahore", "Karachi", "Islamabad", "Rawalpindi", "Faisalabad", "Multan", "Peshawar",
    "Quetta", "Sialkot", "Gujranwala", "Hyderabad", "Pakistan", "India", "UAE", "Dubai",
    "Saudi", "UK", "USA", "Canada", "Australia",
]

NOT_MENTIONED = "Not mentioned"

_purchase_re = re.compile("|".join(PURCHASE_PATTERNS), re.IGNORECASE)
_intent_re = re.compile("|".join(INTENT_PATTERNS), re.IGNORECASE)
_discover_re = re.compile("|".join(DISCOVER_PATTERNS), re.IGNORECASE)
_buy_cue_re = re.compile("|".join(BUY_CUE_PATTERNS), re.IGNORECASE)
_wom_re = re.compile("|".join(WORD_OF_MOUTH_PATTERNS), re.IGNORECASE)
_ad_re = re.compile("|".join(AD_PATTERNS), re.IGNORECASE)
_influencer_re = re.compile("|".join(INFLUENCER_PATTERNS), re.IGNORECASE)
_source_res = {name: re.compile(p, re.IGNORECASE) for name, p in SOURCES.items()}
_location_res = {loc: re.compile(rf"\b{re.escape(loc)}\b", re.IGNORECASE) for loc in LOCATIONS}
_urdu_script_re = re.compile(r"[\u0600-\u06FF]")


def parse_count(value) -> int:
    """Turn YouTube counts like '1.2K', '3M', '15' or '' into numbers."""
    s = str(value).strip().upper().replace(",", "")
    m = re.match(r"^([\d.]+)\s*([KM]?)$", s)
    if not m:
        return 0
    try:
        number = float(m.group(1))
    except ValueError:
        return 0
    return int(number * {"": 1, "K": 1_000, "M": 1_000_000}[m.group(2)])


def _first_match(text: str, patterns: dict) -> str:
    for name, rx in patterns.items():
        if rx.search(text):
            return name
    return NOT_MENTIONED


_CLAUSE_SPLIT = re.compile(r"[,.;!?\n]|\b(?:then|and|but|so|because|after|before|when|while|jab|phir|aur|lekin|magar|ke\s+baad)\b", re.IGNORECASE)


def source_roles(text: str, purchased: bool) -> tuple:
    """Returns (discovered_on, bought_from). The comment is cut into short clauses; a platform named in a
    clause with 'saw / found' words is where they DISCOVERED the brand, and one named in a clause with
    'bought / ordered' words is where they BOUGHT it."""
    discovered, bought = NOT_MENTIONED, NOT_MENTIONED
    for clause in _CLAUSE_SPLIT.split(text):
        if not clause or not clause.strip():
            continue
        platform = _first_match(clause, _source_res)
        if platform == NOT_MENTIONED:
            continue
        if purchased and bought == NOT_MENTIONED and _buy_cue_re.search(clause):
            bought = platform
        elif discovered == NOT_MENTIONED and _discover_re.search(clause):
            discovered = platform
    return discovered, bought


def detect_language(text: str) -> str:
    if _urdu_script_re.search(text):
        return "Urdu script"
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(1 for w in words if w in ROMAN_URDU_STRONG)
    return "Roman Urdu" if hits >= 2 or (hits == 1 and len(words) <= 6) else "English"


def tag_comments(df: pd.DataFrame) -> pd.DataFrame:
    """Add buying-signal, source, how-they-found-it, language, location and reach-weight columns.
    Needs the columns: comment, compound, likes, replies."""
    df = df.copy()
    text = df["comment"].astype(str)

    df["purchased"] = text.apply(lambda t: bool(_purchase_re.search(t)))
    df["intent"] = text.apply(lambda t: bool(_intent_re.search(t))) & ~df["purchased"]
    df["show_off"] = df["purchased"] & (df["compound"] >= 0.05)
    df["source"] = text.apply(lambda t: _first_match(t, _source_res))

    roles = [source_roles(t, p) for t, p in zip(text, df["purchased"])]
    df["discovered_on"] = [r[0] for r in roles]
    df["bought_from"] = [r[1] for r in roles]
    df["word_of_mouth"] = text.apply(lambda t: bool(_wom_re.search(t)))
    df["saw_ad"] = text.apply(lambda t: bool(_ad_re.search(t)))
    df["influencer"] = text.apply(lambda t: bool(_influencer_re.search(t)))
    df["language"] = text.apply(detect_language)

    df["location"] = text.apply(lambda t: _first_match(t, _location_res))
    # A comment with many likes/replies reaches more people; log keeps one viral comment from dominating
    df["weight"] = 1 + np.log1p(df["likes"] + df["replies"])
    return df


# ---------- Summaries ----------
def summarize_group(df: pd.DataFrame, by: str, skip=(NOT_MENTIONED,)) -> pd.DataFrame:
    """Summarize buying interest per source / video / location."""
    cols = [by, "mentions", "purchases", "intents", "show_offs", "avg_sentiment", "reach", "score"]
    sub = df[~df[by].isin(skip)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)

    sub["w_purchase"] = sub["weight"] * sub["purchased"]
    sub["w_intent"] = sub["weight"] * sub["intent"]
    g = sub.groupby(by, as_index=False).agg(
        mentions=("comment", "count"),
        purchases=("purchased", "sum"),
        intents=("intent", "sum"),
        show_offs=("show_off", "sum"),
        avg_sentiment=("compound", "mean"),
        reach=("weight", "sum"),
        w_purchase=("w_purchase", "sum"),
        w_intent=("w_intent", "sum"),
    )
    # Buying counts most, buying interest next, plain mentions least; happier audiences score higher
    g["score"] = (3 * g["w_purchase"] + 2 * g["w_intent"] + g["reach"]) * (1 + g["avg_sentiment"].clip(lower=0))
    g = g.drop(columns=["w_purchase", "w_intent"])
    return g.sort_values("score", ascending=False).round(3).reset_index(drop=True)[cols]


def discovery_summary(df: pd.DataFrame) -> pd.DataFrame:
    """How people say they found the brand: a platform, a friend, an advert or an influencer."""
    groups = []
    for name, g in df[df["discovered_on"] != NOT_MENTIONED].groupby("discovered_on"):
        groups.append((f"Saw it on {name}", g))
    for flag, label in [("word_of_mouth", "Friend / family told them"), ("saw_ad", "Saw an advert"),
                        ("influencer", "Influencer / reviewer")]:
        g = df[df[flag]]
        if len(g):
            groups.append((label, g))
    cols = ["how_they_found_it", "comments", "purchases", "avg_sentiment"]
    if not groups:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame([{
        "how_they_found_it": label, "comments": len(g), "purchases": int(g["purchased"].sum()),
        "avg_sentiment": round(float(g["compound"].mean()), 3),
    } for label, g in groups])
    return out.sort_values("comments", ascending=False).reset_index(drop=True)


def language_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("language").agg(comments=("comment", "count"), purchases=("purchased", "sum"),
                                   avg_sentiment=("compound", "mean")).reset_index()
    g["share_%"] = (g["comments"] / max(len(df), 1) * 100).round(1)
    g["avg_sentiment"] = g["avg_sentiment"].round(3)
    return g.sort_values("comments", ascending=False).reset_index(drop=True)


def coverage(df: pd.DataFrame) -> dict:
    """How much of the evidence is there? Most comments name no source, so this keeps conclusions honest."""
    n = len(df)
    named = (df["source"] != NOT_MENTIONED) | df["word_of_mouth"] | df["saw_ad"] | df["influencer"]
    pct = lambda k: round(k / n * 100, 1) if n else 0.0
    return {
        "comments": int(n),
        "named_any": int(named.sum()), "named_any_%": pct(int(named.sum())),
        "found_on_platform": int((df["discovered_on"] != NOT_MENTIONED).sum()),
        "bought_from_named": int((df["bought_from"] != NOT_MENTIONED).sum()),
        "say_bought": int(df["purchased"].sum()),
        "friend_ad_influencer": int((df["word_of_mouth"] | df["saw_ad"] | df["influencer"]).sum()),
    }


def recommend_budget(summary: pd.DataFrame, budget: float, min_share: float = 0.05) -> pd.DataFrame:
    """Split the budget by score. Every option first gets a small guaranteed share (min_share)
    so no source is cut to zero, and the rest is shared out in proportion to score."""
    out = summary.copy()
    if out.empty:
        return out
    n = len(out)
    floor = min(min_share, 1 / n)
    total = out["score"].sum()
    score_share = out["score"] / total if total > 0 else 1 / n
    out["share_%"] = ((floor + (1 - n * floor) * score_share) * 100).round(1)
    out["budget_rs"] = (out["share_%"] / 100 * budget).round(0)
    return out