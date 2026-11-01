"""
Gemini agents
-------------
Two agents run AT THE SAME TIME on one brand's comments:

  1. Insights agent  - what people like / dislike, questions they ask, what to do about it
  2. Marketing agent - which platforms the brand should invest in, and how the budget should be split
                       (it can also search the web to look at how the brand currently markets itself)

They talk to Google directly over HTTPS (needs only the `requests` package).
Key: https://aistudio.google.com  ->  Get API key  (keep it out of your code and out of GitHub).
If the direct Gemini API rejects the key, the app automatically tries Google's official Python library
(google-genai, if installed) and then the Vertex AI express endpoint with the same key.
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from config import scrub

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
VERTEX_ROOT = "https://aiplatform.googleapis.com/v1"      # Vertex AI "express mode" keys (start with AQ.)
BACKENDS = ["Gemini API", "Gemini SDK", "Vertex AI express"]
DEFAULT_MODEL = "gemini-3.8-flash"


class GeminiError(RuntimeError):
    pass


class GeminiTimeout(GeminiError):
    pass


# ---------- Low level Gemini calls ----------
def _headers(key: str) -> dict:
    return {"x-goog-api-key": key, "Content-Type": "application/json"}


def _error_text(r: requests.Response) -> str:
    try:
        return f"HTTP {r.status_code}: {r.json()['error']['message']}"
    except Exception:
        return f"HTTP {r.status_code}: {r.text[:200]}"


def _version_key(name: str) -> tuple:
    return tuple(int(n) for n in re.findall(r"\d+", name))


def _flash_models(names) -> list:
    ok = [n for n in names if "flash" in n and not any(x in n for x in ("lite", "image", "tts", "live", "audio"))]
    return sorted(ok, key=_version_key)


def pick_model(key: str) -> str:
    """If the chosen model name no longer exists, pick the newest available Gemini Flash model."""
    r = requests.get(f"{API_ROOT}/models", headers=_headers(key), params={"pageSize": 200}, timeout=30)
    if r.status_code != 200:
        raise GeminiError(_error_text(r))
    names = [m.get("name", "").replace("models/", "") for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = _flash_models(names)
    if not flash:
        raise GeminiError("No Gemini Flash model is available for this key.")
    return flash[-1]


def _generate(backend: str, model: str, key: str, body: dict, retries: int = 3,
              timeout: int = 120, max_timeouts: int = 2) -> requests.Response:
    if backend == "Gemini API":
        url = f"{API_ROOT}/models/{model}:generateContent"
        kwargs = {"headers": _headers(key)}
    else:
        url = f"{VERTEX_ROOT}/publishers/google/models/{model}:generateContent"
        kwargs = {"headers": {"Content-Type": "application/json"}, "params": {"key": key}}
    r, timeouts = None, 0
    for attempt in range(retries):
        try:
            r = requests.post(url, json=body, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout:
            timeouts += 1
            if timeouts >= max_timeouts or attempt == retries - 1:
                raise GeminiTimeout(
                    f"Google did not answer in time using {model}. Try again in a minute, or set a different model "
                    "in your .env file (for example GEMINI_MODEL=gemini-3.6-flash)."
                )
            continue
        except requests.exceptions.ConnectionError as e:
            raise GeminiError(f"Could not reach Google: {scrub(e, key)[:150]}")
        if r.status_code in (429, 500, 503) and attempt < retries - 1:
            time.sleep(3 * (attempt + 1))   # busy or rate limited: wait and try again
            continue
        break
    return r


def _is_auth_error(r: requests.Response) -> bool:
    return r.status_code == 401 or "API key not valid" in r.text or "API_KEY_INVALID" in r.text


class _AuthRejected(Exception):
    pass


def _sdk_call(prompt, system, key, model, use_search, temperature):
    """Google's official google-genai library. Returns None if the library is not installed."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None
    client = genai.Client(api_key=key)

    def run(model_name, grounded):
        cfg = {"system_instruction": system, "temperature": temperature}
        if grounded:
            cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        else:
            cfg["response_mime_type"] = "application/json"
        return client.models.generate_content(model=model_name, contents=prompt,
                                              config=types.GenerateContentConfig(**cfg))

    use_model, grounded, switched = model, use_search, False
    for _ in range(4):
        try:
            resp = run(use_model, grounded)
            text = (resp.text or "").strip()
            if not text:
                raise GeminiError("Gemini returned an empty answer.")
            return text, {"model": use_model, "grounded": grounded, "backend": "Gemini SDK"}
        except GeminiError:
            raise
        except Exception as e:
            code, msg = getattr(e, "code", None), scrub(e, key)
            if code == 401 or "API key not valid" in msg or "API_KEY_INVALID" in msg:
                raise _AuthRejected(msg)
            if code == 404 and not switched:               # model name gone: use an available Flash model
                flash = _flash_models(m.name.replace("models/", "") for m in client.models.list())
                if not flash:
                    raise GeminiError(f"Gemini SDK: {msg[:300]}")
                use_model, switched = flash[-1], True
                continue
            if grounded and code in (400, 403, 429):       # web search not allowed: continue without it
                grounded = False
                continue
            raise GeminiError(f"Gemini SDK: {msg[:300]}")
    raise GeminiError("Gemini SDK: could not get an answer.")


def call_gemini(prompt: str, system: str, key: str, model: str = DEFAULT_MODEL,
                use_search: bool = False, temperature: float = 0.3):
    """Returns (text, info). info = {"model": ..., "grounded": True/False, "backend": ...}."""
    if not key:
        raise GeminiError("No Gemini API key was given.")

    def make_body(grounded: bool) -> dict:
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        if grounded:
            body["tools"] = [{"google_search": {}}]
        else:
            body["generationConfig"]["responseMimeType"] = "application/json"
        return body

    last_msg, tried = "", []
    for backend in BACKENDS:
        if backend == "Gemini SDK":
            try:
                result = _sdk_call(prompt, system, key, model, use_search, temperature)
            except _AuthRejected as e:
                last_msg = str(e)
                tried.append(backend)
                continue
            if result is None:          # library not installed: skip this route
                continue
            return result

        tried.append(backend)
        use_model, grounded = model, use_search
        try:
            r = _generate(backend, use_model, key, make_body(grounded),
                          timeout=75 if grounded else 120, max_timeouts=1 if grounded else 2)
        except GeminiTimeout:
            if not grounded:
                raise
            grounded = False                    # web search is too slow right now: answer from the comments only
            r = _generate(backend, use_model, key, make_body(False))
        if _is_auth_error(r):           # this key does not belong to this service: try the next route
            last_msg = _error_text(r)
            continue
        if r.status_code == 404 and backend == "Gemini API":     # model name not found
            use_model = pick_model(key)
            r = _generate(backend, use_model, key, make_body(grounded))
        if grounded and r.status_code in (400, 403, 429):        # web search not allowed: continue without it
            grounded = False
            r = _generate(backend, use_model, key, make_body(False))
        if r.status_code != 200:
            raise GeminiError(f"{backend}: {_error_text(r)}")

        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no answer returned")
            raise GeminiError(f"Gemini returned no answer ({reason}).")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        if not text:
            raise GeminiError("Gemini returned an empty answer.")
        return text, {"model": use_model, "grounded": grounded, "backend": backend}

    raise GeminiError(f"{last_msg} (tried: {', '.join(tried)})")


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    raise GeminiError("Could not read the agent's answer. Please run the agents again.")


# ---------- What the agents are told about the brand ----------
def _clip(text: str, n: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n - 1] + "..."


def _records(df: pd.DataFrame, n: int = 12) -> list:
    if df is None or df.empty:
        return []
    return json.loads(df.head(n).to_json(orient="records"))


def _sample(df: pd.DataFrame, n: int = 20) -> list:
    out = []
    for r in df.sort_values("likes", ascending=False).head(n).itertuples():
        out.append({"text": _clip(r.comment), "likes": int(r.likes), "video": _clip(getattr(r, "video_title", ""), 60)})
    return out


def build_context(brand, niche, budget, comments, videos, type_table, topic_table,
                  source_table, video_table, location_table, discovery_table=None, bought_table=None,
                  coverage=None, language_table=None, ai_mentions=None) -> dict:
    return {
        "brand": brand,
        "niche": niche or "not given",
        "budget_rs": float(budget),
        "totals": {
            "comments": int(len(comments)),
            "videos": int(comments["video_id"].nunique()),
            "positive_%": round(float((comments["compound"] >= 0.05).mean() * 100), 1),
            "negative_%": round(float((comments["compound"] <= -0.05).mean() * 100), 1),
            "say_they_bought": int(comments["purchased"].sum()),
            "want_to_buy": int(comments["intent"].sum()),
            "show_off_comments": int(comments["show_off"].sum()),
        },
        "comment_types": _records(type_table),
        "topics": _records(topic_table),
        "where_people_bought_or_found_brand": _records(source_table),
        "videos_ranked_by_buying_interest": _records(video_table),
        "video_stats": _records(videos[["title", "channel", "views", "likes", "comment_count"]]
                                .sort_values("views", ascending=False), 12) if videos is not None and not videos.empty else [],
        "locations_mentioned": _records(location_table),
        "how_commenters_say_they_found_the_brand": _records(discovery_table),
        "where_commenters_say_they_bought": _records(bought_table),
        "evidence_coverage": coverage or {},
        "comment_languages": _records(language_table),
        "extra_source_mentions_found_by_ai_reader": (ai_mentions or [])[:25],
        "top_liked_positive_comments": _sample(comments[comments["compound"] >= 0.05]),
        "top_liked_negative_comments": _sample(comments[comments["compound"] <= -0.05]),
        "comments_saying_they_bought": _sample(comments[comments["purchased"]], 15),
        "comments_asking_to_buy": _sample(comments[comments["intent"]], 15),
    }


# ---------- The two agents ----------
INSIGHT_SYSTEM = (
    "You are a senior consumer-insights analyst. You ONLY discuss the brand '{brand}'; ignore anything about "
    "other brands. Use ONLY the evidence given to you. Never invent numbers, quotes or facts. If the evidence "
    "is thin, say so. Answer in simple English. Reply with JSON only."
)

INSIGHT_PROMPT = """Brand: {brand}   Niche: {niche}

EVIDENCE (collected from real YouTube comments about this brand):
{data}

Write a JSON object with exactly these keys:
{{
  "summary": "3-4 sentences: what the audience thinks of {brand} overall",
  "what_people_like": ["short bullet points, each tied to the evidence"],
  "what_people_dislike": ["short bullet points, each tied to the evidence"],
  "audience_questions": ["questions or doubts people keep raising"],
  "recommendations": [{{"action": "what {brand} should do", "why": "the evidence behind it"}}]
}}
Give 3-6 items per list."""

MARKETING_SYSTEM = (
    "You are a media planner who decides which advertising platforms a brand should invest in to win more "
    "customers. You ONLY discuss the brand '{brand}'. Base your plan on the comment evidence provided"
    "{search_note}. Never invent statistics. Be honest about confidence. Reply with JSON only."
)

MARKETING_PROMPT = """Brand: {brand}   Niche: {niche}   Total budget: Rs. {budget:,.0f}

EVIDENCE (collected from real YouTube comments about this brand):
{data}

Candidate platforms: YouTube (reviewer / influencer sponsorships and ads), Instagram, Facebook, TikTok,
Google Search ads, Online marketplaces (Daraz, Amazon...), WhatsApp / community groups, Offline stores.
You may add or drop platforms if the evidence supports it.

Remember: some people bought elsewhere but still show off on YouTube; comments saying where they bought or
found the brand show where its customers really come from. This is only what commenters SAY: it is not real
traffic data, and 'evidence_coverage' shows how few comments name a source at all. Mention that limit in
"risks" and lower your confidence if coverage is thin.

Write a JSON object with exactly these keys:
{{
  "summary": "3-4 sentences explaining the plan",
  "allocation": [{{"platform": "name", "percent": number, "reason": "evidence-based reason"}}],
  "brand_marketing_observations": ["how {brand} currently markets itself / where it is active"],
  "risks": ["what could go wrong or what the data cannot tell us"],
  "next_steps": ["practical next actions"],
  "confidence": "low, medium or high"
}}
The allocation percentages must add up to 100."""


def normalize_allocation(allocation) -> list:
    rows = []
    for a in allocation or []:
        try:
            percent = float(a.get("percent", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        name = str(a.get("platform", "")).strip()
        if name and percent > 0:
            rows.append({"platform": name, "percent": percent, "reason": str(a.get("reason", ""))})
    total = sum(r["percent"] for r in rows)
    if total <= 0:
        return []
    for r in rows:
        r["percent"] = round(r["percent"] / total * 100, 1)
    return sorted(rows, key=lambda r: -r["percent"])


def _run_insights(ctx: dict, key: str, model: str) -> dict:
    system = INSIGHT_SYSTEM.format(brand=ctx["brand"])
    prompt = INSIGHT_PROMPT.format(brand=ctx["brand"], niche=ctx["niche"], data=json.dumps(ctx, default=str))
    text, info = call_gemini(prompt, system, key, model, use_search=False)
    result = parse_json(text)
    result["_model"] = info["model"]
    result["_backend"] = info["backend"]
    return result


def _run_marketing(ctx: dict, key: str, model: str, use_search: bool) -> dict:
    note = (" and, where useful, what you find by searching the web about how the brand currently markets itself"
            if use_search else "")
    system = MARKETING_SYSTEM.format(brand=ctx["brand"], search_note=note)
    prompt = MARKETING_PROMPT.format(brand=ctx["brand"], niche=ctx["niche"], budget=ctx["budget_rs"],
                                     data=json.dumps(ctx, default=str))
    text, info = call_gemini(prompt, system, key, model, use_search=use_search)
    result = parse_json(text)
    result["allocation"] = normalize_allocation(result.get("allocation"))
    result["_model"] = info["model"]
    result["_grounded"] = info["grounded"]
    result["_backend"] = info["backend"]
    return result


def run_agents(ctx: dict, key: str, model: str = DEFAULT_MODEL, use_search: bool = True) -> dict:
    """Run both agents at the same time. Returns {"insights": {...}, "marketing": {...}};
    an agent that failed has {"error": "..."} instead."""
    jobs = {
        "insights": lambda: _run_insights(ctx, key, model),
        "marketing": lambda: _run_marketing(ctx, key, model, use_search),
    }
    out = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, future in futures.items():
            try:
                out[name] = future.result()
            except Exception as e:
                out[name] = {"error": scrub(e, key)}
    return out


# ---------- Source finder agent (optional, one extra Gemini call) ----------
SOURCE_SYSTEM = (
    "You are a careful evidence extractor. You ONLY look at comments about the brand '{brand}'. Report only what "
    "a comment CLEARLY states about where the person FOUND (discovered) the brand or WHERE they BOUGHT it. "
    "Never guess or infer. Comments may be English, Urdu or Roman Urdu. Reply with JSON only."
)

SOURCE_PROMPT = """Brand: {brand}

Numbered comments:
{comments}

Write a JSON object with exactly this shape:
{{
  "mentions": [
    {{"id": 0, "discovered_on": "short name or null", "bought_from": "short name or null",
      "how_found": "friend, advert, influencer, search, social media, store, other or null",
      "evidence": "up to 10 words copied from the comment"}}
  ]
}}
Include ONLY comments that clearly say where the person found the brand or where they bought it. Use short names
such as Instagram, TikTok, Daraz, Google, a friend, a store. Skip every other comment."""


def run_source_agent(brand: str, comments: pd.DataFrame, key: str, model: str = DEFAULT_MODEL, limit: int = 150) -> dict:
    """Reads up to `limit` of the most-liked comments and returns the source mentions it finds."""
    try:
        sample = comments.sort_values("weight", ascending=False).head(limit).reset_index(drop=True)
        numbered = [{"id": int(i), "text": _clip(t, 180)} for i, t in enumerate(sample["comment"])]
        system = SOURCE_SYSTEM.format(brand=brand)
        prompt = SOURCE_PROMPT.format(brand=brand, comments=json.dumps(numbered, ensure_ascii=False))
        text, info = call_gemini(prompt, system, key, model, use_search=False, temperature=0.1)
        found = parse_json(text).get("mentions", []) or []

        def clean(value):
            value = str(value).strip() if value is not None else ""
            return "" if value.lower() in ("", "null", "none", "n/a") else value

        rows = []
        for m in found:
            try:
                i = int(m.get("id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if not 0 <= i < len(sample):
                continue
            row = {"comment": _clip(sample.loc[i, "comment"], 200), "likes": int(sample.loc[i, "likes"]),
                   "found_on": clean(m.get("discovered_on")), "bought_from": clean(m.get("bought_from")),
                   "how_found": clean(m.get("how_found")), "evidence": clean(m.get("evidence"))}
            if row["found_on"] or row["bought_from"] or row["how_found"]:
                rows.append(row)
        return {"mentions": rows, "read": int(len(sample)), "_model": info["model"], "_backend": info["backend"]}
    except Exception as e:
        return {"error": scrub(e, key)}