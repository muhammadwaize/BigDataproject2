"""
Checks your Gemini and YouTube API keys WITHOUT starting the app.
It reads the keys from your .env file (or environment) and never prints them in full.

Run (from the project folder):
    python test_keys.py
"""

import re
import time

import requests

from config import get_key

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
VERTEX_URL = "https://aiplatform.googleapis.com/v1/publishers/google/models"


def mask(key: str) -> str:
    return f"{key[:4]}...{key[-3:]}  (length {len(key)})" if key else "(missing)"


def error_message(r: requests.Response) -> str:
    try:
        return r.json()["error"]["message"][:160]
    except Exception:
        return r.text[:160]


def check_gemini(key: str) -> bool:
    print("\n=== Gemini key ===")
    print("Found:", mask(key))
    if not key:
        print("No GEMINI_API_KEY found. Check that .env is saved, is in this folder, and the name is exactly GEMINI_API_KEY.")
        return True
    if any(c in key for c in " \"'\t"):
        print("WARNING: the key contains a space or quote mark. Remove it from the .env file.")

    methods = [
        ("x-goog-api-key header (what the app uses)", {"headers": {"x-goog-api-key": key}}),
        ("?key= in the web address", {"params": {"key": key}}),
        ("Authorization: Bearer header", {"headers": {"Authorization": f"Bearer {key}"}}),
    ]
    working = None
    for label, kwargs in methods:
        params = {"pageSize": 50, **kwargs.pop("params", {})}
        r = requests.get(f"{GEMINI_URL}/models", params=params, timeout=30, **kwargs)
        print(f"  {label:<45} HTTP {r.status_code}  {'WORKS' if r.status_code == 200 else 'failed'}")
        if r.status_code == 200 and working is None:
            working = (label, kwargs, r.json())
        elif r.status_code != 200:
            print("     ", error_message(r))

    if not working:
        print("\nResult: the Gemini API rejected the key with every method.")
        return False

    label, kwargs, data = working
    flash = [m["name"].replace("models/", "") for m in data.get("models", [])
             if "flash" in m.get("name", "") and "generateContent" in m.get("supportedGenerationMethods", [])
             and not any(x in m["name"] for x in ("lite", "image", "tts", "live", "audio"))]
    flash = sorted(flash, key=lambda n: [int(x) for x in re.findall(r"\d+", n)])
    print(f"\nKey is valid using: {label}")
    print("Flash models available:", ", ".join(flash[-4:]) or "none listed")

    print("\nSpeed test (one tiny question to each of the 3 newest Flash models, 45 s limit each):")
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK"}]}]}
    fastest = None
    for model in reversed(flash[-3:]):
        start = time.time()
        try:
            r = requests.post(f"{GEMINI_URL}/models/{model}:generateContent", json=body, timeout=45, **kwargs)
            secs = time.time() - start
            if r.status_code == 200:
                print(f"  {model:<22} answered in {secs:4.1f} s")
                if fastest is None or secs < fastest[1]:
                    fastest = (model, secs)
            else:
                print(f"  {model:<22} HTTP {r.status_code}: {error_message(r)[:90]}")
        except requests.exceptions.Timeout:
            print(f"  {model:<22} NO ANSWER within 45 s (too slow or busy right now)")
        except requests.exceptions.RequestException as e:
            print(f"  {model:<22} network problem: {str(e)[:90]}")
    if fastest:
        print(f"\nFastest working model: {fastest[0]}  ->  use it by writing GEMINI_MODEL={fastest[0]} in your .env file.")
    else:
        print("\nNo model answered. Your key is valid, so this is usually Google being busy: wait a few minutes and run the test again.")
    return True


def check_vertex(key: str) -> None:
    """Some new AQ. keys are Vertex AI 'express mode' keys, which use a different Google address."""
    print("\n=== Same key tried on Vertex AI express mode ===")
    body = {"contents": [{"role": "user", "parts": [{"text": "Reply with the single word OK"}]}]}
    r = requests.post(f"{VERTEX_URL}/gemini-3.8-flash:generateContent", params={"key": key}, json=body, timeout=60)
    print(f"  Test question                                 HTTP {r.status_code}  {'WORKS' if r.status_code == 200 else 'failed'}")
    if r.status_code == 200:
        print("  Your key is a Vertex AI express key. The app will use it automatically.")
    else:
        print("     ", error_message(r))
        print("\nResult: neither service accepts this key. Create a fresh key at aistudio.google.com "
              "(Get API key -> Create API key), copy it with the copy button, and run this test again.")


def check_youtube(key: str) -> None:
    print("\n=== YouTube key ===")
    print("Found:", mask(key))
    if not key:
        print("No YOUTUBE_API_KEY found (that is fine, the app will use yt-dlp instead).")
        return
    try:
        r = requests.get("https://www.googleapis.com/youtube/v3/videos",
                         params={"part": "id", "id": "dQw4w9WgXcQ", "key": key}, timeout=30)
    except requests.exceptions.RequestException as e:
        print("  Could not reach YouTube:", str(e)[:100])
        return
    print(f"  Test request                                  HTTP {r.status_code}  {'WORKS' if r.status_code == 200 else 'failed'}")
    if r.status_code != 200:
        print("     ", error_message(r))
        print("  If it says the API is not enabled, enable 'YouTube Data API v3' in the same Google Cloud project as the key.")


if __name__ == "__main__":
    gemini_key = get_key("GEMINI_API_KEY")
    if not check_gemini(gemini_key):
        check_vertex(gemini_key)
    check_youtube(get_key("YOUTUBE_API_KEY"))
    print()