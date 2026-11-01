"""
Tests your Gemini key with Google's OFFICIAL Python library (google-genai).
It reads the key from your .env file and never prints it.

Install once:   python -m pip install google-genai
Run:            python test_sdk.py
"""

import re

from config import get_key, scrub

key = get_key("GEMINI_API_KEY")
print("Key found:", f"{key[:4]}...{key[-3:]} (length {len(key)})" if key else "NO KEY FOUND in .env")
if not key:
    raise SystemExit(1)

try:
    from google import genai
except ImportError:
    raise SystemExit("google-genai is not installed. Run:  python -m pip install google-genai")


outcomes = []   # "ok", "rejected" or "network"


def try_client(label: str, **client_args) -> bool:
    print(f"\n--- {label} ---")
    try:
        client = genai.Client(api_key=key, **client_args)
        model = "gemini-3.8-flash"
        if not client_args:   # normal Gemini API: ask Google which Flash models this key can use
            names = [m.name.replace("models/", "") for m in client.models.list()]
            flash = sorted((n for n in names if "flash" in n and not any(x in n for x in ("lite", "image", "tts", "live", "audio"))),
                           key=lambda n: [int(x) for x in re.findall(r"\d+", n)])
            print("Key accepted. Flash models:", ", ".join(flash[-4:]) or "none listed")
            if flash:
                model = flash[-1]
        reply = client.models.generate_content(model=model, contents="Reply with the single word OK")
        print(f"WORKS. {model} answered: {(reply.text or '').strip()[:40]}")
        outcomes.append("ok")
        return True
    except Exception as e:
        message = scrub(e, key)
        print("Failed:", message[:300])
        rejected = any(w in message for w in ("API key", "401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID_ARGUMENT"))
        outcomes.append("rejected" if rejected else "network")
        return False


ok = try_client("Gemini API (normal)")
if not ok:
    ok = try_client("Vertex AI express mode", vertexai=True)

if ok:
    print("\nResult: the official library accepts your key.")
elif "rejected" in outcomes:
    print("\nResult: Google rejects this key even through its official library, so the key itself is the problem. "
          "Create a new one in AI Studio.")
else:
    print("\nResult: could not reach Google (internet, firewall or VPN problem). Check your connection and run it again.")