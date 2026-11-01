"""Reads API keys from environment variables or from a local .env file (never commit .env to GitHub)."""

import os
import re


def load_env(path: str = ".env") -> dict:
    values = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_key(name: str) -> str:
    return os.environ.get(name) or load_env().get(name, "")


def scrub(text, *secrets) -> str:
    """Hide API keys in any message (error messages from the network can contain the full web address with the key)."""
    text = str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return re.sub(r"([?&](?:key|api_key)=)[^&\s\"')]+", r"\1***", text)