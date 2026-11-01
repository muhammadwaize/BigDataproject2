# Brand Comment Intelligence & Ad Optimizer

A Streamlit application that collects YouTube comments about a brand and turns them into an overview of customer sentiment, common topics, buying signals, brand discovery sources, and suggested advertising budget allocation.

The app combines keyword-based analysis with two Gemini-powered AI agents:
- **Insights agent:** summarizes what people like or dislike, common questions, and possible actions for the brand.
- **Marketing agent:** recommends platforms to consider for advertising and a suggested budget split. It can use Google Search grounding when enabled.

> **Important:** Results are estimates based on publicly available YouTube videos and comments. Keyword rules can misclassify sarcasm, negation, and context. Mentions of a platform in comments are not verified traffic or sales attribution.

## Features

- Search YouTube for videos related to a brand and optional niche.
- Use the YouTube Data API when a key is configured, with a `yt-dlp` search fallback.
- Collect comments from multiple videos, sorted by popularity or recency.
- Score comments as **Positive**, **Neutral**, or **Negative** using VADER sentiment analysis.
- Categorize comments as praise, complaint, question, suggestion, bought it, wants to buy, or other.
- Identify keyword-based topics such as price, quality, design, performance, delivery, service/trust, durability/comfort, and availability.
- Scan comments for buying intent, purchase claims, show-off comments, mentioned platforms/stores, discovery cues, language, and locations.
- Run two Gemini agents concurrently for customer insights and marketing recommendations.
- Compare comment patterns across videos, sources, topics, locations, and languages.
- Set a separate advertising budget for each saved brand and view suggested allocations.
- Download collected data as CSV or Excel.

## Project structure

```text
.
├── app.py                 # Main Streamlit dashboard
├── agents.py              # Gemini API calls and AI agents
├── app_links.py           # Alternate/standalone YouTube-link sentiment interface
├── brand_insights.py      # Purchase signals, source/discovery analysis, budget helpers
├── brand_pipeline.py      # Brand-to-videos-to-comments collection pipeline
├── config.py              # Reads environment variables and local .env settings
├── topics.py              # Comment types, topic keywords, and summaries
├── youtube_search.py      # YouTube Data API search and yt-dlp fallback
├── youtube_sentiment.py   # Comment cleaning, VADER sentiment, CLI analysis
├── test_keys.py           # Checks API key configuration
├── test_sdk.py            # Tests the optional official Gemini SDK
├── requirements.txt       # Python dependencies
└── .env                   # Your local API keys (create this; do not commit it)
```

## Requirements

- Python 3.10 or newer is recommended.
- Internet access.
- A Google Gemini API key for AI-generated insights and marketing recommendations.
- A YouTube Data API v3 key is optional. Without it, video search can fall back to `yt-dlp`.

## Installation

Open a terminal in the project folder and create a virtual environment.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configure API keys

Create a file named `.env` in the same folder as `app.py`. Add your keys and optional settings:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
YOUTUBE_API_KEY=your_youtube_data_api_key_here
GEMINI_MODEL=gemini-3.8-flash
USE_WEB_SEARCH=true
```

- Get a Gemini key from [Google AI Studio](https://aistudio.google.com/).
- Create a YouTube Data API v3 key in the [Google Cloud Console](https://console.cloud.google.com/). Enable **YouTube Data API v3** for the project if you use this key.
- `YOUTUBE_API_KEY` is optional; omit it or leave it blank to allow the `yt-dlp` fallback.
- `GEMINI_MODEL` is optional. If omitted, the app uses its configured default model and has logic to handle unavailable model names.
- `USE_WEB_SEARCH` controls whether the marketing agent may use Google Search grounding; it defaults to enabled unless set to `0`, `false`, `no`, or `off`.

Do not add real API keys to source files, screenshots, public repositories, or commit history. Add `.env` to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
data/
```

## Run the application

From the project folder, activate the virtual environment and run:

```bash
python -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## How to use the dashboard

1. Open **Collect comments**.
2. Enter a brand name and, optionally, its niche.
3. Choose the maximum number of videos, comments per video, and total comments to collect.
4. Choose whether to keep only videos that mention the brand name. Add video links manually if needed.
5. Start collection and review any notes or skipped videos.
6. Open **Comments & insights** to explore sentiment, comment types, topics, and discovery signals.
7. Open **Ad optimizer** to set the brand's budget and review data-based and AI-generated recommendations.
8. Open **Data & download** to search, inspect, and export the results.

Each brand saved during the current Streamlit session has its own collected comments and budget. The app uses Streamlit session state, so this is not a permanent database; restarting the app may clear the saved in-session brand selections. CSV files are also written by the collection pipeline to the `data/` folder.

## Optional: run the YouTube-link analyzer

`youtube_sentiment.py` also provides a command-line interface for analyzing comments from one or more YouTube videos.

```bash
python youtube_sentiment.py
```

You can paste links interactively, pass links as arguments, or provide a text file containing one URL per line:

```bash
python youtube_sentiment.py "https://www.youtube.com/watch?v=VIDEO_ID"
python youtube_sentiment.py LINK1 LINK2 --limit 1000
python youtube_sentiment.py --file links.txt
```

The alternate Streamlit interface in `app_links.py` is intended for analyzing supplied YouTube links rather than searching for a brand first. Run it with:

```bash
python -m streamlit run app_links.py
```

## API key troubleshooting

Run the key checker from the project folder:

```bash
python test_keys.py
```

To test the optional official Gemini Python SDK, install it and run:

```bash
python -m pip install google-genai
python test_sdk.py
```

The current `requirements.txt` does not list `google-genai`; the main app can use direct HTTPS requests, and the SDK is an optional fallback. If a key is rejected, create or verify the key and ensure the relevant Google API is enabled. If requests time out or are rate-limited, retry later or choose an available model using `GEMINI_MODEL`.

## Analysis notes and limitations

- Sentiment is calculated with VADER, a general-purpose sentiment tool. It may not correctly understand all slang, sarcasm, mixed-language comments, or context.
- Comment types, topics, purchase claims, and discovery/source cues rely partly on keyword patterns in English and Roman Urdu. These are useful indicators, not verified facts.
- A comment saying that a person found or bought a product on a platform does not establish that the platform generated the sale.
- Comment samples are limited by the selected caps and by whether a video has accessible comments.
- AI-generated insights and budget recommendations should be reviewed by a person before making business decisions.
- Budget allocations are recommendations, not actual campaign performance forecasts.
- Do not treat comment-derived city/country or language labels as reliable demographic analytics.

## Dependencies

The main dependencies listed in `requirements.txt` are:

- Streamlit — web dashboard
- Requests — HTTP API requests
- Pandas — tabular data processing
- Matplotlib — charts
- OpenPyXL — Excel exports
- VADER Sentiment — sentiment scoring
- youtube-comment-downloader — retrieving YouTube comments
- yt-dlp — fallback YouTube search


