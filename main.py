
"""
GrokTweetBot: Automated Tweet & Telegram Broadcaster Using Grok AI Summaries

Overview:
---------
This bot automates the process of curating, formatting, and publishing domain-specific news and summaries to Twitter (X) and optionally Telegram. It leverages xAI's Grok large language model to generate tweet-sized summaries based on custom prompts.

Key Components:
---------------
1. **Environment Configuration**:
   - Reads API credentials and Telegram details from `.env` via `dotenv`.
   - Validates required Twitter (X) and Grok credentials are present.

2. **Thread-Safe JSON Cache**:
   - Caches both Grok responses and posted tweets/messages to prevent redundant requests.
   - Uses a reentrant lock (`RLock`) to ensure thread-safe read/write operations to `tweet_cache.json`.

3. **Grok Prompting (LLM Client)**:
   - A detailed `meta_prompt` is used to instruct Grok to return a Python dictionary containing 280-character tweet-sized responses.
   - Each paragraph is required to include a character count in square brackets (e.g., `[274]`).
   - These responses are streamed and parsed using `ast.literal_eval` to ensure safe dictionary parsing.

4. **Text Cleaning**:
   - Removes `[char_count]` annotations and excess whitespace from Grok's output before publishing.

5. **Tweet Posting (via Tweepy)**:
   - Uses Tweepy to post each paragraph as a separate tweet.
   - Each tweet is cached based on content to prevent re-posting identical tweets.

6. **Telegram Integration (Optional)**:
   - If Telegram is configured, the same tweet text is also sent to a configured chat via Telegram's bot API.

7. **Broadcast Loop**:
   - The `process_and_broadcast` method:
     - Accepts a user prompt (e.g., a market sector query).
     - Queries Grok with the structured meta prompt.
     - Cleans and posts each paragraph to Twitter and Telegram.
     - Returns a mapping of tweet and Telegram response metadata for each paragraph.

8. **Time-Based Scheduling (via `schedule`)**:
   - Predefined prompts are triggered at exact HH:MM:SS times using `schedule.every().day.at(...).do(...)`.
   - Prompts cover Indian financial market sectors in rotation every 3 hours.
   - A loop runs every second checking if a scheduled job is due.

Usage:
------
- Run the script continuously (e.g., in a Docker container, VM, or cloud task runner).
- It ensures timely market updates are summarized by Grok, cleaned, and shared on X/Telegram.
- Ensures zero redundancy by caching Grok responses and posted messages.
- Ideal for automated content pipelines in fin-news dissemination, alerting, or thematic channels.
"""
import os
import ast
import json
import logging
import time
import datetime as dt
from pathlib import Path
from threading import RLock
from typing import Dict, Any, Optional

import regex as re
import schedule
import tweepy
import requests
from dotenv import load_dotenv
from xai_sdk import Client
from xai_sdk.chat import user, system
from xai_sdk.search import SearchParameters

# =============================================================
# Configuration & Logging
# =============================================================

load_dotenv()

logger = logging.getLogger("grok_tweet_bot")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s"))
logger.addHandler(_handler)

# -------------------------------------------------------------
# Simple JSON cache (thread‑safe) --------------------------------
# -------------------------------------------------------------

CACHE_FILE = Path("tweet_cache.json")
CACHE_LOCK = RLock()


def _load_cache() -> Dict[str, Any]:
    """Load cache from disk or create the file atomically."""
    with CACHE_LOCK:
        if not CACHE_FILE.exists():
            CACHE_FILE.write_text("{}", encoding="utf-8")
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def _save_cache(cache: Dict[str, Any]):
    with CACHE_LOCK:
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================
# Core Bot -----------------------------------------------------
# =============================================================

class GrokTweetBot:
    """Posts Grok‑summaries to X (Twitter) and Telegram, with caching."""

    # ──────────────────────────────────────────────────────────
    # Initialisation & env handling
    # ──────────────────────────────────────────────────────────

    def __init__(self):
        self._load_env()
        self._validate_env()

        # Grok / xAI client
        self.grok_client = (
            Client(api_key=self.LLM_API_KEY, api_host=self.LLM_API_HOST)
            if self.LLM_API_HOST
            else Client(api_key=self.LLM_API_KEY)
        )

        # Twitter (X) client
        self.twitter_client = tweepy.Client(
            consumer_key=self.API_KEY_X,
            consumer_secret=self.API_KEY_SECRET_X,
            access_token=self.ACCESS_TOKEN_X,
            access_token_secret=self.ACCESS_TOKEN_SECRET_X,
        )

        # Telegram endpoint (optional)
        self.telegram_api_url: Optional[str] = (
            f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}/sendMessage"
            if self.TELEGRAM_BOT_TOKEN
            else None
        )

        # Prompt that forces Grok to produce tweet‑sized paragraphs with counts
        self.meta_prompt = (
            """You are a precise, logical assistant. When answering, adhere strictly to the following:\n\n"
            "1. Paragraph Structure\n"
            "- Divide your response into clear, self-contained paragraphs.\n"
            "- Each paragraph must be ≤280 characters (including spaces and punctuation).\n\n"
            "2. Character Count Verification\n"
            "- After composing each paragraph, append its character count in square brackets, e.g. [274].\n"
            "- Do not exceed 280 characters; adjust wording if necessary.\n\n"
            "3. Dictionary Output\n"
            "- Once all paragraphs are finalized, output only a Python-style dictionary:\n"
            "    {\n"
            "    1: \"First paragraph text…\",\n"
            "    2: \"Second paragraph text…\",\n"
            "    …\n"
            "    n: \"Nth paragraph text…\"\n"
            "    }\n"
            "- No additional commentary, markdown, or logging.\n\n"
            "4. Exact Formatting\n"
            "- Use straight quotes (\") only. No escape characters beyond JSON necessities.\n"
            "- Keys consecutive integers starting at 1. No trailing commas.\n\n"
            "5. Content Requirements\n"
            "- Be specific and factual. Logical flow: intro, development, conclusion.\n"
            "- Avoid filler words; every sentence must add value.\n"""
        )

        # Populate cache into memory once
        self.cache: Dict[str, Any] = _load_cache()

    # ──────────────────────────────────────────────────────────
    # Environment helpers
    # ──────────────────────────────────────────────────────────

    def _load_env(self):
        self.LLM_API_KEY = os.getenv("LLM_API_KEY")
        self.LLM_API_HOST = os.getenv("LLM_API_HOST")
        self.API_KEY_X = os.getenv("API_KEY_X")
        self.API_KEY_SECRET_X = os.getenv("API_KEY_SECRET_X")
        self.ACCESS_TOKEN_X = os.getenv("ACCESS_TOKEN_X")
        self.ACCESS_TOKEN_SECRET_X = os.getenv("ACCESS_TOKEN_SECRET_X")
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    def _validate_env(self):
        required = [
            self.LLM_API_KEY,
            self.API_KEY_X,
            self.API_KEY_SECRET_X,
            self.ACCESS_TOKEN_X,
            self.ACCESS_TOKEN_SECRET_X,
        ]
        if any(v is None for v in required):
            raise EnvironmentError("Missing mandatory Twitter/xAI environment variables.")

        # Telegram is optional, but if one is provided both must exist.
        if bool(self.TELEGRAM_BOT_TOKEN) ^ bool(self.TELEGRAM_CHAT_ID):
            raise EnvironmentError(
                "Provide *both* TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID, or neither."
            )

    # ──────────────────────────────────────────────────────────
    # Utility helpers
    # ──────────────────────────────────────────────────────────

    def _query_grok(
        self, user_prompt: str, model: str = "grok-4", temperature: float = 0.0
    ) -> Dict[int, str]:
        """Ask Grok; cache the structured dict so we never re‑query for same prompt."""
        cache_key = f"grok::{model}::{user_prompt}"
        if (cached := self.cache.get(cache_key)) is not None:
            logger.info("Grok cache hit for prompt → %s", user_prompt[:60])
            return cached

        try:
            chat = self.grok_client.chat.create(
                model=model,
                temperature=temperature,
                messages=[system(self.meta_prompt), user(user_prompt)],
                search_parameters=SearchParameters(mode="auto"),
            )
            chunks: List[str] = []
            for _resp, chunk in chat.stream():
                chunks.append(chunk.content)
            parsed = ast.literal_eval("".join(chunks))
        except Exception:
            logger.exception("Failed Grok query or parsing.")
            raise

        self.cache[cache_key] = parsed
        _save_cache(self.cache)
        return parsed

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove trailing [char_count] markers and redundant whitespace."""
        text = re.sub(r"\s*\[\d{1,3}\]$", "", text.strip())
        text = re.sub(r"\s*([.,;!?])", r"\1", text)
        return text.strip()

    # ---------------------------------------------------------
    # Posting helpers
    # ---------------------------------------------------------

    def _post_tweet(self, text: str) -> Optional[Dict[str, Any]]:
        cache_key = f"tweet::{text}"
        if (cached := self.cache.get(cache_key)) is not None:
            logger.info("Tweet cache hit → skipping X post.")
            return cached
        try:
            response = self.twitter_client.create_tweet(text=text)
            self.cache[cache_key] = response.data
            _save_cache(self.cache)
            logger.info("Tweet posted: %s", response.data)
            return response.data
        except tweepy.TweepyException as exc:
            logger.error("Tweet failed: %s", exc)
            return None

    def _post_telegram(self, text: str) -> Optional[Dict[str, Any]]:
        if not self.telegram_api_url:
            return None  # Telegram not configured
        cache_key = f"telegram::{text}"
        if (cached := self.cache.get(cache_key)) is not None:
            logger.info("Telegram cache hit → skipping Telegram post.")
            return cached
        try:
            resp = requests.post(
                self.telegram_api_url,
                data={"chat_id": self.TELEGRAM_CHAT_ID, "text": text},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.cache[cache_key] = data
            _save_cache(self.cache)
            logger.info("Telegram message sent: message_id=%s", data.get("result", {}).get("message_id"))
            return data
        except Exception as exc:
            logger.error("Telegram post failed: %s", exc)
            return None

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def process_and_broadcast(self, user_prompt: str):
        """End‑to‑end: query Grok → clean → post to X & Telegram."""
        logger.info("Processing prompt: %s", user_prompt)
        paragraphs = self._query_grok(user_prompt)
        results: Dict[int, Dict[str, Any]] = {}

        for idx, para in paragraphs.items():
            clean_text = self._clean_text(para)
            tweet_res = self._post_tweet(clean_text)
            tel_res = self._post_telegram(clean_text)
            results[idx] = {"tweet": tweet_res, "telegram": tel_res}

        return results


# =============================================================
# Scheduling ---------------------------------------------------
# =============================================================

user_prompts: Dict[str, str] = {
    "09:00:00": "What’s the latest on Financials in the Indian financial markets?",
    "12:00:00": "What’s the latest on Energy & Utilities in the Indian financial markets?",
    "15:00:00": "What’s the latest on Information Technology in the Indian financial markets?",
    "18:00:00": "What’s the latest on Telecom & Media in the Indian financial markets?",
    "21:00:00": "What’s the latest on Consumer Goods (FMCG, Auto, Pharma) in the Indian financial markets?",
    "00:00:00": "What’s the latest on Capital Goods, Infrastructure & Real Estate in the Indian financial markets?",
    "03:00:00": "What’s the latest on Metals, Mining & Commodities in the Indian financial markets?",
    "06:00:00": "What’s the latest on Financial Technology & Banking in the Indian financial markets?",
}


def _schedule_jobs(bot: GrokTweetBot):
    for hhmm, prompt in user_prompts.items():
        schedule.every().day.at(hhmm).do(bot.process_and_broadcast, user_prompt=prompt)
        logger.info("Scheduled job at %s for prompt: %s", hhmm, prompt[:60])


def main():
    bot = GrokTweetBot()
    _schedule_jobs(bot)
    logger.info("Scheduler started. Press Ctrl+C to exit.")
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
