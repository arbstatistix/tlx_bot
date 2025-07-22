# 🧠 GrokTweetBot: Auto-Curated AI Tweets from Real-Time Market Insights

**GrokTweetBot** is an autonomous content curation engine that leverages [xAI's Grok](https://x.ai) to summarize sectoral developments from the Indian financial markets and broadcast them as concise, tweet-sized updates. It posts these insights to **Twitter (X)** and optionally **Telegram** — on a precisely timed schedule.

---

## 🚀 Features

- **LLM-Driven Content**: Uses xAI’s `grok-4` to generate summaries in structured, tweet-sized paragraphs.
- **Auto-Broadcast**: Seamless posting to Twitter (X) and optionally Telegram.
- **Fully Cached**: Thread-safe disk caching prevents duplicate tweets or Grok requests.
- **Precise Scheduling**: Time-based triggers (via `schedule`) to post sector-specific insights every 3 hours.
- **Character-Count Aware**: Prompts Grok to self-enforce 280-character constraints and include character counts for validation.

---

## 🧩 How It Works

### 🔁 Automated Loop

```text
Every 3 hours:
  ⮑ Prompt Grok with a sector-specific question
      ⮑ Parse Grok's structured dictionary of tweet-sized paragraphs
          ⮑ Clean the output (strip char counts, tidy punctuation)
              ⮑ Post each paragraph as a tweet
              ⮑ (Optional) Post to Telegram
                  ⮑ Cache results to prevent reposting
```

## 📦 Requirements
- **Python 3.8+
- **xAI SDK (xai_sdk)
- **Tweepy (tweepy)
- **Regex (regex)
- **Schedule (schedule)
- **Requests (requests)
- **python-dotenv
