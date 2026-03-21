#!/usr/bin/env python3
import feedparser
import requests
import json
import time
from groq import Groq, RateLimitError
import os
from datetime import datetime
import re
import numpy as np

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Embedding model — loaded once, reused for all articles
# all-MiniLM-L6-v2: 80MB, 384-dim vectors, fast, accurate enough
print("📐 Loading embedding model...")
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("   ✅ Model ready\n")

CACHE_FILE  = "article_cache.json"
OUTPUT_FILE = "curriculum.json"

FEEDS = [
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feed_anthropic_news.xml",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://www.deepmind.google/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://stability.ai/news/rss",
    "https://ai.meta.com/blog/feed/",
    "https://www.alignmentforum.org/feed.xml",
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://cloud.google.com/blog/products/ai-machine-learning/rss/",
    "https://importai.substack.com/feed",
    "https://www.aisnakeoil.com/feed",
    "https://thegradient.pub/rss/",
    "https://distill.pub/rss.xml",
    "https://pair.withgoogle.com/feed.xml",
]

# =============================================================================
# PROMPT
# =============================================================================

SOUL_PROMPT = """You are an AI curriculum designer analyzing technical articles.

CRITICAL: You MUST return valid JSON. No markdown, no code blocks, just raw JSON.

TOPICS (pick ONE that best fits):
- AI Evaluations & Benchmarking
- Large Language Models
- AI Safety & Alignment
- Agentic AI & Reasoning
- AI Infrastructure & Tooling
- Prompt Engineering & Applications
- Computer Vision & Multimodal AI
- AI Research & Theory

Analyze this article and determine:

1. SKIP ONLY if:
   - Pure job posting with no technical content
   - Event-only announcement (just date/time/location)
   - Press release with zero technical information
   KEEP EVERYTHING ELSE.

2. ARTICLE TYPE:
   - "teaches": Explains a concept from scratch
   - "extends": Assumes knowledge, covers news/opinion/application
   Examples:
   - "What is RLHF and how it works" → teaches
   - "OpenAI's new RLHF improvements" → extends

3. PRIMARY CONCEPT:
   - The ONE main concept, canonical name
   - e.g. "reinforcement learning from human feedback" not "RLHF"

4. SUMMARY:
   - 2-3 sentences, enough context for semantic matching
   - If "extends", mention what concepts it assumes

5. ACCESSIBLE TITLE:
   - Rewrite the title for a smart non-technical professional
   - Formula: "[practical verb phrase for their role] — [the specific angle]"
   - Start with How/Why/When/What
   - Under 12 words total
   - Zero jargon. If you must use a technical term, put it after the dash only.
   - Examples:
     "ACT-based approval directed agents for IDA skeptics"
     → "How to manage AI autonomy in your products — the approval-based approach"
     "Constitutional AI: Harmlessness from AI Feedback"
     → "How to build AI features users can trust — the self-correction approach"
     "Scaling Laws for Neural Language Models"
     → "Why bigger AI models keep getting better — and what it means for your roadmap"

6. CONCEPTS TAUGHT (2-5):
   - Canonical names, confidence 0-1

7. PREREQUISITES (0-3):
   - Canonical names, confidence 0-1

8. DIFFICULTY:
   - foundational / beginner / intermediate / advanced / application
   - technical_depth: 1-10
   - reading_time_minutes: estimate

9. LEARNING OUTCOMES (2-4): concrete outcomes

10. STRATEGIC QUESTIONS (2-3): thought-provoking questions

IMPORTANT: Return ONLY valid JSON.

If skipping:
{{"skip": true}}

Otherwise:
{{
  "skip": false,
  "topic": "Large Language Models",
  "article_type": "teaches",
  "primary_concept": "reinforcement learning from human feedback",
  "summary": "This article explains how RLHF works, covering reward modeling and PPO optimization.",
  "accessible_title": "How to align AI with what humans actually want — the feedback-based approach",
  "concepts_taught": [
    {{"name": "reinforcement learning from human feedback", "confidence": 0.9}}
  ],
  "prerequisites": [
    {{"name": "supervised fine-tuning", "confidence": 0.85}}
  ],
  "difficulty": {{
    "level": "intermediate",
    "technical_depth": 6,
    "reading_time_minutes": 12
  }},
  "learning_outcomes": [
    "Understand how RLHF improves language model outputs"
  ],
  "strategic_questions": [
    "How does RLHF compare to other alignment techniques?"
  ]
}}

Article:
Title: {title}
Content: {content}
"""


# =============================================================================
# CACHE
# =============================================================================

def load_cache() -> dict:
    """
    Cache schema per entry:
    {
      "url": "https://...",
      "title": "...",
      "published": "...",
      "source": "...",
      "cached_at": "2026-03-21T00:00:00Z",
      "skipped": false,
      "embedding": [0.1, 0.2, ...],   # 384 floats, null if skipped
      "curriculum": { ...groq output... }
    }
    """
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cache = json.load(f)
            print(f"📦 Cache loaded: {len(cache)} articles")
            return cache
        except Exception as e:
            print(f"⚠️  Cache load failed ({e}), starting fresh")
    else:
        print("📦 No cache found, starting fresh")
    return {}


def save_cache(cache: dict):
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"💾 Cache saved: {len(cache)} articles")
    except Exception as e:
        print(f"❌ Cache save failed: {e}")


# =============================================================================
# EMBEDDINGS
# =============================================================================

def generate_embedding(text: str) -> list:
    """
    Embed text using all-MiniLM-L6-v2.
    Returns 384 floats quantized to 4dp to reduce file size.
    """
    vector = embedder.encode(text, normalize_embeddings=True)
    return [round(float(x), 4) for x in vector]


def needs_embedding(entry: dict) -> bool:
    """True if this valid cached article is missing an embedding."""
    return (
        not entry.get("skipped")
        and entry.get("curriculum") is not None
        and not entry.get("embedding")
    )


def build_embed_text(entry: dict) -> str:
    """
    Combine summary + accessible_title + concept names for richer embedding.
    This is what we search against when ranking articles for a user.
    """
    curriculum = entry.get("curriculum", {})
    parts = []

    summary = curriculum.get("summary", "")
    if summary:
        parts.append(summary)

    accessible = curriculum.get("accessible_title", "")
    if accessible:
        parts.append(accessible)

    concepts = curriculum.get("concepts_taught", [])
    if concepts:
        parts.append("Concepts: " + ", ".join(c["name"] for c in concepts))

    return " ".join(parts) if parts else entry.get("title", "")


# =============================================================================
# FEED FETCHING
# =============================================================================

def fetch_feed(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Bramble/1.0)"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"    ⚠️  HTTP {response.status_code}")
            return None
        feed = feedparser.parse(response.content)
        if feed.bozo:
            print(f"    ⚠️  Parse warning: {feed.bozo_exception}")
        return feed
    except requests.Timeout:
        print(f"    ⚠️  Timeout")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


def extract_content(entry) -> str:
    content = ""
    if hasattr(entry, "content") and entry.content:
        content = entry.content[0].value
    elif hasattr(entry, "summary_detail") and entry.summary_detail:
        content = entry.summary_detail.value
    elif hasattr(entry, "summary"):
        content = entry.summary
    elif hasattr(entry, "description"):
        content = entry.description

    content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
    content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)
    content = re.sub(r"<[^>]+>", " ", content)
    content = re.sub(r"\s+", " ", content).strip()

    if len(content) > 4000:
        content = content[:4000] + "..."
    return content


# =============================================================================
# GROQ ANALYSIS
# =============================================================================

def analyze_article(title: str, content: str) -> dict | None:
    try:
        prompt = SOUL_PROMPT.format(title=title, content=content)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        result = response.choices[0].message.content.strip()

        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(lines[1:-1]) if len(lines) > 2 else result
            if result.startswith("json"):
                result = result[4:].strip()

        parsed = json.loads(result)

        if parsed.get("skip") is True:
            return {"skip": True}

        if parsed.get("topic"):
            parsed.setdefault("article_type", "teaches")
            parsed.setdefault("accessible_title", title)  # fallback to original
            if "primary_concept" not in parsed:
                if parsed.get("concepts_taught"):
                    parsed["primary_concept"] = parsed["concepts_taught"][0]["name"]
                else:
                    parsed["primary_concept"] = "general"
            parsed.setdefault("summary", "")
            return parsed

        print(f"    ⚠️  Invalid structure")
        return None

    except json.JSONDecodeError as e:
        print(f"    ❌ JSON error: {e}")
        return None
    except RateLimitError:
        print(f"    ⛔ Groq rate limit hit")
        # Set flag so the main loop can stop gracefully
        global rate_limit_hit
        rate_limit_hit = True
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None


# =============================================================================
# CONCEPT REGISTRY
# =============================================================================

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def build_concept_registry(articles: list) -> dict:
    concepts = {}
    for article in articles:
        curriculum = article["curriculum"]
        primary = curriculum.get("primary_concept", "")
        if not primary:
            continue

        concept_id = slugify(primary)
        article_type = curriculum.get("article_type", "teaches")
        topic = curriculum.get("topic", "")

        if concept_id not in concepts:
            concepts[concept_id] = {
                "id": concept_id,
                "name": primary,
                "aliases": set([primary]),
                "topic": topic,
                "teaches": [],
                "extends": [],
                "prerequisite_concepts": set(),
            }
        else:
            concepts[concept_id]["aliases"].add(primary)

        if article_type == "teaches":
            concepts[concept_id]["teaches"].append(article["url"])
        else:
            concepts[concept_id]["extends"].append(article["url"])

        for prereq in curriculum.get("prerequisites", []):
            prereq_id = slugify(prereq["name"])
            if prereq_id and prereq_id != concept_id:
                concepts[concept_id]["prerequisite_concepts"].add(prereq_id)

    for cid in concepts:
        concepts[cid]["aliases"] = list(concepts[cid]["aliases"])
        concepts[cid]["prerequisite_concepts"] = list(concepts[cid]["prerequisite_concepts"])

    return concepts


# =============================================================================
# MAIN
# =============================================================================

print("🌿 Bramble pipeline starting...")
print(f"📡 Feeds: {len(FEEDS)}\n")

# ── Step 1: Load cache ───────────────────────────────────────────────────────
cache = load_cache()
cache_hits     = 0
new_analyses   = 0
new_embeddings = 0
skipped_count  = 0
error_count    = 0
rate_limit_hit = False

# ── Step 2: Fetch feeds ──────────────────────────────────────────────────────
print("\n🔄 Fetching RSS feeds...")
raw_articles = []
feed_stats   = {}

for feed_url in FEEDS:
    feed_name = feed_url.split("/")[2]
    print(f"  📰 {feed_name}")

    feed = fetch_feed(feed_url)
    if not feed or not hasattr(feed, "entries"):
        print(f"    ❌ Failed to fetch")
        feed_stats[feed_name] = 0
        continue

    count = 0
    for entry in feed.entries[:15]:
        if not hasattr(entry, "link") or not entry.link:
            continue
        content = extract_content(entry)
        if len(content.split()) < 300:
            continue
        raw_articles.append({
            "title":     getattr(entry, "title", "Untitled"),
            "url":       entry.link,
            "published": entry.get("published", entry.get("updated", "")),
            "source":    feed.feed.title if hasattr(feed.feed, "title") else feed_name,
            "content":   content,
        })
        count += 1

    feed_stats[feed_name] = count
    print(f"    ✅ {count} articles")
    time.sleep(0.3)

print(f"\n📚 Total fetched: {len(raw_articles)}")

# ── Step 3: Analyze new articles via Groq ───────────────────────────────────
new_count = sum(1 for a in raw_articles if a["url"] not in cache)
print(f"\n🤖 Groq: {new_count} new | {len(raw_articles) - new_count} cached\n")

for i, article in enumerate(raw_articles, 1):
    url = article["url"]

    if url in cache:
        cache_hits += 1
        cache[url]["title"]     = article["title"]
        cache[url]["published"] = article["published"]
        cache[url]["source"]    = article["source"]
        print(f"  [{i}/{len(raw_articles)}] 💾 {article['title'][:60]}...")
        continue

    print(f"  [{i}/{len(raw_articles)}] 🤖 {article['title'][:60]}...")
    analysis = analyze_article(article["title"], article["content"])

    if analysis is None:
        # Check if we hit the rate limit — stop gracefully if so
        if rate_limit_hit:
            print(f"\n⛔ Rate limit reached — stopping analysis, saving progress")
            save_cache(cache)
            break
        error_count += 1
        print(f"    ❌ Failed")
        continue

    if analysis.get("skip"):
        skipped_count += 1
        cache[url] = {
            "title": article["title"], "url": url,
            "published": article["published"], "source": article["source"],
            "cached_at": datetime.utcnow().isoformat() + "Z",
            "skipped": True, "embedding": None, "curriculum": None,
        }
        print(f"    ⏭️  Skipped")
        continue

    cache[url] = {
        "title": article["title"], "url": url,
        "published": article["published"], "source": article["source"],
        "cached_at": datetime.utcnow().isoformat() + "Z",
        "skipped": False, "embedding": None,  # filled in step 4
        "curriculum": analysis,
    }
    new_analyses += 1
    atype   = analysis.get("article_type", "?")
    primary = analysis.get("primary_concept", "")[:35]
    atitle  = analysis.get("accessible_title", "")[:60]
    print(f"    ✅ [{atype}] {primary}")
    print(f"       📌 {atitle}")
    time.sleep(0.3)

# ── Step 4: Generate / backfill embeddings ───────────────────────────────────
to_embed = [url for url, e in cache.items() if needs_embedding(e)]

if to_embed:
    print(f"\n📐 Embedding {len(to_embed)} articles...")
    for j, url in enumerate(to_embed, 1):
        text = build_embed_text(cache[url])
        cache[url]["embedding"] = generate_embedding(text)
        new_embeddings += 1
        if j % 20 == 0 or j == len(to_embed):
            print(f"   {j}/{len(to_embed)} done")
else:
    print("\n📐 All articles already embedded")

# ── Step 5: Save cache ───────────────────────────────────────────────────────
save_cache(cache)

# ── Step 6: Build curriculum from full archive ───────────────────────────────
print("\n📦 Building curriculum...")

analyzed = [
    {
        "title":      e["title"],
        "url":        e["url"],
        "published":  e["published"],
        "source":     e["source"],
        "embedding":  e.get("embedding"),
        "curriculum": e["curriculum"],
    }
    for e in cache.values()
    if not e.get("skipped") and e.get("curriculum") is not None
]

print(f"   {len(analyzed)} articles in archive")

# ── Step 7: Concept registry ─────────────────────────────────────────────────
concepts      = build_concept_registry(analyzed)
teaches_count = sum(1 for a in analyzed if a["curriculum"].get("article_type") == "teaches")
extends_count = sum(1 for a in analyzed if a["curriculum"].get("article_type") == "extends")
orphaned      = [c for c in concepts.values() if not c["teaches"] and c["extends"]]
embedded_n    = sum(1 for a in analyzed if a.get("embedding"))

# ── Step 8: Group by topic ────────────────────────────────────────────────────
topic_counts = {}
topics       = {}

for article in analyzed:
    topic = article["curriculum"]["topic"]
    if topic not in topics:
        topics[topic] = {
            "name": topic, "article_count": 0, "articles": [],
            "levels": {k: [] for k in ["foundational","beginner","intermediate","advanced","application"]},
        }
    topics[topic]["articles"].append(article)
    topics[topic]["article_count"] += 1
    level = article["curriculum"]["difficulty"]["level"].lower()
    if level in topics[topic]["levels"]:
        topics[topic]["levels"][level].append(article)
    topic_counts[topic] = topic_counts.get(topic, 0) + 1

# ── Step 9: Write curriculum.json ────────────────────────────────────────────
curriculum = {
    "generated_at":   datetime.utcnow().isoformat() + "Z",
    "total_articles": len(analyzed),
    "total_concepts": len(concepts),
    "concepts":       concepts,
    "topics":         topics,
    "articles":       analyzed,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(curriculum, f, indent=2)

# ── Step 10: Summary ──────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════╗
║         Bramble Pipeline Complete        ║
╠══════════════════════════════════════════╣
║  This run                                ║
║    Fetched from feeds : {len(raw_articles):<17} ║
║    Cache hits (saved) : {cache_hits:<17} ║
║    New Groq analyses  : {new_analyses:<17} ║
║    New embeddings     : {new_embeddings:<17} ║
║    Skipped            : {skipped_count:<17} ║
║    Errors             : {error_count:<17} ║
╠══════════════════════════════════════════╣
║  Archive                                 ║
║    Total cached       : {len(cache):<17} ║
║    In curriculum      : {len(analyzed):<17} ║
║    With embeddings    : {embedded_n:<17} ║
║    Concepts           : {len(concepts):<17} ║
╚══════════════════════════════════════════╝
""")

print("📚 Topics:")
for t, c in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {t}: {c} articles")

print("\n🧠 Top concepts:")
for concept in sorted(concepts.values(), key=lambda c: len(c["teaches"]) + len(c["extends"]), reverse=True)[:10]:
    t, e = len(concept["teaches"]), len(concept["extends"])
    print(f"   • {concept['name']}: {t} teach, {e} extend {'✅' if t else '⚠️ orphan'}")

print(f"\n✅ {OUTPUT_FILE} written")
print("🌿 Done!")
