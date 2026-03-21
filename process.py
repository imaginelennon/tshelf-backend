#!/usr/bin/env python3
import feedparser
import requests
import json
import time
from groq import Groq
import os
from datetime import datetime
import re

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Cache file — persists between runs in the repo
CACHE_FILE = "article_cache.json"
OUTPUT_FILE = "curriculum.json"

# RSS Feeds
FEEDS = [
    # Major AI Labs
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feed_anthropic_news.xml",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
    "https://www.deepmind.google/blog/rss.xml",

    # Technical Blogs & Research
    "https://huggingface.co/blog/feed.xml",
    "https://stability.ai/news/rss",
    "https://ai.meta.com/blog/feed/",

    # AI Safety & Alignment
    "https://www.alignmentforum.org/feed.xml",

    # Industry/Cloud ML
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://cloud.google.com/blog/products/ai-machine-learning/rss/",

    # Newsletters & Analysis
    "https://importai.substack.com/feed",
    "https://www.aisnakeoil.com/feed",
    "https://thegradient.pub/rss/",

    # Academic/Research-Heavy
    "https://distill.pub/rss.xml",
    "https://pair.withgoogle.com/feed.xml",
]

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
   
   KEEP EVERYTHING ELSE including product announcements, company news, tutorials, research papers, opinion pieces, industry analysis, tool releases.

2. ARTICLE TYPE (critical distinction):
   - "teaches": Article EXPLAINS a concept from scratch. Reader learns the concept by reading this.
   - "extends": Article ASSUMES knowledge of a concept. It's news, opinion, application, or advanced discussion that builds on existing knowledge.
   
   Examples:
   - "What is RLHF and how it works" → teaches
   - "OpenAI's new RLHF improvements announced" → extends
   - "Tutorial: Implementing attention mechanisms" → teaches
   - "Why attention might not be all you need (opinion)" → extends

3. PRIMARY CONCEPT: 
   - The ONE main concept this article is about
   - Use a canonical, normalized name (e.g., "reinforcement learning from human feedback" not "RLHF")
   - Be specific but not overly narrow

4. SUMMARY:
   - 2-3 sentences explaining what this article covers
   - Include enough context for future matching
   - If it's an "extends" article, mention what concepts it assumes

5. CONCEPTS TAUGHT (2-5 concepts):
   - What does this article cover?
   - Use canonical names, be specific
   - Include confidence (0-1)

6. PREREQUISITES (0-3 concepts):
   - What should reader know beforehand?
   - Use canonical concept names
   - Include confidence

7. DIFFICULTY LEVEL:
   - foundational: First principles, no prerequisites
   - beginner: Basic AI familiarity helpful
   - intermediate: Solid AI understanding required
   - advanced: Deep technical knowledge needed
   - application: Real-world implementation focus
   
   Technical depth (1-10): How technical?
   Reading time: Estimate in minutes

8. LEARNING OUTCOMES (2-4 outcomes):
   - What will reader understand after?
   - Be concrete

9. STRATEGIC QUESTIONS (2-3 questions):
   - Thought-provoking
   - Encourage deeper thinking

IMPORTANT: Return ONLY valid JSON. No markdown. No explanations.

If skipping (rare!), return:
{{"skip": true}}

Otherwise return:
{{
  "skip": false,
  "topic": "Large Language Models",
  "article_type": "teaches",
  "primary_concept": "reinforcement learning from human feedback",
  "summary": "This article explains how RLHF works, covering reward modeling, human preference collection, and PPO optimization for language model fine-tuning.",
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
    "Understand how RLHF improves language model outputs",
    "Learn the reward modeling process"
  ],
  "strategic_questions": [
    "How does RLHF compare to other alignment techniques?",
    "What are the limitations of human feedback?"
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
    Load the article cache from disk.
    Structure:
    {
      "https://article-url": {
        "title": "...",
        "url": "...",
        "published": "...",
        "source": "...",
        "cached_at": "2026-03-21T00:00:00Z",
        "curriculum": { ...groq analysis... }   # None if skipped
        "skipped": false
      },
      ...
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
    """Save the cache back to disk."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"💾 Cache saved: {len(cache)} articles")
    except Exception as e:
        print(f"❌ Cache save failed: {e}")


# =============================================================================
# FEED FETCHING
# =============================================================================

def fetch_feed(url):
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

        # Strip markdown code fences if present
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

# ── Step 1: Load cache ──────────────────────────────────────────────────────
cache = load_cache()
cache_hits = 0
new_analyses = 0
skipped_count = 0
error_count = 0

# ── Step 2: Fetch all feeds and collect raw articles ────────────────────────
print("\n🔄 Fetching RSS feeds...")
raw_articles = []   # All candidate articles from feeds this run
feed_stats = {}

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
        word_count = len(content.split())
        if word_count < 300:
            continue

        raw_articles.append({
            "title": getattr(entry, "title", "Untitled"),
            "url": entry.link,
            "published": entry.get("published", entry.get("updated", "")),
            "source": feed.feed.title if hasattr(feed.feed, "title") else feed_name,
            "content": content,
        })
        count += 1

    feed_stats[feed_name] = count
    print(f"    ✅ {count} articles")
    time.sleep(0.3)

print(f"\n📚 Total fetched this run: {len(raw_articles)}")

# ── Step 3: Analyze only uncached articles ───────────────────────────────────
print("\n🤖 Analyzing new articles with Llama 3.3 70B...")

new_article_count = sum(1 for a in raw_articles if a["url"] not in cache)
print(f"   Cache hits: {len(raw_articles) - new_article_count} articles (skipping Groq)")
print(f"   New articles to analyze: {new_article_count}\n")

for i, article in enumerate(raw_articles, 1):
    url = article["url"]

    # ── Cache hit — skip Groq entirely ──────────────────────────────────────
    if url in cache:
        cache_hits += 1
        print(f"  [{i}/{len(raw_articles)}] 💾 cached  {article['title'][:55]}...")
        # Update metadata in case title/source changed (URL is the key)
        cache[url]["title"] = article["title"]
        cache[url]["published"] = article["published"]
        cache[url]["source"] = article["source"]
        continue

    # ── New article — call Groq ──────────────────────────────────────────────
    print(f"  [{i}/{len(raw_articles)}] 🤖 new     {article['title'][:55]}...")
    analysis = analyze_article(article["title"], article["content"])

    if analysis is None:
        error_count += 1
        print(f"    ❌ Analysis failed — not caching")
        continue

    if analysis.get("skip"):
        skipped_count += 1
        # Cache the skip so we don't retry it every run
        cache[url] = {
            "title": article["title"],
            "url": url,
            "published": article["published"],
            "source": article["source"],
            "cached_at": datetime.utcnow().isoformat() + "Z",
            "skipped": True,
            "curriculum": None,
        }
        print(f"    ⏭️  Skipped (cached)")
        continue

    # Valid analysis — cache it
    cache[url] = {
        "title": article["title"],
        "url": url,
        "published": article["published"],
        "source": article["source"],
        "cached_at": datetime.utcnow().isoformat() + "Z",
        "skipped": False,
        "curriculum": analysis,
    }
    new_analyses += 1

    article_type = analysis.get("article_type", "teaches")
    primary = analysis.get("primary_concept", "unknown")[:35]
    print(f"    ✅ {article_type} | {primary} | {analysis['difficulty']['level']}")

    time.sleep(0.3)

# ── Step 4: Save updated cache immediately ───────────────────────────────────
save_cache(cache)

# ── Step 5: Build curriculum from cache (all valid articles, not just today's fetch) ──
print("\n📦 Building curriculum from full cache...")

# Pull every non-skipped article from the cache (not just this run's articles)
# This means curriculum.json always reflects the full historical archive
analyzed = [
    {
        "title": entry["title"],
        "url": entry["url"],
        "published": entry["published"],
        "source": entry["source"],
        "curriculum": entry["curriculum"],
    }
    for entry in cache.values()
    if not entry.get("skipped") and entry.get("curriculum") is not None
]

print(f"   Total valid articles in archive: {len(analyzed)}")

# ── Step 6: Build concept registry ──────────────────────────────────────────
print("\n🧠 Building concept registry...")
concepts = build_concept_registry(analyzed)
print(f"   Found {len(concepts)} unique concepts")

teaches_count = sum(1 for a in analyzed if a["curriculum"].get("article_type") == "teaches")
extends_count = sum(1 for a in analyzed if a["curriculum"].get("article_type") == "extends")
orphaned = [c for c in concepts.values() if len(c["teaches"]) == 0 and len(c["extends"]) > 0]
print(f"   • {teaches_count} teach concepts")
print(f"   • {extends_count} extend concepts")
print(f"   • {len(orphaned)} orphaned concepts")

# ── Step 7: Group into topics ────────────────────────────────────────────────
topic_counts = {}
topics = {}

for article in analyzed:
    topic = article["curriculum"]["topic"]

    if topic not in topics:
        topics[topic] = {
            "name": topic,
            "article_count": 0,
            "articles": [],
            "levels": {
                "foundational": [],
                "beginner": [],
                "intermediate": [],
                "advanced": [],
                "application": [],
            },
        }

    topics[topic]["articles"].append(article)
    topics[topic]["article_count"] += 1
    level = article["curriculum"]["difficulty"]["level"].lower()
    if level in topics[topic]["levels"]:
        topics[topic]["levels"][level].append(article)
    topic_counts[topic] = topic_counts.get(topic, 0) + 1

# ── Step 8: Write curriculum.json ───────────────────────────────────────────
curriculum = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "total_articles": len(analyzed),
    "total_concepts": len(concepts),
    "concepts": concepts,
    "topics": topics,
    "articles": analyzed,
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(curriculum, f, indent=2)

print(f"\n✅ {OUTPUT_FILE} written")

# ── Step 9: Summary ──────────────────────────────────────────────────────────
print(f"""
╔══════════════════════════════════════════╗
║           Bramble Pipeline Done          ║
╠══════════════════════════════════════════╣
║  This run                                ║
║    Fetched from feeds : {len(raw_articles):<17} ║
║    Cache hits (saved) : {cache_hits:<17} ║
║    New analyses       : {new_analyses:<17} ║
║    Skipped            : {skipped_count:<17} ║
║    Errors             : {error_count:<17} ║
╠══════════════════════════════════════════╣
║  Archive                                 ║
║    Total cached       : {len(cache):<17} ║
║    In curriculum      : {len(analyzed):<17} ║
║    Concepts           : {len(concepts):<17} ║
╚══════════════════════════════════════════╝
""")

print("📚 Topics in curriculum:")
for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {topic}: {count} articles")

print("\n🧠 Top concepts by coverage:")
sorted_concepts = sorted(
    concepts.values(),
    key=lambda c: len(c["teaches"]) + len(c["extends"]),
    reverse=True,
)
for concept in sorted_concepts[:10]:
    t = len(concept["teaches"])
    e = len(concept["extends"])
    status = "✅" if t > 0 else "⚠️  orphan"
    print(f"   • {concept['name']}: {t} teach, {e} extend {status}")

print("\n🌿 Done!")
