#!/usr/bin/env python3
import feedparser
import requests
import json
import time
from groq import Groq
import os
from datetime import datetime

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# RSS Feeds - EXPANDED LIST
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
    
    # Emerging/Startup Scene
    
]

# Soul.md prompt - SIMPLIFIED (removed most filtering)
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
   
   KEEP EVERYTHING ELSE including:
   - Product announcements (they explain features!)
   - Company news (often has technical details)
   - Tutorials and guides
   - Research papers
   - Opinion pieces about AI
   - Industry analysis
   - Tool releases

2. TOPIC: Which topic does it teach? (pick ONE from list above)

3. CONCEPTS TAUGHT (2-5 concepts):
   - What does this article cover?
   - Be specific (e.g., "chain-of-thought prompting" not "AI")
   - Include confidence (0-1)

4. PREREQUISITES (0-3 concepts):
   - What should reader know beforehand?
   - Only if truly required
   - Include confidence

5. DIFFICULTY LEVEL:
   - foundational: First principles, no prerequisites
   - beginner: Basic AI familiarity helpful
   - intermediate: Solid AI understanding required
   - advanced: Deep technical knowledge needed
   - application: Real-world implementation focus
   
   Technical depth (1-10): How technical?
   Reading time: Estimate in minutes

6. LEARNING OUTCOMES (2-4 outcomes):
   - What will reader understand after?
   - Be concrete

7. STRATEGIC QUESTIONS (2-3 questions):
   - Thought-provoking
   - Encourage deeper thinking

IMPORTANT: Return ONLY valid JSON. No markdown. No explanations.

If skipping (rare!), return:
{{"skip": true}}

Otherwise return:
{{
  "skip": false,
  "topic": "Large Language Models",
  "concepts_taught": [
    {{"name": "concept name", "confidence": 0.9}}
  ],
  "prerequisites": [
    {{"name": "prerequisite concept", "confidence": 0.85}}
  ],
  "difficulty": {{
    "level": "beginner",
    "technical_depth": 4,
    "reading_time_minutes": 8
  }},
  "learning_outcomes": [
    "outcome 1",
    "outcome 2"
  ],
  "strategic_questions": [
    "question 1",
    "question 2"
  ]
}}

Article:
Title: {title}
Content: {content}
"""

def fetch_feed(url):
    """Fetch and parse RSS feed"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; TShelf/1.0)'}
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

def extract_content(entry):
    """Extract content from feed entry - try multiple fields"""
    content = ""
    
    # Try different content fields in order of preference
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].value
    elif hasattr(entry, 'summary_detail') and entry.summary_detail:
        content = entry.summary_detail.value
    elif hasattr(entry, 'summary'):
        content = entry.summary
    elif hasattr(entry, 'description'):
        content = entry.description
    
    # More aggressive HTML cleanup
    import re
    content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
    content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
    content = re.sub(r'<[^>]+>', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    content = content.strip()
    
    # Limit length for API
    if len(content) > 4000:
        content = content[:4000] + "..."
    
    return content

def analyze_article(title, content):
    """Analyze article with Groq"""
    try:
        prompt = SOUL_PROMPT.format(title=title, content=content)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if result.startswith("```"):
            lines = result.split('\n')
            result = '\n'.join(lines[1:-1]) if len(lines) > 2 else result
            if result.startswith("json"):
                result = result[4:].strip()
        
        # Parse JSON
        parsed = json.loads(result)
        
        # Validate
        if parsed.get('skip') == True:
            return {'skip': True}
        
        if not parsed.get('skip') and parsed.get('topic'):
            return parsed
        else:
            print(f"    ⚠️  Invalid structure")
            return None
        
    except json.JSONDecodeError as e:
        print(f"    ❌ JSON error: {e}")
        print(f"    Response: {result[:200]}")
        return None
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

# Fetch articles
print("🔄 Fetching RSS feeds...")
print(f"📡 Total feeds: {len(FEEDS)}\n")

articles = []
feed_stats = {}

for feed_url in FEEDS:
    feed_name = feed_url.split('/')[2]  # Extract domain
    print(f"  📰 {feed_name}")
    
    feed = fetch_feed(feed_url)
    
    if not feed or not hasattr(feed, 'entries'):
        print(f"    ❌ Failed to fetch")
        feed_stats[feed_name] = 0
        continue
    
    count = 0
    # Get more articles per feed (up to 15 instead of 10)
    for entry in feed.entries[:15]:
        content = extract_content(entry)
        
        content = extract_content(entry)
        
        # Minimum 300 words to ensure substantive content
        word_count = len(content.split())
        if word_count < 300:
            continue
        
        articles.append({
            'title': entry.title,
            'url': entry.link,
            'published': entry.get('published', entry.get('updated', '')),
            'source': feed.feed.title if hasattr(feed.feed, 'title') else feed_name,
            'content': content
        })
        count += 1
    
    feed_stats[feed_name] = count
    print(f"    ✅ {count} articles")
    
    # Small delay between feeds
    time.sleep(0.3)

print(f"\n📚 Total articles fetched: {len(articles)}")
print(f"📊 Articles per source:")
for source, count in sorted(feed_stats.items(), key=lambda x: x[1], reverse=True):
    if count > 0:
        print(f"   • {source}: {count}")

# Analyze articles
print("\n🤖 Analyzing with Llama 3.3 70B...")
analyzed = []
skipped = []
errors = []

for i, article in enumerate(articles, 1):
    print(f"  [{i}/{len(articles)}] {article['title'][:60]}...")
    
    analysis = analyze_article(article['title'], article['content'])
    
    if not analysis:
        errors.append(article['title'])
        print(f"    ❌ Analysis failed")
        continue
    
    if analysis.get('skip'):
        skipped.append(article['title'])
        print(f"    ⏭️  Skipped")
        continue
    
    # Add to results
    analyzed.append({
        'title': article['title'],
        'url': article['url'],
        'published': article['published'],
        'source': article['source'],
        'curriculum': analysis
    })
    
    print(f"    ✅ {analysis['topic']} - {analysis['difficulty']['level']}")
    
    # Rate limiting (slightly faster)
    time.sleep(0.3)

# Build curriculum JSON
print("\n📦 Building curriculum...")

curriculum = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'total_articles': len(analyzed),
    'topics': {},
    'articles': analyzed
}

# Group by topic
topic_counts = {}
for article in analyzed:
    topic = article['curriculum']['topic']
    
    if topic not in curriculum['topics']:
        curriculum['topics'][topic] = {
            'name': topic,
            'article_count': 0,
            'articles': [],
            'levels': {
                'foundational': [],
                'beginner': [],
                'intermediate': [],
                'advanced': [],
                'application': []
            }
        }
    
    curriculum['topics'][topic]['articles'].append(article)
    curriculum['topics'][topic]['article_count'] += 1
    
    # Add to level
    level = article['curriculum']['difficulty']['level'].lower()
    if level in curriculum['topics'][topic]['levels']:
        curriculum['topics'][topic]['levels'][level].append(article)
    
    topic_counts[topic] = topic_counts.get(topic, 0) + 1

# Save
with open('curriculum.json', 'w') as f:
    json.dump(curriculum, f, indent=2)

print(f"\n📝 Curriculum saved to curriculum.json")

# Stats
print(f"\n📊 Processing Stats:")
print(f"   • Fetched: {len(articles)} articles")
print(f"   • Analyzed: {len(analyzed)} articles ({len(analyzed)/len(articles)*100:.1f}%)")
print(f"   • Skipped: {len(skipped)} articles")
print(f"   • Errors: {len(errors)} articles")

print(f"\n📚 Topics discovered:")
for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {topic}: {count} articles")

print("\n✨ Done!")
