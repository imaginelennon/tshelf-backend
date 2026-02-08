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

# RSS Feeds
FEEDS = [
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feed_anthropic_news.xml",
    "https://openai.com/news/rss.xml",
    "https://blog.google/technology/ai/rss/",
]

# Soul.md prompt
SOUL_PROMPT = """You are an AI curriculum designer analyzing technical articles.

CRITICAL: You MUST return valid JSON. No markdown, no code blocks, just raw JSON.

TOPICS (pick ONE):
- AI Evaluations & Benchmarking
- Large Language Models  
- AI Safety & Alignment
- Agentic AI & Reasoning
- AI Infrastructure & Tooling

Analyze this article and determine:

1. SKIP ONLY if:
   - Job postings or recruiting
   - Event announcements (conferences, meetups)
   - Pure marketing/sales content with no learning value
   
   DO NOT SKIP if article teaches ANY technical concept, even if it's:
   - A product announcement that explains how something works
   - News about a feature that includes technical details
   - Company updates that discuss technical implementations

2. TOPIC: Which topic does it teach? (pick ONE from list above)

3. CONCEPTS TAUGHT (2-5 concepts):
   - What core concepts does this article teach?
   - Be specific (e.g., "RLHF training pipeline" not "AI training")
   - Include confidence (0-1)

4. PREREQUISITES (0-3 concepts):
   - What should reader understand before reading?
   - Only include if truly required
   - Include confidence

5. DIFFICULTY LEVEL:
   - foundational: First principles, no prerequisites, fundamental concepts
   - beginner: Basic AI familiarity helpful, introductory
   - intermediate: Solid understanding of AI concepts required
   - advanced: Deep technical knowledge required, expert-level
   - application: Focus on real-world implications, industry impact
   
   Technical depth (1-10): How technical is the writing?
   Reading time: Realistic estimate in minutes

6. LEARNING OUTCOMES (2-4 outcomes):
   - What will reader actually understand?
   - Be concrete and specific

7. STRATEGIC QUESTIONS (2-3 questions):
   - Thought-provoking
   - Encourage critical thinking about implications

IMPORTANT: Return ONLY valid JSON. No markdown code blocks. No explanations.

If skipping, return:
{{"skip": true}}

If not skipping, return:
{{
  "skip": false,
  "topic": "AI Evaluations & Benchmarking",
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
    "specific outcome 1",
    "specific outcome 2"
  ],
  "strategic_questions": [
    "thought-provoking question 1",
    "thought-provoking question 2"
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
        response = requests.get(url, headers=headers, timeout=10)
        return feedparser.parse(response.content)
    except Exception as e:
        print(f"❌ Error fetching {url}: {e}")
        return None

def extract_content(entry):
    """Extract content from feed entry"""
    content = ""
    
    # Try different content fields
    if hasattr(entry, 'content'):
        content = entry.content[0].value
    elif hasattr(entry, 'summary'):
        content = entry.summary
    elif hasattr(entry, 'description'):
        content = entry.description
    
    # Basic cleanup
    content = content.replace('<p>', '').replace('</p>', '\n')
    content = content.replace('<br>', '\n').replace('<br/>', '\n')
    
    # Limit length
    if len(content) > 3000:
        content = content[:3000] + "..."
    
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
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
            result = result.strip()
        
        # Try to parse JSON
        parsed = json.loads(result)
        
        # If it's just {"skip": true}, that's valid
        if parsed.get('skip') == True:
            return {'skip': True}
        
        # Otherwise validate it has required fields
        if not parsed.get('skip') and parsed.get('topic'):
            return parsed
        else:
            print(f"❌ Invalid response structure")
            print(f"Response: {result[:300]}")
            return None
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parse error: {e}")
        print(f"Raw response: {result[:500]}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Fetch articles
print("🔄 Fetching RSS feeds...")
articles = []

for feed_url in FEEDS:
    print(f"  • {feed_url}")
    feed = fetch_feed(feed_url)
    
    if not feed or not hasattr(feed, 'entries'):
        continue
    
    for entry in feed.entries[:10]:  # Limit per feed
        content = extract_content(entry)
        
        # Skip if too short
        if len(content) < 100:
            continue
        
        articles.append({
            'title': entry.title,
            'url': entry.link,
            'published': entry.get('published', ''),
            'source': feed.feed.title if hasattr(feed.feed, 'title') else 'Unknown',
            'content': content
        })

print(f"\n📚 Found {len(articles)} articles to analyze")

# Analyze articles
print("\n🤖 Analyzing with Llama 3.3 70B...")
analyzed = []
topic_counts = {}

for i, article in enumerate(articles, 1):
    print(f"  [{i}/{len(articles)}] {article['title'][:60]}...")
    
    analysis = analyze_article(article['title'], article['content'])
    
    if not analysis or analysis.get('skip'):
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
    
    # Rate limiting
    time.sleep(0.5)

# Build curriculum JSON
print("\n📦 Building curriculum...")

curriculum = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'total_articles': len(analyzed),
    'topics': {},
    'articles': analyzed
}

# Group by topic
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
print(f"\n📚 Topics discovered:")
for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  • {topic}: {count} articles")

print("\n✨ Done!")
