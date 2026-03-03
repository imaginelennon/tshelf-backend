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

# Updated prompt with article_type, primary_concept, and summary
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


def slugify(text):
    """Convert text to URL-friendly slug"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


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
    
    if hasattr(entry, 'content') and entry.content:
        content = entry.content[0].value
    elif hasattr(entry, 'summary_detail') and entry.summary_detail:
        content = entry.summary_detail.value
    elif hasattr(entry, 'summary'):
        content = entry.summary
    elif hasattr(entry, 'description'):
        content = entry.description
    
    # HTML cleanup
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
            # Ensure new fields have defaults
            if 'article_type' not in parsed:
                parsed['article_type'] = 'teaches'
            if 'primary_concept' not in parsed:
                # Fallback to first concept taught
                if parsed.get('concepts_taught'):
                    parsed['primary_concept'] = parsed['concepts_taught'][0]['name']
                else:
                    parsed['primary_concept'] = 'general'
            if 'summary' not in parsed:
                parsed['summary'] = ''
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


def build_concept_registry(articles):
    """Build concept registry from analyzed articles"""
    concepts = {}
    
    for article in articles:
        curriculum = article['curriculum']
        primary = curriculum.get('primary_concept', '')
        if not primary:
            continue
            
        concept_id = slugify(primary)
        article_type = curriculum.get('article_type', 'teaches')
        topic = curriculum.get('topic', '')
        
        if concept_id not in concepts:
            concepts[concept_id] = {
                'id': concept_id,
                'name': primary,
                'aliases': set([primary]),
                'topic': topic,
                'teaches': [],
                'extends': [],
                'prerequisite_concepts': set()
            }
        else:
            # Add alias if different casing/wording
            concepts[concept_id]['aliases'].add(primary)
        
        # Add article to appropriate list
        if article_type == 'teaches':
            concepts[concept_id]['teaches'].append(article['url'])
        else:
            concepts[concept_id]['extends'].append(article['url'])
        
        # Extract prerequisite concepts
        for prereq in curriculum.get('prerequisites', []):
            prereq_id = slugify(prereq['name'])
            if prereq_id and prereq_id != concept_id:
                concepts[concept_id]['prerequisite_concepts'].add(prereq_id)
    
    # Convert sets to lists for JSON serialization
    for concept_id in concepts:
        concepts[concept_id]['aliases'] = list(concepts[concept_id]['aliases'])
        concepts[concept_id]['prerequisite_concepts'] = list(concepts[concept_id]['prerequisite_concepts'])
    
    return concepts


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("🔄 Fetching RSS feeds...")
print(f"📡 Total feeds: {len(FEEDS)}\n")

articles = []
feed_stats = {}

for feed_url in FEEDS:
    feed_name = feed_url.split('/')[2]
    print(f"  📰 {feed_name}")
    
    feed = fetch_feed(feed_url)
    
    if not feed or not hasattr(feed, 'entries'):
        print(f"    ❌ Failed to fetch")
        feed_stats[feed_name] = 0
        continue
    
    count = 0
    for entry in feed.entries[:15]:
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
    
    article_type = analysis.get('article_type', 'teaches')
    primary = analysis.get('primary_concept', 'unknown')[:30]
    print(f"    ✅ {article_type} | {primary} | {analysis['difficulty']['level']}")
    
    time.sleep(0.3)

# Build concept registry
print("\n🧠 Building concept registry...")
concepts = build_concept_registry(analyzed)
print(f"   Found {len(concepts)} unique concepts")

# Count teaches vs extends
teaches_count = sum(1 for a in analyzed if a['curriculum'].get('article_type') == 'teaches')
extends_count = sum(1 for a in analyzed if a['curriculum'].get('article_type') == 'extends')
print(f"   • {teaches_count} articles teach concepts")
print(f"   • {extends_count} articles extend concepts")

# Count orphaned concepts (extends but no teaches)
orphaned = [c for c in concepts.values() if len(c['teaches']) == 0 and len(c['extends']) > 0]
print(f"   • {len(orphaned)} orphaned concepts (extensions without foundational content)")

# Build curriculum JSON
print("\n📦 Building curriculum...")

curriculum = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'total_articles': len(analyzed),
    'total_concepts': len(concepts),
    'concepts': concepts,
    'topics': {},
    'articles': analyzed
}

# Group by topic (for backward compatibility)
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
print(f"   • Analyzed: {len(analyzed)} articles ({len(analyzed)/len(articles)*100:.1f}%)" if articles else "   • Analyzed: 0 articles")
print(f"   • Skipped: {len(skipped)} articles")
print(f"   • Errors: {len(errors)} articles")

print(f"\n📚 Topics discovered:")
for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"   • {topic}: {count} articles")

print(f"\n🧠 Top concepts by coverage:")
sorted_concepts = sorted(concepts.values(), key=lambda c: len(c['teaches']) + len(c['extends']), reverse=True)
for concept in sorted_concepts[:10]:
    teaches = len(concept['teaches'])
    extends = len(concept['extends'])
    status = "✅" if teaches > 0 else "⚠️ orphan"
    print(f"   • {concept['name']}: {teaches} teach, {extends} extend {status}")

print("\n✨ Done!")
