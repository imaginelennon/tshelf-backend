import feedparser
from google import genai
from google.genai import types
import json
import os
from datetime import datetime

# Soul.md prompt
SOUL_PROMPT = """You are an expert AI educator building a learning curriculum.

Your task: Analyze this article and extract learning metadata.

Please provide:

1. TOPIC (which topic does this belong to?)
   - AI Evals & Benchmarking
   - Large Language Models
   - AI Coding Tools
   - AI Safety & Alignment
   - Prompt Engineering
   - Multimodal AI
   - Or suggest a new topic name

2. CONCEPTS TAUGHT (3-7 concepts this article teaches)
   - Use clear, specific names
   - Rate confidence (0-1) that article teaches this

3. PREREQUISITES
   - Which concepts must reader know first?
   - Rate confidence (0-1) for each

4. DIFFICULTY
   - Level (1-5): 1=beginner, 5=advanced
   - Technical depth (1-10)
   - Recommended reading time (minutes)

5. LEARNING OUTCOMES
   - What will reader understand after reading?
   - What becomes unlocked?

6. STRATEGIC QUESTIONS (2-3 questions for reflection)

Return as JSON with this structure:
{
  "topic": "topic name",
  "concepts_taught": [{"name": "concept", "confidence": 0.9}],
  "prerequisites": [{"concept": "prerequisite", "confidence": 0.8}],
  "difficulty": {
    "level": 3,
    "technical_depth": 6,
    "reading_time_minutes": 20
  },
  "learning_outcomes": ["outcome 1", "outcome 2"],
  "strategic_questions": ["question 1", "question 2"]
}
"""

# RSS feeds
FEEDS = [
    "https://www.anthropic.com/rss",
    "https://openai.com/index/rss.xml",
    "https://blog.google/technology/ai/rss/",
]

print("🔄 Starting curriculum generation...")

# Configure Gemini
client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

# Fetch articles
print("\n📡 Fetching RSS feeds...")
articles = []
for feed_url in FEEDS:
    try:
        feed = feedparser.parse(feed_url)
        source_name = feed.feed.get('title', feed_url)
        
        for entry in feed.entries[:10]:  # Last 10 posts per feed
            # Get content
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
            
            # Skip if too short
            if len(content) < 100:
                continue
                
            articles.append({
                "title": entry.title,
                "url": entry.link,
                "content": content[:8000],  # Limit content length
                "published": entry.get('published', ''),
                "source": source_name
            })
        
        print(f"  ✓ {source_name}: {len([e for e in feed.entries[:10]])} articles")
    except Exception as e:
        print(f"  ✗ Error fetching {feed_url}: {e}")
        continue

print(f"\n📊 Total articles to process: {len(articles)}")

# Process with Gemini
print("\n🤖 Processing with Gemini...")
processed = []
for i, article in enumerate(articles, 1):
    try:
        print(f"  [{i}/{len(articles)}] Processing: {article['title'][:50]}...")
        
        prompt = f"{SOUL_PROMPT}\n\n---\n\nArticle Title: {article['title']}\nSource: {article['source']}\n\nContent:\n{article['content']}"
        
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents=prompt
        )
        
        # Extract JSON from response
        response_text = response.text
        
        # Try to find JSON in response
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            response_text = response_text[json_start:json_end].strip()
        
        result = json.loads(response_text)
        
        article['curriculum'] = result
        article.pop('content', None)  # Remove full content to save space
        processed.append(article)
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        continue

print(f"\n✅ Successfully processed: {len(processed)}/{len(articles)} articles")

# Build curriculum structure
print("\n🏗️  Building curriculum structure...")
curriculum = {
    "generated_at": datetime.now().isoformat(),
    "total_articles": len(processed),
    "topics": {},
    "articles": processed
}

# Group by topics
topic_counts = {}
for article in processed:
    topic = article['curriculum'].get('topic', 'Other')
    
    if topic not in curriculum['topics']:
        curriculum['topics'][topic] = {
            "name": topic,
            "article_count": 0,
            "articles": [],
            "levels": {
                "1": [], "2": [], "3": [], "4": [], "5": []
            }
        }
    
    curriculum['topics'][topic]['articles'].append(article)
    curriculum['topics'][topic]['article_count'] += 1
    
    # Add to level
    level = str(article['curriculum']['difficulty']['level'])
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

