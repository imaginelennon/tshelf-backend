import feedparser
from google import genai
from google.genai import types
import json
import os
from datetime import datetime

# Soul.md prompt
SOUL_PROMPT = """You are an expert AI educator building a learning curriculum.

Your task: Analyze this article and extract learning metadata.

IMPORTANT INSTRUCTIONS:

1. QUALITY FILTER - ONLY SKIP IF:
   - Pure advertisement with no content (e.g., "Watch our ad", "See our commercial")
   - Job posting or hiring announcement
   - Event-only announcement (e.g., "Join us at conference X")
   
   KEEP EVERYTHING ELSE, including:
   - Model/feature announcements (even if brief)
   - Blog post summaries or roundups
   - Podcast/video descriptions (they teach concepts)
   - Application examples (sports, conservation, etc)
   - News about AI developments
   
   When in doubt, KEEP the article.
   
   If skipping, return: {"skip": true, "reason": "brief explanation"}

2. TOPIC ASSIGNMENT:
   Choose ONE topic from this list (do NOT create new topics unless truly necessary):
   
   - AI Evals & Benchmarking (testing, evaluation, benchmarks, performance measurement)
   - Large Language Models (LLMs, transformers, GPT, Claude, Gemini, model architecture)
   - AI Safety & Alignment (safety testing, RLHF, constitutional AI, red teaming)
   - Multimodal AI (vision, audio, video, cross-modal models)
   - AI Applications (real-world use cases, industry applications, applied AI)
   - Generative AI (image/video/audio generation, world models, synthesis)
   - AI Research Methods (novel techniques, experiments, research papers)
   
   ONLY create a new topic if the article truly doesn't fit ANY of the above.

3. CONCEPTS TAUGHT (3-5 concepts max):
   - Be specific but not too granular
   - Focus on transferable concepts
   - Rate confidence (0-1) that article teaches this well

4. PREREQUISITES:
   - Only list if confidence > 0.8 (truly needed)
   - Use general concept names
   - Max 3 prerequisites
   - Be realistic - don't over-gatekeep

5. DIFFICULTY:
   - Level 1: Intro/announcement, no background needed
   - Level 2: Some AI familiarity helpful
   - Level 3: Solid understanding of AI concepts required
   - Level 4: Advanced, requires specific technical background
   - Level 5: Expert-level, cutting-edge research
   
   Technical depth (1-10): How technical is the writing?
   Reading time: Realistic estimate in minutes

6. LEARNING OUTCOMES (2-4 outcomes):
   - What will reader actually understand?
   - Be concrete and specific

7. STRATEGIC QUESTIONS (2-3 questions):
   - Thought-provoking
   - Encourage critical thinking about implications

Return as JSON:
{
  "skip": false,
  "topic": "topic name from list above",
  "concepts_taught": [
    {"name": "concept name", "confidence": 0.9}
  ],
  "prerequisites": [
    {"concept": "prerequisite concept", "confidence": 0.85}
  ],
  "difficulty": {
    "level": 2,
    "technical_depth": 4,
    "reading_time_minutes": 8
  },
  "learning_outcomes": [
    "specific outcome 1",
    "specific outcome 2"
  ],
  "strategic_questions": [
    "thought-provoking question 1?",
    "thought-provoking question 2?"
  ]
}

If skipping:
{
  "skip": true,
  "reason": "Pure advertisement with no educational content"
}
"""

# RSS feeds
FEEDS = [
    "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feed_anthropic_news.xml",
    "https://openai.com/news/rss.xml",
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
        feed = feedparser.parse(feed_url, agent='TShelf/1.0')
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
            #if len(content) < 100: continue
                
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
            model='gemini-2.5-flash',
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
        
        # Skip if marked as low quality
        if result.get('skip', False):
            print(f"    ⊘ Skipped: {result.get('reason', 'No reason')}")
            continue
        
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
