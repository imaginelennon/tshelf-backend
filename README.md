# TShelf Backend

Automated curriculum generation from AI research blogs.

## Setup Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `tshelf-backend`
3. Set to **Public** (so the JSON file is accessible)
4. Click "Create repository"

### Step 2: Add Files

1. Download these 3 files from this folder:
   - `process.py`
   - `.github/workflows/daily.yml` (note: create `.github/workflows/` folder first)
   - `README.md`

2. In your new GitHub repo, click "Add file" → "Upload files"
3. Upload all 3 files
4. Click "Commit changes"

### Step 3: Get Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

### Step 4: Add API Key to GitHub

1. In your repo, go to Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GEMINI_API_KEY`
4. Value: Paste your Gemini API key
5. Click "Add secret"

### Step 5: Run First Time

1. Go to "Actions" tab in your repo
2. Click "Update Curriculum" workflow
3. Click "Run workflow" → "Run workflow"
4. Wait 2-3 minutes
5. Check that `curriculum.json` appears in your repo

### Step 6: Get JSON URL

Your curriculum is now available at:
```
https://raw.githubusercontent.com/YOUR_USERNAME/tshelf-backend/main/curriculum.json
```

Replace `YOUR_USERNAME` with your GitHub username.

## How It Works

- Runs daily at 9am UTC
- Fetches latest posts from AI research blogs
- Analyzes with Gemini
- Updates curriculum.json
- Your iOS app fetches this JSON

## Updating Soul.md Prompt

1. Go to your repo on GitHub
2. Click `process.py`
3. Click the pencil icon (Edit)
4. Find the `SOUL_PROMPT` section
5. Edit the text
6. Click "Commit changes"
7. Next run will use the new prompt

## Adding RSS Feeds

Edit `process.py`, find the `FEEDS` list, add new URLs:

```python
FEEDS = [
    "https://www.anthropic.com/news/rss.xml",
    "https://openai.com/blog/rss/",
    "https://blog.google/technology/ai/rss/",
    "https://www.deepmind.com/blog/rss.xml",
    "https://your-new-feed-url/rss.xml",  # Add here
]
```

## Manual Trigger

To run immediately (instead of waiting for daily schedule):
1. Go to Actions tab
2. Click "Update Curriculum"
3. Click "Run workflow"
4. Click green "Run workflow" button

## Costs

- GitHub Actions: Free (2000 minutes/month)
- Gemini API: Free tier (60 requests/minute)
- Total: $0/month for MVP

## Next Steps

Once curriculum.json is generated:
1. Use the raw GitHub URL in your iOS app
2. App fetches and parses the JSON
3. Displays curriculum to users
