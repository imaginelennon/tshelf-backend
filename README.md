# TShelf Backend

Automated curriculum generation from AI research blogs using Groq.

## Setup Instructions

### Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `tshelf-backend`
3. Set to **Public**
4. Click "Create repository"

### Step 2: Get Groq API Key

1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Go to API Keys section
4. Click "Create API Key"
5. Copy the key (starts with `gsk_...`)

### Step 3: Add API Key to GitHub

1. In your repo: Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GROQ_API_KEY`
4. Value: Paste your key
5. Click "Add secret"

### Step 4: Upload Files

1. Create folder structure: `.github/workflows/`
2. Upload:
   - `process.py` (root)
   - `daily.yml` (in `.github/workflows/`)
   - `README.md` (root)

### Step 5: Run First Time

1. Actions tab → "Update Curriculum" → "Run workflow"
2. Wait 1-2 minutes
3. Verify `curriculum.json` appears

### Your JSON URL:
```
https://raw.githubusercontent.com/YOUR_USERNAME/tshelf-backend/main/curriculum.json
```

## Tech Stack

- **Model:** Llama 3.3 70B (via Groq)
- **Free tier:** 14,400 requests/day
- **Cost:** $0/month

## How It Works

- Runs daily at 9am UTC
- Fetches: OpenAI, Google AI, Anthropic blogs
- Analyzes with Llama 3.3 70B
- Generates curriculum.json
- iOS app fetches this JSON

## Costs

**All free:**
- GitHub Actions: 2000 min/month free
- Groq API: 14,400 requests/day free
- No credit card needed
