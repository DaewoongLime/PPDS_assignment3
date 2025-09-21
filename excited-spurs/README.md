# Spurs Excitement Analysis Pipeline ⚽🔥

## 📌 Project Overview
This project builds an **AI-enhanced ETL pipeline** for Tottenham Hotspur matches:

- **Scrape** match data from Transfermarkt
- **Collect** fan reactions from Reddit
- **Clean & process** the data using Pandas
- **Enrich** with DeepSeek API to calculate **fan excitement scores (0–100)**

The output provides per-match excitement levels, tags, summaries, and reasoning from a Spurs fan perspective.

---

## 🏗️ Data Sources
1. **Transfermarkt (Web Scraping)**
   - Match fixtures, competition, results, attendance, home/away info.
   - Example URL: [Tottenham Fixtures 24/25](https://www.transfermarkt.com/tottenham-hotspur/vereinsspielplan/verein/148/saison_id/2025/heim_gast/)

2. **Reddit (API)**
   - Subreddits: `soccer`, `coys`, `PremierLeague`, `soccerhighlights`
   - Extract posts and comments around match dates
   - Metrics: post count, comment count, avg score, engagement, top posts

3. **DeepSeek API (AI Enrichment)**
   - Analyze Tottenham fan excitement using:
     - Match context (result, opponent, competition importance)
     - Reddit fan sentiment and engagement
   - Output JSON:
     ```json
     {
       "excitement_score": 85,
       "tags": ["derby", "last-minute-goal", "home-win"],
       "summary": "A dramatic derby win with fans going wild...",
       "reasons": ["1-goal victory", "London rival", "Reddit buzz"]
     }
     ```

---

## ⚙️ Installation & Usage

### 1. Clone Repo
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables (`.env`)
Create a `.env` file in the root directory:
```
DEEPSEEK_API_KEY=your_key_here
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=excited-spurs/0.1
```

### 4. Run the Pipeline
```bash
python main.py
```

### 5. Check Results
- **Raw data:** `data/raw/`
- **Clean data:** `data/enriched/transfermarkt_matches.csv`, `data/enriched/reddit_metrics.csv`
- **Final AI results:** `data/enriched/excitement_scores.csv`

---

## 📊 Before / After Example
**Before (Raw Reddit JSON snippet):**
```json
{
  "title": "Match Thread: Tottenham vs Arsenal",
  "score": 520,
  "num_comments": 1345
}
```

**After (AI Enriched CSV snippet):**
| date       | home_team | away_team | result | excitement_score | tags                    | summary                          |
|------------|-----------|-----------|--------|-----------------|-------------------------|----------------------------------|
| 2025-02-15 | Spurs     | Arsenal   | 2-1    | 92              | derby, comeback, rivalry| "A dramatic London derby win..." |

---

## 📂 Repository Structure
```
project-root/
├── main.py
├── deepseek_enrichment.py
├── data/
│   ├── raw/          # Original scraped data
│   ├── enriched/     # Clean + AI-enhanced data
├── examples/         # Before/after samples
├── requirements.txt
├── README.md
├── DEEPSEEK_USAGE.md
├── AI_USAGE.md
└── .gitignore
```

---

## 🙌 DeepSeek Enrichment Highlights
- Spurs fan psychology modeled into AI prompt
  - 1-goal wins > blowouts
  - Arsenal/Chelsea/City/Liverpool losses = painful
  - 0-0 draws = worst case
- Reddit volume & sentiment included in scoring
- Output includes 0–100 score, tags, summary, and reasoning

---

## 📑 Documentation
- `DEEPSEEK_USAGE.md` → prompts & enrichment strategies
- `AI_USAGE.md` → AI involvement in development
- `Dev Practices & Submission Instructions.pdf` → style & submission rules
- `Project 3- Data Alchemy.pdf` → assignment guidelines
