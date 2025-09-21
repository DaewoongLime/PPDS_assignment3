# AI_USAGE.md

## 🎯 Purpose
This document records **how AI tools were used** during development, **what AI generated vs. what was human‑written**, problems found in AI suggestions, and how those were fixed. It also captures the **prompt history** that shaped the solution.

---

## 🧰 AI Tools Used
- **Assistant**: ChatGPT (GPT‑5 Thinking)
- **Usage Modes**: brainstorming, requirements clarification, code generation, code review, debugging, documentation drafting

---

## 📦 Project Areas Where AI Was Used
| Area | AI Involvement | Final Ownership |
|---|---|---|
| Requirements shaping (scope, sources, outputs) | High – iterated on idea, constraints, repo layout | Human reviewed & accepted |
| Web scraping (Transfermarkt) | Medium – helped design selectors & fallbacks | Human refined, tested, and validated |
| Reddit data collection | Medium – designed query strategy, deduping, local date filtering | Human tuned limits & windows |
| Data cleaning (pandas) | Medium – column validation, first‑team filtering | Human finalized rules |
| DeepSeek enrichment module | High – prompts, JSON schema, fallback logic, parsing | Human integrated & tested |
| Orchestration (ETL in `main.py`) | Medium – pipeline stages & I/O helpers | Human wiring & final checks |
| Documentation (README / DEEPSEEK_USAGE / AI_USAGE) | High – initial drafts | Human edits & approvals |

---

## 🧩 What Was AI‑Generated vs Human‑Written
- **AI‑generated (first pass, then edited):**
  - Prompt design (system & user prompts for fan‑perspective excitement scoring)
  - JSON response schema and parsing strategy (substring JSON extraction)
  - Baseline excitement heuristic and Spurs‑fan adjustments
  - Reddit query builder (aliases, keyword sets, pattern combinations)
  - ETL scaffolding and helper utilities (`ensure_directory`, `save_json`, `save_csv`)
  - Documentation drafts (README, DEEPSEEK_USAGE, this file)

- **Human‑written / finalized:**
  - Selector fixes for Transfermarkt scraping; robust date parsing & competition extraction
  - Reddit **time_filter=all** + local epoch filtering + window control
  - Outcome derivation (Win/Draw/Loss) and venue‑aware opponent mapping
  - Error handling & logging; environment variable management
  - Repository structure compliance and submission steps

---

## 🐞 Bugs / Gaps in AI Suggestions & Fixes

1. **KeyError: `home_team` / opponent parsing edge cases**  
   - *Issue*: Early scraping logic assumed stable table structure; some rows missed normalized team fields.  
   - *Fix*: Added venue‑aware team resolution, multiple selectors, fallbacks, and duplicate removal.

2. **Reddit search limited to last month**  
   - *Issue*: Using `time_filter="month"` excluded older threads.  
   - *Fix*: Switched to `time_filter="all"` and **locally filtered** by `created_utc` within `[match_date ± window_days]`.

3. **No strict date windowing on Reddit**  
   - *Issue*: Reddit API lacks exact date range; irrelevant results leaked in.  
   - *Fix*: Implemented client‑side epoch filtering and capped per‑query + per‑match totals.

4. **Undefined helpers (`save_json`, `save_csv`)**  
   - *Issue*: AI split helpers but didn’t include implementations.  
   - *Fix*: Implemented helpers with directory creation + UTF‑8 encoding.

5. **DeepSeek misunderstanding score semantics**  
   - *Issue*: Model sometimes flipped score interpretation (e.g., `"2:0"` mapping).  
   - *Fix*: Clarified prompt: **score order matches listing team order**; added explicit **Win/Draw/Loss** in cleaned data to guide the model.

6. **Non‑JSON model outputs**  
   - *Issue*: Occasional preambles or prose broke JSON parsing.  
   - *Fix*: Extracted substring between first `{` and last `}` before `json.loads`, with fallback scoring if parsing fails.

7. **Missing Reddit metrics for specific matches (e.g., PSG vs Spurs, Spurs vs Burnley)**  
   - *Issue*: Outside 1‑month window + alias coverage gaps.  
   - *Fix*: `time_filter="all"` + expanded alias list and keyword patterns; still accept empty metrics with graceful fallbacks.

---

## 🧠 Prompt History (Condensed)
Below are representative prompts provided by the developer that guided AI iterations (lightly edited for brevity/clarity):

- “Help me with this project; idea is to measure excitement for a Tottenham match by scraping Transfermarkt and using Reddit API; aggregate for DeepSeek to evaluate.”  
- “This is the file structure. Tell me step by step. Set up the scraper.”  
- “No more files; everything in `main.py`.” → *(Later reversed for modularity: moved AI code to `deepseek_enrichment.py`.)*  
- “Only the Transfermarkt scraper for now.”  
- “Here is page source. There’s a KeyError—no `home_team`.”  
- “Nothing was scraped; what are you trying to scrape?” → *Selector & parsing fixes.*  
- “Works. Now Reddit reactions.”  
- “`save_json()` and `save_csv()` are not defined.” → *Implemented helpers.*  
- “Results from Reddit are disappointing. What keywords are used?” → *Expanded query builder + aliases.*  
- “Make maximum number of threads possible.” → *Raised limits with dedupe and rate control.*  
- “Give me the whole `main.py`.” → *Provided full ETL flow.*  
- “Now ask DeepSeek per match.” → *Moved AI to `deepseek_enrichment.py` and adapted imports.*  
- “Slightly change the DeepSeek prompt: for A vs B, `2:0` means A=2, B=0.”  
- “No Reddit metrics for PSG vs Spurs and Spurs vs Burnley.”  
- “Probably a time issue; >1 month not fetched.” → *Switch to `time_filter="all"` + local epoch filter.*  
- “DeepSeek doesn’t get win vs loss from score; add explicit result during cleaning.”

---

## 🚦 Limitations & Mitigations
- **Scraping fragility**: Transfermarkt layout may change → *Multiple selectors + minimal assumptions; store raw cells for debugging.*  
- **Reddit coverage variance**: Some matches have sparse data → *Graceful fallbacks; keep metadata of queries; expose top‑post signals.*  
- **LLM variability**: Non‑deterministic outputs → *Low temperature + constrained JSON schema + fallback score.*

---

## 🔁 Repro & Verification
1. Prepare `.env` with `DEEPSEEK_API_KEY`, Reddit credentials.  
2. Run `python main.py` to execute full ETL.  
3. Inspect artifacts:  
   - `data/raw/transfermarkt_raw.csv`  
   - `data/raw/reddit/<match_id>.json`  
   - `data/enriched/transfermarkt_matches.csv`  
   - `data/enriched/reddit_metrics.csv`  
   - `data/enriched/excitement_scores.csv`  
4. Use the provided printer to view top AI results (`print_deepseek_results`).

---

## ✅ Credits & Attribution
- Human developer: project direction, testing, and final sign‑off.  
- AI assistant: scaffolding, prompts, code suggestions, and documentation drafts.
