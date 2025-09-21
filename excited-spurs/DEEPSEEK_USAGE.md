# DEEPSEEK_USAGE.md

## 🎯 Purpose
This document describes how the **DeepSeek API** was integrated into the project and how it enhanced Tottenham Hotspur match data with **fan excitement analysis**.

---

## ⚙️ Prompts Used

### **System Prompt**
```text
You are a Tottenham Hotspur fan and data analyst specializing in measuring match excitement from a Spurs supporter's perspective.

Your job: Analyze match data and Reddit reactions to score how exciting/enjoyable each match was for Tottenham fans specifically (0-100 scale).

Key Tottenham fan psychology to consider:
- WINS are always more exciting than draws/losses (especially against rivals)
- Narrow wins (1-goal) are peak excitement - shows character and drama
- Losses against Arsenal, Chelsea, Man City, Liverpool are particularly painful
- European competition matches matter more than league games
- Late goals, comebacks, and dramatic moments amplify excitement
- Boring 0-0 draws are the worst possible outcome
- High-scoring games are exciting even if we lose (shows attacking football)
- Away wins are more satisfying than home wins

Output requirements:
- excitement_score: 0-100 integer
- tags: 3-5 short descriptive tags
- summary: 2-3 sentences from a Spurs fan perspective
- reasons: 3-4 bullet points explaining your scoring

Be realistic about Spurs fan emotions. Output JSON only.
```

---

### **User Prompt Template**
```text
MATCH ANALYSIS REQUEST

Match Details:
{match_metadata}

Reddit Fan Engagement:
{reddit_metrics}

Sample Fan Reactions:
- Post Titles: {post_titles}
- Comment Samples: {comment_samples}

Baseline Excitement Score: {baseline_score}/70

ANALYSIS INSTRUCTIONS:
- Convert baseline to final 0-100 excitement score for Tottenham fans
- Consider: Did Spurs win/lose? Against whom? How dramatic was it?
- Reddit volume/sentiment indicates fan emotional response
- Adjust baseline based on Spurs-specific context

Scoring Guide:
- 90-100: Legendary matches (big wins, dramatic comebacks, rival thrashings)
- 75-89: Great matches (solid wins, exciting games, good performances)
- 50-74: Decent matches (acceptable results, some entertainment)
- 25-49: Disappointing matches (poor performances, bad losses)
- 0-24: Terrible matches (humiliating defeats, boring draws)

Return JSON only:
{
  "excitement_score": <0-100 integer>,
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "summary": "<Fan perspective 2-3 sentences>",
  "reasons": ["reason 1", "reason 2", "reason 3"]
}
```

---

## 💡 Enrichment Strategy
- **Baseline Score (0–70)**: Calculated using match result, goal difference, competition importance, attendance, and Reddit engagement.  
- **DeepSeek Adjustment (Final 0–100)**: AI adjusts baseline considering:
  - Rivalries (e.g., Arsenal, Chelsea)  
  - Match drama (late goals, comebacks)  
  - Reddit emotional intensity  

---

## ⚠️ Challenges & Solutions
1. **DeepSeek occasionally returned non-JSON text**  
   - **Fix**: Extract JSON substring (`find("{") ... rfind("}")`) before parsing.  
2. **Missing API Key**  
   - **Fix**: Environment variable check with error message if not set.  
3. **Low Reddit activity matches**  
   - **Fix**: Fallback score = baseline × 1.4 (bounded 10–100).  

---

## ✅ Example Output
```json
{
  "excitement_score": 92,
  "tags": ["derby", "last-minute-goal", "home-win"],
  "summary": "Spurs fans witnessed a thrilling derby with a decisive late goal. The atmosphere was electric both in the stadium and online.",
  "reasons": [
    "1-goal narrow win",
    "Rival opponent (Arsenal)",
    "High Reddit engagement",
    "Late decisive goal"
  ]
}
```
