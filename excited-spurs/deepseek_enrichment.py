# deepseek_enrichment.py — DeepSeek enrichment for Spurs excitement scoring
# Python 3.8+ compatible

import os
import re
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "https://api.deepseek.com/chat/completions")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

def _save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV with directory creation"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def _deepseek_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> dict:
    """Make API call to DeepSeek chat completion endpoint"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DEEPSEEK_API_KEY is missing. Add it to your .env file")
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL, 
        "messages": messages, 
        "temperature": temperature
    }
    
    response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def _calculate_spurs_fan_excitement_baseline(match_row: dict, 
                                           metrics: dict) -> float:
    """
    Calculate Tottenham fan excitement baseline (0-70) considering:
    - Spurs fans are more excited by wins than neutrals
    - Losses against rivals are especially painful  
    - Late drama and comebacks are peak excitement
    - European competitions matter more
    """
    posts_count = metrics.get("posts_count", 0)
    comments_count = metrics.get("comments_count", 0)
    avg_post_score = metrics.get("avg_post_score", 0.0)
    comments_per_post = metrics.get("comments_per_post", 0.0)

    # Volume score (max ~25 points)
    import math
    volume_metric = math.log(1 + posts_count + comments_count)
    volume_score = min(25.0, volume_metric * 6.0)

    # Quality/engagement score (max ~15 points)
    quality_score = max(0.0, min(15.0, (avg_post_score/50.0)*7 + 
                                (comments_per_post/20.0)*8))

    # Result-based excitement for Spurs fans (max ~20 points)
    result = (match_row.get("result") or "")
    venue = match_row.get("venue", "")
    spurs_excitement = 0.0
    
    if ":" in result or "-" in result:
        home_goals, away_goals = result.replace("-", ":").split(":")[:2]
        try:
            home_goals, away_goals = int(home_goals), int(away_goals)
            
            if venue == "Home":
                spurs_goals, opponent_goals = home_goals, away_goals
            else:
                spurs_goals, opponent_goals = away_goals, home_goals
            
            # Spurs fan excitement based on result
            if spurs_goals > opponent_goals:  # Win
                goal_diff = spurs_goals - opponent_goals
                if goal_diff == 1:  # Narrow win - most exciting
                    spurs_excitement = 20.0
                elif goal_diff == 2:  # Comfortable win
                    spurs_excitement = 18.0
                elif spurs_goals >= 4:  # Thrashing - exciting but rare
                    spurs_excitement = 19.0
                else:  # Big win
                    spurs_excitement = 15.0
            elif spurs_goals == opponent_goals:  # Draw
                if spurs_goals >= 2:  # High-scoring draw
                    spurs_excitement = 12.0
                else:  # Boring draw
                    spurs_excitement = 6.0
            else:  # Loss - excitement depends on context
                goal_diff = opponent_goals - spurs_goals
                if goal_diff == 1:  # Narrow loss - still dramatic
                    spurs_excitement = 8.0
                elif spurs_goals >= 2:  # High-scoring loss
                    spurs_excitement = 7.0
                else:  # Bad loss
                    spurs_excitement = 3.0
                    
        except Exception:
            spurs_excitement = 5.0

    # Opposition and competition context (max ~15 points)
    competition = (match_row.get("competition") or "").lower()
    opponent = ""
    
    # Get opponent name correctly based on venue
    if venue == "Home":
        opponent = (match_row.get("away_team") or "").lower()
    else:
        opponent = (match_row.get("home_team") or "").lower()
    
    # Big Six rivals - higher stakes
    big_rivals = ["arsenal", "chelsea", "manchester united", "manchester city", 
                  "liverpool"]
    london_rivals = ["arsenal", "chelsea", "west ham", "brentford", "fulham", 
                     "crystal palace"]
    
    # Competition importance
    competition_multiplier = 1.0
    if any(comp in competition for comp in ["champions", "europa", "conference"]):
        competition_multiplier = 1.5
    elif "cup" in competition or "trophy" in competition:
        competition_multiplier = 1.2
    elif "premier league" in competition:
        competition_multiplier = 1.0
    
    context_score = 5.0  # Base score
    if any(rival in opponent for rival in big_rivals):
        context_score = 12.0  # Big rivalry
    elif any(rival in opponent for rival in london_rivals):
        context_score = 8.0   # London derby
    
    context_score *= competition_multiplier
    context_score = min(15.0, context_score)

    # Attendance factor (max ~5 points) - home atmosphere matters
    attendance = match_row.get("attendance") or 0
    attendance_score = min(5.0, attendance/12000.0)  # Scale for smaller stadiums

    total_baseline = (volume_score + quality_score + 
                     spurs_excitement + context_score + attendance_score)
    return round(min(70.0, total_baseline), 2)

# System prompt optimized for Tottenham fan perspective
TOTTENHAM_FAN_SYSTEM_PROMPT = """You are a Tottenham Hotspur fan and data analyst specializing in measuring match excitement from a Spurs supporter's perspective.

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
- excitement_score: 0-100 integer (Tottenham fan excitement level)
- tags: 3-5 short descriptive tags
- summary: 2-3 sentences from a Spurs fan perspective  
- reasons: 3-4 bullet points explaining your scoring

Be realistic about Spurs fan emotions. Output JSON only."""

# User prompt template optimized for fan excitement analysis
TOTTENHAM_FAN_PROMPT_TEMPLATE = """MATCH ANALYSIS REQUEST

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
{{
  "excitement_score": <0-100 integer>,
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "summary": "<Fan perspective 2-3 sentences>",
  "reasons": ["adjustment reason 1", "reason 2", "reason 3", "reason 4"]
}}"""

def _enrich_single_match_for_spurs_fans(match_row: dict, reddit_bundle: dict, 
                                       reddit_metrics: dict) -> dict:
    """
    Process a single match through DeepSeek AI for Tottenham fan excitement scoring.
    
    Uses Spurs-specific prompts and baseline calculation to get realistic
    fan excitement ratings rather than neutral sports analysis.
    """
    posts = reddit_bundle.get("posts", [])
    comments = reddit_bundle.get("comments", [])
    
    # Extract sample content for AI analysis
    post_titles = [post.get("title", "") for post in posts[:5]]
    comment_samples = [comment.get("body", "")[:200] for comment in comments[:8]]

    # Calculate Spurs fan baseline excitement
    baseline_score = _calculate_spurs_fan_excitement_baseline(match_row, reddit_metrics)
    
    # Format match metadata for better AI understanding
    match_metadata = {
        "date": match_row.get("date"),
        "opponent": (match_row.get("away_team") if match_row.get("venue") == "Home" 
                    else match_row.get("home_team")),
        "venue": match_row.get("venue"),
        "competition": match_row.get("competition"),
        "result": match_row.get("result"),
        "attendance": match_row.get("attendance")
    }
    
    # Create Tottenham fan-focused prompt
    user_prompt = TOTTENHAM_FAN_PROMPT_TEMPLATE.format(
        match_metadata=json.dumps(match_metadata, ensure_ascii=False, indent=2),
        reddit_metrics=json.dumps(reddit_metrics, ensure_ascii=False, indent=2),
        post_titles=json.dumps(post_titles, ensure_ascii=False, indent=2),
        comment_samples=json.dumps(comment_samples, ensure_ascii=False, indent=2),
        baseline_score=baseline_score
    )

    try:
        # Make API call to DeepSeek with Spurs fan context
        response = _deepseek_chat([
            {"role": "system", "content": TOTTENHAM_FAN_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], temperature=0.3)  # Slightly more creative for fan perspective
        
        # Parse JSON response safely
        response_text = response["choices"][0]["message"]["content"]
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        
        parsed_data = {}
        if json_start != -1 and json_end != -1:
            json_content = response_text[json_start:json_end+1]
            parsed_data = json.loads(json_content)
            
    except Exception as e:
        print(f"Warning: DeepSeek API call failed: {e}")
        parsed_data = {}
    
    # Fallback scoring with Spurs fan bias if API fails
    fallback_score = max(10, min(100, int(baseline_score * 1.4)))
    
    # Extract results with fan-appropriate defaults
    excitement_score = int(parsed_data.get("excitement_score", fallback_score))
    tags = parsed_data.get("tags", ["spurs-match"])
    summary = parsed_data.get("summary", "Match analysis unavailable")
    reasons = parsed_data.get("reasons", ["API analysis failed"])
    
    # Ensure excitement score is reasonable (0-100)
    excitement_score = max(0, min(100, excitement_score))
    
    return {
        "excitement_score": excitement_score,
        "tags": tags,
        "summary": summary,
        "reasons": reasons,
        "baseline_score": baseline_score  # Keep for debugging
    }

def run_deepseek_enrichment(
    matches_csv: str,
    reddit_metrics_csv: str,
    out_csv: str = "data/enriched/excitement_scores.csv"
):
    """
    Main enrichment function that:
    1. Reads match metadata and Reddit metrics
    2. Loads raw Reddit data bundles for each match
    3. Processes each match through DeepSeek AI
    4. Outputs comprehensive excitement scores and analysis
    """
    # Validate input files exist
    if not os.path.exists(matches_csv):
        raise FileNotFoundError(f"{matches_csv} not found.")
    if not os.path.exists(reddit_metrics_csv):
        raise FileNotFoundError(f"{reddit_metrics_csv} not found. Run the Reddit collection step first.")

    # Load datasets
    matches_df = pd.read_csv(matches_csv)  # Contains competition/result/attendance data
    reddit_df = pd.read_csv(reddit_metrics_csv)  # Contains Reddit engagement metrics

    # Create lookup dictionary for match metadata by match_id
    matches_by_id: Dict[str, Dict[str, Any]] = {
        row["match_id"]: (row._asdict() if hasattr(row, "_asdict") else row.to_dict())
        for _, row in matches_df.iterrows()
    }

    enriched_rows: List[Dict[str, Any]] = []
    
    # Process each match with Reddit data
    for _, reddit_row in reddit_df.iloc[::-1].iterrows():
        match_id = reddit_row["match_id"]
        match_metadata = matches_by_id.get(match_id, {})
        
        # Load raw Reddit bundle data saved by main.py
        reddit_bundle_path = os.path.join("data", "raw", f"reddit_{match_id}.json")
        reddit_bundle = {"posts": [], "comments": []}
        
        if os.path.exists(reddit_bundle_path):
            with open(reddit_bundle_path, "r", encoding="utf-8") as file:
                reddit_bundle = json.load(file)

        # Process through DeepSeek AI with Spurs fan perspective
        enrichment_results = _enrich_single_match_for_spurs_fans(
            match_metadata, 
            reddit_bundle, 
            reddit_row.to_dict()
        )
        
        # Compile final output row
        enriched_rows.append({
            "match_id": match_id,
            "date": match_metadata.get("date"),
            "home_team": match_metadata.get("home_team"),
            "away_team": match_metadata.get("away_team"),
            "competition": match_metadata.get("competition"),
            "result": match_metadata.get("result"),
            "attendance": match_metadata.get("attendance"),
            "posts_count": reddit_row.get("posts_count"),
            "comments_count": reddit_row.get("comments_count"),
            "avg_post_score": reddit_row.get("avg_post_score"),
            "comments_per_post": reddit_row.get("comments_per_post"),
            "excitement_score": enrichment_results["excitement_score"],
            "tags": ", ".join(enrichment_results["tags"]),
            "ai_summary": enrichment_results["summary"],
            "analysis_reasons": " | ".join(enrichment_results["reasons"]),
        })

    # Save enriched dataset
    output_df = pd.DataFrame(enriched_rows)
    _save_csv(output_df, out_csv)
    print(f"✨ DeepSeek excitement scores saved: {out_csv} (rows={len(output_df)})")

def print_deepseek_results(csv_path: str = "data/enriched/excitement_scores.csv", limit: int = 5):
    """
    Pretty-print DeepSeek enrichment results.
    - csv_path: path to the excitement_scores.csv file
    - limit: number of matches to display (default 5)
    """
    if not os.path.exists(csv_path):
        print(f"❗ File not found: {csv_path}. Run enrichment first.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("❗ No rows in results.")
        return

    for _, row in df.head(limit).iterrows():
        print("────────────────────────────")
        print(f"{row['date']} — {row['home_team']} vs {row['away_team']}")
        print(f"Competition : {row['competition']}")
        print(f"Result      : {row['result']}")
        print(f"Attendance  : {row['attendance']}")
        print(f"Excitement  : {row['excitement_score']}/100")
        print(f"Tags        : {row['tags']}")
        print(f"Summary     : {row['ai_summary']}")
        print(f"Reasons     : {row['analysis_reasons']}")
    print("────────────────────────────")