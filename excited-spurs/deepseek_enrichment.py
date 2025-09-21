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

def _calculate_pre_score(match_row: dict, metrics: dict) -> float:
    """Calculate heuristic pre-score in 0–70 range to guide the LLM assessment"""
    posts_count = metrics.get("posts_count", 0)
    comments_count = metrics.get("comments_count", 0)
    avg_post_score = metrics.get("avg_post_score", 0.0)
    comments_per_post = metrics.get("comments_per_post", 0.0)

    # Volume score (max ~25 points)
    import math
    volume_metric = math.log(1 + posts_count + comments_count)
    volume_score = min(25.0, volume_metric * 6.0)

    # Quality score (max ~15 points)
    quality_score = max(0.0, min(15.0, (avg_post_score/50.0)*7 + (comments_per_post/20.0)*8))

    # Drama score (max ~15 points) based on goal difference
    goal_difference = 0
    result = (match_row.get("result") or "")
    if ":" in result or "-" in result:
        home_goals, away_goals = result.replace("-", ":").split(":")[:2]
        try:
            goal_difference = abs(int(home_goals) - int(away_goals))
        except Exception:
            goal_difference = 0
    
    # Drama scoring: 1-goal difference = highest drama
    if goal_difference == 1:
        drama_score = 15.0
    elif goal_difference == 2:
        drama_score = 10.0
    elif goal_difference == 0:  # Draw
        drama_score = 5.0
    else:
        drama_score = 2.0

    # Context score (max ~10 points) - big competitions and rivals
    competition = (match_row.get("competition") or "").lower()
    opponent = (match_row.get("away_team") or "").lower()
    
    is_big_competition = any(keyword in competition for keyword in ["champions", "ucl", "final", "semi"])
    is_big_rival = any(rival in opponent for rival in [
        "arsenal", "chelsea", "manchester", "liverpool", "city", "united", "newcastle"
    ])
    context_score = 10.0 if (is_big_competition or is_big_rival) else 4.0

    # Attendance score (max ~5 points)
    attendance = match_row.get("attendance") or 0
    attendance_score = min(5.0, attendance/10000.0)

    total_score = volume_score + quality_score + drama_score + context_score + attendance_score
    return round(total_score, 2)

# System prompt for DeepSeek AI
SYSTEM_PROMPT = """You are a sports data analyst specializing in fan excitement measurement.
Given match metadata, Reddit engagement signals, and a heuristic pre-score (0–70 range), you must produce:
- A final excitement_score (0–100 scale)
- 3–5 concise descriptive tags
- A 2–3 sentence summary of the match excitement
- 2–4 bullet point reasons explaining your adjustments to the pre-score (+/-)

Be concise, evidence-based, and output strict JSON format only."""

# User prompt template for DeepSeek API
USER_PROMPT_TEMPLATE = """Match Metadata:
{match_meta}

Reddit Engagement Metrics:
{reddit_metrics}

Sample Reddit Content Analysis:
- Top Post Titles: {post_titles}
- Comment Samples: {comment_samples}

Heuristic Pre-Score (0–70 baseline): {pre_score}

Analysis Guidelines:
- Adjust the pre-score to create a final 0–100 excitement_score
- Typical adjustments should be within ±20 points unless extraordinary circumstances
- Consider match drama (scoreline), competition importance, and Reddit reaction volume/quality
- Tags should be concise (1–3 words each)
- Focus on objective fan excitement indicators

Return JSON format ONLY:
{{
  "excitement_score": <integer 0-100>,
  "tags": ["tag1", "tag2", "tag3"],
  "summary": "<2–3 sentences describing fan excitement>",
  "reasons": ["reason 1", "reason 2", "reason 3"]
}}
"""

def _enrich_single_match(match_row: dict, reddit_bundle: dict, reddit_metrics: dict) -> dict:
    """Process a single match through DeepSeek AI for excitement scoring"""
    posts = reddit_bundle.get("posts", [])
    comments = reddit_bundle.get("comments", [])
    
    # Extract sample content for AI analysis
    post_titles = [post.get("title", "") for post in posts[:5]]
    comment_samples = [comment.get("body", "")[:180] for comment in comments[:5]]

    # Calculate baseline pre-score
    pre_score = _calculate_pre_score(match_row, reddit_metrics)
    
    # Format prompt for DeepSeek API
    user_prompt = USER_PROMPT_TEMPLATE.format(
        match_meta=json.dumps(match_row, ensure_ascii=False),
        reddit_metrics=json.dumps(reddit_metrics, ensure_ascii=False),
        post_titles=json.dumps(post_titles, ensure_ascii=False),
        comment_samples=json.dumps(comment_samples, ensure_ascii=False),
        pre_score=pre_score
    )

    # Make API call to DeepSeek
    response = _deepseek_chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ])
    
    # Parse JSON response safely
    response_text = response["choices"][0]["message"]["content"]
    json_start = response_text.find("{")
    json_end = response_text.rfind("}")
    
    parsed_data = {}
    try:
        if json_start != -1 and json_end != -1:
            json_content = response_text[json_start:json_end+1]
            parsed_data = json.loads(json_content)
    except Exception as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        parsed_data = {}
    
    # Fallback scoring if JSON parsing fails
    fallback_score = max(0, min(100, int(pre_score * 1.3)))
    
    return {
        "excitement_score": int(parsed_data.get("excitement_score", fallback_score)),
        "tags": parsed_data.get("tags", []),
        "summary": parsed_data.get("summary", ""),
        "reasons": parsed_data.get("reasons", [])
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
    for _, reddit_row in reddit_df.iterrows():
        match_id = reddit_row["match_id"]
        match_metadata = matches_by_id.get(match_id, {})
        
        # Load raw Reddit bundle data saved by main.py
        reddit_bundle_path = os.path.join("data", "raw", f"reddit_{match_id}.json")
        reddit_bundle = {"posts": [], "comments": []}
        
        if os.path.exists(reddit_bundle_path):
            with open(reddit_bundle_path, "r", encoding="utf-8") as file:
                reddit_bundle = json.load(file)

        # Process through DeepSeek AI
        enrichment_results = _enrich_single_match(
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