# main.py - Spurs fixtures scraper + Reddit reactions + AI enrichment
# Python 3.8+ compatible ETL pipeline

import os
import re
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from deepseek_enrichment import *

# Load .env environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional dependencies
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# Optional dateparser usage
try:
    from dateparser import parse as dparse
except Exception:
    dparse = None

# Optional PRAW usage
try:
    import praw
except Exception:
    praw = None

# --------------------------- Configuration ---------------------------
TEAM = "Tottenham Hotspur"
TRANSFERMARKT_URL = ("https://www.transfermarkt.com/tottenham-hotspur/"
                     "vereinsspielplan/verein/148/saison_id/2025/heim_gast/")

# File paths following assignment structure
RAW_TRANSFERMARKT = "data/raw/transfermarkt_raw.csv"
RAW_REDDIT_DIR = "data/raw/reddit/"
ENRICHED_MATCHES = "data/enriched/transfermarkt_matches.csv"
ENRICHED_REDDIT = "data/enriched/reddit_metrics.csv"
FINAL_SCORES = "data/enriched/excitement_scores.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SpursScraper/0.1)"}


# --------------------------- Helper Functions ---------------------------
def ensure_directory(path: str) -> None:
    """Ensure directory exists for the given file path."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_csv(dataframe: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV with proper encoding."""
    ensure_directory(path)
    dataframe.to_csv(path, index=False, encoding="utf-8")


def save_json(obj: Any, path: str) -> None:
    """Save object to JSON file with proper encoding."""
    ensure_directory(path)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def parse_date(date_string: str) -> Optional[str]:
    """Parse date string and return ISO format date."""
    date_string = (date_string or "").strip()
    if not date_string:
        return None
    
    # Try dateparser first if available
    if dparse:
        parsed_date = dparse(date_string, languages=["en", "de"])
        if parsed_date:
            return parsed_date.date().isoformat()
    
    # Fallback regex parsing
    match = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", date_string)
    if match:
        day, month, year = match.groups()
        year = "20" + year if len(year) == 2 else year
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except Exception:
            pass
    return None

# --------------------------- Main ETL Functions ---------------------------
def scrape_raw_transfermarkt(url: str) -> pd.DataFrame:
    """
    Scrape ALL Transfermarkt match data without filtering.
    
    This is the EXTRACT phase - collect everything first,
    clean in a separate step.
    """
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")

    rows = []
    last_competition = ""
    
    for table_row in soup.select(".responsive-table table tbody tr"):
        cells = table_row.find_all("td")
        if not cells:
            continue

        # Extract result (if match is finished)
        result = None
        for cell in reversed(cells):
            text = cell.get_text(" ", strip=True)
            score_match = re.search(r"\b(\d+\s*[:\-]\s*\d+)\b(.*)", text)
            if score_match:
                score = re.sub(r'\s+', '', score_match.group(1))
                extra_info = score_match.group(2).strip()
                result = f"{score} {extra_info}".strip()
                break

        # Extract competition
        competition = ""
        comp_img = table_row.select_one('td img[alt]')
        if comp_img:
            competition = comp_img.get("alt", "").strip()
        if not competition:
            comp_link = table_row.select_one('a[href*="/wettbewerb/"]')
            if comp_link:
                title = comp_link.get("title") or comp_link.get_text(strip=True)
                competition = title.strip()
        if competition.isdigit() or competition == "":
            competition = last_competition
        else:
            last_competition = competition

        # Extract date
        date_iso = None
        time_tag = table_row.find("time")
        if time_tag and time_tag.get_text(strip=True):
            date_iso = parse_date(time_tag.get_text(strip=True))
        if not date_iso:
            for cell in cells:
                text = cell.get_text(" ", strip=True)
                date_pattern = (r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|"
                               r"[A-Za-z]{3,}\s+\d{1,2},\s*\d{4}")
                if re.search(date_pattern, text):
                    date_iso = parse_date(text)
                    if date_iso:
                        break

        # Extract venue (Home/Away)
        venue = ""
        for cell in cells:
            venue_text = cell.get_text(" ", strip=True).upper()
            if venue_text in ("H", "A"):
                venue = "Home" if venue_text == "H" else "Away"
                break

        # Extract opponent
        opponent = ""
        club_links = [link for link in table_row.select('a[href*="/verein/"]') 
                     if link.get_text(strip=True)]
        for link in club_links:
            name = (link.get("title") or link.get_text(strip=True)).strip()
            if TEAM.lower() not in name.lower():
                opponent = name
                break
        
        if not opponent:
            for cell in cells:
                text = cell.get_text(" ", strip=True)
                if (len(text) > 2 and ":" not in text and 
                    not re.fullmatch(r"\d+", text) and 
                    TEAM.lower() not in text.lower()):
                    opponent = text.strip()
                    break

        # Extract attendance
        attendance = None
        attendance_candidates = []
        for cell in cells:
            raw_text = cell.get_text("", strip=True).replace(".", "")
            raw_text = raw_text.replace(",", "")
            if raw_text.isdigit():
                attendance_candidates.append(int(raw_text))
        if attendance_candidates:
            attendance = max(attendance_candidates)

        # Determine home/away teams
        if venue == "Home":
            home_team, away_team = TEAM, opponent
        elif venue == "Away":
            home_team, away_team = opponent, TEAM
        else:
            home_team, away_team = TEAM, opponent

        # Determine match outcome
        outcome = None
        try:
            goals = result.replace(" ", ":").split(":")
            g_home, g_away = int(goals[0]), int(goals[1])
            if home_team == TEAM:  # Spurs were home
                if g_home > g_away:
                    outcome = "Win"
                elif g_home < g_away:
                    outcome = "Loss"
                else:
                    outcome = "Draw"
            else:  # Spurs were away
                if g_away > g_home:
                    outcome = "Win"
                elif g_away < g_home:
                    outcome = "Loss"
                else:
                    outcome = "Draw"
        except Exception:
            outcome = "Unknown"

        # Create match ID
        date_part = date_iso or 'NODATE'
        home_part = (home_team or 'UNKNOWN').replace(" ", "_")
        away_part = (away_team or 'UNKNOWN').replace(" ", "_")
        match_id = f"{date_part}_{home_part}_vs_{away_part}"
        
        # Store raw data with debugging info
        row_data = {
            "match_id": match_id,
            "date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "competition": competition,
            "result": result,
            "attendance": attendance,
            "outcome": outcome,
            "is_finished": bool(result),
            "raw_cells": str([cell.get_text(" ", strip=True) for cell in cells])
        }
        rows.append(row_data)

    return pd.DataFrame(rows)


def clean_transfermarkt_data(raw_csv: str) -> pd.DataFrame:
    """
    TRANSFORM phase: Clean raw Transfermarkt data.
    
    Filter to finished first-team matches only.
    Remove youth/reserve games and validate data quality.
    """
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} not found. Run scraper first.")
    
    raw_df = pd.read_csv(raw_csv)
    print(f"Raw Transfermarkt data: {len(raw_df)} rows")
    
    # Filter to finished matches
    finished_df = raw_df[raw_df['is_finished'] == True].copy()
    print(f"Finished matches: {len(finished_df)} rows")
    
    # Remove rows with missing essential data
    essential_columns = ['date', 'home_team', 'away_team', 'result']
    clean_df = finished_df.dropna(subset=essential_columns).copy()
    print(f"Complete data: {len(clean_df)} rows")
    
    # First team filter
    youth_pattern = re.compile(
        r"\b(U18|U19|U21|U23|Youth|UEFA U19|PL2|Premier League 2)\b", re.I
    )
    
    def is_first_team_match(row):
        opponent_team = (str(row.get('away_team', '')) if 
                        row.get('venue') == 'Home' else 
                        str(row.get('home_team', '')))
        competition = str(row.get('competition', ''))
        
        if youth_pattern.search(opponent_team) or youth_pattern.search(competition):
            return False
        if re.search(r"\bEFL Trophy\b", competition, re.I):
            return False
        return True
    
    first_team_df = clean_df[clean_df.apply(is_first_team_match, axis=1)].copy()
    print(f"First team matches: {len(first_team_df)} rows")
    
    # Clean attendance data
    def validate_attendance(attendance):
        if pd.isna(attendance) or attendance == 0:
            return None
        if 1000 <= attendance <= 200000:
            return int(attendance)
        return None
    
    first_team_df['attendance'] = first_team_df['attendance'].apply(validate_attendance)
    
    # Remove debugging columns and duplicates
    final_columns = ['match_id', 'date', 'home_team', 'away_team', 
                    'venue', 'competition', 'result', 'attendance', 'outcome']
    final_df = first_team_df[final_columns].drop_duplicates(subset=['match_id'])
    final_df = final_df.reset_index(drop=True)
    
    print(f"Final clean dataset: {len(final_df)} rows")
    return final_df


def initialize_reddit() -> Any:
    """Initialize Reddit API client."""
    if not praw:
        raise RuntimeError("praw not installed. Run: pip install praw")
    
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "excited-spurs/0.1")
    
    if not (client_id and client_secret):
        error_msg = ("Please set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET "
                    "environment variables in .env")
        raise RuntimeError(error_msg)
    
    return praw.Reddit(client_id=client_id, client_secret=client_secret, 
                      user_agent=user_agent)


def to_epoch_time(datetime_obj: datetime) -> int:
    """Convert datetime to epoch timestamp."""
    return int(datetime_obj.replace(tzinfo=timezone.utc).timestamp())


def generate_team_aliases(team_name: str) -> List[str]:
    """Generate comprehensive team abbreviations and aliases."""
    base_name = team_name.strip()
    lowercase_name = base_name.lower()
    
    common_aliases: Dict[str, List[str]] = {
        "tottenham hotspur": ["Tottenham", "Spurs", "TOT"],
        "manchester city": ["Manchester City", "Man City", "MCFC", "City"],
        "manchester united": ["Manchester United", "Man United", "Man Utd", 
                             "MUFC", "United"],
        "newcastle united": ["Newcastle", "NUFC"],
        "west ham united": ["West Ham", "WHUFC", "West Ham United"],
        "brighton & hove albion": ["Brighton", "BHAFC", "Brighton & Hove Albion"],
        "liverpool": ["Liverpool", "LFC"],
        "arsenal": ["Arsenal", "AFC"],
        "chelsea": ["Chelsea", "CFC"],
        "crystal palace": ["Crystal Palace", "Palace", "CPFC"],
        "nottingham forest": ["Nottingham Forest", "Forest", "NFFC"],
        "afc bournemouth": ["AFC Bournemouth", "Bournemouth"],
        "leicester city": ["Leicester", "Leicester City", "LCFC"],
        "southampton": ["Southampton", "Saints"],
        "aston villa": ["Aston Villa", "Villa", "AVFC"],
        "paris saint-germain": ["Paris Saint-Germain", "PSG"],
        "villarreal cf": ["Villarreal", "Villarreal CF"],
        "bristol rovers": ["Bristol Rovers"],
        "burnley fc": ["Burnley", "Burnley FC"],
    }
    
    # Build alias set
    cleaned_name = re.sub(r"\s*\([^)]*\)", "", base_name).strip()
    aliases = {base_name, cleaned_name, cleaned_name.replace("FC", "").strip()}
    
    if lowercase_name in common_aliases:
        aliases.update(common_aliases[lowercase_name])
    
    simplified_name = re.sub(r"\b(FC|CF|AFC|C\.F\.)\b", "", cleaned_name, 
                           flags=re.I).strip()
    if simplified_name:
        aliases.add(simplified_name)
    
    return sorted({alias for alias in aliases if alias})


def build_reddit_queries(home_team: str, away_team: str) -> List[str]:
    """Generate Reddit search queries for match threads."""
    home_aliases = generate_team_aliases(home_team)
    away_aliases = generate_team_aliases(away_team)
    
    home_query = " OR ".join(f'"{h}"' if " " in h else h for h in home_aliases)
    away_query = " OR ".join(f'"{a}"' if " " in a else a for a in away_aliases)

    keywords = [
        '"Match Thread"', '"Post Match Thread"', '"Post-Match Thread"',
        '"Pre Match Thread"', '"Pre-Match Thread"',
        '"Full Time"', "FT", '"Player Ratings"', "Highlights"
    ]
    keywords_query = " OR ".join(keywords)

    match_patterns = [
        f'({home_query}) vs ({away_query})',
        f'({away_query}) vs ({home_query})',
    ]

    queries: List[str] = []
    
    # Add keyword + pattern combinations
    for pattern in match_patterns:
        queries.append(f'{keywords_query} {pattern}')
        queries.append(f'"Match Thread" {pattern}')
        queries.append(f'"Post Match Thread" {pattern}')
        queries.append(f'"Full Time" {pattern}')
    
    # Add pattern-only queries
    for pattern in match_patterns:
        queries.append(pattern)
    
    # Add Tottenham-specific queries
    queries.append(f'(Tottenham OR Spurs) vs ({away_query})')
    queries.append(f'({away_query}) vs (Tottenham OR Spurs)')

    # Remove duplicates while preserving order
    unique_queries: List[str] = []
    seen = set()
    for query in queries:
        query_key = query.lower()
        if query_key not in seen:
            seen.add(query_key)
            unique_queries.append(query)
    
    return unique_queries


def collect_reddit_data(subreddit_list: List[str], home_team: str, 
                        away_team: str, date_iso: str, window_days: int = 3,
                        per_query_limit: int = 100, max_posts_total: int = 600,
                        sleep_seconds: float = 0.25) -> Dict[str, Any]:
    """
    Collect Reddit posts and comments for a match.

    Strategy:
      - Use time_filter="all" (no 1-month cap)
      - Locally filter by created_utc within [start_time, end_time]
    """
    reddit = initialize_reddit()
    match_date = datetime.fromisoformat(date_iso)
    start_time = to_epoch_time(match_date - timedelta(days=window_days))
    end_time   = to_epoch_time(match_date + timedelta(days=window_days))

    queries = build_reddit_queries(home_team, away_team)
    seen_post_ids: set[str] = set()
    posts: List[Dict[str, Any]] = []

    for subreddit_name in subreddit_list:
        subreddit = reddit.subreddit(subreddit_name)
        for query in queries:
            # KEY CHANGE: time_filter="all" (was "month")
            for submission in subreddit.search(
                query=query,
                sort="new",
                time_filter="all",
                limit=per_query_limit
            ):
                post_id = submission.id
                if post_id in seen_post_ids:
                    continue

                created_time = int(getattr(submission, "created_utc", 0))
                # strict local date filter
                if not (start_time <= created_time <= end_time):
                    continue

                posts.append({
                    "id": post_id,
                    "subreddit": subreddit_name,
                    "title": submission.title,
                    "score": int(getattr(submission, "score", 0)),
                    "num_comments": int(getattr(submission, "num_comments", 0)),
                    "created_utc": created_time,
                    "url": submission.url,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "query": query,
                })
                seen_post_ids.add(post_id)

                if len(posts) >= max_posts_total:
                    break

            if len(posts) >= max_posts_total:
                break
            time.sleep(sleep_seconds)

        if len(posts) >= max_posts_total:
            break

    # Rank by engagement and recency
    posts.sort(key=lambda p: (p["score"], p["num_comments"], p["created_utc"]), reverse=True)

    # Comment samples from top posts
    comments: List[Dict[str, Any]] = []
    top_posts_limit = min(12, len(posts))
    for post in posts[:top_posts_limit]:
        submission = reddit.submission(id=post["id"])
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:80]:
            comments.append({
                "post_id": post["id"],
                "id": comment.id,
                "body": (getattr(comment, "body", "") or "")[:1000],
                "score": int(getattr(comment, "score", 0)),
                "created_utc": int(getattr(comment, "created_utc", 0)),
            })
        time.sleep(sleep_seconds)

    return {
        "posts": posts,
        "comments": comments,
        "meta": {
            "queries_tried": queries,
            "subreddits": subreddit_list,
            "window_days": window_days,
            "search_time_filter": "all",
            "filtered_by_created_utc": True,
        },
    }


def save_raw_reddit_data(reddit_bundle: Dict[str, Any], match_id: str) -> str:
    """Save raw Reddit data bundle to structured directory."""
    ensure_directory(RAW_REDDIT_DIR)
    file_path = os.path.join(RAW_REDDIT_DIR, f"{match_id}.json")
    save_json(reddit_bundle, file_path)
    return file_path


def clean_reddit_data(raw_reddit_directory: str, 
                     matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    TRANSFORM phase: Process raw Reddit data into clean metrics.
    
    Calculate standardized engagement metrics from raw Reddit bundles.
    """
    metrics_rows = []
    
    for _, match_row in matches_df.iterrows():
        match_id = match_row["match_id"]
        raw_file_path = os.path.join(raw_reddit_directory, f"{match_id}.json")
        
        if not os.path.exists(raw_file_path):
            print(f"Warning: No Reddit data found for {match_id}")
            continue
            
        # Load raw Reddit bundle
        with open(raw_file_path, "r", encoding="utf-8") as file:
            reddit_bundle = json.load(file)
        
        # Calculate engagement metrics
        posts = reddit_bundle.get("posts", [])
        comments = reddit_bundle.get("comments", [])
        posts_count = len(posts)
        comments_count = len(comments)
        
        if posts_count > 0:
            avg_post_score = round(sum(p["score"] for p in posts) / posts_count, 2)
            comments_per_post = round(comments_count / posts_count, 2)
            top_post = max(posts, key=lambda p: p["score"])
            high_engagement_posts = len([p for p in posts if p["score"] > 50])
            avg_actual_comments = sum(p["num_comments"] for p in posts) / posts_count
        else:
            avg_post_score = 0.0
            comments_per_post = 0.0
            top_post = {}
            high_engagement_posts = 0
            avg_actual_comments = 0.0
        
        metrics_row = {
            "match_id": match_id,
            "date": match_row["date"],
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "competition": match_row["competition"],
            "result": match_row["result"],
            "posts_count": posts_count,
            "comments_count": comments_count,
            "avg_post_score": avg_post_score,
            "comments_per_post": comments_per_post,
            "high_engagement_posts": high_engagement_posts,
            "avg_actual_comments_per_post": round(avg_actual_comments, 2),
            "top_post_title": top_post.get("title", ""),
            "top_post_score": top_post.get("score", 0),
            "top_post_url": top_post.get("permalink", ""),
            "subreddits_covered": len(set(p["subreddit"] for p in posts)),
            "collection_metadata": json.dumps(reddit_bundle.get("meta", {}))
        }
        metrics_rows.append(metrics_row)
    
    clean_df = pd.DataFrame(metrics_rows)
    print(f"Clean Reddit metrics: {len(clean_df)} matches processed")
    return clean_df


def run_reddit_pipeline(matches_csv: str, limit_matches: int = 5,
                       subreddit_list: Optional[List[str]] = None,
                       window_days: int = 3) -> None:
    """
    Execute complete Reddit data collection and processing pipeline.
    
    1. EXTRACT: Collect raw Reddit data
    2. TRANSFORM: Process into clean metrics
    3. LOAD: Save to enriched directory
    """
    if not os.path.exists(matches_csv):
        error_msg = f"{matches_csv} not found. Run Transfermarkt pipeline first."
        raise FileNotFoundError(error_msg)
    
    matches_df = pd.read_csv(matches_csv).sort_values(by='date', ascending=False)
    print(f"Collecting Reddit data for {min(limit_matches, len(matches_df))} matches...")

    if subreddit_list is None:
        subreddit_list = ["soccer", "coys", "PremierLeague", "soccerhighlights"]

    # Step 1: EXTRACT - Collect raw Reddit data
    match_subset = matches_df.head(limit_matches)
    iterable = match_subset.iterrows()
    
    if TQDM:
        iterable = tqdm(iterable, total=len(match_subset), 
                       desc="Reddit data collection")

    for _, row in iterable:
        reddit_bundle = collect_reddit_data(
            subreddit_list=subreddit_list,
            home_team=row["home_team"],
            away_team=row["away_team"],
            date_iso=row["date"],
            window_days=window_days,
            per_query_limit=100,
            max_posts_total=600,
            sleep_seconds=0.25
        )
        
        raw_file_path = save_raw_reddit_data(reddit_bundle, row['match_id'])
        print(f"Raw Reddit data saved: {raw_file_path}")

    # Step 2: TRANSFORM - Clean and process raw data
    print("Processing raw Reddit data...")
    clean_reddit_df = clean_reddit_data(RAW_REDDIT_DIR, match_subset)
    
    # Step 3: LOAD - Save clean data
    save_csv(clean_reddit_df, ENRICHED_REDDIT)
    print(f"Clean Reddit metrics saved: {ENRICHED_REDDIT} "
          f"({len(clean_reddit_df)} rows)")


def main() -> None:
    """
    Main ETL pipeline orchestration function.
    
    Executes: Extract → Transform → Load → Enrich workflow
    Following assignment requirements exactly.
    """
    print("Starting Spurs excitement analysis pipeline...")
    
    # STEP 1: EXTRACT - Scrape raw Transfermarkt data
    print("\n=== STEP 1: EXTRACT ===")
    print("Scraping raw Transfermarkt data...")
    raw_transfermarkt_df = scrape_raw_transfermarkt(TRANSFERMARKT_URL)
    
    if raw_transfermarkt_df.empty:
        print("ERROR: No data scraped. Check page structure.")
        return
    
    save_csv(raw_transfermarkt_df, RAW_TRANSFERMARKT)
    print(f"Raw Transfermarkt data saved: {RAW_TRANSFERMARKT} "
          f"({len(raw_transfermarkt_df)} rows)")

    # STEP 2: TRANSFORM - Clean Transfermarkt data
    print("\n=== STEP 2: TRANSFORM ===")
    print("Cleaning Transfermarkt data...")
    clean_transfermarkt_df = clean_transfermarkt_data(RAW_TRANSFERMARKT)
    save_csv(clean_transfermarkt_df, ENRICHED_MATCHES)
    print(f"Clean matches saved: {ENRICHED_MATCHES} "
          f"({len(clean_transfermarkt_df)} rows)")

    # STEP 3: EXTRACT & TRANSFORM - Reddit data pipeline
    print("\n=== STEP 3: REDDIT PIPELINE ===")
    run_reddit_pipeline(
        matches_csv=ENRICHED_MATCHES,
        limit_matches=10,
        subreddit_list=["soccer", "coys", "PremierLeague", "soccerhighlights"],
        window_days=3
    )

    # STEP 4: ENRICH - DeepSeek AI enhancement
    print("\n=== STEP 4: ENRICH ===")
    print("Running DeepSeek AI enrichment...")
    run_deepseek_enrichment(
        matches_csv=ENRICHED_MATCHES,
        reddit_metrics_csv=ENRICHED_REDDIT,
        out_csv=FINAL_SCORES
    )
    
    # STEP 5: Summary
    print("\n=== PIPELINE COMPLETE ===")
    print("File structure:")
    print("  data/raw/ - Raw scraped data (debugging/audit trail)")  
    print("  data/enriched/ - Clean processed data")
    print(f"  {FINAL_SCORES} - Final AI-enhanced results")
    print("\nPipeline completed successfully!")

    # STEP 6: Display final results
    print_deepseek_results(limit=10)


if __name__ == "__main__":
    # main()
    print_deepseek_results(limit=10)