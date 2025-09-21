# main.py — Spurs fixtures scraper + max-coverage Reddit reactions (single file)
# Python 3.8+ compatible (using typing Optional/List, avoiding | syntax)

import os
import re
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from deepseek_enrichment import run_deepseek_enrichment

# Load .env environment variables (REDDIT_* etc)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Make it work even without tqdm
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

# --------------------------- Basic Configuration ---------------------------------------
TEAM = "Tottenham Hotspur"
URL  = "https://www.transfermarkt.com/tottenham-hotspur/vereinsspielplan/verein/148/saison_id/2025/heim_gast/"
OUT_MATCHES  = "data/raw/transfermarkt_matches.csv"
OUT_REDDIT   = "data/enriched/reddit_metrics.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SpursScraper/0.1)"}

# ------------------------ Utilities / Common Functions ------------------------------------
def ensure_dir_for(path: str):
    """Ensure directory exists for the given file path"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str):
    """Save DataFrame to CSV with proper encoding"""
    ensure_dir_for(path)
    df.to_csv(path, index=False, encoding="utf-8")

def save_json(obj: Any, path: str):
    """Save object to JSON file with proper encoding"""
    ensure_dir_for(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_date(s: str) -> Optional[str]:
    """Parse date string and return ISO format date"""
    s = (s or "").strip()
    if not s:
        return None
    if dparse:
        dt = dparse(s, languages=["en","de"])
        if dt:
            return dt.date().isoformat()
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", s)
    if m:
        day, month, year = m.groups()
        year = "20"+year if len(year)==2 else year
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except Exception:
            pass
    return None

# --------------------- Transfermarkt Scraper ---------------------------------
def fetch_finished_matches(url: str) -> pd.DataFrame:
    """
    Scrape Tottenham 2025 season 'finished' matches → keep only first team matches
    Returns columns: match_id, date, home_team, away_team, venue, competition, result, attendance
    """
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows = []
    last_competition = ""  # Use previous value when competition name is omitted in same block
    
    for tr in soup.select(".responsive-table table tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # Result (finished matches only)
        result = None
        for td in reversed(tds):
            txt = td.get_text(" ", strip=True)
            m = re.search(r"\b(\d+)\s*[:\-]\s*(\d+)\b", txt)
            if m:
                result = m.group(0).replace(" ", "")
                break
        if not result:
            continue

        # Competition: img alt → /wettbewerb/ link → fallback to previous value
        competition = ""
        comp_img = tr.select_one('td img[alt]')
        if comp_img:
            competition = comp_img.get("alt", "").strip()
        if not competition:
            comp_a = tr.select_one('a[href*="/wettbewerb/"]')
            if comp_a:
                competition = (comp_a.get("title") or comp_a.get_text(strip=True)).strip()
        if competition.isdigit() or competition == "":
            competition = last_competition
        else:
            last_competition = competition

        # Date
        date_iso = None
        time_tag = tr.find("time")
        if time_tag and time_tag.get_text(strip=True):
            date_iso = parse_date(time_tag.get_text(strip=True))
        if not date_iso:
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|[A-Za-z]{3,}\s+\d{1,2},\s*\d{4}", txt):
                    date_iso = parse_date(txt)
                    if date_iso:
                        break
        if not date_iso:
            continue

        # Home/Away (H/A)
        venue = ""
        for td in tds:
            v = td.get_text(" ", strip=True).upper()
            if v in ("H", "A"):
                venue = "Home" if v == "H" else "Away"
                break
        if not venue:
            continue

        # Opponent team
        opponent = ""
        club_links = [a for a in tr.select('a[href*="/verein/"]') if a.get_text(strip=True)]
        for a in club_links:
            name = (a.get("title") or a.get_text(strip=True)).strip()
            if TEAM.lower() not in name.lower():
                opponent = name
                break
        if not opponent:
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if len(txt) > 2 and ":" not in txt and not re.fullmatch(r"\d+", txt):
                    if TEAM.lower() not in txt.lower():
                        opponent = txt.strip()
                        break
        if not opponent:
            continue

        # Attendance (filter to normal range)
        attendance = None
        candidates = []
        for td in tds:
            raw = td.get_text("", strip=True).replace(".", "").replace(",", "")
            if raw.isdigit():
                n = int(raw)
                if 1000 <= n <= 200000:  # Approximate first team match attendance range
                    candidates.append(n)
        if candidates:
            attendance = max(candidates)

        # Determine home/away team names
        if venue == "Home":
            home_team, away_team = TEAM, opponent
        else:
            home_team, away_team = opponent, TEAM

        # First team filter (remove U18/U19/U21/U23/PL2/Youth/EFL Trophy)
        youth_pattern = re.compile(r"\b(U18|U19|U21|U23|Youth|UEFA U19|PL2|Premier League 2)\b", re.I)
        if youth_pattern.search(opponent) or youth_pattern.search(competition):
            continue
        if re.search(r"\bEFL Trophy\b", competition, re.I):
            continue

        match_id = f"{date_iso}_{home_team}_vs_{away_team}".replace(" ", "_")
        rows.append({
            "match_id": match_id,
            "date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "competition": competition,
            "result": result,
            "attendance": attendance
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["match_id"]).reset_index(drop=True)
    return df

# --------------------- Reddit reactions (maximum collection) ---------------------------
def _init_reddit():
    """Initialize Reddit API client"""
    if not praw:
        raise RuntimeError("praw is not installed. Run: pip install praw")
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "excited-spurs/0.1")
    if not (client_id and client_secret):
        raise RuntimeError("Please set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET environment variables in .env")
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

def _to_epoch(dt: datetime) -> int:
    """Convert datetime to epoch timestamp"""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def _aliases_for(team_name: str) -> List[str]:
    """Generate comprehensive team abbreviations/aliases"""
    base = team_name.strip()
    low = base.lower()
    COMMON: Dict[str, List[str]] = {
        "tottenham hotspur": ["Tottenham", "Spurs", "TOT"],
        "manchester city": ["Manchester City", "Man City", "MCFC", "City"],
        "manchester united": ["Manchester United", "Man United", "Man Utd", "MUFC", "United"],
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
    cleaned = re.sub(r"\s*\([^)]*\)", "", base).strip()
    aliases = {base, cleaned, cleaned.replace("FC","").strip()}
    if low in COMMON:
        aliases.update(COMMON[low])
    simplified = re.sub(r"\b(FC|CF|AFC|C\.F\.)\b", "", cleaned, flags=re.I).strip()
    if simplified:
        aliases.add(simplified)
    return sorted({a for a in aliases if a})

def _build_queries(home_team: str, away_team: str) -> List[str]:
    """Generate various search queries including Match Thread / Post-Match / FT / general match expressions"""
    home_aliases = _aliases_for(home_team)
    away_aliases = _aliases_for(away_team)
    home_or = " OR ".join(f'"{h}"' if " " in h else h for h in home_aliases)
    away_or = " OR ".join(f'"{a}"' if " " in a else a for a in away_aliases)

    keywords = [
        '"Match Thread"', '"Post Match Thread"', '"Post-Match Thread"',
        '"Pre Match Thread"', '"Pre-Match Thread"',
        '"Full Time"', "FT", '"Player Ratings"', "Highlights"
    ]
    keywords_or = " OR ".join(keywords)

    patterns = [
        f'({home_or}) vs ({away_or})',
        f'({away_or}) vs ({home_or})',
    ]

    queries: List[str] = []
    for p in patterns:
        queries.append(f'{keywords_or} {p}')
        queries.append(f'"Match Thread" {p}')
        queries.append(f'"Post Match Thread" {p}')
        queries.append(f'"Full Time" {p}')
    for p in patterns:
        queries.append(p)
    queries.append(f'(Tottenham OR Spurs) vs ({away_or})')
    queries.append(f'({away_or}) vs (Tottenham OR Spurs)')

    # Remove duplicates while preserving order
    unique_queries: List[str] = []
    seen = set()
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)
    return unique_queries

def fetch_match_reddit_bundle(
    subreddit_list: List[str],
    home_team: str,
    away_team: str,
    date_iso: str,
    window_days: int = 3,
    per_query_limit: int = 100,
    max_posts_total: int = 600,
    sleep_sec: float = 0.25
) -> Dict[str, Any]:
    """
    Collect maximum number of threads:
      - Multiple queries (abbreviations/aliases/keywords/bidirectional) × multiple subreddits
      - search time_filter='month' for broad coverage → precise filter by created_utc for ±window_days
      - Remove duplicates then accumulate up to max_posts_total, expand comment samples from top posts
    """
    reddit = _init_reddit()
    date_dt = datetime.fromisoformat(date_iso)
    start = _to_epoch(date_dt - timedelta(days=window_days))
    end   = _to_epoch(date_dt + timedelta(days=window_days))

    queries = _build_queries(home_team, away_team)
    seen_ids = set()
    posts: List[Dict[str, Any]] = []

    for sub_name in subreddit_list:
        sub = reddit.subreddit(sub_name)
        for query in queries:
            for submission in sub.search(query=query, sort="new", time_filter="month", limit=per_query_limit):
                post_id = submission.id
                if post_id in seen_ids:
                    continue
                created = int(getattr(submission, "created_utc", 0))
                if not (start <= created <= end):
                    continue
                posts.append({
                    "id": post_id,
                    "subreddit": sub_name,
                    "title": submission.title,
                    "score": int(getattr(submission, "score", 0)),
                    "num_comments": int(getattr(submission, "num_comments", 0)),
                    "created_utc": created,
                    "url": submission.url,
                    "permalink": f"https://reddit.com{submission.permalink}",
                    "query": query,
                })
                seen_ids.add(post_id)
                if len(posts) >= max_posts_total:
                    break
            if len(posts) >= max_posts_total:
                break
            time.sleep(sleep_sec)
        if len(posts) >= max_posts_total:
            break

    # Sort by score/comments/time
    posts.sort(key=lambda p: (p["score"], p["num_comments"], p["created_utc"]), reverse=True)

    # Comment samples (top 12 posts × 80 comments each)
    comments: List[Dict[str, Any]] = []
    for post in posts[: min(12, len(posts))]:
        submission = reddit.submission(id=post["id"])
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:80]:
            body = getattr(comment, "body", "") or ""
            comments.append({
                "post_id": post["id"],
                "id": comment.id,
                "body": body[:1000],
                "score": int(getattr(comment, "score", 0)),
                "created_utc": int(getattr(comment, "created_utc", 0)),
            })
        time.sleep(sleep_sec)

    return {
        "posts": posts, 
        "comments": comments, 
        "meta": {
            "queries_tried": queries,
            "subreddits": subreddit_list,
            "window_days": window_days
        }
    }

def compute_reddit_metrics(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate Reddit engagement metrics from collected data"""
    posts = bundle.get("posts", [])
    comments = bundle.get("comments", [])
    posts_count = len(posts)
    comments_count = len(comments)
    avg_post_score = round(sum(p["score"] for p in posts)/posts_count, 2) if posts_count else 0.0
    comments_per_post = round(comments_count / posts_count, 2) if posts_count else 0.0
    top_post = max(posts, key=lambda p: p["score"], default=None)
    return {
        "posts_count": posts_count,
        "comments_count": comments_count,
        "avg_post_score": avg_post_score,
        "comments_per_post": comments_per_post,
        "top_post_title": (top_post or {}).get("title", ""),
        "top_post_score": (top_post or {}).get("score", 0),
        "top_post_url": (top_post or {}).get("permalink", ""),
    }

def run_reddit_for_matches(
    matches_csv: str,
    limit_matches: int = 6,
    subreddit_list: Optional[List[str]] = None,
    window_days: int = 3
):
    """
    Read scraped matches CSV and collect Reddit data for top N matches
    - Raw JSON: data/raw/reddit_<match_id>.json
    - Summary metrics: data/enriched/reddit_metrics.csv
    """
    if not os.path.exists(matches_csv):
        raise FileNotFoundError(f"{matches_csv} not found. Please run Transfermarkt scraper first.")
    df = pd.read_csv(matches_csv)

    if subreddit_list is None:
        subreddit_list = ["soccer", "coys", "PremierLeague", "soccerhighlights"]

    metrics_rows = []
    iterable = df.head(limit_matches).iterrows()
    if TQDM:
        iterable = tqdm(iterable, total=min(limit_matches, len(df)), desc="Reddit collection")

    for _, row in iterable:
        bundle = fetch_match_reddit_bundle(
            subreddit_list=subreddit_list,
            home_team=row["home_team"],
            away_team=row["away_team"],
            date_iso=row["date"],
            window_days=window_days,
            per_query_limit=100,
            max_posts_total=600,
            sleep_sec=0.25
        )
        raw_path = f"data/raw/reddit_{row['match_id']}.json"
        save_json(bundle, raw_path)

        metrics = compute_reddit_metrics(bundle)
        metrics.update({
            "match_id": row["match_id"],
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "competition": row["competition"],
            "result": row["result"],
        })
        metrics_rows.append(metrics)

    output_df = pd.DataFrame(metrics_rows)
    save_csv(output_df, OUT_REDDIT)
    print(f"✅ Reddit metrics saved: {OUT_REDDIT} (rows={len(output_df)})")

# --------------------------- Entry Point -------------------------------------
def main():
    """Main function to run the complete scraping and analysis pipeline"""
    # 1) Transfermarkt scraping
    matches_df = fetch_finished_matches(URL)
    if matches_df.empty:
        print("❗ Nothing was scraped. Possible page structure change → need to check selectors.")
        return
    save_csv(matches_df, OUT_MATCHES)
    print(f"✅ Saved: {OUT_MATCHES} (rows={len(matches_df)})")
    print(matches_df.head(3).to_string(index=False))

    # 2) Reddit (maximum collection) — safely run example with top 6 matches only
    run_reddit_for_matches(
        matches_csv=OUT_MATCHES,
        limit_matches=6,                 # Increase as needed
        subreddit_list=["soccer", "coys", "PremierLeague", "soccerhighlights"],
        window_days=3
    )

    # 3) DeepSeek enrichment (writes data/enriched/excitement_scores.csv)
    run_deepseek_enrichment(
        matches_csv=OUT_MATCHES,
        reddit_metrics_csv=OUT_REDDIT,
        out_csv="data/enriched/excitement_scores.csv"
    )

if __name__ == "__main__":
    main()