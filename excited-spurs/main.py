# main.py — Spurs fixtures scraper + Reddit reactions + AI enrichment
# Single-file pipeline (Python 3.8+) with English comments

import os
import re
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from deepseek_enrichment import run_deepseek_enrichment, print_deepseek_results

# --------------------------- Environment ---------------------------
# Loads .env for Reddit / LLM keys if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional progress bar
try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

# Optional fuzzy date parsing (helpful for month-name formats)
try:
    from dateparser import parse as dparse
except Exception:
    dparse = None

# Reddit API (PRAW)
try:
    import praw
except Exception:
    praw = None


# --------------------------- Config ---------------------------
TEAM = "Tottenham Hotspur"
TRANSFERMARKT_URL = (
    "https://www.transfermarkt.com/tottenham-hotspur/"
    "vereinsspielplan/verein/148/saison_id/2025/heim_gast/"
)

# File paths
RAW_TRANSFERMARKT = "data/raw/transfermarkt_raw.csv"
RAW_REDDIT_DIR = "data/raw/reddit"
ENRICHED_MATCHES = "data/enriched/transfermarkt_matches.csv"
ENRICHED_REDDIT = "data/enriched/reddit_metrics.csv"
FINAL_SCORES = "data/enriched/excitement_scores.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SpursScraper/0.1)"}


# --------------------------- Helpers ---------------------------
def ensure_directory(path: str) -> None:
    """Create parent directory for a file path (or the directory itself)."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV (UTF-8, no index)."""
    ensure_directory(path)
    df.to_csv(path, index=False, encoding="utf-8")

def save_json(obj: Any, path: str) -> None:
    """Save dict/list JSON with UTF-8 and indentation."""
    ensure_directory(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_date(s: str) -> Optional[str]:
    """
    Robust date parser for Transfermarkt rows.
    Handles:
      - ISO-like strings (yyyy-mm-dd)
      - dd.mm.yyyy / dd/mm/yyyy
      - 'Aug 13, 2025' or 'Wed, 13 Aug 2025' (via dateparser)
      - bare yyyymmdd tokens inside noisy text
    Returns ISO date (yyyy-mm-dd) or None.
    """
    s = (s or "").strip()
    if not s:
        return None

    # 1) direct ISO yyyy-mm-dd
    m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date().isoformat()
        except Exception:
            pass

    # 2) dd.mm.yyyy or dd/mm/yyyy
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{2,4})\b", s)
    if m:
        d_, mo_, y_ = m.groups()
        y_ = int("20" + y_) if len(y_) == 2 else int(y_)
        try:
            return datetime(y_, int(mo_), int(d_)).date().isoformat()
        except Exception:
            pass

    # 3) Month-name formats (best-effort)
    if dparse:
        dt = dparse(s, languages=["en", "de"])
        if dt:
            return dt.date().isoformat()

    # 4) bare yyyymmdd anywhere in the string
    m = re.search(r"\b(20\d{2})(0[1-9]|1[0-2])([0-3]\d)\b", s)
    if m:
        y, mo, d = map(int, m.groups())
        try:
            return datetime(y, mo, d).date().isoformat()
        except Exception:
            pass

    return None

def to_epoch_time(dt: datetime) -> int:
    """Convert naive datetime to epoch seconds (UTC)."""
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# --------------------------- Transfermarkt Scraper ---------------------------
def scrape_raw_transfermarkt(url: str) -> pd.DataFrame:
    """Extract raw rows from Transfermarkt (finished + in-progress)."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    def _extract_date_from_tr(tr) -> Optional[str]:
        """Prefer <time datetime=...>, else scan text across tds."""
        t = tr.find("time")
        if t:
            iso_attr = (t.get("datetime") or "").strip()
            if iso_attr:
                iso = parse_date(iso_attr)
                if iso:
                    return iso
            txt = t.get_text(" ", strip=True)
            iso = parse_date(txt)
            if iso:
                return iso
        txt_all = " | ".join(td.get_text(" ", strip=True) for td in tr.find_all("td"))
        return parse_date(txt_all)

    rows, last_comp = [], ""
    for tr in soup.select(".responsive-table table tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # Result (finished only) — detect any "d:d" or "d-d"
        result = None
        for td in reversed(tds):
            txt = td.get_text(" ", strip=True)
            m = re.search(r"\b(\d+)\s*[:\-]\s*(\d+)\b", txt)
            if m:
                result = m.group(0).replace(" ", "")
                break

        # Competition — prefer icon alt/title, fallback to previous row
        comp = ""
        comp_img = tr.select_one('td img[alt]')
        if comp_img:
            comp = comp_img.get("alt", "").strip()
        if not comp:
            comp_a = tr.select_one('a[href*="/wettbewerb/"]')
            if comp_a:
                comp = (comp_a.get("title") or comp_a.get_text(strip=True)).strip()
        comp = comp if comp and not comp.isdigit() else last_comp
        if comp:
            last_comp = comp

        # Date
        date_iso = _extract_date_from_tr(tr)

        # Venue (H/A)
        venue = ""
        for td in tds:
            v = td.get_text(" ", strip=True).upper()
            if v in ("H", "A"):
                venue = "Home" if v == "H" else "Away"
                break

        # Opponent — first non-Spurs club link, else textual fallback
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
                if len(txt) > 2 and ":" not in txt and TEAM.lower() not in txt.lower():
                    opponent = txt.strip()
                    break

        # Attendance — plausible values only; ignore 8-digit yyyymmdd tokens
        attendance = None
        candidates = []
        for td in tds:
            raw = td.get_text("", strip=True).replace(".", "").replace(",", "")
            if raw.isdigit():
                if len(raw) == 8 and raw.startswith("20"):
                    continue  # ignore dates like yyyymmdd
                n = int(raw)
                if 1000 <= n <= 200000:
                    candidates.append(n)
        if candidates:
            attendance = max(candidates)

        # Home/Away explicit team names
        if venue == "Home":
            home_team, away_team = TEAM, opponent
        else:
            home_team, away_team = opponent, TEAM

        # Spurs outcome label (Win/Loss/Draw)
        outcome = "Unknown"
        try:
            gh, ga = map(int, result.replace("-", ":").split(":"))
            if home_team == TEAM:
                outcome = "Win" if gh > ga else "Loss" if gh < ga else "Draw"
            else:
                outcome = "Win" if ga > gh else "Loss" if ga < gh else "Draw"
        except Exception:
            pass

        match_id = f"{date_iso}_{home_team.replace(' ', '_')}_vs_{away_team.replace(' ', '_')}"
        rows.append({
            "match_id": match_id,
            "date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "venue": venue,
            "competition": comp,
            "result": result,
            "attendance": attendance,
            "outcome": outcome,
            "is_finished": bool(result),
        })
    return pd.DataFrame(rows)


def clean_transfermarkt_data(raw_csv: str) -> pd.DataFrame:
    """Transform raw Transfermarkt rows → first-team finished matches with validated fields."""
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} not found")
    raw = pd.read_csv(raw_csv)

    # Finished + essential fields present
    finished = raw[raw["is_finished"] == True].copy()
    finished = finished.dropna(subset=["date", "home_team", "away_team", "result"])

    # Filter out youth/reserve competitions
    youth = re.compile(r"\b(U18|U19|U21|U23|Youth|UEFA U19|PL2|Premier League 2)\b", re.I)

    def is_first_team(row):
        opp = row["away_team"] if row["venue"] == "Home" else row["home_team"]
        if youth.search(str(opp)) or youth.search(str(row["competition"])):
            return False
        if re.search(r"\bEFL Trophy\b", str(row["competition"]), re.I):
            return False
        return True

    ft = finished[finished.apply(is_first_team, axis=1)].copy()

    # Validate attendance range
    def val_att(x):
        try:
            x = int(x)
            return x if 1000 <= x <= 200000 else None
        except Exception:
            return None

    ft["attendance"] = ft["attendance"].apply(val_att)

    cols = ["match_id", "date", "home_team", "away_team", "venue",
            "competition", "result", "attendance", "outcome"]
    return ft[cols].drop_duplicates(subset=["match_id"]).reset_index(drop=True)


# --------------------------- Reddit layer ---------------------------
def initialize_reddit() -> Any:
    """Initialize PRAW client from environment variables."""
    if not praw:
        raise RuntimeError("praw not installed. pip install praw")
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "excited-spurs/0.1")
    if not (cid and csec):
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)

def generate_team_aliases(team_name: str) -> List[str]:
    """Return a list of common aliases for a team to improve Reddit search recall."""
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
        "burnley": ["Burnley", "Burnley FC"],
    }
    cleaned = re.sub(r"\s*\([^)]*\)", "", base).strip()
    aliases = {base, cleaned, cleaned.replace("FC", "").strip()}
    if low in COMMON:
        aliases.update(COMMON[low])
    simp = re.sub(r"\b(FC|CF|AFC|C\.F\.)\b", "", cleaned, flags=re.I).strip()
    if simp:
        aliases.add(simp)
    return sorted(a for a in aliases if a)

def build_reddit_queries(home_team: str, away_team: str) -> List[str]:
    """Build a list of Lucene-ish queries for subreddit.search across 'all' time."""
    H = generate_team_aliases(home_team)
    A = generate_team_aliases(away_team)
    H_or = " OR ".join(f'"{h}"' if " " in h else h for h in H)
    A_or = " OR ".join(f'"{a}"' if " " in a else a for a in A)

    KW = [
        '"Match Thread"', '"Post Match Thread"', '"Post-Match Thread"',
        '"Pre Match Thread"', '"Pre-Match Thread"', '"Full Time"', "FT",
        '"Player Ratings"', "Highlights"
    ]
    KW_or = " OR ".join(KW)

    patterns = [
        f'({H_or}) vs ({A_or})',
        f'({A_or}) vs ({H_or})',
    ]

    q: List[str] = []
    for p in patterns:
        q.append(f'{KW_or} {p}')
        q.append(f'"Match Thread" {p}')
        q.append(f'"Post Match Thread" {p}')
        q.append(f'"Full Time" {p}')
    for p in patterns:
        q.append(p)
    q.append(f'(Tottenham OR Spurs) vs ({A_or})')
    q.append(f'({A_or}) vs (Tottenham OR Spurs)')

    # Deduplicate while preserving order
    seen, uq = set(), []
    for s in q:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uq.append(s)
    return uq

def collect_reddit_data(subreddit_list: List[str], home_team: str,
                        away_team: str, date_iso: str, window_days: int = 3,
                        per_query_limit: int = 100, max_posts_total: int = 600,
                        sleep_seconds: float = 0.25) -> Dict[str, Any]:
    """
    Collect Reddit posts/comments for a match.
    Strategy:
      - Use time_filter='all' (so older than a month is reachable)
      - Locally filter to [date - window_days, date + window_days] by created_utc
    """
    reddit = initialize_reddit()
    match_date = datetime.fromisoformat(date_iso)
    start_time = to_epoch_time(match_date - timedelta(days=window_days))
    end_time = to_epoch_time(match_date + timedelta(days=window_days))

    queries = build_reddit_queries(home_team, away_team)
    seen_ids: set = set()
    posts: List[Dict[str, Any]] = []

    for sub_name in subreddit_list:
        sub = reddit.subreddit(sub_name)
        for q in queries:
            try:
                for s in sub.search(query=q, sort="new", time_filter="all", limit=per_query_limit):
                    pid = s.id
                    if pid in seen_ids:
                        continue
                    created = int(getattr(s, "created_utc", 0))
                    if not (start_time <= created <= end_time):
                        continue
                    posts.append({
                        "id": pid,
                        "subreddit": sub_name,
                        "title": s.title,
                        "score": int(getattr(s, "score", 0)),
                        "num_comments": int(getattr(s, "num_comments", 0)),
                        "created_utc": created,
                        "url": s.url,
                        "permalink": f"https://reddit.com{s.permalink}",
                        "query": q,
                    })
                    seen_ids.add(pid)
                    if len(posts) >= max_posts_total:
                        break
            except Exception:
                # Be resilient to occasional API hiccups / rate limits
                pass
            if len(posts) >= max_posts_total:
                break
            time.sleep(sleep_seconds)
        if len(posts) >= max_posts_total:
            break

    # Rank by engagement + recency
    posts.sort(key=lambda p: (p["score"], p["num_comments"], p["created_utc"]), reverse=True)

    # Sample comments from the top posts
    comments: List[Dict[str, Any]] = []
    for p in posts[: min(12, len(posts))]:
        try:
            subm = reddit.submission(id=p["id"])
            subm.comments.replace_more(limit=0)
            for c in subm.comments[:80]:
                comments.append({
                    "post_id": p["id"],
                    "id": c.id,
                    "body": (getattr(c, "body", "") or "")[:1000],
                    "score": int(getattr(c, "score", 0)),
                    "created_utc": int(getattr(c, "created_utc", 0)),
                })
            time.sleep(sleep_seconds)
        except Exception:
            continue

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
    """Persist raw Reddit bundle as JSON under data/raw/reddit/<match_id>.json."""
    ensure_directory(RAW_REDDIT_DIR)
    path = os.path.join(RAW_REDDIT_DIR, f"{match_id}.json")
    save_json(reddit_bundle, path)
    return path

def clean_reddit_data(raw_reddit_directory: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw Reddit bundles → standardized per-match metrics."""
    rows = []
    for _, row in matches_df.iterrows():
        match_id = row["match_id"]
        raw_path = os.path.join(raw_reddit_directory, f"{match_id}.json")
        if not os.path.exists(raw_path):
            print(f"Warning: no Reddit data for {match_id}")
            continue

        with open(raw_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        posts = bundle.get("posts", [])
        comments = bundle.get("comments", [])
        pc, cc = len(posts), len(comments)

        if pc:
            avg_post_score = round(sum(p.get("score", 0) for p in posts) / pc, 2)
            comments_per_post = round(cc / pc, 2)
            top_post = max(posts, key=lambda p: p.get("score", 0))
            high_engagement_posts = sum(1 for p in posts if p.get("score", 0) > 50)
            avg_actual_comments = round(sum(p.get("num_comments", 0) for p in posts) / pc, 2)
            subs_covered = len({p.get("subreddit") for p in posts})
        else:
            avg_post_score = 0.0
            comments_per_post = 0.0
            top_post = {}
            high_engagement_posts = 0
            avg_actual_comments = 0.0
            subs_covered = 0

        rows.append({
            "match_id": match_id,
            "date": row.get("date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "competition": row.get("competition"),
            "result": row.get("result"),
            "posts_count": pc,
            "comments_count": cc,
            "avg_post_score": avg_post_score,
            "comments_per_post": comments_per_post,
            "high_engagement_posts": high_engagement_posts,
            "avg_actual_comments_per_post": avg_actual_comments,
            "top_post_title": top_post.get("title", ""),
            "top_post_score": top_post.get("score", 0),
            "top_post_url": top_post.get("permalink", ""),
            "subreddits_covered": subs_covered,
            "collection_metadata": json.dumps(bundle.get("meta", {})),
        })

    df = pd.DataFrame(rows)
    print(f"Clean Reddit metrics: {len(df)} matches processed")
    return df


# --------------------------- Selection UI ---------------------------
def _print_match_menu(matches_df: pd.DataFrame) -> None:
    """Print numbered menu of matches (newest first)."""
    print("\n=== Select matches to fetch Reddit data ===")
    for i, r in enumerate(matches_df.itertuples(index=False), start=1):
        print(f"{i:>2}. {r.date} — {r.home_team} vs {r.away_team} | "
              f"{r.competition} | {r.result} | outcome={getattr(r,'outcome','')}")
    print("\nEnter numbers like '1,3,5' or ranges like '2-4'. "
          "Press ENTER to fetch ALL matches.")

def _parse_selection(user_input: str, max_n: int) -> List[int]:
    """Parse '1,3,5' or '2-4' input into a sorted list of unique 1-based indices."""
    s = (user_input or "").strip()
    if not s:
        return []
    picks = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            try:
                a, b = map(int, tok.split("-", 1))
                if a > b:
                    a, b = b, a
                for j in range(a, b + 1):
                    if 1 <= j <= max_n:
                        picks.add(j)
            except Exception:
                continue
        else:
            try:
                j = int(tok)
                if 1 <= j <= max_n:
                    picks.add(j)
            except Exception:
                continue
    return sorted(picks)


# ---------------- Reddit pipeline for selected matches ----------------
def run_reddit_pipeline_for_selected(matches_df: pd.DataFrame,
                                     selected_indices: List[int],
                                     subreddit_list: Optional[List[str]] = None,
                                     window_days: int = 3) -> None:
    """
    Process only the user-selected matches.
    - If raw JSON exists, reuse it; otherwise fetch and save.
    - reddit_metrics.csv is re-initialized (overwritten) with ONLY the selected matches.
    """
    if subreddit_list is None:
        subreddit_list = ["soccer", "coys", "PremierLeague", "soccerhighlights"]
    if not selected_indices:
        print("No matches selected. Skipping Reddit collection.")
        return

    # Take rows by 1-based indices based on newest-first menu
    newest_first = matches_df.sort_values(by="date", ascending=False).reset_index(drop=True)
    chosen = newest_first.iloc[[i - 1 for i in selected_indices if 1 <= i <= len(newest_first)]]
    if chosen.empty:
        print("Selected indices out of range. Skipping.")
        return

    print(f"Collecting Reddit data for {len(chosen)} selected match(es)...")
    iterator = tqdm(chosen.iterrows(), total=len(chosen), desc="Reddit data") if TQDM else chosen.iterrows()

    # Extract phase — prefer existing raw, otherwise fetch
    for _, row in iterator:
        raw_path = os.path.join(RAW_REDDIT_DIR, f"{row['match_id']}.json")
        if os.path.exists(raw_path):
            print(f"Use existing raw: {raw_path}")
            continue
        bundle = collect_reddit_data(
            subreddit_list=subreddit_list,
            home_team=row["home_team"],
            away_team=row["away_team"],
            date_iso=row["date"],
            window_days=window_days,
            per_query_limit=100,
            max_posts_total=600,
            sleep_seconds=0.25
        )
        save_raw_reddit_data(bundle, row["match_id"])
        print(f"Saved raw: {raw_path}")

    # Transform phase — compute metrics ONLY for chosen matches
    print("Processing raw Reddit data for selected matches...")
    processed = clean_reddit_data(RAW_REDDIT_DIR, chosen)

    # Load phase — initialize (overwrite) metrics with selected-only
    save_csv(processed, ENRICHED_REDDIT)
    print(f"✅ Initialized reddit metrics: {ENRICHED_REDDIT} ({len(processed)} rows; selected only)")


# --------------------------- Orchestration ---------------------------
def main() -> None:
    print("Starting Spurs excitement analysis pipeline...")

    # STEP 1: EXTRACT — Transfermarkt
    print("\n=== STEP 1: EXTRACT (Transfermarkt) ===")
    raw_df = scrape_raw_transfermarkt(TRANSFERMARKT_URL)
    if raw_df.empty:
        print("❗ No data scraped. Check page structure.")
        return
    save_csv(raw_df, RAW_TRANSFERMARKT)
    print(f"Saved raw Transfermarkt: {RAW_TRANSFERMARKT} ({len(raw_df)} rows)")

    # STEP 2: TRANSFORM — clean matches
    print("\n=== STEP 2: TRANSFORM (Clean matches) ===")
    clean_df = clean_transfermarkt_data(RAW_TRANSFERMARKT)
    save_csv(clean_df, ENRICHED_MATCHES)
    print(f"Saved clean matches: {ENRICHED_MATCHES} ({len(clean_df)} rows)")

    # STEP 3: Reddit — user selection; ENTER = ALL
    print("\n=== STEP 3: REDDIT (select matches) ===")
    menu_df = clean_df.sort_values(by="date", ascending=False).reset_index(drop=True)
    _print_match_menu(menu_df)
    user_raw = input("\nSelect matches (e.g., 1,3,5 or 2-4). ENTER = ALL: ")
    selected = _parse_selection(user_raw, len(menu_df))
    if not user_raw.strip():
        selected = list(range(1, len(menu_df) + 1))
        print(f"(ENTER) Fetching ALL {len(selected)} matches.")

    if selected:
        run_reddit_pipeline_for_selected(
            matches_df=clean_df,
            selected_indices=selected,
            subreddit_list=["soccer", "coys", "PremierLeague", "soccerhighlights"],
            window_days=3
        )
    else:
        print("No valid indices. Skipping Reddit collection.")

    # STEP 4: ENRICH — DeepSeek scoring (uses existing metrics CSV)
    print("\n=== STEP 4: ENRICH (DeepSeek) ===")
    run_deepseek_enrichment(
        matches_csv=ENRICHED_MATCHES,
        reddit_metrics_csv=ENRICHED_REDDIT,
        out_csv=FINAL_SCORES
    )

    # STEP 5: Summary and preview
    print("\n=== PIPELINE COMPLETE ===")
    print("Outputs:")
    print(f"  - {RAW_TRANSFERMARKT}")
    print(f"  - {ENRICHED_MATCHES}")
    print(f"  - {RAW_REDDIT_DIR}/<match_id>.json")
    print(f"  - {ENRICHED_REDDIT}")
    print(f"  - {FINAL_SCORES}")
    print_deepseek_results(limit=10)


if __name__ == "__main__":
    main()