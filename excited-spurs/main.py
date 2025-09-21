# main.py - Spurs fixtures scraper + Reddit reactions + AI enrichment
# Python 3.8+ compatible single-file pipeline

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

# Load .env (Reddit / DeepSeek keys)
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

# Optional date parsing
try:
    from dateparser import parse as dparse
except Exception:
    dparse = None

# Reddit API (PRAW)
try:
    import praw
except Exception:
    praw = None


# ============================ CONFIG ==================================
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


# ============================ HELPERS =================================
def ensure_directory(path: str) -> None:
    d = path if os.path.isdir(path) else os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_directory(path)
    df.to_csv(path, index=False, encoding="utf-8")

def save_json(obj: Any, path: str) -> None:
    ensure_directory(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    if not s:
        return None
    if dparse:
        dt = dparse(s, languages=["en", "de"])
        if dt:
            return dt.date().isoformat()
    m = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", s)
    if m:
        d_, m_, y = m.groups()
        y = "20" + y if len(y) == 2 else y
        try:
            return datetime(int(y), int(m_), int(d_)).date().isoformat()
        except Exception:
            return None
    return None

def to_epoch_time(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


# ====================== TRANSFERMARKT SCRAPER =========================
def scrape_raw_transfermarkt(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    rows, last_comp = [], ""
    for tr in soup.select(".responsive-table table tbody tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        # Result (finished only)
        result = None
        for td in reversed(tds):
            txt = td.get_text(" ", strip=True)
            m = re.search(r"\b(\d+)\s*[:\-]\s*(\d+)\b", txt)
            if m:
                result = m.group(0).replace(" ", "")
                break

        # Competition
        comp = ""
        comp_img = tr.select_one('td img[alt]')
        if comp_img:
            comp = comp_img.get("alt", "").strip()
        if not comp:
            comp_a = tr.select_one('a[href*="/wettbewerb/"]')
            if comp_a:
                comp = (comp_a.get("title") or comp_a.get_text(strip=True)).strip()
        if comp.isdigit() or comp == "":
            comp = last_comp
        else:
            last_comp = comp

        # Date
        date_iso = None
        t = tr.find("time")
        if t and t.get_text(strip=True):
            date_iso = parse_date(t.get_text(strip=True))
        if not date_iso:
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if re.search(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}|[A-Za-z]{3,}\s+\d{1,2},\s*\d{4}", txt):
                    date_iso = parse_date(txt); break

        # Venue (H/A)
        venue = ""
        for td in tds:
            v = td.get_text(" ", strip=True).upper()
            if v in ("H","A"):
                venue = "Home" if v=="H" else "Away"; break

        # Opponent
        opponent = ""
        club_links = [a for a in tr.select('a[href*="/verein/"]') if a.get_text(strip=True)]
        for a in club_links:
            name = (a.get("title") or a.get_text(strip=True)).strip()
            if TEAM.lower() not in name.lower():
                opponent = name; break
        if not opponent:
            for td in tds:
                txt = td.get_text(" ", strip=True)
                if len(txt)>2 and ":" not in txt and TEAM.lower() not in txt.lower():
                    opponent = txt.strip(); break

        # Attendance (max numeric in row; later we validate)
        attendance = None
        nums = []
        for td in tds:
            raw = td.get_text("", strip=True).replace(".","").replace(",","")
            if raw.isdigit():
                nums.append(int(raw))
        if nums: attendance = max(nums)

        # Home/Away teams
        if venue == "Home":
            home_team, away_team = TEAM, opponent
        else:
            home_team, away_team = opponent, TEAM

        # Outcome (Win/Loss/Draw for Spurs)
        outcome = "Unknown"
        try:
            gh, ga = map(int, result.replace("-", ":").split(":"))
            if home_team == TEAM:
                outcome = "Win" if gh>ga else "Loss" if gh<ga else "Draw"
            else:
                outcome = "Win" if ga>gh else "Loss" if ga<gh else "Draw"
        except Exception:
            pass

        match_id = f"{date_iso}_{home_team.replace(' ','_')}_vs_{away_team.replace(' ','_')}"
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
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"{raw_csv} not found")
    raw = pd.read_csv(raw_csv)

    finished = raw[raw["is_finished"]==True].copy()
    essentials = ["date","home_team","away_team","result"]
    finished = finished.dropna(subset=essentials)

    # first team filter
    youth = re.compile(r"\b(U18|U19|U21|U23|Youth|UEFA U19|PL2|Premier League 2)\b", re.I)
    def is_first(row):
        opp = row["away_team"] if row["venue"]=="Home" else row["home_team"]
        if youth.search(str(opp)) or youth.search(str(row["competition"])):
            return False
        if re.search(r"\bEFL Trophy\b", str(row["competition"]), re.I):
            return False
        return True
    ft = finished[finished.apply(is_first, axis=1)].copy()

    def val_att(x):
        try:
            x = int(x)
            return x if 1000 <= x <= 200000 else None
        except Exception:
            return None
    ft["attendance"] = ft["attendance"].apply(val_att)

    cols = ["match_id","date","home_team","away_team","venue","competition","result","attendance","outcome"]
    return ft[cols].drop_duplicates(subset=["match_id"]).reset_index(drop=True)


# ============================ REDDIT =================================
def initialize_reddit() -> Any:
    if not praw:
        raise RuntimeError("praw not installed. pip install praw")
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT","excited-spurs/0.1")
    if not (cid and csec):
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)

def generate_team_aliases(team_name: str) -> List[str]:
    base = team_name.strip()
    low = base.lower()
    COMMON: Dict[str, List[str]] = {
        "tottenham hotspur": ["Tottenham","Spurs","TOT"],
        "manchester city": ["Manchester City","Man City","MCFC","City"],
        "manchester united": ["Manchester United","Man United","Man Utd","MUFC","United"],
        "newcastle united": ["Newcastle","NUFC"],
        "west ham united": ["West Ham","WHUFC","West Ham United"],
        "brighton & hove albion": ["Brighton","BHAFC","Brighton & Hove Albion"],
        "liverpool": ["Liverpool","LFC"],
        "arsenal": ["Arsenal","AFC"],
        "chelsea": ["Chelsea","CFC"],
        "crystal palace": ["Crystal Palace","Palace","CPFC"],
        "nottingham forest": ["Nottingham Forest","Forest","NFFC"],
        "afc bournemouth": ["AFC Bournemouth","Bournemouth"],
        "leicester city": ["Leicester","Leicester City","LCFC"],
        "southampton": ["Southampton","Saints"],
        "aston villa": ["Aston Villa","Villa","AVFC"],
        "paris saint-germain": ["Paris Saint-Germain","PSG"],
        "villarreal cf": ["Villarreal","Villarreal CF"],
        "bristol rovers": ["Bristol Rovers"],
        "burnley fc": ["Burnley","Burnley FC"],
        "burnley": ["Burnley","Burnley FC"],
    }
    cleaned = re.sub(r"\s*\([^)]*\)", "", base).strip()
    aliases = {base, cleaned, cleaned.replace("FC","").strip()}
    if low in COMMON: aliases.update(COMMON[low])
    simp = re.sub(r"\b(FC|CF|AFC|C\.F\.)\b","", cleaned, flags=re.I).strip()
    if simp: aliases.add(simp)
    return sorted(a for a in aliases if a)

def build_reddit_queries(home_team: str, away_team: str) -> List[str]:
    H = generate_team_aliases(home_team)
    A = generate_team_aliases(away_team)
    H_or = " OR ".join(f'"{h}"' if " " in h else h for h in H)
    A_or = " OR ".join(f'"{a}"' if " " in a else a for a in A)

    KW = [
        '"Match Thread"','"Post Match Thread"','"Post-Match Thread"',
        '"Pre Match Thread"','"Pre-Match Thread"','"Full Time"',"FT",
        '"Player Ratings"',"Highlights"
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

    seen, uq = set(), []
    for s in q:
        k = s.lower()
        if k not in seen:
            seen.add(k); uq.append(s)
    return uq

def collect_reddit_data(subreddit_list: List[str], home_team: str, 
                        away_team: str, date_iso: str, window_days: int = 3,
                        per_query_limit: int = 100, max_posts_total: int = 600,
                        sleep_seconds: float = 0.25) -> Dict[str, Any]:
    """
    Use time_filter='all' and locally filter by created_utc within [start,end].
    """
    reddit = initialize_reddit()
    match_date = datetime.fromisoformat(date_iso)
    start_time = to_epoch_time(match_date - timedelta(days=window_days))
    end_time   = to_epoch_time(match_date + timedelta(days=window_days))

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
                pass
            if len(posts) >= max_posts_total:
                break
            time.sleep(sleep_seconds)
        if len(posts) >= max_posts_total:
            break

    posts.sort(key=lambda p: (p["score"], p["num_comments"], p["created_utc"]), reverse=True)

    # collect sample comments
    comments: List[Dict[str, Any]] = []
    for p in posts[: min(12, len(posts))]:
        try:
            subm = reddit.submission(id=p["id"])
            subm.comments.replace_more(limit=0)
            for c in subm.comments[:80]:
                comments.append({
                    "post_id": p["id"],
                    "id": c.id,
                    "body": (getattr(c, "body","") or "")[:1000],
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
    ensure_directory(RAW_REDDIT_DIR)
    path = os.path.join(RAW_REDDIT_DIR, f"{match_id}.json")
    save_json(reddit_bundle, path)
    return path

def clean_reddit_data(raw_reddit_directory: str, matches_df: pd.DataFrame) -> pd.DataFrame:
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


# ============== USER SELECTION FOR REDDIT FETCH ======================
def _print_match_menu(matches_df: pd.DataFrame) -> None:
    print("\n=== Select matches to fetch Reddit data ===")
    for i, r in enumerate(matches_df.itertuples(index=False), start=1):
        print(f"{i:>2}. {r.date} — {r.home_team} vs {r.away_team} | "
              f"{r.competition} | {r.result} | outcome={getattr(r,'outcome','')}")

    print("\nEnter numbers like '1,3,5' or ranges like '2-4'. "
          "Press ENTER to skip Reddit fetching.")

def _parse_selection(user_input: str, max_n: int) -> List[int]:
    s = (user_input or "").strip()
    if not s: return []
    picks = set()
    for tok in s.split(","):
        tok = tok.strip()
        if not tok: continue
        if "-" in tok:
            try:
                a,b = map(int, tok.split("-",1))
                if a>b: a,b=b,a
                for j in range(a,b+1):
                    if 1<=j<=max_n: picks.add(j)
            except Exception:
                continue
        else:
            try:
                j=int(tok)
                if 1<=j<=max_n: picks.add(j)
            except Exception:
                continue
    return sorted(picks)

def run_reddit_pipeline_for_selected(matches_df: pd.DataFrame,
                                     selected_indices: List[int],
                                     subreddit_list: Optional[List[str]] = None,
                                     window_days: int = 3) -> None:
    if subreddit_list is None:
        subreddit_list = ["soccer","coys","PremierLeague","soccerhighlights"]
    if not selected_indices:
        print("No matches selected. Skipping Reddit collection.")
        return

    # Take rows by 1-based indices, newest first for display consistency
    sorted_df = matches_df.sort_values(by="date", ascending=False).reset_index(drop=True)
    chosen = sorted_df.iloc[[i-1 for i in selected_indices if 1 <= i <= len(sorted_df)]]
    if chosen.empty:
        print("Selected indices out of range. Skipping.")
        return

    print(f"Collecting Reddit data for {len(chosen)} selected match(es)...")
    iterator = tqdm(chosen.iterrows(), total=len(chosen), desc="Reddit data") if TQDM else chosen.iterrows()
    for _, row in iterator:
        raw_path = os.path.join(RAW_REDDIT_DIR, f"{row['match_id']}.json")
        if os.path.exists(raw_path):
            print(f"Skip (raw exists): {raw_path}")
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

    # Process only chosen matches, then merge with existing metrics
    print("Processing raw Reddit data for selected matches...")
    processed = clean_reddit_data(RAW_REDDIT_DIR, chosen)
    if os.path.exists(ENRICHED_REDDIT):
        cur = pd.read_csv(ENRICHED_REDDIT)
        cur = cur[~cur["match_id"].isin(processed["match_id"])]
        merged = pd.concat([cur, processed], ignore_index=True)
    else:
        merged = processed
    save_csv(merged, ENRICHED_REDDIT)
    print(f"Clean Reddit metrics saved: {ENRICHED_REDDIT} ({len(merged)} rows)")


# =========================== MAIN ORCHESTRATION ======================
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

    # STEP 2: TRANSFORM — Clean matches
    print("\n=== STEP 2: TRANSFORM (Clean matches) ===")
    clean_df = clean_transfermarkt_data(RAW_TRANSFERMARKT)
    save_csv(clean_df, ENRICHED_MATCHES)
    print(f"Saved clean matches: {ENRICHED_MATCHES} ({len(clean_df)} rows)")

    # STEP 3: Reddit — user selects which matches to fetch
    print("\n=== STEP 3: REDDIT (select matches) ===")
    menu_df = clean_df.sort_values(by="date", ascending=False).reset_index(drop=True)
    _print_match_menu(menu_df)
    selected = _parse_selection(input("\nSelect matches (e.g., 1,3,5 or 2-4). ENTER to skip: "), len(menu_df))
    if selected:
        run_reddit_pipeline_for_selected(
            matches_df=clean_df,
            selected_indices=selected,
            subreddit_list=["soccer","coys","PremierLeague","soccerhighlights"],
            window_days=3
        )
    else:
        print("No selection. Skipping Reddit collection.")

    # STEP 4: ENRICH — DeepSeek scoring (uses existing metrics CSV)
    print("\n=== STEP 4: ENRICH (DeepSeek) ===")
    run_deepseek_enrichment(
        matches_csv=ENRICHED_MATCHES,
        reddit_metrics_csv=ENRICHED_REDDIT,
        out_csv=FINAL_SCORES
    )

    # STEP 5: Show summary
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