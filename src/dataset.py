import numpy as np
import pandas as pd

from .config import (
    CANDIDATES_NEG_RANK,
    N_USERS,
    NEG_PER_POS,
    RAW_DIR,
    SEED,
    TEST_DAYS,
    VAL_DAYS,
    WATCH_POS_PCT,
)
from .download import ensure_kion


def load_raw():
    ensure_kion()

    interactions = pd.read_csv(RAW_DIR / "interactions.csv", parse_dates=["last_watch_dt"])
    users = pd.read_csv(RAW_DIR / "users.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")

    interactions = interactions.rename(columns={"last_watch_dt": "timestamp"})
    interactions["watched_pct"] = interactions["watched_pct"].fillna(0)
    interactions["total_dur"] = interactions["total_dur"].fillna(0)

    # позитив = относительно нормальный просмотр, не клик на 2 секунды
    interactions["label"] = (interactions["watched_pct"] >= WATCH_POS_PCT).astype(np.int8)

    # у фильма несколько жанров через запятую, беру первый
    items["genre"] = items["genres"].fillna("unknown").astype(str).str.split(",").str[0].str.strip()
    items["genre"] = items["genre"].replace("", "unknown")
    items["content_type"] = items["content_type"].fillna("unknown")

    for col in ("age", "sex", "income"):
        users[col] = users[col].fillna("unknown")
    users["kids"] = users["kids_flg"].fillna(0) if "kids_flg" in users.columns else 0

    return interactions, users, items


def sample_users(interactions, n_users=N_USERS, seed=SEED):
    """случайные юзеры, но все их просмотры. Иначе фичи по истории будут кривые."""
    rng = np.random.default_rng(seed)
    all_users = interactions["user_id"].unique()
    n = min(n_users, len(all_users))
    chosen = rng.choice(all_users, size=n, replace=False)
    sampled = interactions[interactions["user_id"].isin(chosen)].copy()
    return sampled.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)


def add_negatives(positives, catalog, seed, k):
    """к каждому просмотру добавляем k тайтлов, которые этот юзер вообще не смотрел"""
    rng = np.random.default_rng(seed)
    seen = positives.groupby("user_id")["item_id"].agg(set).to_dict()
    catalog = np.asarray(catalog)
    rows = []

    for row in positives.itertuples(index=False):
        banned = seen.get(row.user_id, set())
        picks = []
        tries = 0
        while len(picks) < k and tries < 120:
            tries += 1
            cand = int(rng.choice(catalog))
            if cand not in banned and cand not in picks:
                picks.append(cand)
        for item_id in picks:
            rows.append(
                {
                    "user_id": int(row.user_id),
                    "item_id": item_id,
                    "timestamp": row.timestamp,
                    "label": 0,
                    "total_dur": 0,
                    "watched_pct": 0,
                }
            )

    pos = positives[["user_id", "item_id", "timestamp", "total_dur", "watched_pct"]].copy()
    pos["label"] = 1
    return pd.concat([pos, pd.DataFrame(rows)], ignore_index=True)


def make_splits(n_users=N_USERS):
    interactions, users, items = load_raw()
    sampled = sample_users(interactions, n_users=n_users)

    sampled = sampled.merge(items[["item_id", "genre", "content_type"]], on="item_id", how="left")
    sampled["genre"] = sampled["genre"].fillna("unknown")

    max_dt = sampled["timestamp"].max()
    cut_test = max_dt - pd.Timedelta(days=TEST_DAYS)
    cut_val = cut_test - pd.Timedelta(days=VAL_DAYS)
    cut_train = cut_val - pd.Timedelta(days=VAL_DAYS)

    def positives_between(start, end=None):
        mask = (sampled["timestamp"] >= start) & (sampled["label"] == 1)
        if end is not None:
            mask &= sampled["timestamp"] < end
        cols = ["user_id", "item_id", "timestamp", "total_dur", "watched_pct"]
        return sampled.loc[mask, cols]

    train_pos = positives_between(cut_train, cut_val)
    val_pos = positives_between(cut_val, cut_test)
    test_pos = positives_between(cut_test, None)

    catalog = sampled["item_id"].unique()
    train = add_negatives(train_pos, catalog, seed=SEED, k=NEG_PER_POS)
    val = add_negatives(val_pos, catalog, seed=SEED + 1, k=NEG_PER_POS)
    test = add_negatives(test_pos, catalog, seed=SEED + 2, k=NEG_PER_POS)
    # отдельный набор для NDCG/Recall: больше негативов
    test_rank = add_negatives(test_pos, catalog, seed=SEED + 3, k=CANDIDATES_NEG_RANK)

    keep_users = users[users["user_id"].isin(sampled["user_id"].unique())]
    keep_items = items[items["item_id"].isin(sampled["item_id"].unique())]

    print(
        f"users={sampled.user_id.nunique()} events={len(sampled)} "
        f"train={len(train)} val={len(val)} test={len(test)} rank={len(test_rank)}"
    )
    print(f"даты: train={cut_train.date()} val={cut_val.date()} test={cut_test.date()} max={max_dt.date()}")

    return {
        "all_events": sampled,
        "train": train,
        "val": val,
        "test": test,
        "test_rank": test_rank,
        "users": keep_users,
        "items": keep_items,
        "cut_train": cut_train,
        "cut_val": cut_val,
        "cut_test": cut_test,
    }
