import numpy as np
import pandas as pd

from .config import EXTRA_WINDOWS, MANUAL_WINDOWS


def _as_key(value):
    # groupby по одной колонке отдаёт скаляр, по двум — tuple
    if isinstance(value, tuple):
        return value
    return (value,)


def count_before(history, pairs, keys, windows, prefix):
    """сколько событий с таким ключом было в окне (t-W, t).
    Важно: берём только timestamp < t, иначе утечка."""
    out = pd.DataFrame(index=pairs.index)
    for w in windows:
        out[f"{prefix}_cnt_{w}d"] = 0

    if history.empty or pairs.empty:
        return out

    hist = history.dropna(subset=keys + ["timestamp"])
    grouped = {}
    for key, group in hist.groupby(keys, sort=False):
        grouped[_as_key(key)] = np.sort(group["timestamp"].to_numpy())

    key_arr = pairs[keys].to_numpy()
    ts = pairs["timestamp"].to_numpy()
    counts = {w: np.zeros(len(pairs), dtype=np.int32) for w in windows}

    for i in range(len(pairs)):
        times = grouped.get(tuple(key_arr[i]))
        if times is None:
            continue
        t = ts[i]
        # всё, что раньше t
        end = int(np.searchsorted(times, t, side="left"))
        if end == 0:
            continue
        past = times[:end]
        for w in windows:
            start = int(np.searchsorted(past, t - np.timedelta64(w, "D"), side="left"))
            counts[w][i] = end - start

    for w in windows:
        out[f"{prefix}_cnt_{w}d"] = counts[w]
    return out


def add_categories(pairs, users, items):
    frame = pairs.merge(users[["user_id", "age", "sex", "income", "kids"]], on="user_id", how="left")
    frame = frame.merge(items[["item_id", "genre", "content_type"]], on="item_id", how="left")
    for col in ("age", "sex", "income", "genre", "content_type"):
        frame[col] = frame[col].fillna("unknown")
    frame["kids"] = frame["kids"].fillna(0)
    return frame


def build_features(pairs, all_events, users, items, windows):
    base = add_categories(pairs, users, items)
    hist = all_events[["user_id", "item_id", "timestamp", "genre"]].copy()

    parts = [
        base,
        count_before(hist, base, ["user_id"], windows, "user"),
        count_before(hist, base, ["item_id"], windows, "item"),
        count_before(hist, base, ["user_id", "item_id"], windows, "ui"),
        count_before(hist, base, ["user_id", "genre"], windows, "ugenre"),
    ]
    feat = pd.concat(parts, axis=1)

    # доли: как часто этот тайтл / жанр относительно активности юзера
    for w in windows:
        user_cnt = feat[f"user_cnt_{w}d"].replace(0, np.nan)
        feat[f"ui_share_{w}d"] = feat[f"ui_cnt_{w}d"] / user_cnt
        feat[f"ugenre_share_{w}d"] = feat[f"ugenre_cnt_{w}d"] / user_cnt
        # насколько тайтл популярнее среднего в этой таблице
        feat[f"item_pop_rel_{w}d"] = feat[f"item_cnt_{w}d"] / (feat[f"item_cnt_{w}d"].mean() + 1e-6)

    return feat.replace([np.inf, -np.inf], np.nan).fillna(0)


def feature_columns(frame):
    cats = ["age", "sex", "income", "genre", "content_type"]
    skip = set(cats + ["user_id", "item_id", "timestamp", "label", "total_dur", "watched_pct"])
    nums = [c for c in frame.columns if c not in skip]
    return nums, cats


def manual_features(pairs, all_events, users, items):
    # обычный набор, который чаще всего считают руками
    return build_features(pairs, all_events, users, items, MANUAL_WINDOWS)


def extra_features(pairs, all_events, users, items):
    # те же фичи, но окон больше — это и есть автоматизация без нейронок
    return build_features(pairs, all_events, users, items, EXTRA_WINDOWS)
