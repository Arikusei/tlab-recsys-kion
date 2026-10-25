import json
import time

import pandas as pd

from src.config import N_USERS, REPORTS_DIR
from src.dataset import make_splits
from src.features import extra_features, feature_columns, manual_features
from src.model import classify_metrics, fit_catboost, ranking_metrics, top_features


def make_all_features(builder, splits, name):
    t0 = time.perf_counter()
    print(f"считаю фичи: {name}")
    train = builder(splits["train"], splits["all_events"], splits["users"], splits["items"])
    val = builder(splits["val"], splits["all_events"], splits["users"], splits["items"])
    test = builder(splits["test"], splits["all_events"], splits["users"], splits["items"])
    rank = builder(splits["test_rank"], splits["all_events"], splits["users"], splits["items"])
    print(f"  готово за {time.perf_counter() - t0:.1f} сек")
    return train, val, test, rank


def evaluate(model, val, test, rank, num_cols, cat_cols):
    return {
        "n_features": len(num_cols) + len(cat_cols),
        "val": classify_metrics(model, val, num_cols, cat_cols),
        "test": classify_metrics(model, test, num_cols, cat_cols),
        "ranking": ranking_metrics(model, rank, num_cols, cat_cols),
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("готовлю данные")
    splits = make_splits(n_users=N_USERS)

    man_tr, man_va, man_te, man_rk = make_all_features(manual_features, splits, "ручные окна 7 и 30")
    extra_tr, extra_va, extra_te, extra_rk = make_all_features(extra_features, splits, "окна 1/7/14/30/60")

    man_num, man_cat = feature_columns(man_tr)
    extra_num, extra_cat = feature_columns(extra_tr)

    print("учу catboost на ручных фичах")
    man_model = fit_catboost(man_tr, man_va, man_num, man_cat)

    print("учу catboost на расширенном наборе")
    extra_model = fit_catboost(extra_tr, extra_va, extra_num, extra_cat)

    report = {
        "dataset": "MTS KION, 4000 users, full history",
        "n_users": int(splits["all_events"].user_id.nunique()),
        "n_events": int(len(splits["all_events"])),
        "cuts": {
            "train": str(splits["cut_train"].date()),
            "val": str(splits["cut_val"].date()),
            "test": str(splits["cut_test"].date()),
        },
        "target": "label=1 if watched_pct >= 10, else random unseen item",
        "manual": evaluate(man_model, man_va, man_te, man_rk, man_num, man_cat),
        "extra_windows": evaluate(extra_model, extra_va, extra_te, extra_rk, extra_num, extra_cat),
        "important_features": [
            {"name": name, "importance": round(val, 3)}
            for name, val in top_features(extra_model, extra_num, extra_cat)
        ],
    }

    out = REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for method, block in (("manual_7_30", report["manual"]), ("extra_windows", report["extra_windows"])):
        rows.append(
            {
                "method": method,
                "n_features": block["n_features"],
                "test_logloss": block["test"]["logloss"],
                "test_auc": block["test"]["roc_auc"],
                **block["ranking"],
            }
        )
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "metrics.csv", index=False)

    print("test manual:", report["manual"]["test"])
    print("test extra:", report["extra_windows"]["test"])
    print("rank extra:", report["extra_windows"]["ranking"])
    print("сохранил", out)


if __name__ == "__main__":
    main()
