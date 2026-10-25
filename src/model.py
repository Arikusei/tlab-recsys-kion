import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score

from .config import CATBOOST_DEPTH, CATBOOST_ITERS, CATBOOST_LR, SEED, TOP_K


def fit_catboost(train, val, num_cols, cat_cols):
    model = CatBoostClassifier(
        iterations=CATBOOST_ITERS,
        depth=CATBOOST_DEPTH,
        learning_rate=CATBOOST_LR,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        auto_class_weights="Balanced",
    )
    train_pool = Pool(train[num_cols + cat_cols], label=train["label"], cat_features=cat_cols)
    val_pool = Pool(val[num_cols + cat_cols], label=val["label"], cat_features=cat_cols)
    # early stopping по val, test тут не используем
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def classify_metrics(model, frame, num_cols, cat_cols):
    proba = model.predict_proba(frame[num_cols + cat_cols])[:, 1]
    y = frame["label"].to_numpy()
    return {
        "logloss": float(log_loss(y, proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, proba)),
        "n": int(len(frame)),
    }


def ranking_metrics(model, frame, num_cols, cat_cols, k=TOP_K):
    """NDCG и Recall считаем по каждому юзеру, потом среднее"""
    proba = model.predict_proba(frame[num_cols + cat_cols])[:, 1]
    tmp = frame[["user_id", "label"]].copy()
    tmp["score"] = proba

    ndcgs = []
    recalls = []
    for _, group in tmp.groupby("user_id"):
        group = group.sort_values("score", ascending=False)
        n_pos = int(group["label"].sum())
        if n_pos == 0:
            continue
        top = group.head(k)
        recalls.append(int(top["label"].sum()) / n_pos)

        dcg = 0.0
        for rank, lab in enumerate(top["label"].to_numpy(), start=1):
            if lab:
                dcg += 1.0 / np.log2(rank + 1)
        ideal = sum(1.0 / np.log2(r + 1) for r in range(1, min(n_pos, k) + 1))
        ndcgs.append(dcg / ideal if ideal else 0.0)

    return {
        f"ndcg@{k}": float(np.mean(ndcgs) if ndcgs else 0.0),
        f"recall@{k}": float(np.mean(recalls) if recalls else 0.0),
        "ranked_users": int(len(ndcgs)),
    }


def top_features(model, num_cols, cat_cols, n=15):
    names = num_cols + cat_cols
    imp = model.get_feature_importance()
    order = np.argsort(imp)[::-1][:n]
    return [(names[i], float(imp[i])) for i in order]
