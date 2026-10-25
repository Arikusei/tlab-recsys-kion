import zipfile
from urllib.request import urlretrieve

from .config import DATA_DIR, RAW_DIR, ZIP_URL


def ensure_kion():
    """скачать KION, если csv ещё нет"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions = RAW_DIR / "interactions.csv"
    if interactions.exists():
        return RAW_DIR

    zip_path = DATA_DIR / "data_original.zip"
    if not zip_path.exists():
        print("качаю датасет...")
        urlretrieve(ZIP_URL, zip_path)

    print("распаковываю...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)

    if not interactions.exists():
        raise FileNotFoundError("после распаковки нет interactions.csv")
    return RAW_DIR
