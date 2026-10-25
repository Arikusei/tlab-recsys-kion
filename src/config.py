from pathlib import Path

# пути
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "data_original"
REPORTS_DIR = ROOT / "reports"

SEED = 42

# полный датасет тяжёлый, берём часть юзеров
# историю каждого выбранного юзера оставляем целиком
N_USERS = 4000

# на 1 просмотр добавляем 4 случайных непросмотренных тайтла
NEG_PER_POS = 4

# val и test — последние 2 недели, train — неделя перед ними
TEST_DAYS = 7
VAL_DAYS = 7

# позитив, если досмотрели хотя бы 10%
WATCH_POS_PCT = 10.0

# для ранжирования: 1 позитив + 50 негативов
CANDIDATES_NEG_RANK = 50
TOP_K = 10

CATBOOST_ITERS = 250
CATBOOST_DEPTH = 6
CATBOOST_LR = 0.08

# руками обычно берут неделю и месяц
MANUAL_WINDOWS = (7, 30)

# идея: нагенерить больше окон, пусть модель сама выберет
EXTRA_WINDOWS = (1, 7, 14, 30, 60)

ZIP_URL = (
    "https://github.com/irsafilo/KION_DATASET/raw/"
    "f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"
)
