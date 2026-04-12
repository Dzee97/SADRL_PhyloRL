from pathlib import Path
from evaluation import compare_over_checkpoints

METRIC = "match_raxml_count"

BASE_DIR = Path("output/")

S100 = "Size9Samples100Train100Test20"
S100_EVAL = "Size9Samples100Test20Validation"
S100_DIR = BASE_DIR / S100
S100_EVAL_VALIDATE = "evaluate_Size9Samples100Train100Test20_topk1"

# Experiment 3.2: Change gamma under target ent 0.3, replay size 10000, with new features
PLOT_DIR = BASE_DIR / "Experiment_3.2"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
       # {"dir": S100_DIR / "SQL_0d399a4b" / S100_EVAL_VALIDATE, "label": "γ=0.99"},
        {"dir": S100_DIR / "SQL_4ca876ab" / S100_EVAL_VALIDATE, "label": "γ=0.9"},
       # {"dir": S100_DIR / "SQL_2fa4c376" / S100_EVAL_VALIDATE, "label": "γ=0.5"},
       # {"dir": S100_DIR / "SQL_4841ac65" / S100_EVAL_VALIDATE, "label": "γ=0.0"}
    ],
    metric=METRIC,
    train_dataset=S100,
    eval_dataset=S100_EVAL,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)
