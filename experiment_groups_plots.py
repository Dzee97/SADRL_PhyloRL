from pathlib import Path
from evaluation import compare_over_checkpoints

BASE_DIR = Path("output/")
S100 = "Size9Samples100Train100Test20"
S100_DIR = BASE_DIR / S100
S100_EVAL_SELF = "evaluate_Size9Samples100Train100Test20_topk1"

# Experiment 1: Change gamma under target ent 0.3, replay size 10000
PLOT_DIR = BASE_DIR / "Experiment_1"
compare_over_checkpoints(
    eval_configs=[
        {"dir": S100_DIR / "SQL_6c83fa9a" / S100_EVAL_SELF, "label": "γ=0.99"},
        {"dir": S100_DIR / "SQL_9e25499f" / S100_EVAL_SELF, "label": "γ=0.9"},
        {"dir": S100_DIR / "SQL_0ad02242" / S100_EVAL_SELF, "label": "γ=0.5"},
        {"dir": S100_DIR / "SQL_3bdac2ee" / S100_EVAL_SELF, "label": "γ=0.0"}
    ],
    metric="match_raxml_count",
    train_dataset=S100,
    eval_dataset=S100,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)
