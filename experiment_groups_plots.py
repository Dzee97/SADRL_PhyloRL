from pathlib import Path
from evaluation import compare_over_checkpoints

METRIC = "match_raxml_count"

BASE_DIR = Path("output/")

S1 = "Size9Samples1Train100Test20"
S1_DIR = BASE_DIR / S1
S1_EVAL_SELF = "evaluate_Size9Samples1Train100Test20_topk1"

# Experiment 1: Vanilla DQN vs extensions on single MSA (done)
PLOT_DIR = BASE_DIR / "Experiment_1"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
        {"dir": S1_DIR / "DQN_7f72e865" / S1_EVAL_SELF, "label": "DQN: Vanilla"},
        {"dir": S1_DIR / "DQN_cd6d8fe7" / S1_EVAL_SELF, "label": "DQN: PER + Double Q"},
        {"dir": S1_DIR / "SQL_03e60329" / S1_EVAL_SELF, "label": "SQL: Target Ent: 0.3"},
        {"dir": S1_DIR / "SQL_e427aee4" / S1_EVAL_SELF, "label": "SQL: Target Ent: 0.5"}
    ],
    metric=METRIC,
    train_dataset=S1,
    eval_dataset=S1,
    algorithm_name="DQN + SQL",
    plot_dir=PLOT_DIR
)

S100 = "Size9Samples100Train100Test20"
S100_EVAL = "Size9Samples100Test20Validation"
S100_DIR = BASE_DIR / S100
S100_EVAL_VALIDATE = "evaluate_Size9Samples100Test20Validation_topk1"

# Experiment 2: Compare SQL on multiple MSAs with and without regularization, target ent 0.3
PLOT_DIR = BASE_DIR / "Experiment_2"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
        {"dir": S100_DIR / "SQL_03e60329" / S100_EVAL_VALIDATE, "label": "No regularization"},  # x
        {"dir": S100_DIR / "SQL_9e25499f" / S100_EVAL_VALIDATE, "label": "Dropout + AdamW"},  # x
    ],
    metric=METRIC,
    train_dataset=S100,
    eval_dataset=S100_EVAL,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)

# Experiment 3.1: Change gamma under target ent 0.3, replay size 10000
PLOT_DIR = BASE_DIR / "Experiment_3.1"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
        {"dir": S100_DIR / "SQL_6c83fa9a" / S100_EVAL_VALIDATE, "label": "γ=0.99"},  # x
        {"dir": S100_DIR / "SQL_9e25499f" / S100_EVAL_VALIDATE, "label": "γ=0.9"},  # x
        {"dir": S100_DIR / "SQL_0ad02242" / S100_EVAL_VALIDATE, "label": "γ=0.5"},  # x
        {"dir": S100_DIR / "SQL_3bdac2ee" / S100_EVAL_VALIDATE, "label": "γ=0.0"}   # x
    ],
    metric=METRIC,
    train_dataset=S100,
    eval_dataset=S100_EVAL,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)

# Experiment 3.2: Change gamma under target ent 0.3, replay size 10000, with new features
PLOT_DIR = BASE_DIR / "Experiment_3.2"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
        {"dir": S100_DIR / "SQL_0d399a4b" / S100_EVAL_VALIDATE, "label": "γ=0.99"},
        {"dir": S100_DIR / "SQL_21ebac9e" / S100_EVAL_VALIDATE, "label": "γ=0.9"},
        {"dir": S100_DIR / "SQL_2fa4c376" / S100_EVAL_VALIDATE, "label": "γ=0.5"},
        {"dir": S100_DIR / "SQL_4841ac65" / S100_EVAL_VALIDATE, "label": "γ=0.0"}
    ],
    metric=METRIC,
    train_dataset=S100,
    eval_dataset=S100_EVAL,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)

S100_EVAL_VALIDATE_top5 = "evaluate_Size9Samples100Test20Validation_topk5"
S100_EVAL_VALIDATE_top10 = "evaluate_Size9Samples100Test20Validation_topk10"

# Experiment 3.3: Compare top1 to top5 evaluation
PLOT_DIR = BASE_DIR / "Experiment_3.3"
PLOT_DIR.mkdir(exist_ok=True)
compare_over_checkpoints(
    eval_configs=[
        {"dir": S100_DIR / "SQL_21ebac9e" / S100_EVAL_VALIDATE, "label": "Greedy evaluation"},
        {"dir": S100_DIR / "SQL_21ebac9e" / S100_EVAL_VALIDATE_top5, "label": "Guided top-5 evaluation"},
    ],
    metric=METRIC,
    train_dataset=S100,
    eval_dataset=S100_EVAL,
    algorithm_name="SQL",
    plot_dir=PLOT_DIR
)
