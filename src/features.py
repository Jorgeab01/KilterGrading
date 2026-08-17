"""
Feature sets and parameters for each training iteration.

Each entry in ITERATIONS is a numbered configuration: columns to use as
features and which XGBRegressor parameters to train with.

Once an iteration is added, don't edit it. Add a new number instead. If
you edit an old one, the saved model and its notes in ITERATIONS.md stop
matching the code that made them.
"""

BASE_FEATURES = [
    "angle",
    "num_holds",
    "num_start",
    "num_middle",
    "num_foot",
    "num_finish",
    "avg_nearest_hold_dist",
]


ITERATIONS = {
    1: {
        "features": BASE_FEATURES,
        "model_params": {},
    },
}