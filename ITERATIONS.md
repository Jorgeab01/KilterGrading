# Iterations

## 002 - Geometry features

Added five features describing the shape of the route instead of just
counting holds.

**Validation MAE:** 2.07

Parameter tuning was tried on top of these features and still
didn't move the number meaningfully (2.06 with defaults -> 2.057),
confirming the bottleneck was the features, not the model, at both iterations.

**Features**
- angle, num_holds, num_start, num_middle, num_foot, num_finish,
  avg_nearest_hold_dist (from iteration 1)
- `max_dist_between_nearest_holds`: the single biggest reach between holds, not the average
- `width`, `height`: how far the route spreads sideways and up
- `foot_ratio`, `middle_ratio`: hold-type proportions

**Model:** `models/it002.json`

## 001 - Baseline

First XGBoost model using the initial set of route features.

**Validation MAE:** 2.42

**Features**
- angle
- num_holds
- num_start
- num_middle
- num_foot
- num_finish
- avg_nearest_hold_dist

**Model:** `models/it001.json`

---