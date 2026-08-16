# KilterGrading

![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.13-blue)

Predicts the difficulty grade of a Kilter Board climbing problem from its
hold layout and wall angle, trained on real climbs graded by the community.

## Why

Grading a climbing problem is subjective. Different climbers disagree
depending on height, strength style, and experience. On a Kilter Board, the
displayed grade is actually the average of every climber who's logged that
problem, which means a newly set route has no grade at all until enough
people have climbed it and voted.

KilterGrading predicts a starting grade from the hold layout and wall
angle alone, before any of that community feedback exists, using
gradient boosting trained on 190k already graded climbs.

## How

Each climb is converted into a feature vector: wall angle,
number of holds by role (start, middle, foot, finish), and the average
distance between holds. These features and the real community grade
train a gradient boosting model (XGBoost) that learns the relationship
between hold layout and difficulty from 190k already graded climbs.

## Data

The database isn't included in this repo.
Download `kilter_splits.sqlite` from Hugging Face and place it in `data/`:

https://huggingface.co/datasets/Vilin97/KilterBoard

## Pipeline

- [x] `src/data_parser.py`:
    Reads the database, converts each route into a
    feature vector (angle, hold counts by role, average distance
    between holds), and saves `data/{train,val,test}_features.csv`.

- [ ] `src/train_model.py`:
    Trains a gradient boosting model and evaluates
    it against val/test.