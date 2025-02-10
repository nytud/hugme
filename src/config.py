import os

RESULTS_DIR = os.getenv("RESULTS_DIR", "./results")

DATASETS = os.getenv("DATASETS", "./datasets/")

NEEEDLE_FILE = DATASETS + "nih_needle.txt"
