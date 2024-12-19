import pandas as pd
import os

if not os.path.exists("test.tsv"):
    print("Downloading Frames dataset...")
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
    df.to_csv("test.tsv", sep="\t", index=False)
else:
    print("Loading Frames dataset from local file...")
    df = pd.read_csv("test.tsv", sep="\t")
