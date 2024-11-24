import pandas as pd


def extract_keywords(links):
    keywords = []
    for link in links.split(","):
        if "https://en.wikipedia.org/wiki/" in link:
            keywords.append(link.split("https://en.wikipedia.org/wiki/")[1])
    return ",".join(keywords)


def get_frames_dataset():
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "reasoning_types", "wiki_links"]]
    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

    return df_filtered


def get_frames_filtereddataset():
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    # filter for 3 wiki links or less
    df_filtered = df[df["wiki_links"].str.count(",") <= 2]

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df_filtered.loc[:, ["Prompt", "reasoning_types", "wiki_links"]]

    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

    return df_filtered
