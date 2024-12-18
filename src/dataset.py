import pandas as pd
import random


def extract_keywords(links):
    keywords = []
    for link in links.split(","):
        if "https://en.wikipedia.org/wiki/" in link:
            keywords.append(link.split("https://en.wikipedia.org/wiki/")[1])
    return ",".join(keywords)


def get_frames_dataset():
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "Answer", "reasoning_types", "wiki_links"]]
    # keep track of the number of wiki links as ground truth

    return df_filtered


def get_frames_filtereddataset():
    print("Reading dataset...")
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
    print(f"Initial dataframe shape: {df.shape}")

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "reasoning_types", "wiki_links"]]

    # Ensure Prompt column contains strings
    df_filtered["Prompt"] = df_filtered["Prompt"].astype(str)

    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

    print(f"Final dataframe shape: {df_filtered.shape}")
    print(f"Prompt column type: {df_filtered['Prompt'].dtype}")

    return df_filtered


def get_frames_relevant_dataset():
    print("Reading dataset...")
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
    print(f"Initial dataframe shape: {df.shape}")

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "reasoning_types", "Answer", "wiki_links"]]

    # Ensure Prompt column contains strings
    df_filtered["Prompt"] = df_filtered["Prompt"].astype(str)

    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)
    df_filtered = df_filtered[
        (df_filtered["query_count"] >= 2) & (df_filtered["query_count"] <= 7)
    ]

    print(f"Final dataframe shape: {df_filtered.shape}")
    print(f"Prompt column type: {df_filtered['Prompt'].dtype}")

    return df_filtered


def get_condensed_frames_dataset(samples_per_query=5):
    """
    Creates a condensed dataset containing a fixed number of questions for each unique query_count.

    Args:
        df_filtered (pd.DataFrame): The filtered dataset with a 'query_count' column.
        samples_per_query (int): The number of questions to sample for each unique value in 'query_count'.

    Returns:
        pd.DataFrame: A condensed dataset.
    """
    print("Creating condensed dataset...")
    print("Reading dataset...")
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
    print(f"Initial dataframe shape: {df.shape}")
    df = df[df["Unnamed: 0"] != 11]
    df = df[df["Unnamed: 0"] != 59]
    df = df[df["Unnamed: 0"] != 0]
    df = df[df["Unnamed: 0"] != 4]
    df = df[df["Unnamed: 0"] != 1]
    df = df[df["Unnamed: 0"] != 2]

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "Answer", "reasoning_types", "wiki_links"]]

    # Ensure Prompt column contains strings
    df_filtered["Prompt"] = df_filtered["Prompt"].astype(str)

    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)
    df_filtered = df_filtered[
        (df_filtered["query_count"] >= 2) & (df_filtered["query_count"] <= 7)
    ]
    # Initialize an empty list to store sampled data
    sampled_dfs = []

    # Iterate over each unique query_count value
    for query in df_filtered["query_count"].unique():
        # Filter rows for the current query_count value
        subset = df_filtered[df_filtered["query_count"] == query]

        # Randomly sample up to `samples_per_query` rows
        sampled = subset.sample(n=min(samples_per_query, len(subset)), random_state=42)
        sampled_dfs.append(sampled)

    # Concatenate all sampled dataframes
    condensed_df = pd.concat(sampled_dfs)

    print(f"Condensed dataframe shape: {condensed_df.shape}")
    return condensed_df


def get_random_question():
    """
    Returns one random question from the dataset.

    Args:
        df (pd.DataFrame): The dataset to select the question from.

    Returns:
        str: A random question (Prompt) from the dataset.
    """
    print("Reading dataset...")
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")
    print(f"Initial dataframe shape: {df.shape}")

    # Keep only the 'Prompt' and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "Answer", "reasoning_types", "wiki_links"]]

    # Ensure Prompt column contains strings
    df_filtered["Prompt"] = df_filtered["Prompt"].astype(str)

    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

    random_index = random.randint(0, len(df_filtered) - 1)
    new_df = df_filtered.iloc[[random_index]]
    return new_df


# def main():
#     d = get_condensed_frames_dataset(5)
#     print(d.head())


# if __name__ == "__main__":
#     main()
