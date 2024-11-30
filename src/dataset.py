import pandas as pd


def extract_keywords(links):
    keywords = []
    for link in links.split(","):
        if "https://en.wikipedia.org/wiki/" in link:
            keywords.append(link.split("https://en.wikipedia.org/wiki/")[1])
    return ",".join(keywords)


def get_frames_dataset():
    df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

    # Keep only the 'Prompt', Answer and 'wiki_links' columns
    df_filtered = df.loc[:, ["Prompt", "Answer", "reasoning_types", "wiki_links"]]
    # keep track of the number of wiki links as ground truth
    df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1

    df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

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

if __name__ == '__main__':
    get_frames_dataset()