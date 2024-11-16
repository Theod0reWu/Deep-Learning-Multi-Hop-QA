import pandas as pd

df = pd.read_csv("hf://datasets/google/frames-benchmark/test.tsv", sep="\t")

# filter for 3 wiki links or less
df_filtered = df[df["wiki_links"].str.count(",") <= 2]

# print(df_filtered.head(2))

# Keep only the 'Prompt' and 'wiki_links' columns
df_filtered = df_filtered.loc[:, ["Prompt", "reasoning_types", "wiki_links"]]

# keep track of the number of wiki links as ground truth
df_filtered["query_count"] = df_filtered["wiki_links"].str.count(",") + 1


def extract_keywords(links):
    keywords = []
    for link in links.split(","):
        if "https://en.wikipedia.org/wiki/" in link:
            keywords.append(link.split("https://en.wikipedia.org/wiki/")[1])
    return ",".join(keywords)


df_filtered["keywords"] = df_filtered["wiki_links"].apply(extract_keywords)

print(df_filtered.head(2))
print(df_filtered.shape)
