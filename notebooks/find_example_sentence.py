# %%
import pandas as pd
# %%
df = pd.read_csv("../data/org_data_processing/DATA_senses_sentences_scored.csv")
df.head()
# %%

# search for a term
search_term = " sorrow"

# find something high concreteness containing that term
filtered_df = df[df['text'].str.contains(search_term, case=False, na=False)]
sorted_df = filtered_df.sort_values(by='concreteness_normalized', ascending=False)
for i, row in sorted_df[['fileid', 'text', 'concreteness_normalized', 'imageability_normalized']].head(10).iterrows():
    print(f"FileID: {row['fileid']}")
    print(f"Concreteness: {row['concreteness_normalized']}, Imageability: {row['imageability_normalized']}")
    print(f"Sentence: {row['text']}")
    print("-----")

# %%
# now search for sentences with city names
cities = ["New York", "Los Angeles", "Chicago", "Houston", "Washington", "Miami", "Seattle", "Boston", "San Francisco", 
          "Denver", "London", "Paris", "Berlin", "Tokyo", "Sydney", "Toronto", "Vancouver", "Rome", "Madrid", "Dublin", "Moscow", "Beijing", "Mumbai", 
          "Cairo", "Cape Town", "Rio de Janeiro", "Buenos Aires", "Lima", "Santiago", "Bogota", "Caracas",
          "Mexico City", "São Paulo", "Lisbon", "Athens", "Istanbul", "Dubai", "Singapore", "Hong Kong", "Seoul", "Bangkok", "Jakarta"]
# same thing:
filtered_df = df[df['text'].str.contains('|'.join(cities), case=True, na=False)]
sorted_df = filtered_df.sort_values(by='concreteness_normalized', ascending=False)
for i, row in sorted_df[['fileid', 'text', 'concreteness_normalized', 'imageability_normalized']].head(20).iterrows():
    print(f"FileID: {row['fileid']}")
    print(f"Concreteness: {row['concreteness_normalized']}, Imageability: {row['imageability_normalized']}")
    print(f"Sentence: {row['text']}")
    print("-----")
# %%

# now we want to find sentence low in concreteness but high in imageability
sorted_df = df.sort_values(by='concreteness_normalized', ascending=True).head(300)
sorted_df = sorted_df.sort_values(by='imageability_normalized', ascending=False)
for i, row in sorted_df[['fileid', 'text', 'concreteness_normalized', 'imageability_normalized']].head(10).iterrows():
    print(f"FileID: {row['fileid']}")
    print(f"Concreteness: {row['concreteness_normalized']}, Imageability: {row['imageability_normalized']}")
    print(f"Sentence: {row['text']}")
    print("-----")  


# %%
