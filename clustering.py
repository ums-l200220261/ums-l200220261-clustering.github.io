import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

file_path = 'chat.csv'
data = pd.read_csv(file_path)
# Define a function to parse and clean the data
def parse_message(row):
    # Pattern to identify timestamp, sender, and message
    pattern = r"(\d{2}/\d{2}/\d{2} \d{2}\.\d{2}) - (.+?): (.+)"
    match = re.match(pattern, row)
    if match:
        return match.groups()
    return None, None, None

# Apply parsing to the dataset
data_cleaned = data.iloc[:, 0].dropna().apply(parse_message)
parsed_data = pd.DataFrame(data_cleaned.tolist(), columns=["Timestamp", "Sender", "Message"])

# Drop rows where parsing failed (empty rows)
parsed_data = parsed_data.dropna().reset_index(drop=True)

# Display the first few rows of the cleaned data
parsed_data.head()

# Step 1: Preprocessing - Convert messages into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to top 500 words for simplicity
X = vectorizer.fit_transform(parsed_data["Message"])

# Step 2: Define a function to perform clustering and analyze results
def perform_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Top keywords per cluster
    top_keywords = []
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    for i in range(n_clusters):
        top_keywords.append([terms[ind] for ind in order_centroids[i, :3]])  # Top 3 words per cluster
    
    return clusters, top_keywords

# Perform clustering for 3, 4, and 5 clusters
results = {}
for n in [3, 4, 5]:
    clusters, keywords = perform_clustering(X, n_clusters=n)
    parsed_data[f"Cluster_{n}"] = clusters  # Add cluster labels to the dataframe
    results[n] = keywords

# Display the top keywords for each clustering result
results

# Menampilkan beberapa baris dari dataframe yang sudah diberi label cluster
print("\n=== Parsed Data with Clustering ===")
print(parsed_data.head())

# Menampilkan top keywords untuk setiap cluster
print("\n=== Top Keywords per Cluster ===")
for n_clusters, keywords in results.items():
    print(f"\nNumber of Clusters: {n_clusters}")
    for i, words in enumerate(keywords):
        print(f"Cluster {i}: {', '.join(words)}")
