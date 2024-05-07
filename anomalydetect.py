import re
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import base64

# Define known timestamp patterns
timestamp_patterns = [
    r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',  # ISO 8601, common log format
    r'\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}:\d{2}',      # U.S. format with slashes
    r'\d{2}-\w{3}-\d{4}\s\d{2}:\d{2}:\d{2}',      # U.K. format with dashes and month abbreviation
    r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',       # Apache log format
    r'\w{3}\s+\d{2}\s\d{2}:\d{2}:\d{2}',          # Syslog format
    # Add more patterns as needed
]

# Streamlit app
def main():
    st.title("Log Anomaly Detection")

    # Upload log file
    uploaded_file = st.file_uploader("Upload a log file (CSV or TXT)", type=["csv", "txt", "log"])

    if uploaded_file:
        # Read the entire file into a single string
        log_data = uploaded_file.getvalue().decode("utf-8")

        # Extract timestamps and log segments
        log_segments = extract_log_segments(log_data)

        # Flag anomalies
        anomalies, log_pca = flag_anomalies(log_segments)

        # Plotting the results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(log_pca[:, 0], log_pca[:, 1], label="Normal Segments")
        ax.scatter(log_pca[anomalies.index, 0], log_pca[anomalies.index, 1], color="red", label="Anomalies")
        ax.set_title("PCA of Log Segments")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        ax.legend()
        st.pyplot(fig)

        # Display anomalies
        if not anomalies.empty:
            st.write("Anomalous log segments:")
            for _, row in anomalies.iterrows():
                st.write(f"Timestamp: {row['timestamp']}")
                st.text(row['log'])  # Display the anomalous log segment
        else:
            st.write("No significant anomalies detected.")

# Function to detect timestamps and split log into segments
def extract_log_segments(log_data):
    timestamps = []
    for pattern in timestamp_patterns:
        timestamps.extend(re.findall(pattern, log_data))
    segments = re.split('|'.join(timestamp_patterns), log_data)[1:]  # Skip the first split
    return list(zip(timestamps, segments))

# Function to flag anomalies in log segments
def flag_anomalies(log_segments):
    df = pd.DataFrame(log_segments, columns=['timestamp', 'log'])
    vectorizer = TfidfVectorizer(stop_words="english")
    log_vectors = vectorizer.fit_transform(df['log'])
    pca = PCA(n_components=2)
    log_pca = pca.fit_transform(log_vectors.toarray())
    similarity_matrix = cosine_similarity(log_vectors)
    similarity_scores = np.mean(similarity_matrix, axis=1)
    df['similarity'] = similarity_scores
    anomaly_threshold = np.quantile(similarity_scores, 0.05)  # Adjust the quantile as needed
    anomalies = df[df['similarity'] < anomaly_threshold]
    return anomalies, log_pca

if __name__ == "__main__":
    main()
