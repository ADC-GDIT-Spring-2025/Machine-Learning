import pandas as pd
from sklearn.feature_extraction.text import TfidVectorizer
df = pd.read_csv('data/filtered.csv')

def chunking(text, chunk_size, overlap_size):
    chunks = []
    start_idx = 0

    while start_idx < (len(text) - 1):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append([text[start_idx:end_idx]])
        start_idx += (chunk_size - overlap_size)
    return chunks

def tf_idf(chunks, top_n):
    # add tf-idf analysis for each chunk
    tf_idf = []

    for index, row in df.iterrows():
        chunks = row['ner_chunks']
        tf_idf_row = []

        vectorizer = TfidfVectorizer()

        for chunk in chunks:
            tf_idf_matrix = vectorizer.fit_transform([chunk])
            feature_names = vectorizer.get_feature_names_out()

            scores = zip(feature_names, tf_idf_matrix.toarray().flatten())
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            top_words = [word for word, score in sorted_scores[:top_n]]
            tf_idf_row.append(top_words)

        tf_idf.append(tf_idf)

    return tf_idf


def df_chunking(chunk_size, overlap_size, top_n):
    df['ner_chunks'] = None

    for index, row in df.iterrows():
        body_text = str(row['body_clean'])
        ner_chunk = chunking(body_text, chunk_size, overlap_size)
        tf_idf = tf_idf(ner_chunk, top_n)
        df.at[index, 'ner_chunks'] = ner_chunk
        df.at[index, 'tf_idf'] = tf_idf



df_chunking(15, 3, 5)
df.to_csv('data/emails_with_ner_chunking.csv', index=False)
