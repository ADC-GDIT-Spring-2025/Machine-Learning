import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv('data/filtered.csv')


"""
calculating tf-idf scores
"""
def tf_idf_calc(chunks, top_n):
    tf_idf_results = []
    
    # Each chunk is already a list from your chunking function
    for chunk in chunks:
        # Convert chunk to string if it's a list with one string
        chunk_text = chunk[0] if isinstance(chunk, list) and len(chunk) == 1 else str(chunk)
        
        # Ensure there's actual content to analyze
        if not chunk_text or chunk_text.isspace():
            tf_idf_results.append([])
            continue
            
        vectorizer = TfidfVectorizer(stop_words=None)
        
        try:
            tf_idf_matrix = vectorizer.fit_transform([chunk_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores
            scores = zip(feature_names, tf_idf_matrix.toarray().flatten())
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            top_words = [word for word, score in sorted_scores[:top_n]]
            tf_idf_results.append(top_words)
        except ValueError:
            # Handle empty vocabulary
            print(f"Warning: Empty vocabulary for chunk: '{chunk_text[:50]}...'")
            tf_idf_results.append([])
            
    return tf_idf_results


"""
chunking body of text all at once
"""
def chunking(text, chunk_size, overlap_size):
    chunks = []
    start_idx = 0

    while start_idx < (len(text) - 1):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append([text[start_idx:end_idx]])
        start_idx += (chunk_size - overlap_size)
    return chunks

def df_chunking(chunk_size, overlap_size, top_n):
    df['ner_chunks'] = None
    df['tf_idf'] = None

    for index, row in df.iterrows():
        body_text = str(row['body_clean'])
        ner_chunk = chunking(body_text, chunk_size, overlap_size)
        tf_idf = tf_idf_calc(ner_chunk, top_n)
        df.at[index, 'ner_chunks'] = ner_chunk
        df.at[index, 'tf_idf'] = tf_idf



# need to debug!!
# extra brackets, chunks are not one single string
"""
chunking body of text by words
"""
def chunking_word(text, chunk_size, overlap_size):
    chunks = []
    start_idx = 0

    while start_idx < (len(text) - 1):
        end_idx = min(start_idx + chunk_size, len(text))
        chunk = [''.join(text[start_idx:end_idx])]
        chunks.append(chunk)
        start_idx += (chunk_size - overlap_size)
    return chunks

def df_chunking_word(chunk_size, overlap_size, top_n):
    df['ner_chunks_word'] = None

    for index, row in df.iterrows():
        body_text = str(row['body_tokens']).split()
        ner_chunk = chunking_word(body_text, chunk_size, overlap_size)
        # tf_idf = tf_idf(ner_chunk, top_n)
        df.at[index, 'ner_chunks_word'] = ner_chunk
        # df.at[index, 'tf_idf'] = tf_idf


df_chunking(150, 3, 5)
# df_chunking_word(15, 3, 5)
df.to_csv('data/emails_with_ner_chunking.csv', index=False)
