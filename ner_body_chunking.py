import pandas as pd
from gliner import GLiNER
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/filtered.csv')
df = df[0:20]

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

labels = ["date", "location", "person", "action", "finance", "legal", "event", "product", "organization"]

# potential chunk sizes and overlap sizes for automated testing
chunk_sizes = [300, 400, 500, 600]
overlap_size = 100

results = []

"""
Extracting Named Entities with GLiNER
"""
def apply_gliner_labeling(row, chunk_size):
    text = str(row['content_clean'])

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    all_entities = []

    for chunk in chunks:
        try:
            entities = model.predict_entities(chunk, labels, threshold=0.5)
            all_entities.extend(entities)
        except Exception as e:
            print(f"Error processing chunk: {e}")
            
    entity_dict = {}
    for entity in all_entities:
        entity_type = entity['label']
        entity_text = entity['text']
        if entity_type not in entity_dict:
            entity_dict[entity_type] = []
        entity_dict[entity_type].append(entity_text)
    return entity_dict


"""
Smart Chunking that Avoids Entity Splitting
"""

def ner_chunking(text, entities, chunk_size, overlap_size):
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))

        # expand chunk to prevent cutting words in half
        # ensure we're not at the last chunk
        if end_idx < len(text):  
            # extend until reaching a non-alphanumeric char
            while end_idx < len(text) and text[end_idx].isalnum(): 
                end_idx += 1

        # making sure named entities aren't split across chunks
        for entity_list in entities.values():
            for entity in entity_list:
                entity_start = text.find(entity)
                entity_end = entity_start + len(entity)

                # if entity starts near the chunk end, extend the chunk to include it
                if entity_start < end_idx and entity_end > end_idx:
                    end_idx = entity_end

        # adding chunk and move index
        chunks.append([text[start_idx:end_idx]])
        start_idx += (chunk_size - overlap_size)

    return chunks

"""
Calculate TF-IDF Scores for Each Chunk
"""
def tf_idf_calc(chunks, top_n):
    tf_idf_results = []

    for chunk in chunks:
        if not chunk:
            tf_idf_results.append([])
            continue
        
        vectorizer = TfidfVectorizer(stop_words='english')

        try:
            tf_idf_matrix = vectorizer.fit_transform(chunk)
            feature_names = vectorizer.get_feature_names_out()

            # getting scores
            scores = zip(feature_names, tf_idf_matrix.toarray().flatten())
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

            top_words = [word for word, score in sorted_scores[:top_n]]
            tf_idf_results.append(top_words)
        except ValueError:
            # in case of an empty vocabulary
            tf_idf_results.append([])

    return tf_idf_results

"""
Evaluate Chunking Quality Using Entity Consistency Score
"""
def compute_entity_consistency(df):
    entity_scores = []

    for _, row in df.iterrows():
        entities = row['named_entities']
        chunks = row['ner_chunks']
        retained_count = 0
        counted_entities = set()

        # totalling entity occurrences
        entity_count = sum(len(v) for v in entities.values()) 
        # print(f"entity_count: {entity_count}")
        for v in entities.values():
            for ent in v:
                # print(f"ent: {ent}\n")
                if ent not in counted_entities:
                    for chunk in chunks:
                        # print(f"chunk: {chunk}\n")
                        if str(ent) in str(chunk):
                            # print("***** found ent: {ent} ***** \n")
                            retained_count += 1
                            counted_entities.add(ent)
                            break
        # print(f"retained_count: {retained_count}")

        # ECS: % of entities retained in chunks
        ecs = round((retained_count / entity_count), 2) if entity_count > 0 else 0
        entity_scores.append(ecs)

    df['entity_consistency_score'] = entity_scores
    return df

"""
Evaluate Overlap Between GLiNER Entities and Top TF-IDF Terms
"""
def compute_entity_tf_idf_overlap(df):
    overlap_scores = []

    for _, row in df.iterrows():
        entities = set([ent for entity_list in row['named_entities'].values() for ent in entity_list])
        tf_idf_terms = set([term for chunk in row['tf_idf'] for term in chunk])

        common_terms = entities.intersection(tf_idf_terms)
        overlap_score = round((len(common_terms) / len(entities)),2) if len(entities) > 0 else 0

        overlap_scores.append(overlap_score)

    df['entity_tf_idf_overlap_score'] = overlap_scores
    return df

"""
Applying NER-Based Chunking & TF-IDF Scoring to the Dataset
"""
def process_dataframe(df, chunk_size, overlap_size, top_n):

    print(f"Running chunk_size={chunk_size}, overlap={overlap_size}...")
    df['named_entities'] = df.apply(lambda row: apply_gliner_labeling(row, chunk_size), axis=1)
    df['ner_chunks'] = df.apply(lambda row: ner_chunking(str(row['content_clean']), row['named_entities'], chunk_size, overlap_size), axis=1)
    df['tf_idf'] = df.apply(lambda row: tf_idf_calc(row['ner_chunks'], top_n), axis=1)
    df = compute_entity_consistency(df)
    df = compute_entity_tf_idf_overlap(df)
    avg_ecs = df['entity_consistency_score'].mean()
    avg_overlap = df['entity_tf_idf_overlap_score'].mean()  
    results.append((chunk_size, avg_ecs, avg_overlap))
    df.to_csv(f"data/emails_chunk_{chunk_size}.csv", index=False)

# testing multiple chunk sizes and find the best one
for chunk_size in chunk_sizes:
    process_dataframe(df.copy(), chunk_size, overlap_size, top_n=5)

# finding the best chunk size based on avg ECS and overlap scores
results_df = pd.DataFrame(results, columns=["Chunk Size", "Avg ECS", "Avg TF-IDF Overlap"])
results_df.to_csv("data/chunking_evaluation_results.csv", index=False)

best_config = results_df.sort_values(by=["Avg ECS", "Avg TF-IDF Overlap"], ascending=False).iloc[0]
best_chunk_size = int(best_config["Chunk Size"])
print(f"\nBest Chunk Size: {best_chunk_size}")

# saving the best chunking res
best_df = pd.read_csv(f"data/emails_chunk_{best_chunk_size}.csv")
best_df.to_csv("data/best_chunking_strategy.csv", index=False)
print("\nBest chunking strategy saved in 'data/best_chunking_strategy.csv'")