import pandas as pd
import spacy
from collections import deque

nlp = spacy.load("en_core_web_sm")
df = pd.read_csv('data/filtered.csv')
df_rows = df.shape[0]
train_rows = 1

df_train = df.iloc[0:train_rows]
df_train.to_csv("data/chunk_train.csv", index=True)
df_test = df.iloc[train_rows:df_rows]
df_test.to_csv("data/chunk_test.csv", index=True)


"""
1. body chunking

produces body chunks of various chunk sizes and sliding window overlaps, uses body_tokens from pre-processed data
"""
def body_chunking_helper(tokens, chunk_size, overlap):
    chunks = []
    queue = deque(maxlen=chunk_size)

    for token in tokens:
        queue.append(token)
        # print(token)

        if len(queue) == chunk_size:
            chunks.append(list(queue))
            if overlap > 0:
                for _ in range(chunk_size - overlap):
                    queue.popleft()

    return chunks

def body_chunking(chunk_size, overlap):
    # assuming body_tokens column contains list of tokenized words from pre-processing
    # eval(x) ensures that the tokens are stored as list of strings, necessary because pandas stores the list as one whole string
    body_chunk = df['body_tokens'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    # print(type(df['body_tokens']))
    # print(body_chunk)
    body_chunk = body_chunk.apply(lambda body_tokens: body_chunking_helper(body_tokens, chunk_size, overlap))
    file_name = f"chunks/body_chunks/bd_{chunk_size}_{overlap}.csv"
    body_chunk.to_csv(file_name, index=True)


"""
2. spaCy data format

* needs debugging *
convert chunking csv data into spaCy's training format
"""
def prepare_training_data(body_chunk_df):
    training_data = []

    for _, row in body_chunk_df.iterrows():
        text = row['body_clean']
        tokens = row['body_tokens'].split(',')
        labels = row['body_tokens_labeled'].split(',')

        entities = []
        s_idx = 0

        for token, label in zip(tokens, labels):
            token_start = text.find(token, s_idx)
            token_end = token_start + len(token)
            print(label)

            if label != '0':
                entities.append((token_start, token_end, label))

            s_idx = token_end + 1

        training_data.append((text, {"entities": entities}))

    # print(training_data)
    return training_data


"""
3. training

train model based on processed training data
"""
def train_model(training_data, n_iter):
    return


"""
4. testing & metrics

run model on test data, compare to ground truth labels, use F1 score to find metrics
"""
def test_model(testing_data):
    return



def main():
    # test_chunking = ['the', 'cat', 'jumped', 'over', 'the', 'brown', 'fox', '!']
    # print(body_chunking_helper(test_chunking, 3, 2))
    # body_chunking(10, 6)
    # body_chunking(10, 5)
    # body_chunking(5, 2)
    prepare_training_data(df_train)

main()