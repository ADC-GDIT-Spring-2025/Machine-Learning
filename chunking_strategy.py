import pandas as pd
from collections import deque

df = pd.read_csv('data/filtered.csv')
# print(df.head())


"""
TO chunking
extracting only first email in "To" column
"""
def to_chunking(): 
    to_chunk = df['To'].str.split(', ').str[0]
    to_chunk.to_csv("chunks/to_chunk.csv", index=True)


"""
BCC / CC chunking
extracting all except for first email in "To" column

NOTE not sure how to distinguish between bcc and cc
"""
def b_cc_chunking():
    # accounts for rows where To has missing values--is pre-processed data supposed to look like this?
    b_cc_chunk = df['To'].fillna('')
    b_cc_chunk = b_cc_chunk.str.split(', ').apply(lambda x: ', '.join(x[1:]) if len(x) > 1 else '')
    b_cc_chunk.to_csv("chunks/b_cc_chunk.csv", index=True)


"""
FROM chunking
extracting entire from line

NOTE does it make sense to extract entire subject line as one chunk bc it's relatively short / contextually concise?
"""
def from_chunking():
    from_chunk = df['From']
    from_chunk.to_csv("chunks/from_chunk.csv", index=True)


"""
SUBJECT chunking
extracting entire subject line

NOTE does it make sense to extract entire subject line as one chunk bc it's relatively short / contextually concise?
"""
def subj_chunking():
    subj_chunk = df['subject_clean']
    subj_chunk.to_csv("chunks/subj_chunk.csv", index=True)


"""
BODY chunking
produces body chunks of various chunk sizes and sliding window overlaps, uses body_tokens from pre-processed data
"""
def body_chunking_helper(tokens, chunk_size, overlap):
    #for i in range(len(tokens)):
        #print(tokens[i])
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



def main():
    to_chunking()
    b_cc_chunking()
    from_chunking()
    subj_chunking()
    # test_chunking = ['the', 'cat', 'jumped', 'over', 'the', 'brown', 'fox', '!']
    # print(body_chunking_helper(test_chunking, 3, 2))
    body_chunking(10, 6)
    body_chunking(10, 5)
    body_chunking(5, 2)


main()
