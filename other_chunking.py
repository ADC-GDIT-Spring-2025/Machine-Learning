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
SUBJECT chunking
extracting entire subject line

NOTE does it make sense to extract entire subject line as one chunk bc it's relatively short / contextually concise?
"""
def subj_chunking():
    subj_chunk = df['subject_clean']
    subj_chunk.to_csv("chunks/subj_chunk.csv", index=True)



def main():
    to_chunking()
    b_cc_chunking()
    subj_chunking()



main()