import pandas as pd
from helpers import gliner_labeling
from helpers import ner_chunking

df = pd.read_csv('data/filtered.csv')
chunk_size = 150
overlap_size = 3
tf_idf_num = 5
is_words_chunk = False

def body_chunking_strategy():
    if is_words_chunk:
        ner_chunking.df_chunking_word(df, chunk_size, overlap_size, tf_idf_num)
    else:
        ner_chunking.df_chunking(df, chunk_size, overlap_size, tf_idf_num)
    
    gliner_labeling.gliner_labeling(df)
    df_output = pd.read_csv('data/body_processing.csv')
    df_output.drop(['subject_clean', 'content_tokens', 'subject_tokens'], axis=1)
    df_output.to_csv('data/body_processing.csv')
    

body_chunking_strategy()

    
