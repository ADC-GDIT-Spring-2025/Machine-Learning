import pandas as pd
import warnings
from gliner import GLiNER

df = pd.read_csv('data/filtered.csv')
test_df = df[0:5]

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

labels = ["date", "location", "person", "action", "finance", "legal"]

def apply_gliner_labeling(row):
    print(f"****** labeling row {row.name} *****")
    text = str(row['content_clean'])

    chunk_size = 300
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    all_entities = []
    for i, chunk in enumerate(chunks):
        # print(f"chunk {i}: {chunk} \t {len(chunk)}\n")
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


def gliner_labeling(df):
    print("***** labeling with gliner *****")
    entity_dicts = df.apply(apply_gliner_labeling, axis=1)
    entities_df = pd.DataFrame(entity_dicts.tolist())
    pd.concat([df, entities_df], axis=1).to_csv('data/body_processing.csv', index=False)  


# entity_dicts = df.apply(apply_gliner_labeling, axis=1)
# entities_df = pd.DataFrame(entity_dicts.tolist())
# pd.concat([df, entities_df], axis=1).to_csv('data/emails_with_gliner_labeling.csv', index=False)