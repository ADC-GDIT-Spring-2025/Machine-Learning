import pandas as pd
from gliner import GLiNER

df = pd.read_csv('data/filtered.csv')
df_test = df.iloc[0:1]
df_test['body_clean'].astype("string")


# other labels: legal terms, department/team, location, financial data
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
entity_types = [
    "action_item", "person", "meeting_time"
]



examples = [
    # Action item examples with context
    {
        "text": "Can you please prepare the slides for tomorrow's meeting?",
        "entities": [{"text": "prepare", "label": "action_item"}]
    },
    {
        "text": "We need to schedule a follow-up discussion next week.",
        "entities": [{"text": "schedule", "label": "action_item"}]
    },
    {
        "text": "Please send the report to the team by EOD.",
        "entities": [{"text": "send", "label": "action_item"}]
    },
    {
        "text": "Make sure to review the documentation before submitting.",
        "entities": [{"text": "review", "label": "action_item"}]
    },
    
    # Person examples
    {
        "text": "I spoke with Philip about the project timeline.",
        "entities": [{"text": "Philip", "label": "person"}]
    },
    
    # Meeting time examples
    {
        "text": "Let's meet Monday at 10:00 AM to discuss this further.",
        "entities": [{"text": "Monday at 10:00 AM", "label": "meeting_time"}]
    }
]

"""
creating examples 
"""
sample_texts = df_test['body_clean'].iloc[:5].tolist()

# Create examples from your actual data
custom_examples = []
for text in sample_texts:
    # Focus on sentences with action verbs
    sentences = text.split('.')
    for sentence in sentences:
        if any(verb in sentence.lower() for verb in ["prepare", "schedule", "send", "review", "complete", "holding"]):
            custom_examples.append({
                "text": sentence.strip(),
                "entities": [{"text": sentence.strip(), "label": "action_item"}]
            })

# Use these examples if any were found
if custom_examples:
    print(f"Created {len(custom_examples)} custom examples from your data")
    examples.extend(custom_examples)

"""
examples = {
    "action_item": ["prepare", "hold", "schedule", "send", "review"],
    "person": ['philip', 'george fichards', 'thomas richards', 'amy'],
    "meeting_time": ["3pm", "tomorrow", "Monday", "10:00 AM"]
}
"""

for entity_type in entity_types:
    df[f'{entity_type}_extracted'] = [[] for _ in range(len(df))]

for index, row in df_test.iterrows():
    entities = model.predict_entities(row['body_clean'], ["person", "meeting_time"], threshold=0.5)
    action_entities = model.predict_entities(row['body_clean'], ["action_item"], threshold=0.05)
    all_entities = entities + action_entities

    for entity in all_entities:
        df.at[index, f'{entity["label"]}_extracted'].append(entity["text"])
        print(entity["text"], "=>", entity["label"])


df.to_csv('data/emails_with_gliner_labeling.csv', index=False)