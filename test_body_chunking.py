"""
This script tests the body_chunking evaluation methodology on a smaller dataset--data set consists
of just a couple of lines of text (in comparison to a whole csv file)

For the time being, alleviates complexities of dealing with how data is represented in dataframe
and resulting effects on how data is used with spacy
"""

import spacy
from spacy.training.example import Example
import random
from collections import deque
from sklearn.metrics import f1_score


# 1. tokenize and label text
def tokenize_and_label(text, nlp):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    token_labels = [token.ent_type_ if token.ent_type_ else '0' for token in doc]
    return tokens, token_labels

# 2. chunk tokens w/ chunk and overlap size
def chunk_text(tokens, chunk_size, overlap_size):
    chunks = []
    queue = deque(maxlen=chunk_size)

    for token in tokens:
        queue.append(token)
        if len(queue) == chunk_size:
            chunks.append(list(queue))
            if overlap_size > 0:
                for _ in range(chunk_size - overlap_size):
                    queue.popleft()

    return chunks

# 3. convert chunking data into spaCy format for training data
def format_training_data(chunks, text, nlp):
    training_data = []

    for chunk in chunks:
        doc = nlp(" ".join(chunk))
        entities = []

        for ent in doc.ents:
            start = doc.text.find(ent.text)
            end = start + len(ent.text)
            entities.append((start, end, ent.label_))
        training_data.append((doc.text, {"entities": entities}))

    return training_data

# 4. training model with spacy formated chunked data
# need help implementing logic for training model using spaCy
"""
def train_model(training_data, n_iter=10):
    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()

    for epoch in range(n_iter):
        random.shuffle(training_data)
        losses = {}

        for text, annotations in training_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

    nlp.to_disk("tuned_model")
    return nlp
"""


# 5. evaluate model on test data using F1 score
def evaluate_model(nlp, test_data):
    """
    true_labels = []
    pred_labels = []

    for text, annotations in test_data:
        doc = nlp(text)
        true_entities = annotations["entities"]
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]

        true_labels.extend([ent[2] for ent in true_entities])
        pred_labels.extend([ent[2] for ent in pred_entities])

    f1 = f1_score(true_labels, pred_labels, average="macro")
    return f1
    """


def main(train_text, test_text, chunk_size, overlap_size):
    nlp = spacy.load("en_core_web_sm")

    """
    tokens, token_labels = tokenize_and_label(train_text, nlp)
    chunks = chunk_text(tokens, chunk_size, overlap_size)
    training_data = format_training_data(chunks, train_text, nlp)
    trained_model = train_model(training_data)

    test_tokens, test_labels = tokenize_and_label(test_text, nlp)
    test_chunks = chunk_text(test_tokens, chunk_size, overlap_size)
    test_data = format_training_data(test_chunks, test_text, nlp)

    f1 = evaluate_model(trained_model, test_data)
    print("F1 Score: {}".format(f1))
    """


train_text = "Sally Hanson was born in California. She worked as a surf instruction in San Diego."
test_text = "Matt Johnson was born in New York. He worked as a waitress in Manhatten."
main(train_text, test_text, 3, 2)






