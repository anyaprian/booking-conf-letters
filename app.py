import streamlit as st
from transformers import BertTokenizerFast, DistilBertForTokenClassification
import torch

def get_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = DistilBertForTokenClassification.from_pretrained("Anna567/booking-conf-letters")
    return tokenizer,model


tokenizer, model = get_model()

user_input = st.text_area('Text')
button = st.button("Find Entities")

ids_to_labels = {0: 'O',
                 1: 'B-HOTEL_NAME',
                 2: 'I-HOTEL_NAME',
                 3: 'B-PROPERTY_PHONE',
                 4: 'I-PROPERTY_PHONE',
                 5: 'B-PIN_CODE',
                 6: 'B-RESERVATION_NUMBER',
                 7: 'B-CHECK_OUT_TIME',
                 8: 'I-CHECK_OUT_TIME',
                 9: 'B-CHECK_IN_DATE',
                 10: 'I-CHECK_IN_DATE',
                 11: 'B-CHECK_IN_TIME',
                 12: 'I-CHECK_IN_TIME',
                 13: 'B-CHECK_OUT_DATE',
                 14: 'I-CHECK_OUT_DATE',
                 15: 'B-LOCATION',
                 16: 'I-LOCATION',
                 17: 'B-MAX_CAPACITY',
                 18: 'I-MAX_CAPACITY',
                 19: 'B-GUEST_NAME',
                 20: 'I-GUEST_NAME',
                 21: 'B-TOTAL_PRICE',
                 22: 'I-TOTAL_PRICE',
                 23: 'B-NUMBER_OF_GUESTS',
                 24: 'I-NUMBER_OF_GUESTS',
                 25: 'B-CANCELLATION_FEE',
                 26: 'I-CANCELLATION_FEE',
                 27: 'B-CANCELATION_FEE',
                 28: 'I-CANCELATION_FEE'}

def bio_to_dict(tokens, tags):
    """Convert BIO tagged text to a dictionary of entities and their spans"""
    entities = {}
    entity = None
    start = None
    for i in range(len(tokens)):
        if tags[i][0] == "B":
            if entity is not None:
                entities[entity] = (start, i-1)
            entity = tags[i][2:]
            start = i
        elif tags[i][0] == "I":
            if entity is None:
                entity = tags[i][2:]
                start = i
        else:
            if entity is not None:
                entities[entity] = (start, i-1)
                entity = None
    if entity is not None:
        entities[entity] = (start, len(tokens)-1)
    
    # Create the output dictionary
    output = {}
    for entity, span in entities.items():
        start, end = span
        output[entity] = " ".join(tokens[start:end+1]).strip('.').strip('!')
    return output

if user_input and button :    
    inputs = tokenizer(user_input.split(),
                        is_split_into_words=True, 
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=256,
                        return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    tags = []
    for token_pred, mapping in zip(wp_preds, inputs["offset_mapping"].squeeze().tolist()):
      if mapping[0] == 0 and mapping[1] != 0:
        tags.append(token_pred[1])
      else:
        continue
    tokens = user_input.split()
    st.write(bio_to_dict(tokens, tags))
    del (
            model,
            tokenizer,
            logits,
            outputs
        )
