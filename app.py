import streamlit as st
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

@st.cache_data()
def get_model():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    model = BertForTokenClassification.from_pretrained("Anna567/booking-conf-letters")
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
                 7: 'B-CHECK_IN_DATE_TIME',
                 8: 'I-CHECK_IN_DATE_TIME',
                 9: 'B-CHECK_OUT_DATE_TIME',
                 10: 'I-CHECK_OUT_DATE_TIME',
                 11: 'B-LOCATION',
                 12: 'I-LOCATION',
                 13: 'B-MAX_CAPACITY',
                 14: 'I-MAX_CAPACITY',
                 15: 'B-GUEST_NAME',
                 16: 'I-GUEST_NAME',
                 17: 'B-TOTAL_PRICE',
                 18: 'I-TOTAL_PRICE',
                 19: 'B-NUMBER_OF_GUESTS',
                 20: 'I-NUMBER_OF_GUESTS',
                 21: 'B-CANCELLATION_FEE',
                 22: 'I-CANCELLATION_FEE'}

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
        output[entity] = " ".join(tokens[start:end+1])
    
    return output

if user_input and button :    
    inputs = tokenizer(user_input.split(),
                        is_split_into_words=True, 
                        return_offsets_mapping=True, 
                        padding='max_length', 
                        truncation=True, 
                        max_length=512,
                        return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)
    flattened_predictions = torch.argmax(active_logits, axis=1)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
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
