from djl_python import Input, Output
import torch
import logging
import math
import os
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer


def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location)
   
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_location, 
        device_map="balanced_low_0", 
        #load_in_8bit=True
    )
    model.requires_grad_(False)
    model.eval()
    
    return model, tokenizer


model = None
tokenizer = None
generator = None


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    input_sentences = data["inputs"]
    params = data["parameters"]
    
    # preprocess
    input_ids = tokenizer(input_sentences, return_tensors="pt").input_ids
    # pass inputs with all kwargs in data
    if params is not None:
        outputs = model.generate(input_ids, **params)
    else:
        outputs = model.generate(input_ids)

    # postprocess the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    result = {"outputs": prediction}
    return Output().add_as_json(result)
