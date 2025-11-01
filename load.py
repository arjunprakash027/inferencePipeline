

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from inferencePipeline import loadPipeline

if __name__ == '__main__':

    pipeline = loadPipeline()

    questions = [{'questionID': 123, 'question': 'what is the capital of Ireland?'}, 
                 {'questionID': 456, 'question': 'what is the capital of Italy?'}]

    answers = pipeline(questions)

    print(answers)