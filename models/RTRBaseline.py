import json
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import embedding
import evaluation
import QAModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch.multiprocessing as mp


def model_parallel_t5(rank, world_size):
    #[ ] for evaluation
    print(f"Running DDP with model parallel t5 on rank {rank}.")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

    # setup mp_model and devices for this process
    device_map = {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12],
             1: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}

    t5_model.parallelize(device_map)
    # ddp_mp_model = DDP(t5_model)

    print('ddp model loaded')

    #[ ] write inputs

    input_ids = tokenizer(inputs, return_tensors="pt").input_ids
    
    input_ids = input_ids.to('cuda')

    outputs = t5_model.generate(input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    t5_model.deparallelize()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)