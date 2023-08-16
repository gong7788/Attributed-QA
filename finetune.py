# finetune T5 base model for classification

import argparse
import glob
import os
import json
import time
import logging
import random
import re

import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from dataloader import doc2dialEvalDataset, ExtendedDoc2dialEvalDataset
from torch.utils.data import Subset
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
from evaluation import infer_autoais_batch, format_for_autoais_batch


def run(**kwargs):
    test = kwargs.get('test', False)
    if test:
        pass 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    model.to(device)

    batch_size = kwargs.get('batch_size', 4)
    num_epochs = kwargs.get('num_epochs', 5)
    
    csv_file = 'data/doc2dial/new_dataset/DEFAULT/DEFAULT_ModelAnswer.csv'
    label_file = 'data/doc2dial/human_label.csv'
    dataset = ExtendedDoc2dialEvalDataset(csv_file, label_file)

    # choose first 100 examples
    dataset = Subset(dataset, range(100))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #[ ] shuffle true or false?
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW optimizer

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
            for batch in dataloader:
                questions = batch['question']
                model_answer = batch['model_answer']
                retrived_docs = batch['retrived_doc']
                labels = batch['label']
                print('labels: ', labels)
                labels = tokenizer(labels,
                                padding=True,
                                return_tensors="pt").input_ids
                
                labels[labels == tokenizer.pad_token_id] = -100
                labels = torch.tensor(labels).to(device)
                print('labels: ', labels)

                example_list = format_for_autoais_batch(questions, model_answer, retrived_docs)

                optimizer.zero_grad()

                encoding = tokenizer(example_list, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True, 
                                      max_length=1024)
                encoding = {k: v.to(model.device) for k, v in encoding.items()}

                print('input_ids: ', encoding['input_ids'].shape)

                # outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'], max_new_tokens=512)
                outputs = model(encoding['input_ids'], attention_mask=encoding['attention_mask'], labels=labels)
                print('outputs: ', outputs.logits.shape)

                loss = outputs.loss
                print('loss: ', loss.item())

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")

    # save model
    torch.save(model.state_dict(), 'model/flant5_ft.pth')

if __name__ == "__main__":
    run(test=True)