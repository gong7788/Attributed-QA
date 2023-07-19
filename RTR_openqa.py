import argparse
import torch
import os
from configparser import ConfigParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

import pandas as pd
import ast
from embedding import embedding
from dataloader import OpenQADataset
from QAModel import local_answer_model
from models.retriever import retrieve_openqa



def main(path, *args, **kwargs):
    # get args
    qa_model_name = kwargs['qa_model_name']
    batch_size = kwargs['batch_size']
    test_mode = kwargs['test_mode']
    output_path = kwargs['output_path']
    embedding_model = kwargs['embedding_model']
    new_method = kwargs['new_method']
    topk = kwargs['topk']

    if 'openqa' not in path:
        raise ValueError('Not supported dataset')
    # load data
    dataset = OpenQADataset(path)

    df = pd.DataFrame(columns=['question', 'true_answer', 'model_answer', 'true_ref', 'retrived_doc', 'retrieve_docs_idx', 'human_rating'])

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(device)
    if qa_model_name == 'Intel/fid_flan_t5_base_nq':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    else:
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    for batch in dataloader:
        # Perform your training/inference operations on the batch
        questions = batch['question']
        true_answers = batch['answer']
        true_refs = batch['passage']
        human_ratings = batch['human_rating']
        batch_paragraphs = batch['paragraphs']

        retrieve_docs, retrieve_docs_idx = retrieve_openqa(questions, batch_paragraphs, model=embedding_model, new_method=new_method, topk=topk)

        model_answers = local_answer_model(model, tokenizer, questions, retrieve_docs, device)

        for idx, answer in enumerate(model_answers):
            df.loc[len(df)] = [questions[idx], 
                                true_answers[idx], 
                                answer,
                                true_refs[idx],
                                retrieve_docs[idx],
                                retrieve_docs_idx[idx],
                                human_ratings[idx]]

    if not test_mode:
        df.to_csv(output_path+'model_answers.csv', index=False)
    else:
        print('Test mode, save to test.csv')
        df.to_csv('data/openqa/TEST/test.csv', index=False)


if __name__ == "__main__":
    print("Run!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='DEFAULT', help='load config settings')
    parser.add_argument('-t', '--test', action='store_true', help='run in test mode')
    
    args = parser.parse_args()
    setting = args.config

    config_object = ConfigParser()
    config_object.read("config_openqa.ini")
    userinfo = config_object[setting]
    test = args.test

    data_path = userinfo['data_path']
    embedding_model = userinfo['embedding_model']
    qa_model_name = userinfo['qa_model']
    new_method = userinfo.getboolean('new_method')
    topk = int(userinfo['topk'])

    output_path = 'data/openqa/qa_test.csv'
    if 't5-large' in qa_model_name:
        output_path = 'data/openqa/t5large/' + setting

    if not os.path.exists(qa_model_name) and not test:
        os.makedirs(qa_model_name)

    main(data_path, qa_model_name=qa_model_name, batch_size=2, test_mode=test, output_path=output_path, embedding_model=embedding_model, new_method=new_method, topk=topk)
