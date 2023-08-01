import datetime
import pandas as pd
import json

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import torch.multiprocessing as mp
import os
import time
import argparse
from torch.utils.data import DataLoader
from dataloader import doc2dialEvalDataset
from configparser import ConfigParser

import evaluation
from utils import timeit

# failed_doc_ids = [645,1448,1523,1573,8159]
AUTOAIS = "google/t5_xxl_true_nli_mixture"


def read_data(data_paths):
    """Reads the data from the given paths."""
    dfs = []
    for path in data_paths:
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs)

def load_doc_file(file_path):
    """
    Load doc2dial_doc.json file
    """
    #check if file is json
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            doc2dial_doc = json.load(f)
        return doc2dial_doc
    else:
        raise ValueError('File is not json')
    
def eval(df, qa_df, doc2dial_doc, test=False, test_num=1, eval_id=None):

    hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
    hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS)
    print('model loaded')

    total_autoais = 0
    cnt = 0
    null_answer_cnt = 0
    eval_df = pd.DataFrame(columns=['id', 'autoais', 'f1', 'att_f1'])

    for index, row in df.iterrows():
        # qa_pair = qa_df.iloc[cnt]
        qa_pair = qa_df[qa_df['question'] == df.iloc[cnt]['question']].iloc[0]

        if row['question'] != qa_pair['question']:
            # should never happen
            print('question not match')
            # cnt += 1
            # continue

        model_answer = row['answer']

        # if str(model_answer) == 'nan':
        #     print('answer is NaN')
        #     cnt += 1
        #     continue

        retrived_doc = row['passage(context)']
        retrived_doc = retrived_doc.replace('/n', '\n')
        # print('retrived_doc: ', retrived_doc)

        sp_list = evaluation.get_ref(qa_pair, doc2dial_doc)
        ref_list = [item['text_sp'] for item in sp_list]
        true_ref = '\n'.join(ref_list)
        # print('true_ref: ', true_ref)

        #autoais
        example = {}
        example['question'] = row['question']
        example['answer'] = model_answer 
        example['passage'] = retrived_doc

        # measure time cost 
        autoais = evaluation.infer_autoais(example, hf_tokenizer, hf_model)
        total_autoais += autoais == 'Y' 
        #f1 score
        # print('model_answer: ', model_answer)
        try:
            f1 = evaluation.compute_f1(model_answer, qa_pair['answer'])
        except:
            print('cnt: ', cnt)
            null_answer_cnt += 1
        #attribution_f1 score (retrieved answer, true answer)
        att_f1 = evaluation.compute_f1(retrived_doc, true_ref)

        eval_df.loc[len(eval_df)] = [cnt, autoais, f1, att_f1]

        if test and cnt == test_num:
            break
        #set max number of evaluation
        elif cnt == 5000:
            break
        #save checkpoint
        elif not test and cnt % 1000 == 0 and cnt != 0:
            checkpoint_dir = 'data/doc2dial/eval_{}_cp'.format(eval_id)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint = os.path.join(checkpoint_dir, 'eval_{}_cp{}.csv'.format(eval_id, cnt))
            eval_df.to_csv(checkpoint, index=False)
            print('checkpoint saved at: ', cnt)
        elif not test and cnt % 1000 == 0 and cnt != 0:
            print('Running at: ', cnt)
        cnt += 1

    avg_autoais = total_autoais / cnt
    print('AUTOAIS: {}, f1: {}, att f1: {} '.format(avg_autoais, eval_df['f1'].mean(), eval_df['att_f1'].mean()))
    print('null answer cnt: ', null_answer_cnt)

    if test:
        output_path = 'data/doc2dial/eval_test.csv'
    else:
        output_path = 'data/doc2dial/eval_{}.csv'.format(eval_id)
    
    eval_df.to_csv(output_path, index=False)

@timeit
def infer_autoais(path, output_path, batch_size, *args, **kwargs):
    # check path form
    if not path.endswith('_withModelAnswer.csv'):
        raise ValueError('File path is not correct')
    if not os.path.exists(path):
        raise ValueError('File path does not exist')
    
    cnt = 0
    if 'top5' in output_path and cnt == 0:
        print('check point will work')

    dataset = doc2dialEvalDataset(path)
    print('size: ', len(dataset))
    # inference task, so shuffle is False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    df = pd.DataFrame(columns=['question', 'answer', 'model_answer', 'true_ref_str', 'retrived_doc', 'answer_f1', 'answer_prec', 'answer_recall', 'autoais_retrevied(model_answer)', 'att_f1', 'att_prec', 'att_recall', 'autoais_true_answer', 'ref_range'])

    tokenizer = T5Tokenizer.from_pretrained(AUTOAIS, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(AUTOAIS)
    print('model loaded')
    # AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS, device_map="auto")

    # first_batch = next(iter(dataloader))
    # questions = first_batch['question']
    # answers = first_batch['answer']
    # model_answer = first_batch['model_answer']
    # retrived_docs = first_batch['retrived_doc']
    # true_refs = first_batch['ref_string']
    # true_ref_range = first_batch['true_ref_position']
    # retrieved_doc_range = first_batch['retrieved_doc_position']

    for batch in dataloader:
        cnt += len(batch)
    #     # question,answer,model_answer,ref,retrived_doc,doc_id
        questions = batch['question']
        answers = batch['answer']
        model_answer = batch['model_answer']
        retrived_docs = batch['retrived_doc']
        true_refs = batch['ref_string']
        true_ref_range = batch['true_ref_position']
        # retrieved_doc_range = batch['retrieved_doc_position']

        # answer f1 
        ans_f1 = []
        ans_prec = []
        ans_recall = []
        for i in range(len(questions)):
            try:
                f1, prec, recall = evaluation.compute_f1(model_answer[i], answers[i], return_prec_recall=True)
                ans_f1.append(f1)
                ans_prec.append(prec)
                ans_recall.append(recall)
            except:
                print('cnt: ', i)
                ans_f1.append(0)
                ans_prec.append(0)
                ans_recall.append(0)
        # attribution f1
        att_f1 = []
        att_prec = []
        att_recall = []
        for i in range(len(questions)):
            try:
                f1, prec, recall = evaluation.compute_f1(retrived_docs[i], true_refs[i], return_prec_recall=True)
                att_f1.append(f1)
                att_prec.append(prec)
                att_recall.append(recall)
            except:
                print('cnt: ', i)
                att_f1.append(0)
                att_prec.append(0)
                att_recall.append(0)

        autoais_retrived = evaluation.infer_autoais_batch(questions, model_answer, retrived_docs, tokenizer, model)

        # [x] another autoais for true refs -> need a function to get true refs ??? 
        autoais_true_answer = evaluation.infer_autoais_batch(questions, answers, retrived_docs, tokenizer, model)

        for i in range(len(questions)):
            df.loc[len(df)] = [questions[i], 
                                answers[i], 
                                model_answer[i],
                                true_refs[i], 
                                retrived_docs[i],
                                ans_f1[i],
                                ans_prec[i],
                                ans_recall[i],
                                autoais_retrived[i],
                                att_f1[i],
                                att_prec[i],
                                att_recall[i],
                                autoais_true_answer[i],
                                true_ref_range[i]]
        
        #save checkpoint
        if 'top5' in output_path and cnt % 200 == 0 and cnt != 0:
            checkpoint = output_path.replace('eval', 'checkpoint')
            df.to_csv(checkpoint, index=False)
            print('checkpoint saved at: ', cnt)

    if test_mode:
        output_path = 'data/doc2dial/eval_test.csv'
        print('output save in : ', output_path)
    else:
        df.to_csv(output_path, index=False)


def run_model_parallel(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='DEFAULT', help='load config settings')
    parser.add_argument('--path', type=str, help='path to model folder')
    parser.add_argument('-t', '--test', action='store_true', help='run in test mode')

    args = parser.parse_args()
    setting = args.config
    print('setting: ', setting)
    test_mode = args.test

    config_object = ConfigParser()
    config_object.read("config1.ini")
    userinfo = config_object[setting]
    path = userinfo["path"]

    if test_mode:
        print('Test mode')
        file_path = 'data/doc2dial/TEST/test_withModelAnswer.csv'
        output_path = 'data/doc2dial/TEST/eval_CPU.csv'

        infer_autoais(file_path, output_path, batch_size=1)

    else:
        print('target_folder: ', path)

        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            file_name = subfolder + '_withModelAnswer.csv'
            eval_file = 'eval.csv'
            file_path = os.path.join(subfolder_path, file_name)
            eval_path = os.path.join(subfolder_path, eval_file)
            # Check if the file exists
            if not os.path.exists(file_path):
                print('File does not exist: ', file_path)
            elif os.path.exists(eval_path):
                print('Eval file exists: ', eval_path)
            else:
                if setting == 'DEFAULT' and subfolder == 'doc2dial_1000_top1':
                    print('Skip: ', subfolder)
                    continue
                if setting == 'DEFAULT' and subfolder == 'doc2dial_500_top5':
                    print('Skip: ', subfolder)
                    continue
                if setting == 'DEFAULT' and subfolder == 'DEFAULT':
                    print('Skip: ', subfolder)
                    continue
                # if setting == 'fid' and subfolder == 'fid_250_top1_new':
                #     print('Skip: ', subfolder)
                #     continue
                # if setting == 'fid' and subfolder == 'fid_500_top1_new':
                #     print('Skip: ', subfolder)
                #     continue
                # if setting == 'fid' and subfolder == 'fid_500_top1':
                #     print('Skip: ', subfolder)
                #     continue
                if 'top5' in subfolder:
                    print('Skip: ', subfolder)
                    continue

                output_path = os.path.join(subfolder_path, 'eval.csv')
                if test_mode:
                    print('Output path should be: ', output_path)
                # infer_autoais(file_path, subfolder, batch_size=8)
                else:
                    # record current time 
                    print('Subfolder: ', subfolder)
                    infer_autoais(file_path, output_path, batch_size=8)


  

    # data_paths = ['data/doc2dial/result_4.csv']
    # doc_file_path = 'data/doc2dial/doc2dial_doc.json'
    # qa_file_path = 'data/doc2dial/doc2dial_qa_train.csv'
    # max_num = 5000

    # df = read_data(data_paths)
    # df = df.dropna(subset=['answer'])
    # qa_df = pd.read_csv(qa_file_path)
    # doc2dial_doc = load_doc_file(doc_file_path)

    # eval(df, qa_df, doc2dial_doc, test=True, test_num=1, eval_id=4)


    # n_gpus = torch.cuda.device_count()
    # if n_gpus < 2:
    #     print(f"Requires at least 2 GPUs to run, but got {n_gpus}.")
    # else:
    #     # run_demo(demo_basic, 2)
    #     world_size = n_gpus//2
    #     run_model_parallel(demo_model_parallel_t5, world_size)