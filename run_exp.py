import argparse
import logging
import torch
import os
import pandas as pd
import random
from configparser import ConfigParser

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

from dataloader import doc2dialDataset
from models.retriever import retrieve_only
from QAModel import local_answer_model
from utils import timeit
from tqdm import tqdm


@timeit
def run_RTR_Model(data, qa_model_name, batch_size, output_path, test_mode=False, **kwargs):
    #csv_file = 'data/doc2dial/qa_train_dmv.csv'
    use_cuda = kwargs.get('use_cuda', False)
    split_num = kwargs.get('split_num', None)
    flag = True

    if 'doc2dial' in data_path:
        dataset = doc2dialDataset(data)
        print('size of dataset: {}'.format(len(dataset)))
    if 'openqa' in data_path:
        raise ValueError('Not this dataset')
    
    # output_path = data_path.replace('withRefs', 'withModelAnswer')
    df = pd.DataFrame(columns=['question', 'answer', 'model_answer', 'ref', 'retrived_doc', 'doc_id', 'dial_id'])
    # [x] pending question,answer,ref,passage(context),doc_id

    print('Info: qa_model_name: {} , batch_size: {}, size: {}'.format(qa_model_name, batch_size, len(dataset)))
    cnt = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

    model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name).to(device)
    if qa_model_name == 'Intel/fid_flan_t5_base_nq':
        tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    else:
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

    for batch in tqdm(dataloader, desc='Batch', total=len(dataloader)):
        # Perform your training/inference operations on the batch
        questions = batch['question']
        answers = batch['answer']
        refs = batch['ref']
        retrieve_docs = batch['retrived_doc']
        doc_ids = batch['doc_id']
        dial_ids = batch['dial_id']
        cnt += len(questions)

        model_answers = local_answer_model(model, tokenizer, questions, retrieve_docs, device)

        for idx, answer in enumerate(model_answers):
            df.loc[len(df)] = [questions[idx], 
                                answers[idx], 
                                answer,
                                refs[idx], 
                                retrieve_docs[idx],
                                doc_ids[idx],
                                dial_ids[idx]]
        

        if cnt == 512 and '1000' in output_path and 'new' in output_path:
            print("save point at index: {}".format(cnt))
            checkpoint_path = '/'.join(output_path.split('/')[:-1]) + '/checkpoint.csv'
            df.to_csv(checkpoint_path, index=False)
            
    if not test_mode:
        df.to_csv(output_path, index=False)
    else:
        print('Test mode, save to test.csv')
        df.to_csv('data/doc2dial/TEST/new_test.csv', index=False)


if __name__ == "__main__":
    print("Running run_exp.py...")
    logging.basicConfig(level=logging.WARNING)
    logging.info("Start Running")

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--which', type=str, default='config.ini', help='which config settings should be load' )
    parser.add_argument('--config', type=str, default='DEFAULT', help='load config settings')
    parser.add_argument('-t', '--test', action='store_true', help='run in test mode')
    parser.add_argument('-s', '--save_dir', type=str, default=None, help='save directory'
    )
    parser.add_argument('-f', '--using_fid', action='store_true', help='using fid model')
    parser.add_argument('-r', '--random', action='store_true', help='randomly select attribution')
    
    args = parser.parse_args()
    which = args.which
    setting = args.config
    rand = args.random
    if rand:
        which = 'random.ini'
        print('Randomly select attribution')

    config_object = ConfigParser()
    config_object.read(which)
    userinfo = config_object[setting]
    test = args.test
    using_fid = args.using_fid

    logging.info('Running experiment with config: {}'.format(setting))

    if which == 'config2.ini' or which == 'random.ini':
        data_path = 'data/doc2dial/new_dataset/train_set_norandom.csv'
    else:
        data_path = userinfo['data_path']
    
    chunk_size = int(userinfo['chunk_size'])
    chunk_overlap = int(userinfo['chunk_overlap'])
    embedding_model = userinfo['embedding_model']
    qa_model_name = userinfo['qa_model'] if not using_fid else 'Intel/fid_flan_t5_base_nq'
    new_method = userinfo['new_method']
    topk = int(userinfo['topk'])

    directory = None
    if which == 'config.ini':
        if 't5base' in setting and 'fid' not in setting:
            directory = 'data/doc2dial/t5base/' + setting
        elif 't5small' in setting and 'fid' not in setting:
            directory = 'data/doc2dial/t5small/' + setting
        elif 'fid' in setting:
            directory = 'data/doc2dial/fid/' + setting
        elif data_path == 'data/doc2dial/qa_test_dmv.csv':
            directory = 'data/doc2dial/doc2dial_testset/' + setting
        elif 'doc2dial' in data_path:
            directory = 'data/doc2dial/' + setting
        elif 'openqa' in data_path:
            directory = 'data/openqa/' + setting
        else:
            raise ValueError('dataset should either doc2dial or openqa')
    elif which == 'config2.ini' and not args.save_dir:
        directory = 'data/doc2dial/new_dataset/' + setting
    elif args.save_dir:
        directory = args.save_dir + '/' + setting
    elif which == 'random.ini':
        directory = 'data/doc2dial/random/' + setting

    if not os.path.exists(directory):
        os.mkdir(directory)
    
    # if file exists
    # save_path = directory + '/'+ setting +'_withRefs.csv'
    # output_path = save_path.replace('withRefs', 'withModelAnswer')
    output_path = directory + '/'+ setting +'_xModelAnswer.csv'
    
    # if os.path.exists(output_path):
    #     print('file exists, skip')
    # else:
    if new_method == 'True':
        print('x', output_path)
        print('Info: seting: {}, test_mode: {}, chunk_size: {}, chunk_overlap: {}, qa_model: {}, embedding_model: {}, new_method: {}, topk: {}'.format(setting, args.test, chunk_size, chunk_overlap, qa_model_name, embedding_model, new_method, topk))
        print('Running retriever')
        result_df = retrieve_only(data_path, 
                    cs=chunk_size, 
                    c_overlap=chunk_overlap, 
                    save_dir=None,
                    new_method=True,
                    embedding_model=embedding_model,
                    topk=topk,
                    test_mode=test,
                    random=rand)
            # print('Created file with reference')
        print('Running QA model')
        run_RTR_Model(result_df, qa_model_name, batch_size=16, output_path=output_path, test_mode=test, use_cuda=True)