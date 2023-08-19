import os 
import pandas as pd
import argparse
from tqdm import tqdm
from utils import timeit, cos_single
import torch

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import SentenceTransformer


@timeit
def main(path, **kwargs):
    test_mode = kwargs.get('test', False)
    att_mode = kwargs.get('att_mode', False)
    if att_mode:
        print('In attribution setting')

    for subfolder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, subfolder)):
            continue
        subfolder_path = os.path.join(path, subfolder)

        # if subfolder_path == 'data/doc2dial/new_dataset/DEFAULT':
        #     print('Skipping DEFAULT')
        #     continue
        # if '_new' not in subfolder_path:
        #     # for new method experiments only
        #     continue

        # print('Processing: {}'.format(subfolder_path))
        
        if att_mode:
            eval_file = 'eval_one_att.csv'
        else:
            eval_file = 'eval.csv'
        eval_path = os.path.join(subfolder_path, eval_file)

        # print('eval_path: ', eval_path)
        # if os.path.exists(eval_path):
        #     print('eval.csv already exists')

        df = pd.read_csv(eval_path)
        extra_eval = pd.DataFrame(columns=['BLEU', 'ROUGHLS', 'BLEU_ANS', 'ROUGHLS_ANS'])
        chencherry = SmoothingFunction()
        #iterate df
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            true_answer = row['answer'] if not pd.isna(row['answer']) else ''
            model_answer = row['model_answer'] if not pd.isna(row['model_answer']) else ''
            true_ref_string = row['true_ref_str'] if not pd.isna(row['true_ref_str']) else ''
            if att_mode:
                pred_refs = row['attribution'] if not pd.isna(row['attribution']) else ''
            else:
                pred_refs = row['retrived_doc'] if not pd.isna(row['retrived_doc']) else ''

            BLEUScore = sentence_bleu(true_ref_string, pred_refs, smoothing_function=chencherry.method2)
            ROUGHLScore = sentence_bleu(true_ref_string, pred_refs, smoothing_function=chencherry.method4)

            #for answer
            BLEUScore_ans = sentence_bleu(true_answer, model_answer, smoothing_function=chencherry.method2)
            ROUGHLScore_ans = sentence_bleu(true_answer, model_answer, smoothing_function=chencherry.method4)

            extra_eval.loc[len(extra_eval)] = [BLEUScore, ROUGHLScore, BLEUScore_ans, ROUGHLScore_ans]

            if test_mode and index == 10:
                break
        
        #save to csv
        if test_mode:
            extra_eval.to_csv(os.path.join(subfolder_path, 'test_extra_eval.csv'), index=False)
            break

        if att_mode:
            extra_eval.to_csv(os.path.join(subfolder_path, 'extra_eval_att.csv'), index=False)
        else:
            extra_eval.to_csv(os.path.join(subfolder_path, 'extra_eval.csv'), index=False)

def compute_cos_sim(path, eval_file=None, **kwargs):
    att_mode = kwargs.get('att_mode', False)

    output_path = '/'.join(path.split('/')[:-1]) + '/cos_sim_att.csv'

    model_name = 'sentence-transformers/gtr-t5-base'
    model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    # if os.path.exists(path):
    #     print('path exists: ', path)

    chunk_size = kwargs.get('chunk_size', 10)

    total = 1000//chunk_size
    # progress_bar = tqdm(total=total, desc="Processing Chunks")

    # cos_sim = pd.DataFrame(columns=['cos_sim', 'cos_sim_ans'])

    # for chunk_df in pd.read_csv(path, chunksize=chunk_size):
    #     true_answer = chunk_df['answer'].fillna('')
    #     model_answer = chunk_df['model_answer'].fillna('')
    #     true_ref_string_list = chunk_df['true_ref_str'].fillna('')
    #     if att_mode:
    #         pred_refs_list = chunk_df['attribution'].fillna('')
    #     else:
    #         pred_refs_list = chunk_df['retrived_doc'].fillna('')


    #     cos_sim_ref = cos_single(model, true_ref_string_list, pred_refs_list)
    #     cos_sim_ans = cos_single(model, true_answer, model_answer)

    #     for i in range(len(cos_sim_ref)):
    #         cos_sim.loc[len(cos_sim)] = [cos_sim_ref[i], cos_sim_ans[i]]
    #     # cos_sim.loc[len(cos_sim)] = [cos_sim_ref, cos_sim_ans]

    #     progress_bar.update(total//100)
    #     # break
    
    # progress_bar.close()


    if att_mode:
        print('Saving to: ', output_path)
        # cos_sim.to_csv(output_path, index=False)
    else:
        pass
        # cos_sim.to_csv(path.replace('eval.csv', 'cos_sim.csv'), index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, help='path of target file')
    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('-a', '--att', action='store_true', help='in attribution seeting')
    args = parser.parse_args()

    target_path = args.target
    test_mode = args.test
    att_mode = args.att

    folders = []
    for subfolder in os.listdir(target_path):
        if not os.path.isdir(os.path.join(target_path, subfolder)):
            continue
        subfolder_path = os.path.join(target_path, subfolder)
        eval_file = 'eval_one_att.csv'
        eval_path = os.path.join(subfolder_path, eval_file)
        if os.path.exists(eval_path):
            folders.append(eval_path)

    # TODO: run attribution 
    # eval_one_att.csv
    if att_mode:
        # print('att mode')
        # main(target_path, att_mode=att_mode)

        for subfolder in folders:
            compute_cos_sim(subfolder, att_mode=att_mode)
    
    else:
        selected_folder = []
        # redo for new method experiments
        for folder in folders:
            if '_new' in folder:
                selected_folder.append(folder)
        
        main(target_path, test = test_mode)

        for subfolder in selected_folder:
            compute_cos_sim(subfolder)