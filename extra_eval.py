import os 
import pandas as pd
import argparse
from tqdm import tqdm
from utils import timeit, cos_single

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction


@timeit
def main(path, **kwargs):
    test_mode = kwargs.get('test', False)

    for subfolder in os.listdir(path):
        if not os.path.isdir(os.path.join(path, subfolder)):
            continue
        subfolder_path = os.path.join(path, subfolder)

        if subfolder_path == 'data/doc2dial/new_dataset/DEFAULT':
            print('Skipping DEFAULT')
            continue

        print('Processing: {}'.format(subfolder_path))
        
        eval_file = 'eval.csv'
        eval_path = os.path.join(subfolder_path, eval_file)
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
        
        extra_eval.to_csv(os.path.join(subfolder_path, 'extra_eval.csv'), index=False)

def compute_cos_sim(path, eval_file=None, **kwargs):
    if os.path.exists(path):
        print('path exists: ', path)

    chunk_size = kwargs.get('chunk_size', 10)

    total = 1000//chunk_size
    progress_bar = tqdm(total=total, desc="Processing Chunks")

    cos_sim = pd.DataFrame(columns=['cos_sim', 'cos_sim_ans'])

    for chunk_df in pd.read_csv(path, chunksize=chunk_size):
        true_answer = chunk_df['answer'].fillna('')
        model_answer = chunk_df['model_answer'].fillna('')
        true_ref_string_list = chunk_df['true_ref_str'].fillna('')
        pred_refs_list = chunk_df['retrived_doc'].fillna('')

        cos_sim_ref = cos_single(true_ref_string_list, pred_refs_list)
        cos_sim_ans = cos_single(true_answer, model_answer)

        for i in range(len(cos_sim_ref)):
            cos_sim.loc[len(cos_sim)] = [cos_sim_ref[i], cos_sim_ans[i]]
        # cos_sim.loc[len(cos_sim)] = [cos_sim_ref, cos_sim_ans]

        progress_bar.update(total//100)
        # break
    
    progress_bar.close()
    cos_sim.to_csv(path.replace('eval.csv', 'cos_sim.csv'), index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, help='path of target file')
    parser.add_argument('--test', action='store_true', help='test mode')
    args = parser.parse_args()

    target_path = args.target
    test_mode = args.test

    folders = []
    for subfolder in os.listdir(target_path):
        if not os.path.isdir(os.path.join(target_path, subfolder)):
            continue
        subfolder_path = os.path.join(target_path, subfolder)
        eval_file = 'eval.csv'
        eval_path = os.path.join(subfolder_path, eval_file)
        folders.append(eval_path)

    # main(target_path, test = test_mode)

    for subfolder in folders:
        compute_cos_sim(subfolder)