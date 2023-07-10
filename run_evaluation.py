import evaluation
import pandas as pd
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
# import torch
import os

data_paths = ['data/doc2dial/result_4.csv']
doc_file_path = 'data/doc2dial/doc2dial_doc.json'
qa_file_path = 'data/doc2dial/doc2dial_qa_train.csv'
max_num = 5000

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
        elif cnt == max_num:
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

if __name__ == "__main__":
    print('Running evaluation...')
    df = read_data(data_paths)
    df = df.dropna(subset=['answer'])
    qa_df = pd.read_csv(qa_file_path)
    doc2dial_doc = load_doc_file(doc_file_path)

    eval(df, qa_df, doc2dial_doc, test=False, test_num=10, eval_id=4)