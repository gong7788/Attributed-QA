import json

import pandas as pd
import ast

from torch.utils.data import Dataset


class doc2dialDataset(Dataset):
    def __init__(self, data):
        self.doc_data = json.load(open('data/doc2dial/doc2dial_doc.json', 'r'))
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError('Not supported data type.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        refs_ID = [term['sp_id'] for term in ast.literal_eval(self.data.loc[idx, 'ref'])]

        doc_file_span = self.doc_data['doc_data']['dmv'][self.data.loc[idx, 'doc_id']]['spans']
        ll = [doc_file_span[i] for i in refs_ID]

        true_ref_string = [term['text_sp'] for term in ll]
        concatenated_string = ''.join(string for sublist in true_ref_string for string in sublist)
        string_list = '##'.join(string for string in true_ref_string)

        try: 
            sample = {'question': self.data.loc[idx, 'question'],
                    'answer': self.data.loc[idx, 'answer'],
                    'ref': self.data.loc[idx, 'ref'],
                    'ref_string': concatenated_string,
                    'string_list': string_list,
                    'retrived_doc': self.data.loc[idx, 'passage(context)'],
                    'doc_id': self.data.loc[idx, 'doc_id'],
                    'dial_id': self.data.loc[idx, 'dial_id']
                    }
        except:
            raise ValueError('Error in getting item', self.data.loc[idx, 'dial_id'])

        return sample

class OpenQADataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # question,answer,passage,human_rating,paragraphs
        sample= {'question': self.data.loc[idx, 'question'],
                'answer': self.data.loc[idx, 'answer'],
                'passage': self.data.loc[idx, 'passage'],
                'human_rating': self.data.loc[idx, 'human_rating'],
                'paragraphs': self.data.loc[idx, 'paragraphs']
                }
        return sample
    
class doc2dialEvalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.doc_data = json.load(open('data/doc2dial/doc2dial_doc.json', 'r'))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # preprocess
        true_ref_position, concatenated_string = self.preprocess(idx)

        # sentences = self.data.loc[idx, 'retrived_doc'].split('\n')
        # starts = [p.find(sentence) for sentence in sentences]
        # ends = [start + len(sentence) - 1 for start, sentence in zip(starts, sentences)]

        # question,answer,model_answer,ref,retrived_doc,doc_id
        sample= {'question': self.data.loc[idx, 'question'],
                'answer': self.data.loc[idx, 'answer'],
                'model_answer': str(self.data.loc[idx, 'model_answer']),
                'ref': self.data.loc[idx, 'ref'],
                'ref_string': concatenated_string,
                'retrived_doc': self.data.loc[idx, 'retrived_doc'],
                'doc_id': self.data.loc[idx, 'doc_id'],
                'dial_id': self.data.loc[idx, 'dial_id'],
                'true_ref_position': str(true_ref_position),
                # 'retrieved_doc_s': starts,
                # 'retrieved_doc_e': ends
                }
        return sample

    def preprocess(self, idx):
        ref_data = ast.literal_eval(self.data.loc[idx, 'ref'])

        if isinstance(ref_data, list):
            if all(isinstance(item, dict) for item in ref_data):
                refs_ID = [term['sp_id'] for term in ast.literal_eval(self.data.loc[idx, 'ref'])]
                doc_file_span = self.doc_data['doc_data']['dmv'][self.data.loc[idx, 'doc_id']]['spans']
                ll = [doc_file_span[i] for i in refs_ID]

                true_ref_position = [(term['start_sp'], term['end_sp']) for term in ll]
                true_ref_string = [term['text_sp'] for term in ll]
                concatenated_string = ' '.join(string for string in true_ref_string)

                return true_ref_position, concatenated_string
                
            elif all(isinstance(item, list) and all(isinstance(sub_item, dict) for sub_item in item) for item in ref_data):
                refs_ID = [[term['sp_id'] for term in _] for _ in ast.literal_eval(self.data.loc[idx, 'ref'])]
                doc_file_span = self.doc_data['doc_data']['dmv'][self.data.loc[idx, 'doc_id']]['spans']
                ll = [[doc_file_span[i] for i in l] for l in refs_ID]
                # p = self.doc_data['doc_data']['dmv'][self.data.loc[idx, 'doc_id']]['doc_text']

                true_ref_position = [[(term['start_sp'], term['end_sp']) for term in _] for _ in ll]
                true_ref_string = [[term['text_sp'] for term in _] for _ in ll]
                concatenated_string = ' '.join(string for sublist in true_ref_string for string in sublist)
                return true_ref_position, concatenated_string
            else:
                print("The data has a mixed structure of strings and lists.")
                return None


class ExtendedDoc2dialEvalDataset(doc2dialEvalDataset):
    # since we only label the 'data/doc2dial/new_dataset/DEFAULT/DEFAULT_ModelAnswer.csv' only 
    def __init__(self, csv_file, label_csv):
        super().__init__(csv_file)
        self.label_data = pd.read_csv(label_csv)
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx) #call parent class method
        sample['label'] = self.label_data.loc[idx, 'label']

        return sample


# csv_file = 'data/doc2dial/qa_train_dmv.csv'
# dataset = doc2dialDataset(csv_file)
# batch_size = 16
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# for i, batch in enumerate(dataloader):
#     print('ith sample: ', i) 
#     # Perform your training/inference operations on the batch
#     questions = batch['question']
#     answers = batch['answer']
#     refs = batch['ref']



# data_path = 'data/doc2dial/'
# file_path = data_path + 'doc2dial_doc.json'
# with open(file_path, 'r') as f:
#     data = json.load(f)

# def filter_and_write_to_json(data, keys, output_file) -> None:
#     # Create a new dictionary with only the desired keys
#     filtered_data = {key: data[key] for key in keys if key in data}

#     # Write the filtered data to a JSON file
#     with open(output_file, 'w') as file:
#         json.dump(filtered_data, file)

# doc_list = list(data['doc_data']['ssa'].keys())

# keys_to_filter = ['title', 'doc_text']
# filter_and_write_to_json(data['doc_data']['ssa'][doc_list[0]], keys_to_filter, 'data/doc1.json')