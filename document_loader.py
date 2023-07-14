import json
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class doc2dialDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, str):
            self.data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError('Not supported data type.')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'question': self.data.loc[idx, 'question'],
                  'answer': self.data.loc[idx, 'answer'],
                  'ref': self.data.loc[idx, 'ref'],
                  'retrived_doc': self.data.loc[idx, 'passage(context)'],
                  'doc_id': self.data.loc[idx, 'doc_id'],
                  'dial_id': self.data.loc[idx, 'dial_id']
                }
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