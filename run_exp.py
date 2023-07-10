import RTRBaseline
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from document_loader import doc2dialDataset, OpenQADataset

def run_RTR_Model(data_path, batch_size, test_mode):
    #csv_file = 'data/doc2dial/qa_train_dmv.csv'

    if 'doc2dial' in data_path:
        dataset = doc2dialDataset(data_path)
    if 'openqa' in data_path:
        dataset = OpenQADataset(data_path)

    shuffle = True

    if test_mode:
        batch_size = 2
        shuffle = False

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in dataloader:
        # Perform your training/inference operations on the batch
        questions = batch['question']
        answers = batch['answer']
        refs = batch['ref']
        #[ ] retrieve docs = batch['retrived_doc']

        print(questions)
        print(answers)
        print(refs)

        break

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Start Experiment")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--new_method", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    
    args = parser.parse_args()
    logging.info("Experiment id: {}".format(args.exp_id))

    run_RTR_Model(data_path=args.data_path, batch_size=args.batch_size, test_mode=args.test)