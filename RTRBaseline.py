import json
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

#load qa pairs
file_path = 'data/doc2dial/doc2dial_qa_train.csv'
df = pd.read_csv(file_path)

#load json file
file_path = 'data/doc2dial/doc2dial_doc.json'
with open(file_path, 'r') as f:
    doc2dial_doc = json.load(f)
#get doc text
# doc_text = doc2dial_doc['doc_data'][domain][doc_id]['doc_text']


def load_doc_file(file_path) -> dict:
    """
    Load doc2dial_doc.json file
    
    """
    with open(file_path, 'r') as f:
        doc2dial_doc = json.load(f)
    return doc2dial_doc

def RTRBaseline() -> None:
    pass

if __name__ == 'main':
    print('Running RTR Baseline...')
    file_path = 'data/doc2dial/doc2dial_doc.json'