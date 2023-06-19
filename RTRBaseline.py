import json
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import pandas as pd
import embedding
import evaluation
import QAModel
from typing import Dict, List

######################global variables######################
#test number
test_num = 1
#load qa pairs
qa_file_path = 'data/doc2dial/doc2dial_qa_train.csv'
#load json file
doc_file_path = 'data/doc2dial/doc2dial_doc.json'
#chunk size
cs = 500
#chunk overlap
c_overlap = 0
#embedding model
embedding_model = 'sentence-transformers/gtr-t5-base'
#qa model
qa_model = 'google/flan-t5-large'
#chain type
chain_type = 'stuff'
#how many retrieved docs to use
topk = 1

#get doc text
# doc_text = doc2dial_doc['doc_data'][domain][doc_id]['doc_text']
############################################################

def load_qa_file(file_path) -> Dict:
    """
    Load doc2dial_qa_train.csv file
    """
    #check if file is csv
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        return df
    else:
        raise ValueError('File is not csv')


def load_doc_file(file_path) -> Dict:
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

def seaerch_doc(doc, db) -> List[Document]:
    query = doc['question']
    docs = db.similarity_search(query)
    return docs


def RTRBaseline(qa_set, doc2dial_doc, test=True) -> None:
    """
    qa_set: dataframe of qa pairs {'question', 'answer', 'domain', 'doc_id', 'references', 'dial_id'}
    doc2dial_doc: dict (json file)

        get doc text:
        doc_text = doc2dial_doc['doc_data'][domain][doc_id]['doc_text']
    """

    #iterate through qa_set
    last_doc_name = ''
    for index, row in qa_set.iterrows():
        doc_text = doc2dial_doc['doc_data'][row['domain']][row['doc_id']]['doc_text']

        #get embeddings
        doc_name = row['doc_id']
        embeddings = HuggingFaceEmbeddings(model_name = embedding_model)
        #check if using same doc
        if doc_name != last_doc_name:
            #build new index
            document = Document(page_content=doc_text, metadata={"source": row['doc_id']})
            split_documents = embedding.split([document], cs=cs, co=c_overlap)
            db = embedding.embedding(split_documents, model=embedding_model)
            last_doc_name = doc_name
            if not test:
                embedding.save_db(db)
            print('new index created at: ', index)
        else: 
            #TODO not need load index every time
            #load index
            db = embedding.load_db(embeddings)
            print('index loaded at: ', index)

        #seaerch doc
        result_docs = seaerch_doc(row, db) # list of retrieved documents

        ref_list = evaluation.get_ref(row, doc2dial_doc) # true references

        #get answer
        model_answer = QAModel.answer_from_local_model(row['question'], result_docs, model_name=qa_model, ct=chain_type, topk=1)

        #save answer
        #FIXME
        example = {}
        example['question'] = doc1['question']
        example['answer'] = model_answer #TODO should be model result?
        example["passage"] = result_docs[:topk] #TODO only first document. If use all documents, need to change 

        if test and index == test_num:
            break
        else:
            continue
    
    #evaluation process
    #TODO iterate through answer file
    #TODO compare em, f1, autoais 

if __name__ == '__main__':
    print('Running RTR Baseline...')
    df = load_qa_file(qa_file_path)
    doc2dial_doc = load_doc_file(doc_file_path)
    RTRBaseline(df, doc2dial_doc)