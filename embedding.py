import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import Document


#split document into chunks
def split(document) -> Document:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
    )
    # remove empty documents
    split_documents = text_splitter.split_documents(document)
    return split_documents

def embedding(documents, embedding_model='sentence-transformers/gtr-t5-base') -> FAISS:
    #load embeddings
    embeddings = HuggingFaceEmbeddings(model_name = embedding_model)
    db = FAISS.from_documents(documents, embeddings)
    return db

def load_db(embeddings):
    new_db = FAISS.load_local("data/faiss_index", embeddings)
    return new_db

if __name__ == 'main':
    print('Testing...')