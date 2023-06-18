import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


#split document into chunks
def split(document, cs=500, co=0) -> Document:
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = cs,
    chunk_overlap = co
    )
    # remove empty documents
    split_documents = text_splitter.split_documents(document)
    return split_documents

def embedding(documents, model='sentence-transformers/gtr-t5-base') -> FAISS:
    #load embeddings
    embeddings = HuggingFaceEmbeddings(model_name = model)
    db = FAISS.from_documents(documents, embeddings)
    return db

def save_db(db):
    db.save_local("data/faiss_index")

def load_db(embeddings):
    new_db = FAISS.load_local("data/faiss_index", embeddings)
    return new_db
