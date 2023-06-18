import re

def get_ref(doc, doc2dial_doc):
    refs_ID = re.findall(r"\d+", doc['references'])
    refs_ID = [int(i) for i in refs_ID]
    sp_list = [doc2dial_doc['doc_data'][doc['domain']][doc['doc_id']]['spans'][str(i)] for i in refs_ID]
    return sp_list