import timeit
import pandas as pd
from langchain.docstore.document import Document

import embedding
from RTRBaseline import load_doc_file, load_qa_file, seaerch_doc, find_pre_next_doc
from embedding import embedding
import os
import ast

def BaseRetriever():
    raise NotImplementedError()

@timeit
def retrieve_openqa(batch_question, batch_paragraphs, model='sentence-transformers/gtr-t5-base', new_method=False, topk=1):
    """
    retriever for OpenQA dataset
    """
    batch_result = []
    batch_idx = []
    if len(batch_question) != len(batch_paragraphs):
        raise ValueError('batch_question and batch_paragraphs should have same length')
    
    for i in range(len(batch_question)):
        paragraphs = batch_paragraphs[i]
        question = batch_question[i]
        paragraph_list = ast.literal_eval(paragraphs)
        docs = [Document(page_content=paragraph_list[i], metadata={'p_idx':i}) for i in range(len(paragraph_list))]
        # build index or read index
        db = embedding(docs, model=model)
        
        res = db.similarity_search(question)

        if new_method:
            if topk != 1:
                raise ValueError('topk should be 1 when using new_method')
            
            idx = res[0].metadata['p_idx']
            if idx == 0:
                res_idx = [idx, idx+1]
            elif idx == len(paragraph_list) - 1:
                res_idx = [idx-1, idx]
            else:
                res_idx = [idx-1, idx, idx+1]

            res_content = [paragraph_list[i] for i in res_idx]

            batch_result.append(res_content)
            batch_idx.append(res_idx)

        else:
            res_content = [doc.page_content for doc in res][:topk]
            res_idx = [doc.metadata['p_idx'] for doc in res][:topk]
            batch_result.append(res_content)
            batch_idx.append(res_idx)

    return batch_result, batch_idx

def retrieve_only(data_path, cs, c_overlap, save_dir, embedding_model='sentence-transformers/gtr-t5-base', new_method=False, topk=1, test_mode=False):
    last_doc_name = ''
    #question,answer,ref,doc_id,dial_id
    result_df = pd.DataFrame(columns=['question', 'answer', 'ref', 'passage(context)', 'doc_id', 'dial_id'])
    total_split_docs = 0
    cnt = 0
    failed_doc_ids = []
    doc2dial_doc = load_doc_file('data/doc2dial/doc2dial_doc.json')
    
    qa_set = pd.read_csv(data_path)

    for index, row in qa_set.iterrows():
        if test_mode and index == 10:
            break

        doc_text = doc2dial_doc['doc_data']['dmv'][row['doc_id']]['doc_text']

        #get embeddings
        doc_name = row['doc_id']
        #check if using same doc
        if doc_name != last_doc_name:
            #build new index
            document = Document(page_content=doc_text, metadata={"source": row['doc_id']})
            split_documents = embedding.split([document], cs=cs, co=c_overlap)
            total_split_docs += len(split_documents)
            cnt += 1

            db = embedding.embedding(split_documents, model=embedding_model)
            last_doc_name = doc_name

        #seaerch doc
        try:
            result_docs = seaerch_doc(row, db) # list of retrieved documents
        except:
            # if question is Nan, skip
            failed_doc_ids.append(index)
            print('search failed at: ', index)
        
        if new_method and result_docs is not None:
            #find pre and next doc
            #now fix num to 1, otherwise may larger than max token length
            idx_list = find_pre_next_doc(split_documents, result_docs, num=1)
            result_docs = [split_documents[i] for i in idx_list]
            # if used new method, topk should not 1, but len(result_docs)
            topk = len(result_docs)
        # ref_list = evaluation.get_ref(row, doc2dial_doc) # true references

        #save answer

        result_docs_list = result_docs[:topk]
        passage = '\n'.join([doc.page_content for doc in result_docs_list])

        result_df.loc[len(result_df)] = [row['question'], row['answer'], row['ref'], passage, row['doc_id'], row['dial_id']]
    
    return result_df
    # write to csv
    # if test_mode:
    #     print(save_dir)
        
    # result_df.to_csv(save_dir, index=False)

    # return result_df, failed_doc_ids


