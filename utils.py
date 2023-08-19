import datetime
import time
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
import nltk
import re

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function '{func.__name__}' started at {datetime.datetime.now()}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' ended at {datetime.datetime.now()}")
        print(f"Function '{func.__name__}' took {execution_time:.6f} seconds to execute.")
        return result
    return wrapper

def get_title(passage):
    return passage.split('\n')[0]

def cos_single(model, sentence1, sentence2):
    if isinstance(sentence1, pd.Series):
        sentence1 = sentence1.tolist()
    if isinstance(sentence2, pd.Series):
        sentence2 = sentence2.tolist()
    # Encode the sentences into embeddings
    sentence1_embedding = model.encode(sentence1)
    sentence2_embedding = model.encode(sentence2)

    # Calculate the cosine similarity between the embeddings
    similarity_score = cos_sim(sentence1_embedding, sentence2_embedding)[0]

    return similarity_score.to('cpu').detach().numpy()

def extract_one_sentence(chunk, answer, method='BLEU'):
    if re.match(r'\d. ', chunk) is not None:
        # if chunk start with like '1. ', remove it
        chunk = chunk[3:]

    sentences = nltk.sent_tokenize(chunk)
    best_sentence = '' 
    best_score = 0

    ref_tokens = nltk.word_tokenize(answer)
    
    if method == 'BLEU':
        if len(sentences) == 1:
            return sentences[0]
        
        chencherry = SmoothingFunction()
        for sentence in sentences:
            if re.match(r'\d\.', sentence) is not None:
                continue
            
            sentence_tokens = nltk.word_tokenize(sentence)
            score = sentence_bleu([ref_tokens], sentence_tokens, smoothing_function=chencherry.method2)
            
            if score > best_score:
                best_score = score
                best_sentence = sentence

    return best_sentence
