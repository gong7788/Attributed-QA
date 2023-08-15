import datetime
import time
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd

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

def cos_single(sentence1, sentence2):
    model_name = 'sentence-transformers/gtr-t5-base'
    model = SentenceTransformer(model_name, device='cuda')

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

