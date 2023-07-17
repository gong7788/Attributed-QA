import re
from utils import compute_time_cost
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

def get_ref(doc, doc2dial_doc):
    refs_ID = re.findall(r"\d+", doc['references'])
    refs_ID = [int(i) for i in refs_ID]
    sp_list = [doc2dial_doc['doc_data'][doc['domain']][doc['doc_id']]['spans'][str(i)] for i in refs_ID]
    return sp_list

PASSAGE_FORMAT = re.compile("« ([^»]*) » « ([^»]*) » (.*)")

def format_example_for_autoais(example):
  return "premise: {} hypothesis: The answer to the question '{}' is '{}'".format(
      example["passage"], example["question"], example["answer"])

def format_for_autoais_batch(questions, answers, refs):
  example_list = []
  for i in range(len(questions)):
    question = questions[i].replace("##", " ")
    answer = answers[i].replace("##", " ")
    example = "premise: {} hypothesis: The answer to the question '{}' is '{}'".format(
      refs[i], question, answer)
    example_list.append(example)
  return example_list

def infer_autoais(example, tokenizer, model):
  """Runs inference for assessing AIS between a premise and hypothesis.

  Args:
    example: Dict with the example data.
    tokenizer: A huggingface tokenizer object.
    model: A huggingface model object.

  Returns:
    A string representing the model prediction.
  """
  input_text = format_example_for_autoais(example)
  input_ids = tokenizer(input_text, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  result = tokenizer.decode(outputs[0], skip_special_tokens=True)
  inference = "Y" if result == "1" else "N"
  example["autoais"] = inference
  return inference

@compute_time_cost
def infer_autoais_batch(questions, answers, refs, tokenizer, model):
  example_list = format_for_autoais_batch(questions, answers, refs)

  #batch inference
  input_ids = tokenizer(example_list, return_tensors="pt", padding=True, truncation=True, max_length=1024)

  outputs = model.generate(input_ids['input_ids'], attention_mask=input_ids['attention_mask'])

  results = tokenizer.batch_decode(outputs, skip_special_tokens=True)     

  return results


def score_predictions(predictions, nq_answers):
  """Scores model predictions against AutoAIS and NQ answers.

  Args:
    predictions: A dict from questions to prediction rows.
    nq_answers: A dict from questions to lists of NQ reference answers.
    passages: A dict from identifiers from the attribution corpus to the
      corresponding paragraphs.

  Returns:
    a dict of metric values, keyed by metric names
  """
  AUTOAIS = "google/t5_xxl_true_nli_mixture"
  hf_tokenizer = T5Tokenizer.from_pretrained(AUTOAIS)
  hf_model = T5ForConditionalGeneration.from_pretrained(AUTOAIS)

  autoais = 0
  target_answers = []
  predicted_answers = []
  for question, answers in nq_answers.items():
    target_answers.append(answers)
    example = predictions.get(question, None)
    if example is None:
    #   logging.error("Did not find prediction for '%s'", question)
      predicted_answers.append("")
      continue
    predicted_answers.append(example["answer"])
    if not example["passage"]:
      continue
    inference = infer_autoais(example, hf_tokenizer, hf_model)
    autoais += inference == "Y"

  scores = {}
  scores["AutoAIS"] = autoais / len(target_answers)
#   for metric, score in squad(target_answers, predicted_answers).items():
#     scores[f"SQuAD ({metric})"] = score
  return scores

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(model_answer, truth):
    pred_tokens = normalize_text(model_answer).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec), prec, rec