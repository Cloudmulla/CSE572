from dotenv import load_dotenv
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import spacy
import numpy as np

#Change working directory to be same as script directory to make everything work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

QUERY_CACHE_PATH = "cached_queries.json"

with open(QUERY_CACHE_PATH, "r") as f:
    answered_queries = json.load(f)


#Simply average together all the query latencies
def eval_avg_latency():
    latencies = []
    questions = [question for convo in answered_queries for question in convo["questions"]]
    for question in questions:
        avg_latency = sum(question["latencies"]) / len(question["latencies"])
        question["avg_latency"] = avg_latency
        latencies.append(avg_latency)
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency

sentence_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def similarity_helper(sentences):
    embeddings = sentence_transformer_model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)
    similarities = cosine_similarity(embeddings, [centroid])
    return float(np.mean(similarities))
def eval_consistency():
    consistency_scores = []
    questions = [question for convo in answered_queries for question in convo["questions"]]
    for question in questions:
        consistency_score = similarity_helper(question["given_answers"])
        question["consistency_score"] = consistency_score
        consistency_scores.append(consistency_score)
    return sum(consistency_scores) / len(consistency_scores)

from dateutil import parser
import unicodedata

spacy_model = spacy.load("en_core_web_sm")
similarity_threshold = 0.8
def accuracy_helper(question, answer_index):
    given = question["given_answers"][answer_index]
    expected = question["answer_text"]
    full_question = question["completed_question"] if "completed_question" in question else question["question"]

    #Start off by normalizing the texts
    given = unicodedata.normalize("NFKC", given)
    expected = unicodedata.normalize("NFKC", expected)
    expected = re.sub(r"[^\w\s]", "", expected.lower())
    given = re.sub(r"[^\w\s]", "", given.lower())

    #If we have an exact match we can just evaluate accuracy as true right off the bat.
    match = expected == given
    if match:
        return 1
   

    # #Make sure that the set of entities in expected is a subset of given
    # expected_entities = {entity.text.lower() for entity in spacy_model(expected).ents}
    # given_entities = {entity.text.lower() for entity in spacy_model(given).ents}
    # match = expected_entities.issubset(given_entities) or given_entities.issubset(expected_entities)
    # if match:
    #     return 1
    # elif len(expected_entities) == len(given_entities):
    #     print(f"Accuracy scored as 0 between '{expected}' and {given} with entities {expected_entities} and {given_entities}")
    #     return 0
    # #print(f"Did not get match using exact string matching or spacy between \"{expected}\" and \"{given}\", trying semantic similarity.")

    #Fallback to semantic similarity in case we don't get a match, possibly due to slightly differing names for same thing.
    similarity_score = similarity_helper([expected, given])
    match = similarity_score >= similarity_threshold
    if match:
        return 1
    else:
        print(f"No match between f{expected} and {given}")
        return 0

    #Sometimes model likes to give long answer and even repeat the question so including the question might make meanings match up.
    similarity_score = similarity_helper([f"{full_question} {expected}", given])
    match = similarity_score >= similarity_threshold
    if match:
        return 1
    
    return 0

def eval_accuracies():
    accuracies = []
    accuracies_multihop = []
    questions = [question for convo in answered_queries for question in convo["questions"]]
    for question in questions:
        question["accuracies"] = []
        for answer_index, given_answer in enumerate(question["given_answers"]):
            accuracy = accuracy_helper(question, answer_index)
            question["accuracies"].append(accuracy)
        question["avg_accuracy"] = sum(question["accuracies"]) / len(question["accuracies"])
        accuracies.append(question["avg_accuracy"])
        if question["turn"] != 0:
            accuracies_multihop.append(question["avg_accuracy"])
    
    accuracy = sum(accuracies) / len(accuracies)
    accuracy_multihop = sum(accuracies_multihop) / len(accuracies_multihop)
    return accuracy, accuracy_multihop

p1_accuracy, p1_accuracy_multihop = eval_accuracies()
consistency_score = eval_consistency()
avg_response_latency = eval_avg_latency()
metrics_obj = {"Overall P@1": p1_accuracy, "Multi-hop P@1": p1_accuracy_multihop, "Consistency Score": consistency_score, "Avg. Response Latency": avg_response_latency}

#Note we put together queries with intermediate computations and metrics obj together in same file with metrics at the top for easier viewing.
METRICS_CACHE_PATH = "cached_metrics.json"
answered_queries.insert(0, metrics_obj)
with open(METRICS_CACHE_PATH, "w") as file:
    json.dump(answered_queries, file)

#Now print out all of our compiled Metrics
for key, value in metrics_obj.items():
    print(f"{key}: {value}")


