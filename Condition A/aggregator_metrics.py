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

#Load the query slices and combine them all together. Load env variables as well to validate workload.
load_dotenv("../.env")
NUM_SLICES = int(os.environ["NUM_SLICES"])
QUERY_CACHE_PATH = "cached_queries"
answered_queries = []
for i in range(NUM_SLICES):
    filepath = f"{QUERY_CACHE_PATH}/slice_{i}_{NUM_SLICES}.json"
    if not os.path.isfile(filepath):
        raise RuntimeError(f"The slice file at {filepath} is missing!")
    
    with open(filepath, "r") as f:
        slice_data = json.load(f)
    answered_queries = answered_queries + slice_data

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

spacy_model = spacy.load("en_core_web_sm")
similarity_threshold = 0.8
def accuracy_helper(expected, given, full_question):
    #Start off by normalizing the texts
    expected = re.sub(r"[^\w\s]", "", expected.lower())
    given = re.sub(r"[^\w\s]", "", given.lower())

    #Make sure that the set of entities in expected is a subset of given
    expected_entities = {entity.text.lower() for entity in spacy_model(expected).ents}
    given_entities = {entity.text.lower() for entity in spacy_model(given).ents}
    match = expected_entities.issubset(given_entities)
    if match:
        return 1
    
    #Fallback to semantic similarity in case we don't get a match, possibly due to slightly differing names for same thing.
    similarity_score = similarity_helper([expected, given])
    match = similarity_score >= similarity_threshold
    if match:
        return 1

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
        for given_answer in question["given_answers"]:
            full_question = question["completed_question"] if "completed_question" in question else question["question"]
            accuracy = accuracy_helper(question["answer_text"], given_answer, full_question)
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


