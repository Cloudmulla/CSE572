import os
import json
import random
import numpy as np
from dotenv import load_dotenv 
from groq import Groq
import time

#Change working directory to be same as script directory to make everything work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

load_dotenv("../.env") #Load variables from .env file

#Initialize constants
DATASET_PATH = "../ConvQuestions/test_set/test_set_ALL.json"
random_state = 1
random.seed(random_state)
np.random.seed(random_state)
NUM_SAMPLES = 1
REPETITIONS = 3

#A lot of our constants are from the .env file because they have to do with managing API Calls
API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ["MODEL"]
OUTPUT_TOKENS = int(os.environ["MAX_OUTPUT_TOKENS"])
DELAY = float(os.environ["REQUEST_DELAY"])
QUERY_CACHE_PATH = "cached_queries.json"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 4))
RETRY_BACKOFF_SECONDS = float(os.environ.get("RETRY_BACKOFF_SECONDS", 2.0))

#Load dataset and then sample conversations
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
sampled_convos = random.sample(dataset, NUM_SAMPLES)
saved_progress = None
if os.path.exists(QUERY_CACHE_PATH):
    with open(QUERY_CACHE_PATH, "r") as f:
        saved_progress = json.load(f)

#Prompt user to ask if they want to start from scratch or use saved cache. Also prompt to ask them how many tokens they'd like to use.
queries_to_answer = None
if saved_progress:
    completed_queries = [convo for convo in saved_progress if convo.get("_query_complete", False)]
    print(f"Previous progress found with {len(completed_queries)}/{len(saved_progress)} conversations queried.")
    response = input("Type 1 to use previous progress and type 2 to start from scratch: ").strip()
    if response == "1":
        queries_to_answer = saved_progress
        print("Using previous progress...")
    elif response == "2":
        queries_to_answer = sampled_convos
        print("Starting from scratch...")
    else:
        raise ValueError("That is not a valid option, please rerun the program (sorry didn't feel like doing a loop and stuff).")
else:
    print("No previous progress found, starting from scratch...")
    queries_to_answer = sampled_convos
print()

#Validate the workload, assuming 5 requests per conversation
unanswered_queries = [convo for convo in queries_to_answer if not convo.get("_query_complete", False)]
tokens_per_convo = int(os.environ["TOKENS_PER_CONVO"])
total_convos = len(unanswered_queries)
rpm = 60 / DELAY
total_requests = REPETITIONS * total_convos * 5
usage_limits = {
    "Requests per Minute": (rpm, int(os.environ["RPM"])),
    "Tokens per Minute": (rpm / 5 * tokens_per_convo, int(os.environ["TPM"])),
    "Requests per Day": (total_requests, int(os.environ["RPD"])),
    "Total Tokens (daily limit)": (total_convos * tokens_per_convo, int(os.environ["TPD"]))
}
print("Displaying estimated usage/limit for various metrics (does not include retries due to failed requests).")
print("-------------------------------------------")
limit_exceeded = False
for metric, (usage, limit) in usage_limits.items():
    print(f"{metric}: {usage}/{limit}")
print()

print(f"Expected running time for all queries is {total_requests * (DELAY + 0.5)} seconds (note that it may take longer).")
print("Also Note that the program will stop early and save progress if you hit the daily limits for requests or tokens.")
confirmation = input("After reviewing all of the above details CAREFULLY, type \'proceed\' to start run the queries: ").strip()
if confirmation != "proceed":
    print("Cancelled.")
    exit(0)

#Note that we are modify the convo data in place and inserting the LLM answers inside. Preserves all question data we might need later.
groq_client = Groq(api_key=API_KEY)

def save_progress(slice_data, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as file:
        json.dump(slice_data, file)
    os.replace(tmp_path, path)


def is_retryable_error(error):
    error_text = str(error).lower()
    retryable_signals = [
        "output_parse_failed",
        "rate",
        "429",
        "timeout",
        "connection",
        "internal",
        "500",
        "502",
        "503",
        "504",
    ]
    return any(signal in error_text for signal in retryable_signals)


def request_with_retries(history):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(
                model=MODEL,
                messages=history,
                max_tokens=OUTPUT_TOKENS
            )
        except Exception as error:
            last_error = error
            should_retry = attempt < MAX_RETRIES and is_retryable_error(error)
            if not should_retry:
                break
            backoff = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(
                f"Warning: request failed ({type(error).__name__}: {error}). "
                f"Retry {attempt + 1}/{MAX_RETRIES} in {backoff:.1f}s."
            )
            time.sleep(backoff)

    raise last_error

#TODO: Implement the graph context stuff.
def query_model(convo, REPETITIONS):
    #Initialize empty list for answers and latencies in each question
    for question in convo["questions"]:
        question["given_answers"] = []
        question["latencies"] = []
    
    #Run through each conversation multiple times so we can check consistency later
    for i in range(REPETITIONS):
        history = [{"role" : "system", "content": f"Answer questions about {convo['seed_entity_text']} in as few words as possible."}]
        #Query the model and update history on each question
        for question in convo["questions"]:
            time.sleep(DELAY) #Rate limiting
            question_text = question["question"]
            history.append({"role": "user", "content": question_text})

            start_time = time.time()
            response = request_with_retries(history)
            end_time = time.time()

            extracted_response = response.choices[0].message.content
            if extracted_response:
                history.append({"role":"assistant", "content": extracted_response})
            question["given_answers"].append(extracted_response)
            question["latencies"].append(end_time - start_time)

    convo["_query_complete"] = True

queries_start = time.time()

#Now execute the queries
for index, query in enumerate(queries_to_answer):
    if query.get("_query_complete", False):
        print(f"Skipping convo {index}; already complete from checkpoint.")
        continue
    try:
        query_model(query, REPETITIONS)
        print(f"Finished querying for convo {index} (convo-id: {query['conv_id']})")
    except Exception as error:
        print(f"Failed convo {index} after retries: {type(error).__name__}: {error}")
        continue
    finally:
        save_progress(queries_to_answer, QUERY_CACHE_PATH)

queries_end = time.time()
print()
print(f"Altogether, queries took {queries_end - queries_start} seconds.")

#Store final checkpointed results in file (including completion markers for resumability)
save_progress(queries_to_answer, QUERY_CACHE_PATH)

incomplete_count = sum(1 for convo in queries_to_answer if not convo.get("_query_complete", False))
if incomplete_count > 0:
    print(f"Warning: {incomplete_count} convos are incomplete and can be retried by rerunning the script.")
else:
    print("All convos are complete.")