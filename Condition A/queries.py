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
NUM_SAMPLES = 100
REPETITIONS = 3

#A lot of our constants are from the .env file because they have to do with managing API Calls
API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ["MODEL"]
OUTPUT_TOKENS = int(os.environ["MAX_OUTPUT_TOKENS"])
DELAY = float(os.environ["REQUEST_DELAY"])
NUM_SLICES = int(os.environ["NUM_SLICES"])
SLICE_NUM = int(os.environ["SLICE_NUM"])
QUERY_CACHE_PATH = "cached_queries"
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 4))
RETRY_BACKOFF_SECONDS = float(os.environ.get("RETRY_BACKOFF_SECONDS", 2.0))

#Load dataset and then sample conversations
with open(DATASET_PATH, 'r') as f:
    dataset = json.load(f)
sampled_convos = random.sample(dataset, NUM_SAMPLES)

#Slice up the convos for each worker and assign ourselves a slice based on config
slice_size = int(NUM_SAMPLES / NUM_SLICES)
if slice_size == 0:
    raise ValueError("0 size slices are not acceptable. Either decrease number of slices or increase number of samples")
slices = [sampled_convos[i:i+slice_size] for i in range(0, len(sampled_convos), slice_size)]
if len(slices) > NUM_SLICES:
    remainder_slices = slices[NUM_SLICES:]
    for i, rslice in enumerate(remainder_slices):
        slices[i].extend(rslice)
    slices = slices[:NUM_SLICES]
for i, slice in enumerate(slices):
    print(f"Samples in slice {i}: {len(slice)}")
print()
slice = slices[SLICE_NUM]

#Validate the workload, assuming 5 questions per conversation
if SLICE_NUM < 0 or NUM_SLICES <= SLICE_NUM or NUM_SLICES <= 0:
    raise ValueError(f"Invalid configuration NUM_SLICES={NUM_SLICES} and SLICE_NUM={SLICE_NUM}")
tokens_per_convo = int(os.environ["TOKENS_PER_CONVO"])
rpm = 60 / DELAY
total_requests = REPETITIONS * len(slice) * 5
limit_checks = {
    "Requests per Minute": (rpm, int(os.environ["RPM"])),
    "Requests per Day": (total_requests, int(os.environ["RPD"])),
    "Tokens per Minute": (rpm / 5 * tokens_per_convo, int(os.environ["TPM"])),
    "Tokens per Day": (total_requests / 5 * tokens_per_convo, int(os.environ["TPD"]))
}
print("Displaying estimated usage/limit for various metrics.")
print("-------------------------------------------")
limit_exceeded = False
for metric, (usage, limit) in limit_checks.items():
    print(f"{metric}: {usage}/{limit}")
    if usage > limit:
        limit_exceeded = True
print()
if limit_exceeded:
    print("Your configuration exceeds one or more of the above API limits.")
    exit(0)

print(f"Expected running time is at LEAST {total_requests * DELAY} seconds (just for the waiting delays, not even including API call latencies).")
confirmation = input(f"Proceed with running slice {SLICE_NUM} ({len(slice)} convos, {REPETITIONS} repetitions each)? (Type yes to proceed) ").strip()
if confirmation != "yes":
    print("Cancelled.")
    exit(0)

queries_start = time.time()

#Note that we are modify the convo data in place and inserting the LLM answers inside. Preserves all question data we might need later.
groq_client = Groq(api_key=API_KEY)


def save_slice_progress(slice_data, path):
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
    
#Now execute the queries
os.makedirs(QUERY_CACHE_PATH, exist_ok=True)
filepath = f"{QUERY_CACHE_PATH}/slice_{SLICE_NUM}_{NUM_SLICES}.json"
if os.path.isfile(filepath):
    with open(filepath, "r") as file:
        existing_slice = json.load(file)
    if len(existing_slice) == len(slice):
        slice = existing_slice
        completed_count = sum(1 for convo in slice if convo.get("_query_complete"))
        print(f"Resuming from checkpoint: {completed_count}/{len(slice)} convos already complete.")
    else:
        print("Existing checkpoint has mismatched size; starting a fresh run for this slice.")

for index, query in enumerate(slice):
    if query.get("_query_complete"):
        print(f"Skipping convo {index}; already complete from checkpoint.")
        continue

    try:
        query_model(query, REPETITIONS)
        print(f"Finished querying for convo {index} of the slice!")
    except Exception as error:
        print(f"Failed convo {index} after retries: {type(error).__name__}: {error}")
        continue
    finally:
        save_slice_progress(slice, filepath)

queries_end = time.time()
print()
print(f"Altogether, queries took {queries_end - queries_start} seconds.")

#Store final checkpointed results in file (including completion markers for resumability)
save_slice_progress(slice, filepath)

incomplete_count = sum(1 for convo in slice if not convo.get("_query_complete"))
if incomplete_count > 0:
    print(f"Warning: {incomplete_count} convos are incomplete and can be retried by rerunning the script.")
else:
    print("All convos in this slice are complete.")