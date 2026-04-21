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
NUM_SAMPLES = 20
REPETITIONS = 3

#A lot of our constants are from the .env file because they have to do with managing API Calls
API_KEY = os.environ["GROQ_API_KEY"]
MODEL = os.environ["MODEL"]
OUTPUT_TOKENS = int(os.environ["MAX_OUTPUT_TOKENS"])
DELAY = float(os.environ["REQUEST_DELAY"])
NUM_SLICES = int(os.environ["NUM_SLICES"])
SLICE_NUM = int(os.environ["SLICE_NUM"])
QUERY_CACHE_PATH = "cached_queries"

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
            response = groq_client.chat.completions.create(model=MODEL, messages=history, max_tokens=OUTPUT_TOKENS)
            end_time = time.time()

            extracted_response = response.choices[0].message.content
            if extracted_response:
                history.append({"role":"assistant", "content": extracted_response})
            question["given_answers"].append(extracted_response)
            question["latencies"].append(end_time - start_time)
    
#Now execute the queries
for index, query in enumerate(slice):
    query_model(query, REPETITIONS)
    print(f"Finished querying for convo {index} of the slice!")
queries_end = time.time()
print()
print(f"Altogether, queries took {queries_end - queries_start} seconds.")

#Store results in file
os.makedirs(QUERY_CACHE_PATH, exist_ok=True)
filepath = f"{QUERY_CACHE_PATH}/slice_{SLICE_NUM}_{NUM_SLICES}.json"
with open(filepath, "w") as file:
    json.dump(slice, file)