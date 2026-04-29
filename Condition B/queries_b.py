import os
import json
import random
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from SPARQLWrapper import SPARQLWrapper, JSON
import time

# Change working directory to be same as script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

load_dotenv("../.env")

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH = "../ConvQuestions/test_set/test_set_ALL.json"
random_state = 1
random.seed(random_state)
np.random.seed(random_state)
NUM_SAMPLES   = 100
REPETITIONS   = 3

API_KEY              = os.environ["GROQ_API_KEY"]
MODEL                = os.environ["MODEL"]
OUTPUT_TOKENS        = int(os.environ["MAX_OUTPUT_TOKENS"])
DELAY                = float(os.environ["REQUEST_DELAY"])
NUM_SLICES           = int(os.environ["NUM_SLICES"])
SLICE_NUM            = int(os.environ["SLICE_NUM"])
QUERY_CACHE_PATH     = "cached_queries_b"          # separate cache from Condition A
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES", 4))
RETRY_BACKOFF_SECONDS = float(os.environ.get("RETRY_BACKOFF_SECONDS", 2.0))

# ── Dataset loading & slicing ────────────────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)
sampled_convos = random.sample(dataset, NUM_SAMPLES)

slice_size = int(NUM_SAMPLES / NUM_SLICES)
if slice_size == 0:
    raise ValueError("0 size slices. Decrease NUM_SLICES or increase NUM_SAMPLES.")

slices = [sampled_convos[i:i + slice_size] for i in range(0, len(sampled_convos), slice_size)]
if len(slices) > NUM_SLICES:
    for i, rslice in enumerate(slices[NUM_SLICES:]):
        slices[i].extend(rslice)
    slices = slices[:NUM_SLICES]

for i, s in enumerate(slices):
    print(f"Samples in slice {i}: {len(s)}")
print()

if SLICE_NUM < 0 or NUM_SLICES <= SLICE_NUM or NUM_SLICES <= 0:
    raise ValueError(f"Invalid configuration NUM_SLICES={NUM_SLICES} and SLICE_NUM={SLICE_NUM}")

slice = slices[SLICE_NUM]

# ── Usage / limit validation ─────────────────────────────────────────────────
tokens_per_convo = int(os.environ["TOKENS_PER_CONVO"])
rpm = 60 / DELAY
total_requests = REPETITIONS * len(slice) * 5
limit_checks = {
    "Requests per Minute": (rpm,                          int(os.environ["RPM"])),
    "Requests per Day":    (total_requests,               int(os.environ["RPD"])),
    "Tokens per Minute":   (rpm / 5 * tokens_per_convo,  int(os.environ["TPM"])),
    "Tokens per Day":      (total_requests / 5 * tokens_per_convo, int(os.environ["TPD"])),
}

print("Estimated usage / limit (excludes retries):")
print("-------------------------------------------")
limit_exceeded = False
for metric, (usage, limit) in limit_checks.items():
    print(f"  {metric}: {usage}/{limit}")
    if usage > limit:
        limit_exceeded = True
print()

if limit_exceeded:
    print("Configuration exceeds one or more API limits. Adjust .env before proceeding.")
    exit(0)

print(f"Expected minimum running time: {total_requests * DELAY:.0f}s")
confirmation = input(
    f"Proceed with Condition B – slice {SLICE_NUM} ({len(slice)} convos, {REPETITIONS} reps)? "
    f"(Type 'yes' to proceed) "
).strip()
if confirmation != "yes":
    print("Cancelled.")
    exit(0)

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=API_KEY)

sparql_client = SPARQLWrapper(
    "https://query.wikidata.org/sparql",
    agent="CSE572-ConditionB/1.0 (ASU Data Mining Project; Python SPARQLWrapper)"
)
sparql_client.setReturnFormat(JSON)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_slice_progress(slice_data, path):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(slice_data, f)
    os.replace(tmp_path, path)


def is_retryable_error(error):
    error_text = str(error).lower()
    retryable_signals = [
        "output_parse_failed", "rate", "429", "timeout",
        "connection", "internal", "500", "502", "503", "504",
    ]
    return any(signal in error_text for signal in retryable_signals)


def request_with_retries(history):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return groq_client.chat.completions.create(
                model=MODEL,
                messages=history,
                max_tokens=OUTPUT_TOKENS,
            )
        except Exception as error:
            last_error = error
            if attempt >= MAX_RETRIES or not is_retryable_error(error):
                break
            backoff = RETRY_BACKOFF_SECONDS * (2 ** attempt)
            print(
                f"  Warning: request failed ({type(error).__name__}: {error}). "
                f"Retry {attempt + 1}/{MAX_RETRIES} in {backoff:.1f}s."
            )
            time.sleep(backoff)
    raise last_error


# ── get_graph: fetch Wikidata triples for the seed entity ────────────────────
def get_graph(convo):
    """
    Returns a list of human-readable triples from Wikidata for the
    conversation's seed entity, e.g.:
        ["The Fast and the Furious | director | Rob Cohen", ...]

    These triples are injected into the LLM system prompt as static
    knowledge-graph context (Condition B).
    """
    qid = convo["seed_entity"].split("/")[-1].split("?")[0]

    graph_query = f"""
SELECT ?propLabel ?valLabel WHERE {{
    wd:{qid} ?p ?val .
    ?prop wikibase:directClaim ?p .
    FILTER(STRSTARTS(STR(?p), STR(wdt:)))
    SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 20
"""
    # FIX 1: actually assign the query before calling .query()
    sparql_client.setQuery(graph_query)
    # FIX 2: wikibase:directClaim (no space) — already correct above
    # FIX 3: removed exit(0) debug call — return triples instead

    try:
        graph_results = sparql_client.query().convert()
    except Exception as e:
        print(f"  Warning: SPARQL query failed for {qid}: {e}. Continuing without graph context.")
        return []

    triples = []
    for result in graph_results["results"]["bindings"]:
        prop = result["propLabel"]["value"]
        val  = result["valLabel"]["value"]
        triples.append(f"{convo['seed_entity_text']} | {prop} | {val}")

    return triples


def format_graph_context(triples):
    """Serialize the triple list into a string for the LLM system prompt."""
    if not triples:
        return ""
    lines = "\n".join(f"  - {t}" for t in triples)
    return f"\n\nKnowledge graph facts about this topic:\n{lines}"


# ── Core query function (Condition B) ────────────────────────────────────────
def query_model(convo, repetitions):
    for question in convo["questions"]:
        question["given_answers"] = []
        question["latencies"]     = []

    # --- Fetch static KG context once per conversation ---
    graph_triples = get_graph(convo)
    graph_context = format_graph_context(graph_triples)
    convo["graph_triples"] = graph_triples   # save for inspection later

    system_prompt = (
        f"Answer questions about {convo['seed_entity_text']} in as few words as possible."
        f"{graph_context}"
    )

    for _ in range(repetitions):
        history = [{"role": "system", "content": system_prompt}]

        for question in convo["questions"]:
            time.sleep(DELAY)
            history.append({"role": "user", "content": question["question"]})

            start = time.time()
            response = request_with_retries(history)
            end   = time.time()

            answer = response.choices[0].message.content
            if answer:
                history.append({"role": "assistant", "content": answer})

            question["given_answers"].append(answer)
            question["latencies"].append(end - start)

    convo["_query_complete"] = True


# ── Main loop ─────────────────────────────────────────────────────────────────
os.makedirs(QUERY_CACHE_PATH, exist_ok=True)
filepath = f"{QUERY_CACHE_PATH}/slice_{SLICE_NUM}_{NUM_SLICES}.json"

if os.path.isfile(filepath):
    with open(filepath, "r") as f:
        existing_slice = json.load(f)
    if len(existing_slice) == len(slice):
        slice = existing_slice
        completed = sum(1 for c in slice if c.get("_query_complete"))
        print(f"Resuming from checkpoint: {completed}/{len(slice)} convos complete.")
    else:
        print("Checkpoint size mismatch — starting fresh.")

queries_start = time.time()

for index, query in enumerate(slice):
    if query.get("_query_complete"):
        print(f"Skipping convo {index} (already complete).")
        continue
    try:
        query_model(query, REPETITIONS)
        print(f"Finished convo {index} (conv_id: {query['conv_id']})")
    except Exception as error:
        print(f"Failed convo {index}: {type(error).__name__}: {error}")
    finally:
        save_slice_progress(slice, filepath)

queries_end = time.time()
print(f"\nAll queries took {queries_end - queries_start:.1f}s total.")

save_slice_progress(slice, filepath)

incomplete = sum(1 for c in slice if not c.get("_query_complete"))
if incomplete:
    print(f"Warning: {incomplete} convos incomplete — rerun to retry.")
else:
    print("All convos in this slice complete. ✅")
