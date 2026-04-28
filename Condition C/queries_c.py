import os
import json
import random
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
import spacy
import time

# Change working directory to be same as script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

load_dotenv("../.env")

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH  = "../ConvQuestions/test_set/test_set_ALL.json"
random_state  = 1
random.seed(random_state)
np.random.seed(random_state)
NUM_SAMPLES   = 100
REPETITIONS   = 3

API_KEY               = os.environ["GROQ_API_KEY"]
MODEL                 = os.environ["MODEL"]
OUTPUT_TOKENS         = int(os.environ["MAX_OUTPUT_TOKENS"])
DELAY                 = float(os.environ["REQUEST_DELAY"])
NUM_SLICES            = int(os.environ["NUM_SLICES"])
SLICE_NUM             = int(os.environ["SLICE_NUM"])
QUERY_CACHE_PATH      = "cached_queries_c"
MAX_RETRIES           = int(os.environ.get("MAX_RETRIES", 4))
RETRY_BACKOFF_SECONDS = float(os.environ.get("RETRY_BACKOFF_SECONDS", 2.0))

# Neo4j connection settings — update password to match yours
NEO4J_URI      = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]  # required in .env

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
    "Requests per Minute": (rpm,                                   int(os.environ["RPM"])),
    "Requests per Day":    (total_requests,                        int(os.environ["RPD"])),
    "Tokens per Minute":   (rpm / 5 * tokens_per_convo,           int(os.environ["TPM"])),
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
    f"Proceed with Condition C - slice {SLICE_NUM} ({len(slice)} convos, {REPETITIONS} reps)? "
    f"(Type 'yes' to proceed) "
).strip()
if confirmation != "yes":
    print("Cancelled.")
    exit(0)

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=API_KEY)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
nlp = spacy.load("en_core_web_sm")

# ── Neo4j helpers ─────────────────────────────────────────────────────────────
def clear_conversation_graph(tx, conv_id):
    """Remove all nodes/edges for this conversation before starting fresh."""
    tx.run("MATCH (n {conv_id: $conv_id}) DETACH DELETE n", conv_id=conv_id)


def add_turn_to_graph(tx, conv_id, turn_num, question, answer, entities):
    """
    Insert a Turn node and Entity nodes into the graph.
    Each entity found in the answer is linked to the Turn node.
    """
    # Create the Turn node
    tx.run("""
        MERGE (t:Turn {conv_id: $conv_id, turn: $turn_num})
        SET t.question = $question, t.answer = $answer
    """, conv_id=conv_id, turn_num=turn_num, question=question, answer=answer)

    # Create Entity nodes and link them to the Turn
    for entity_text, entity_label in entities:
        tx.run("""
            MERGE (e:Entity {conv_id: $conv_id, name: $name})
            SET e.type = $label
            WITH e
            MATCH (t:Turn {conv_id: $conv_id, turn: $turn_num})
            MERGE (t)-[:MENTIONS {turn: $turn_num}]->(e)
        """, conv_id=conv_id, name=entity_text, label=entity_label, turn_num=turn_num)


def get_graph_context(tx, conv_id, current_turn):
    """
    Query the graph for all entities and answers from previous turns.
    Returns a formatted string to inject into the LLM prompt.
    """
    result = tx.run("""
        MATCH (t:Turn {conv_id: $conv_id})-[:MENTIONS]->(e:Entity)
        WHERE t.turn < $current_turn
        RETURN t.turn AS turn, t.question AS question, 
               t.answer AS answer, collect(e.name) AS entities
        ORDER BY t.turn
    """, conv_id=conv_id, current_turn=current_turn)

    rows = result.data()
    if not rows:
        return ""

    lines = ["Knowledge graph context from previous turns:"]
    for row in rows:
        lines.append(f"  Turn {row['turn']}: Q: {row['question']}")
        lines.append(f"           A: {row['answer']}")
        if row["entities"]:
            lines.append(f"           Entities: {', '.join(row['entities'])}")
    return "\n".join(lines)


# ── NLP entity extraction ─────────────────────────────────────────────────────
def extract_entities(text):
    """
    Use spaCy to extract named entities from a text.
    Returns list of (entity_text, entity_label) tuples.
    e.g. [("Vin Diesel", "PERSON"), ("2001", "DATE")]
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ── Groq helpers ──────────────────────────────────────────────────────────────
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


# ── Core query function (Condition C) ─────────────────────────────────────────
def query_model(convo, repetitions):
    conv_id = str(convo["conv_id"])

    for question in convo["questions"]:
        question["given_answers"] = []
        question["latencies"]     = []

    for rep in range(repetitions):
        # Clear the graph for this conversation before each repetition
        with neo4j_driver.session() as session:
            session.execute_write(clear_conversation_graph, conv_id)

        base_system = f"Answer questions about {convo['seed_entity_text']} in as few words as possible."
        history = [{"role": "system", "content": base_system}]

        for question in convo["questions"]:
            time.sleep(DELAY)
            turn_num      = question["turn"]
            question_text = question["question"]

            # --- Fetch dynamic graph context from previous turns ---
            with neo4j_driver.session() as session:
                graph_context = session.execute_read(get_graph_context, conv_id, turn_num)

            # Rebuild system prompt with updated graph context for this turn
            if graph_context:
                system_content = f"{base_system}\n\n{graph_context}"
                # Update system message in history with latest graph context
                history[0] = {"role": "system", "content": system_content}

            history.append({"role": "user", "content": question_text})

            start = time.time()
            response = request_with_retries(history)
            end   = time.time()

            answer = response.choices[0].message.content or ""
            if answer:
                history.append({"role": "assistant", "content": answer})

            question["given_answers"].append(answer)
            question["latencies"].append(end - start)

            # --- Extract entities and update the graph ---
            entities = extract_entities(answer)
            with neo4j_driver.session() as session:
                session.execute_write(
                    add_turn_to_graph,
                    conv_id, turn_num, question_text, answer, entities
                )

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
neo4j_driver.close()

incomplete = sum(1 for c in slice if not c.get("_query_complete"))
if incomplete:
    print(f"Warning: {incomplete} convos incomplete — rerun to retry.")
else:
    print("All convos in this slice complete.")
