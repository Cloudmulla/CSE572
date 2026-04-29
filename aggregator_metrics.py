import os
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
import spacy
import numpy as np
import unicodedata

# Change working directory to be same as script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ── Load models (shared across all conditions) ────────────────────────────────
print("Loading NLP models...")
sentence_transformer_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
spacy_model = spacy.load("en_core_web_sm")
similarity_threshold = 0.8
print("Models loaded.\n")

# ── Data loading helpers ──────────────────────────────────────────────────────
def load_condition_a(path="Condition A/cached_queries.json"):
    """Load Condition A results from a single file."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping Condition A.")
        return None
    with open(path, "r") as f:
        data = json.load(f)
    # Skip metrics object if present at index 0
    if isinstance(data[0], dict) and "Overall P@1" in data[0]:
        data = data[1:]
    completed = [c for c in data if c.get("_query_complete")]
    print(f"  Condition A: {len(completed)} completed conversations loaded.")
    return completed


def load_condition_slices(cache_dir, num_slices=4):
    """
    Load and merge all slice files for Condition B or C.
    Expects files named slice_0_4.json, slice_1_4.json, etc.
    """
    if not os.path.exists(cache_dir):
        print(f"  Warning: {cache_dir} not found, skipping.")
        return None

    all_convos = []
    for i in range(num_slices):
        filepath = os.path.join(cache_dir, f"slice_{i}_{num_slices}.json")
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found — slice {i} missing.")
            continue
        with open(filepath, "r") as f:
            slice_data = json.load(f)
        completed = [c for c in slice_data if c.get("_query_complete")]
        print(f"    slice_{i}: {len(completed)}/{len(slice_data)} completed.")
        all_convos.extend(completed)

    print(f"  Total: {len(all_convos)} conversations loaded.")
    return all_convos if all_convos else None


# ── Metric helpers ────────────────────────────────────────────────────────────
def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def similarity_helper(sentences):
    # Filter out empty strings to avoid encoding errors
    sentences = [s for s in sentences if s and s.strip()]
    if len(sentences) < 2:
        return 0.0
    embeddings = sentence_transformer_model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)
    similarities = cosine_similarity(embeddings, [centroid])
    return float(np.mean(similarities))


def accuracy_helper(question, answer_index):
    raw_answer = question["given_answers"][answer_index]
    if not raw_answer or not raw_answer.strip():
        return 0

    given    = normalize_text(raw_answer)
    expected = normalize_text(question["answer_text"])
    full_question = question.get("completed_question", question["question"])

    if expected == given:
        return 1

    expected_entities = {e.text.lower() for e in spacy_model(expected).ents}
    given_entities    = {e.text.lower() for e in spacy_model(given).ents}
    if expected_entities and expected_entities.issubset(given_entities):
        return 1

    similarity_score = similarity_helper([expected, given])
    if similarity_score >= similarity_threshold:
        return 1

    similarity_score = similarity_helper([f"{full_question} {expected}", given])
    if similarity_score >= similarity_threshold:
        return 1

    return 0


# ── Per-condition metric computation ─────────────────────────────────────────
def compute_metrics(convos, condition_name):
    print(f"\nComputing metrics for {condition_name}...")
    questions = [q for convo in convos for q in convo["questions"]]

    # Filter out questions with no answers
    questions = [q for q in questions if q.get("given_answers")]

    # ── Accuracy ──
    accuracies       = []
    accuracies_multihop = []
    for question in questions:
        question["accuracies"] = []
        for i, _ in enumerate(question["given_answers"]):
            question["accuracies"].append(accuracy_helper(question, i))
        question["avg_accuracy"] = sum(question["accuracies"]) / len(question["accuracies"])
        accuracies.append(question["avg_accuracy"])
        if question["turn"] != 0:
            accuracies_multihop.append(question["avg_accuracy"])

    overall_p1   = sum(accuracies) / len(accuracies) if accuracies else 0
    multihop_p1  = sum(accuracies_multihop) / len(accuracies_multihop) if accuracies_multihop else 0

    # ── Consistency ──
    consistency_scores = []
    for question in questions:
        score = similarity_helper(question["given_answers"])
        question["consistency_score"] = score
        consistency_scores.append(score)
    consistency = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

    # ── Latency ──
    latencies = []
    for question in questions:
        if question.get("latencies"):
            avg_lat = sum(question["latencies"]) / len(question["latencies"])
            question["avg_latency"] = avg_lat
            latencies.append(avg_lat)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    metrics = {
        "Condition":         condition_name,
        "Conversations":     len(convos),
        "Overall P@1":       round(overall_p1, 4),
        "Multi-hop P@1":     round(multihop_p1, 4),
        "Consistency Score": round(consistency, 4),
        "Avg. Latency (s)":  round(avg_latency, 3),
    }

    return metrics, convos


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data for all conditions...")
print("=" * 60)

conditions_data = {}

# Condition A — single file
print("\nCondition A:")
data_a = load_condition_a("Condition A/cached_queries.json")
if data_a:
    conditions_data["A"] = data_a

# Condition B — 4 slices
print("\nCondition B:")
data_b = load_condition_slices("Condition B/cached_queries_b", num_slices=4)
if data_b:
    conditions_data["B"] = data_b

# Condition C — 4 slices
print("\nCondition C:")
data_c = load_condition_slices("Condition C/cached_queries_c", num_slices=4)
if data_c:
    conditions_data["C"] = data_c

if not conditions_data:
    print("\nNo data found. Make sure the cached results are in the correct folders.")
    exit(1)

# ── Compute metrics for each available condition ──────────────────────────────
all_metrics = []
updated_data = {}

for condition_key, convos in conditions_data.items():
    metrics, updated_convos = compute_metrics(convos, f"Condition {condition_key}")
    all_metrics.append(metrics)
    updated_data[condition_key] = updated_convos

# ── Save results per condition ────────────────────────────────────────────────
for condition_key, convos in updated_data.items():
    output_path = f"Condition {condition_key}/cached_metrics_{condition_key}.json"
    os.makedirs(f"Condition {condition_key}", exist_ok=True)
    output = [m for m in all_metrics if m["Condition"] == f"Condition {condition_key}"] + convos
    with open(output_path, "w") as f:
        json.dump(output, f)
    print(f"\nSaved metrics for Condition {condition_key} to {output_path}")

# ── Print comparison table ────────────────────────────────────────────────────
print("\n")
print("=" * 60)
print("RESULTS COMPARISON TABLE")
print("=" * 60)

col_width = 20
headers = ["Metric", "Cond. A", "Cond. B", "Cond. C"]
print("".join(h.ljust(col_width) for h in headers))
print("-" * (col_width * len(headers)))

metric_keys = ["Conversations", "Overall P@1", "Multi-hop P@1", "Consistency Score", "Avg. Latency (s)"]
for key in metric_keys:
    row = [key]
    for cond in ["A", "B", "C"]:
        m = next((x for x in all_metrics if x["Condition"] == f"Condition {cond}"), None)
        row.append(str(m[key]) if m else "N/A")
    print("".join(str(cell).ljust(col_width) for cell in row))

print("=" * 60)

# Save combined summary
summary_path = "metrics_summary.json"
with open(summary_path, "w") as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nSummary saved to {summary_path}")
