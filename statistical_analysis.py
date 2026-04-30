import json
import os
import numpy as np
from scipy import stats

# Change working directory to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# ── Load metrics data ─────────────────────────────────────────────────────────
def load_question_scores(metrics_path):
    """Extract per-question accuracy and consistency scores from cached metrics."""
    with open(metrics_path, "r") as f:
        data = json.load(f)

    # Skip the metrics summary object at index 0 if present
    if isinstance(data[0], dict) and "Overall P@1" in data[0]:
        data = data[1:]

    accuracies   = []
    consistencies = []
    latencies    = []

    for convo in data:
        for q in convo.get("questions", []):
            if q.get("avg_accuracy") is not None:
                accuracies.append(q["avg_accuracy"])
            if q.get("consistency_score") is not None:
                consistencies.append(q["consistency_score"])
            if q.get("avg_latency") is not None:
                latencies.append(q["avg_latency"])

    return np.array(accuracies), np.array(consistencies), np.array(latencies)


print("Loading data...")
acc_a, con_a, lat_a = load_question_scores("Condition A/cached_metrics_A.json")
acc_b, con_b, lat_b = load_question_scores("Condition B/cached_metrics_B.json")
acc_c, con_c, lat_c = load_question_scores("Condition C/cached_metrics_C.json")

print(f"  Condition A: {len(acc_a)} questions")
print(f"  Condition B: {len(acc_b)} questions")
print(f"  Condition C: {len(acc_c)} questions")
print()

# ── Cohen's d ─────────────────────────────────────────────────────────────────
def cohens_d(group1, group2):
    """Calculate Cohen's d effect size between two groups."""
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
    if pooled_std == 0:
        return 0.0
    return mean_diff / pooled_std

def interpret_d(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:   return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else:         return "large"

def interpret_p(p):
    """Interpret p-value significance."""
    if p < 0.001:  return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else:          return "ns"

# ── Pairwise t-tests ──────────────────────────────────────────────────────────
def run_ttest(name, g1, g2, label1, label2):
    t_stat, p_val = stats.ttest_ind(g1, g2)
    d = cohens_d(g1, g2)
    sig = interpret_p(p_val)
    effect = interpret_d(d)
    print(f"  {label1} vs {label2}:")
    print(f"    t = {t_stat:.4f}, p = {p_val:.4f} {sig}")
    print(f"    Cohen's d = {d:.4f} ({effect} effect)")
    print()
    return t_stat, p_val, d

# ── ANOVA ─────────────────────────────────────────────────────────────────────
def run_anova(name, g1, g2, g3):
    f_stat, p_val = stats.f_oneway(g1, g2, g3)
    sig = interpret_p(p_val)
    print(f"  F = {f_stat:.4f}, p = {p_val:.4f} {sig}")
    print()
    return f_stat, p_val

# ── Run all analyses ──────────────────────────────────────────────────────────
print("=" * 60)
print("STATISTICAL ANALYSIS RESULTS")
print("=" * 60)

results = {}

# ── ACCURACY ─────────────────────────────────────────────────────────────────
print("\n── ACCURACY (Overall P@1) ──────────────────────────────────")
print(f"  Mean A: {np.mean(acc_a):.4f} | Mean B: {np.mean(acc_b):.4f} | Mean C: {np.mean(acc_c):.4f}")
print()
print("  One-way ANOVA:")
run_anova("accuracy", acc_a, acc_b, acc_c)
print("  Pairwise t-tests:")
run_ttest("accuracy", acc_a, acc_b, "A", "B")
run_ttest("accuracy", acc_a, acc_c, "A", "C")
run_ttest("accuracy", acc_b, acc_c, "B", "C")

# ── CONSISTENCY ───────────────────────────────────────────────────────────────
print("── CONSISTENCY SCORE ───────────────────────────────────────")
print(f"  Mean A: {np.mean(con_a):.4f} | Mean B: {np.mean(con_b):.4f} | Mean C: {np.mean(con_c):.4f}")
print()
print("  One-way ANOVA:")
run_anova("consistency", con_a, con_b, con_c)
print("  Pairwise t-tests:")
run_ttest("consistency", con_a, con_b, "A", "B")
run_ttest("consistency", con_a, con_c, "A", "C")
run_ttest("consistency", con_b, con_c, "B", "C")

# ── LATENCY ───────────────────────────────────────────────────────────────────
print("── RESPONSE LATENCY ────────────────────────────────────────")
print(f"  Mean A: {np.mean(lat_a):.4f}s | Mean B: {np.mean(lat_b):.4f}s | Mean C: {np.mean(lat_c):.4f}s")
print()
print("  One-way ANOVA:")
run_anova("latency", lat_a, lat_b, lat_c)
print("  Pairwise t-tests:")
run_ttest("latency", lat_a, lat_b, "A", "B")
run_ttest("latency", lat_a, lat_c, "A", "C")
run_ttest("latency", lat_b, lat_c, "B", "C")

# ── Summary table ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SUMMARY — PAIRWISE COMPARISONS")
print("=" * 60)
print("Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
print()

comparisons = [("A vs B", acc_a, acc_b, con_a, con_b, lat_a, lat_b),
               ("A vs C", acc_a, acc_c, con_a, con_c, lat_a, lat_c),
               ("B vs C", acc_b, acc_c, con_b, con_c, lat_b, lat_c)]

col = 18
header = ["Comparison", "Accuracy p", "Acc d", "Consist. p", "Con d", "Latency p", "Lat d"]
print("".join(h.ljust(col) for h in header))
print("-" * (col * len(header)))

for label, a1, a2, c1, c2, l1, l2 in comparisons:
    _, pa, da = run_ttest.__wrapped__(None, a1, a2, "", "") if False else (None, stats.ttest_ind(a1, a2)[1], cohens_d(a1, a2))
    _, pc, dc = None, stats.ttest_ind(c1, c2)[1], cohens_d(c1, c2)
    _, pl, dl = None, stats.ttest_ind(l1, l2)[1], cohens_d(l1, l2)
    row = [
        label,
        f"{pa:.4f}{interpret_p(pa)}", f"{da:.3f}({interpret_d(da)})",
        f"{pc:.4f}{interpret_p(pc)}", f"{dc:.3f}({interpret_d(dc)})",
        f"{pl:.4f}{interpret_p(pl)}", f"{dl:.3f}({interpret_d(dl)})",
    ]
    print("".join(str(c).ljust(col) for c in row))

print()

# ── Save results ──────────────────────────────────────────────────────────────
stat_results = {}
for label, a1, a2, c1, c2, l1, l2 in comparisons:
    _, pa, da = None, float(stats.ttest_ind(a1, a2)[1]), float(cohens_d(a1, a2))
    _, pc, dc = None, float(stats.ttest_ind(c1, c2)[1]), float(cohens_d(c1, c2))
    _, pl, dl = None, float(stats.ttest_ind(l1, l2)[1]), float(cohens_d(l1, l2))
    stat_results[label] = {
        "accuracy":    {"p_value": pa, "cohens_d": da, "significance": interpret_p(pa), "effect": interpret_d(da)},
        "consistency": {"p_value": pc, "cohens_d": dc, "significance": interpret_p(pc), "effect": interpret_d(dc)},
        "latency":     {"p_value": pl, "cohens_d": dl, "significance": interpret_p(pl), "effect": interpret_d(dl)},
    }

with open("statistical_analysis.json", "w") as f:
    json.dump(stat_results, f, indent=2)

print("Results saved to statistical_analysis.json")
