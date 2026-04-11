import os
import re
import json

TEST_DIR = r"c:\Users\ilhan\Desktop\SER_PROJECT\Test\TestsResults"
FRONTEND_DATA_PATH = r"c:\Users\ilhan\Desktop\SER_PROJECT\Frontend\src\data\realWorldResults.ts"

mastermind_txt = os.path.join(TEST_DIR, "mastermind_benchmark_results.txt")

with open(mastermind_txt, "r", encoding="utf-8") as f:
    text = f.read()

# Parse accuracy
acc_match = re.search(r"Accuracy\s*:\s*([\d.]+)", text)
accuracy = float(acc_match.group(1)) if acc_match else 0.0

# Parse precision, recall, fscore
lines = text.split("\n")
parsed_metrics = []

in_table = False
for line in lines:
    if "Angry" in line and "|" in line:
        in_table = True
    if in_table and "----" in line:
        in_table = False
    if in_table:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 4:
            parsed_metrics.append({
                "emotion": parts[0],
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1": float(parts[3])
            })

ts_content = f"""// AUTO-GENERATED FROM TEST RESULTS
export const MastermindMetrics = {{
    accuracy: {accuracy},
    metrics: {json.dumps(parsed_metrics, indent=4)}
}};
"""
os.makedirs(os.path.dirname(FRONTEND_DATA_PATH), exist_ok=True)
with open(FRONTEND_DATA_PATH, "w", encoding="utf-8") as f:
    f.write(ts_content)
    
print("Data Extraction Complete!")
