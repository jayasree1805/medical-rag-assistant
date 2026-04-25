import pandas as pd

INPUT_FILE = "data/medquad.csv"
OUTPUT_FILE = "data/medical_kb.txt"

MIN_ANSWER_LENGTH = 50    # characters
MAX_ANSWER_LENGTH = 1500  # characters

df = pd.read_csv(INPUT_FILE)

chunks = []
skipped = 0

for _, row in df.iterrows():
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()

    # Skip empty, nan, or out-of-range rows
    if not question or not answer:
        skipped += 1
        continue
    if question == "nan" or answer == "nan":
        skipped += 1
        continue
    if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
        skipped += 1
        continue

    # Question embedded WITH answer so retrieval matches query → question → answer
    # Use a separator that won't appear in medical text
    chunk = f"Q: {question}\nA: {answer}"
    chunks.append(chunk)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(chunks))  # unambiguous separator, not ##

print(f"Saved {len(chunks)} chunks! Skipped {skipped}.")