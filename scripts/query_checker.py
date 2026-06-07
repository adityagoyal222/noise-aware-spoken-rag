import pandas as pd
from pathlib import Path

chunks_df = pd.concat(
    [pd.read_csv(p) for p in Path("data/aligned_chunks").glob("*_aligned.csv")],
    ignore_index=True,
)
queries_df = pd.read_csv("data/eval/retrieval_eval_queries.csv")

all_meetings = set(chunks_df["file_id"].unique())
covered = set(queries_df["audio_id"].unique())
missed = all_meetings - covered

print(f"Total meetings: {len(all_meetings)}")
print(f"Covered: {len(covered)}")
print(f"Missed: {len(missed)}")

# Check language distribution in missed meetings
for aid in sorted(missed)[:10]:
    sample = chunks_df[chunks_df["file_id"] == aid]["text"].head(3).tolist()
    print(f"\n{aid}:")
    for t in sample:
        print(f"  {t[:80]}")