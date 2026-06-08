from pathlib import Path
import pandas as pd

INPUT_DIR = Path("hanwoo")

OUTPUT_DIR = Path("hanwoo_sample")
OUTPUT_DIR.mkdir(exist_ok=True)

# 최종 저장할 데이터 개수
N_ROWS = 1000

# 한 번에 읽을 행 개수
CHUNK_SIZE = 100_000

RANDOM_STATE = 42

for csv_path in INPUT_DIR.glob("*.csv"):
    print(f"Processing: {csv_path.name}")

    sampled_chunks = []

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
        sample_size = min(N_ROWS, len(chunk))
        sampled_chunk = chunk.sample(n=sample_size, random_state=RANDOM_STATE)
        sampled_chunks.append(sampled_chunk)

    sampled_df = pd.concat(sampled_chunks, ignore_index=True)
    final_sample_size = min(N_ROWS, len(sampled_df))
    sampled_df = sampled_df.sample(n=final_sample_size, random_state=RANDOM_STATE)
    
    output_path = OUTPUT_DIR / f"sample_{csv_path.name}"
    sampled_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_path}")

print("Done.")