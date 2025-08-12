import os
import shutil

SRC_DIR = "/app/program"   # where the worker unpacks the participant ZIP in ingestion
DST_DIR = "/app/output"    # becomes prediction_result.zip and later /app/input/res in scoring

def main():
    os.makedirs(DST_DIR, exist_ok=True)

    copied = 0
    for root, _, files in os.walk(SRC_DIR):
        # Preserve relative structure under SRC_DIR in case you expect <stem>/*.wav
        rel_root = os.path.relpath(root, SRC_DIR)
        out_root = os.path.join(DST_DIR, rel_root) if rel_root != "." else DST_DIR
        os.makedirs(out_root, exist_ok=True)

        for f in files:
            if f.startswith("."):
                continue
            if f.lower().endswith(".wav"):
                shutil.copy2(os.path.join(root, f), os.path.join(out_root, f))
                copied += 1

    # Optional: basic guardrail so empty submissions fail clearly
    if copied == 0:
        # Write a small marker; scorer will likely error anyway if nothing is present
        with open(os.path.join(DST_DIR, "EMPTY.txt"), "w") as fp:
            fp.write("No .wav files found in submission.")

if __name__ == "__main__":
    main()