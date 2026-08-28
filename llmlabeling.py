import csv
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "train.csv"
OUTPUT_CSV = "llm_generated_labels.csv"
MAX_WORKERS = 4  # Number of parallel API requests

# Local LLM Server Endpoint (Ollama / LM Studio / vLLM)
API_URL = "http://localhost:8080/v1/chat/completions"
MODEL_NAME = "qwen3.8:27b"  # Recommended model for Greek & general multilingual medical report parsing

TARGET_COLS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

DEFAULT_VALUES = {col: 0.15 for col in TARGET_COLS}

SYSTEM_PROMPT = """You are an expert multilingual radiologist assistant. Analyze the given knee MRI report (written in Greek or English) and evaluate the presence of 12 pathologies.

Translate findings into probability scores between 0.00 and 1.00:
- 0.90 to 1.00: Explicitly present / positive finding.
- 0.50: Ambiguous or uncertain text.
- 0.15: unmentioned baseline.
- 0.00: Explicitly negative or ruled out or Normal or intact.
ACL tear: A high-grade partial or full-thickness tear of the anterior cruciate ligament, meaning complete discontinuity of the ligament, or more than 50 percent of fibers disrupted, with or without secondary signs such as characteristic pivot-shift bone contusions. Mild signal change, degeneration, or thickening without discontinuity is graded negative.

MCL tear: A high-grade partial or complete acute tear of the medial collateral ligament, with disrupted fibers and edema within and adjacent to the ligament. Low-grade sprains and chronic or remote stress changes are graded negative.

Medial meniscus tear: Abnormal signal that definitely contacts the meniscal surface on at least two images, or a morphologic abnormality such as a truncated, diminutive, or displaced fragment, involving the medial meniscus. Intrasubstance degeneration that does not reach the surface is negative.

Lateral meniscus tear: The same criteria applied to the lateral meniscus.

Medial compartment osteoarthritis: A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than 50 percent of cartilage thickness, in the medial compartment, with or without underlying subchondral marrow changes.

Lateral compartment osteoarthritis: The same criteria applied to the lateral compartment.

Patellofemoral compartment osteoarthritis: The same criteria applied to the patellofemoral compartment.

Joint effusion: A moderate or large amount of fluid distending the joint.

Synovitis: Inflammation and thickening of the synovial lining of the joint.

Baker (popliteal) cyst: A moderate or large fluid collection in the characteristic location behind the knee.

Contusion: A bone contusion, seen as bone marrow edema-like signal from impact, without a discrete fracture line.

Acute fracture: An acute cortical break or fracture line.

Each study in the annotated reference set was independently labeled by two subspecialty-trained MSK radiologists, with disagreements adjudicated by a third radiologist to produce a single consensus ground truth. Labels are assigned at the level of the whole examination, for a single knee.

pay attention to negation words like 
NEGATIONS = [
    r'no\b', r'without', r'sin', r'ausente', r'normal', r'intact', r'intacto',
    r'salim', r'yok', r'denied', r'unremarkable', r'negativo', r'negative', r'kein', r'keine', r'preservad'
]
also watchout for greek reports

Return ONLY a valid JSON object mapping each exact English key to a float (0.00 to 1.00):
"ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"

Do not add extra text, notes, or markdown formatting outside the raw JSON."""


def query_local_llm(report_text: str) -> dict:
    """Send report to local OpenAI-compatible endpoint and parse returned JSON."""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Analyze this report:\n"""\n{report_text}\n"""',
            },
        ],
        "reasoning_effort": "xhigh",
    "chat_template_kwargs": {
        "enable_thinking": True,
        "preserve_thinking": True,
        },
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=600,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"].strip()

        if "```" in content:
            content = re.sub(r"```json|```", "", content).strip()

        parsed_scores = json.loads(content)

        result = {}
        for col in TARGET_COLS:
            val = parsed_scores.get(col, 0.15)
            result[col] = (
                float(val) if isinstance(val, (int, float, str)) else 0.15
            )
        return result

    except Exception as e:
        print(f"\n[Warning] API call failed or JSON parsing error: {e}")
        return None


def process_report(row, writer, csv_file, csv_lock):
    """Processes an individual row and safely writes the output under a thread lock."""
    uid = str(row["StudyInstanceUID"])
    report = row["Report"]

    scores = query_local_llm(report)
    if scores is not None:
        record = {"StudyInstanceUID": uid, **scores}

        # Acquire lock for thread-safe CSV writing
        with csv_lock:
            writer.writerow(record)
            csv_file.flush()

    return uid


def main():
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    print(f"Loaded {len(df)} total reports from {INPUT_CSV}")

    # Checkpoint Detection: Track already processed StudyInstanceUIDs
    processed_uids = set()
    file_exists = (
        os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    )

    if file_exists:
        try:
            existing_df = pd.read_csv(
                OUTPUT_CSV, usecols=["StudyInstanceUID"], encoding="utf-8-sig"
            )
            processed_uids = set(existing_df["StudyInstanceUID"].astype(str))
            print(
                f"Resuming progress: Found {len(processed_uids)} completed items in {OUTPUT_CSV}."
            )
        except Exception as e:
            print(f"Could not read existing checkpoint file ({e}). Starting fresh.")
            file_exists = False

    # Filter out rows already completed
    unprocessed_rows = [
        row
        for _, row in df.iterrows()
        if str(row["StudyInstanceUID"]) not in processed_uids
    ]
    print(f"Remaining items to process: {len(unprocessed_rows)}")

    if not unprocessed_rows:
        print("All reports are already processed!")
        return

    fieldnames = ["StudyInstanceUID"] + TARGET_COLS
    csv_lock = threading.Lock()

    with open(
        OUTPUT_CSV, mode="a", newline="", encoding="utf-8-sig"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
            csv_file.flush()

        # Concurrent Execution Pool (4 Workers)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    process_report, row, writer, csv_file, csv_lock
                )
                for row in unprocessed_rows
            ]

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Extracting Labels ({MAX_WORKERS} Threads)",
            ):
                pass

    print(
        f"\nProcessing completed or caught up! Output saved to {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()