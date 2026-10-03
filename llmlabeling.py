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
API_URL = "http://localhost:8888/v1/chat/completions"
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

SYSTEM_PROMPT = """
Magnetic resonance imaging (MRI) is the gold standard modality for evaluating non-osseous and subchondral internal derangements of the knee joint. A standardized knee MRI protocol relies primarily on proton density fat-saturated (PD-FS), T2-weighted fat-saturated, and un-suppressed T1-weighted sequences acquired in the sagittal, coronal, and axial planes. These pulse sequences allow for precise characterization of soft tissue soft-tissue integrity, hyaline cartilage thickness, bone marrow signal alterations, and fluid dynamics.

---

## 1. Anterior Cruciate Ligament (ACL) Pathologies

### Anatomy and Biomechanics

The anterior cruciate ligament (ACL) is a primary intra-articular, extrasynovial stabilizer of the knee, composed of two functional bundles: the anteromedial (AM) and posterolateral (PL) bundles. It originates from the posteromedial surface of the lateral femoral condyle and inserts into the anterior intercondylar area of the tibia. Its primary biomechanical role is preventing anterior translation of the tibia relative to the femur and restraining internal rotation and valgus stress.

### MRI Findings and Diagnostics

An intact ACL appears as a taut, continuous band of low signal intensity on all pulse sequences, running parallel to the intercondylar roof (Blumensaat's line).

* **Complete Tear or High-Grade Disruption (>50% Fiber Loss):** Characterized by complete physical discontinuity of ligamentous fibers, replacement of the normal ligament contour by a fluid-filled gap, or a wavy, disorganized appearance of residual fibers. Primary direct signs include non-visualization of the ligament, severe fiber fraying with horizontal orientation of the distal stump, or fluid replacing the normal footprint. Secondary signs include an anterior tibial translation $> 5\text{ mm}$, a steep PCL angle, and the classic pivot-shift bone contusion pattern (edema in the lateral femoral condyle and posterior lateral tibial plateau).
* **Low-Grade Sprains, Interstitial Tears, and Mucoid Degeneration:** Low-grade injuries (Grade 1/2 sprains) manifest as periligamentous fluid and internal hyperintensity without significant loss of structural continuity or bundle collapse. Interstitial tears demonstrate linear signal elevation within an intact outer sheath. Mucoid degeneration displays a characteristic "celery stalk" appearance—where the ligament is thickened with diffuse, high T2 signal extending parallel to intact fibers—often accompanied by intra-ligamentous cysts without joint instability.

### Diagnostic Threshold Criteria

For quantitative classification, a definitive pathologic rating (1.00) requires a complete full-thickness disruption or high-grade tearing exceeding 50% of the ligamentous fibers. Conversely, mild sprains, low-grade partial tears ($<50\%$), interstitial tearing, and mucoid degeneration are classified as low-threshold or non-actionable baseline findings (0.00).

---

## 2. Medial Collateral Ligament (MCL) Pathologies

### Anatomy and Biomechanics

The medial collateral ligament (MCL) complex is the chief passive stabilizer against valgus load on the knee. It consists of a superficial layer (tibial collateral ligament), a deep layer (composed of the meniscofemoral and meniscotibial ligaments attached directly to the medial meniscus), and the posterior oblique ligament (POL).

### MRI Findings and Diagnostics

The normal MCL is visualized on coronal T1 and T2/PD-FS images as a thin, continuous, low-signal band extending from the medial femoral epicondyle to the proximal medial tibial metaphysis, roughly 4–5 cm below the joint line.

* **High-Grade, Complete, or Severe Moderate Tears:** Characterized by wavy, discontinuous ligamentous fibers, complete fluid filling across the ligamentous pathway, or proximal/distal avulsion. A Grade III injury demonstrates complete tearing of both superficial and deep layers with significant surrounding soft-tissue fluid extravasation. Moderate (Grade II) high-impact partial disruptions display partial fiber tearing with fluid extending into the ligament sheath and adjacent soft tissues.
* **Low-Grade Sprains and Chronic Changes:** Grade I MCL sprains demonstrate high signal intensity in the soft tissues superficial to the ligament on T2/PD-FS images, but the underlying fibrous band remains uniform, continuous, and of normal thickness. Chronic post-traumatic changes present as smooth focal thickening or calcification at the femoral attachment (Pellegrini-Stieda lesion) without acute fluid accumulation.

### Diagnostic Threshold Criteria

High-grade, complete, or severe moderate disruptions (Grade II/III) that breach ligament continuity meet the criteria for pathology (1.00). In contrast, isolated low-grade Grade I sprains, chronic fibrous thickening, and asymptomatic residual scaring are rated as low-threshold status (0.00).

---

## 3. Medial Meniscus Pathologies

### Anatomy and Biomechanics

The medial meniscus is a fibrocartilaginous, semi-lunar (C-shaped) structure covering approximately 60% of the medial tibial plateau. It absorbs shock, distributes load, and enhances joint congruency. Its posterior horn and posterior root attach firmly to the intercondylar tibia, making it less mobile and more susceptible to shear forces during weight-bearing flexion and rotation compared to the lateral meniscus.

### MRI Findings and Diagnostic Grading

In standard PD-FS and T1 images, a normal meniscus is uniformly low in signal intensity (hypointense).

* **Grade 1 Signal:** Small, non-linear, focal intra-substance high signal intensity that does not contact an articular surface. Represents early intracellular mucoid or hyaline degeneration.
* **Grade 2 Signal:** Linear intra-substance high signal intensity that extends through the meniscal body or horn but stops short of extending to either the superior or inferior articular surface. Represents more advanced horizontal degeneration.
* **Grade 3 Signal (True Surface Tear):** Linear or complex high signal intensity that unequivocally extends to at least one articular surface (superior, inferior, or free inner margin) on at least two consecutive sagittal or coronal images.
* **Displaced Fragments and Root Tears:** Severe tear patterns include bucket-handle tears (where the inner fragment displaces into the intercondylar notch, yielding the "double ACL" sign), flap/parrot-beak tears, and radial root tears. Root tears lead to loss of circumferential hoop stress, resulting in meniscal extrusion ($>3\text{ mm}$ peripheral displacement beyond the tibial margin).

### Diagnostic Threshold Criteria

Only Grade 3 tears reaching the articular surface, radial root disruptions, and displaced structural fragments meet the definitive threshold (1.00). Grade 1 and Grade 2 intrasubstance signals, mucoid changes, and non-surfacing degenerative fissures are categorized as negative/degenerative baseline findings (0.00).

---

## 4. Lateral Meniscus Pathologies

### Anatomy and Biomechanics

The lateral meniscus is more circular and covers a larger portion of the lateral tibial plateau than its medial counterpart. It is loosely attached along its periphery and is interrupted posterolaterally by the popliteal hiatus, through which the popliteus tendon passes. This increased mobility renders it less susceptible to simple degenerative wear from linear weight-bearing, but prone to complex acute mechanical tearing and congenital developmental variations.

### MRI Findings and Diagnostics

On sagittal and coronal planes, the anterior and posterior horns of the lateral meniscus appear equal or subequal in size, creating a "bow-tie" appearance across at least two 3-mm sagittal slices.

* **Articular-Surfacing Tears:** Similar to the medial side, a true lateral meniscal tear requires fluid-bright PD/T2 signal or explicit morphological deformity reaching the superior, inferior, or free-margin articular boundary. Patterns include complex vertical longitudinal tears, radial tears along the free edge, and displaced flap tears.
* **Intrasubstance Degeneration and Discoid Meniscus:** Intrasubstance signal increase without surface involvement represents grade 1/2 mucoid degeneration. A common congenital variant is the discoid lateral meniscus, characterized by continuous meniscal tissue bridging the anterior and posterior horns across three or more 3-mm sagittal slices or extending deep into the intercondylar notch on coronal images.

### Diagnostic Threshold Criteria

Definitive classification (1.00) requires a Grade 3 tear with surface extension or structural fragment displacement. Asymptomatic discoid variants without secondary tears, alongside isolated Grade 1/2 intrasubstance signal changes, fall under the zero-scoring baseline (0.00).

---

## 5. Medial Compartment Osteoarthritis (Medial OA)

### Anatomy and Pathology

The medial compartment comprises the medial femoral condyle and the medial tibial plateau. Due to the normal mechanical axis of the lower extremity passing slightly medial to the knee center (varus alignment force), the medial compartment bears up to 60–70% of static weight-bearing loads, making it the most frequent site for primary osteoarthritis.

### MRI Findings and Chondral Grading

Hyaline cartilage is best assessed on PD-FS sagittal and coronal sequences, where normal cartilage displays an intermediate signal with a smooth, continuous surface interface against hyperintense joint fluid.

* **Grade 1/2 (Mild/Moderate Chondrosis):** Surface irregularity, focal blistering, superficial fissuring, or partial-thickness cartilage defects involving $<50\%$ or $>50\%$ of cartilage depth without reaching subchondral bone.
* **Grade 3/4 (Severe/Full-Thickness Chondral Loss):** Grade 3 represents deep cartilage loss extending $>50\%$ of depth down to the subchondral plate. Grade 4 (full-thickness denudation) presents with complete loss of hyaline cartilage, exposing underlying subchondral bone. This is accompanied by secondary radiographic signs of osteoarthritis: osteophytosis, subchondral bone sclerosis, subchondral cysts (geodes), and reactive marrow edema.

### Diagnostic Threshold Criteria

Severe osteoarthritis (1.00) is defined strictly by the presence of Grade 3/4 lesions or complete full-thickness cartilage denudation within the medial compartment. Mild-to-moderate chondrosis, superficial fissuring, or Grade 1/2 changes are categorized as zero-score findings (0.00).

---

## 6. Lateral Compartment Osteoarthritis (Lateral OA)

### Anatomy and Pathology

The lateral compartment includes the lateral femoral condyle and lateral tibial plateau. Isolated lateral compartment osteoarthritis is less common than medial OA and is frequently secondary to lateral meniscectomy, developmental valgus malalignment, or prior high-energy trauma (such as depressed lateral tibial plateau fractures).

### MRI Findings and Diagnostics

Cartilage loss in the lateral compartment follows the same MRI grading criteria as the medial side.

* **High-Grade Degeneration (Grade 3/4):** Denudation of the lateral femoral or tibial articular cartilage, exposing raw subchondral bone. On T2/PD fat-sat images, full-thickness loss appears as complete loss of intermediate-signal cartilage, allowing joint fluid to directly abut low-signal subchondral cortical bone. Secondary features include marginal osteophytes along the lateral joint line, flattened weight-bearing surfaces, subchondral sclerosis, and reactive subchondral bone marrow edema.
* **Low-Grade Chondromalacia:** Superficial matrix signal alteration, minor focal cartilage swelling, or shallow non-full-thickness surface fissures.

### Diagnostic Threshold Criteria

Full-thickness cartilage loss or Grade 3/4 chondral destruction establishes a positive rating (1.00). Superficial chondral defects, non-full-thickness fissuring, and Grade 1/2 chondromalacia receive a zero rating (0.00).

---

## 7. Patellofemoral Osteoarthritis (PF OA)

### Anatomy and Pathology

The patellofemoral compartment consists of the articulation between the posterior surface of the patella (medial and lateral facets separated by a median ridge) and the anterior trochlear groove of the distal femur. This compartment experiences massive compressive forces during knee flexion (e.g., ascending stairs, deep squatting). Maltracking, patellar tilt, and trochlear dysplasia accelerate degenerative articular loss.

### MRI Findings and Diagnostics

Evaluated primarily in the axial plane on T2/PD fat-saturated images.

* **Severe/Full-Thickness Loss (Grade 3/4):** High-grade chondral degradation characterized by full-thickness loss of hyaline cartilage along the patellar facets (most commonly the median ridge or lateral facet) or the central/lateral trochlear groove. Exposed bone, subchondral cystic transformation, severe osteophytosis along the patellar margins, and reactive marrow edema within the patella or trochlea confirm advanced disease.
* **Mild/Moderate Chondromalacia Patellae:** Grade 1 chondromalacia presents as focal cartilage softening or hyperintensity. Grade 2 involves shallow surface fissuring or partial-thickness defects extending $<50\%$ of cartilage thickness.

### Diagnostic Threshold Criteria

A score of 1.00 requires high-grade or full-thickness cartilage denudation (Grade 3/4). Early-stage softening, superficial fissuring, or Grade 1/2 chondromalacia patellae receive a zero score (0.00).

---

## 8. Joint Effusion

### Physiology and Pathophysiology

Synovial fluid is normally present in tiny volumes (1–5 mL) within the knee joint capsule to lubricate articular surfaces and nourish non-vascularized cartilage. An abnormal elevation in fluid volume—a joint effusion—is a non-specific response to intra-articular pathology, including trauma, infection, inflammatory arthritis, or chronic wear.

### MRI Evaluation and Distinctions

Effusions are characterized on T2-weighted and PD fat-saturated images as bright, hyperintense fluid distending the intra-articular spaces: the suprapatellar pouch, the lateral and medial recesses, and the posterior capsular space.

* **Pathologic Moderate-to-Large Effusion:** Marked expansion of the suprapatellar recess (>10–15 mm in anteroposterior dimension) with convex bowing of the quadriceps tendon anteriorly and displacement of the suprapatellar fat pad. Fluid collections distending the capsular boundaries cause significant joint tightness and mechanical restriction.
* **Physiological or Small Effusion:** Physiological or minimal fluid collections restricted to thin, sliver-like hyperintensities in the suprapatellar pouch ($<5\text{ mm}$ thickness) or deep to the patellar tendon.

### Diagnostic Threshold Criteria

A positive classification (1.00) is assigned strictly when a moderate or large joint effusion is present. Reports specifying "small," "minimal," "trace," "some," or "little" intra-articular fluid are categorized under the zero threshold (0.00).

---

## 9. Synovitis

### Pathophysiology

Synovitis involves chronic or acute inflammation, hypervascularity, and structural hypertrophy of the synovial membrane lining the joint capsule. It can be secondary to osteoarthritic mechanical irritation, crystalline arthropathies (gout/pseudogout), rheumatoid arthritis, or pigmented villonodular synovitis (PVNS).

### MRI Findings and Diagnostics

Synovitis is evaluated on T2-weighted, PD-FS, and contrast-enhanced T1-weighted images.

* **Synovial Thickening and Hypertrophy:** Unlike a simple serous effusion (which appears as homogeneous, low T1 / high T2 fluid), synovitis presents with thick, irregular, nodular, or frond-like synovial projections extending into the joint cavity. On non-contrast PD-FS images, hypertrophic synovium demonstrates intermediate signal intensity that distorts normal capsular margins. Post-contrast T1 FS images demonstrate immediate, intense enhancement of the inflamed synovial tissue, distinguishing it from non-enhancing fluid.
* **Absence of Synovitis:** Smooth, paper-thin lining of the capsule with crystal-clear interface between fluid and capsular wall.

### Diagnostic Threshold Criteria

Explicit mention or visualization of synovial thickening, hypertrophy, or diffuse nodular synovial proliferation fulfills the positive pathological score (1.00). Smooth synovial boundaries without inflammatory thickening are rated as negative/baseline (0.15 if unmentioned, 0.00 if explicitly normal).

---

## 10. Baker's Cyst (Popliteal Cyst)

### Anatomy and Pathophysiology

A Baker's cyst (popliteal cyst) represents a fluid-filled distension of the gastrocnemio-semimembranosus bursa located in the posteromedial popliteal fossa. It communicates with the knee joint through a narrow, valve-like neck situated between the medial head of the gastrocnemius muscle and the semimembranosus tendon. It develops when intra-articular pressure rises secondary to chronic effusions, meniscus tears, or advanced chondral disease.

### MRI Findings and Diagnostics

* **Morphology:** Visualized on axial and sagittal T2/PD-FS images as a well-circumscribed, comma- or crescent-shaped fluid collection extending distally in the posteromedial soft tissues. The pathognomonic diagnostic feature is the identification of the "cyst neck" passing between the semimembranosus tendon and the medial gastrocnemius tendon.
* **Complications:** A ruptured Baker's cyst displays ill-defined hyperintense fluid extending inferiorly along the fascial planes of the calf, mimicking deep vein thrombosis (DVT) clinically.

### Diagnostic Threshold Criteria

The explicit presence of a popliteal cyst or gastrocnemio-semimembranosus bursa distension rates as positive (1.00). Complete absence or negative posteromedial space findings rate as zero (0.00).

---

## 11. Bone Contusion (Osseous Marrow Edema)

### Pathophysiology

Bone contusions (bone bruises) represent microtrabecular microfractures within subchondral cancellous bone, accompanied by intertrabecular hemorrhage and edema. They result from acute compressive or shear impact forces applied to the joint during traumatic events.

### MRI Findings and Diagnostics

Bone marrow edema appears as poorly marginated, geographic areas of high signal intensity on T2-weighted, STIR, or PD fat-saturated sequences, with corresponding ill-defined low signal on T1-weighted sequences.

* **Traumatic Impact Contusion:** Characterized by acute, geographic edema distribution directly corresponding to mechanism of injury. The classic "pivot-shift" contusion displays edema in the mid-to-anterior lateral femoral condyle and posterior lateral tibial plateau following an acute ACL tear. Direct blow contusions display localized edema superficial to cortical bone.
* **Non-Traumatic/Reactive Edema:** Diffuse, chronic subchondral marrow edema directly underlying end-stage cartilage loss (osteoarthritis) or avascular necrosis without acute impact mechanics represents reactive biomechanical stress rather than a traumatic contusion.

### Diagnostic Threshold Criteria

A positive score (1.00) requires marrow edema associated with a clear traumatic impact, impaction mechanism, or acute bone bruise pattern. Non-impact reactive edema or total absence of edema yields a score of 0.00.

---

## 12. Osseous Fracture (Cortical & Subchondral)

### Pathophysiology

A knee fracture represents a structural breach in the osseous architecture. In the knee, fractures involve the distal femoral condyles, tibial plateau, patella, or fibular head. They may range from frank, displaced macrotraumatic cortical breaks to subtle subchondral insufficiency fractures (SIFK).

### MRI Findings and Diagnostics

* **Acute Cortical Fracture:** Characterized by a crisp, low-signal line interrupting the low-signal cortical bone on both T1 and T2 images, frequently accompanied by cortical step-off deformity, articular surface depression, and extensive surrounding marrow edema. Subchondral insufficiency fractures present as a focal, thin, crescentic low-signal line parallel to the articular surface surrounded by severe edema, often progressing to subchondral collapse or osteonecrosis.
* **Stress/Bone Bruise without Fracture:** Isolated marrow edema without a distinct low-signal cortical breach line, step-off deformity, or articular collapse.

### Diagnostic Threshold Criteria

A positive score (1.00) requires an acute cortical breach, structural step-off deformity, displaced cortical fragment, or subchondral fracture accompanied by articular surface collapse. Intact cortical bone, simple bone bruises without fracture lines, and uncollapsed subchondral insufficiency edema rate as 0.00.

---

## Comprehensive Pathology Summary Matrix

| Pathology | Primary MRI Diagnostic Criteria | Threshold Score 1.00 (Positive/Severe) | Threshold Score 0.00 (Negative/Mild/Degenerative) |
| --- | --- | --- | --- |
| **ACL** | Discontinuity on PD-FS/T2; fluid-filled gap; secondary pivot-shift edema. | Complete tear or $>50\%$ fiber disruption. | Grade 1/2 sprain, interstitial tear, mucoid degeneration. |
| **MCL** | Periligamentous fluid; fiber waviness/disruption on coronal views. | High-grade, complete, or severe moderate disruption. | Low-grade Grade I sprain, chronic thickening, intact fibers. |
| **Medial Meniscus** | High PD/T2 signal reaching superior/inferior/free articular surface. | Grade 3 surfacing tear, root disruption, displaced fragment. | Grade 1/2 intrasubstance signal, mucoid degeneration. |
| **Lateral Meniscus** | Signal extension to articular surface on multiple sagittal/coronal slices. | Grade 3 surfacing tear, root tear, bucket-handle displacement. | Grade 1/2 intrasubstance signal, discoid variant without tear. |
| **Medial OA** | Chondral depth loss along medial femoral condyle / tibial plateau. | Grade 3/4 cartilage loss or full-thickness bone exposure. | Grade 1/2 chondromalacia, shallow superficial fissuring. |
| **Lateral OA** | Cartilage loss along lateral condyle / tibial plateau; joint narrowing. | Grade 3/4 cartilage loss or full-thickness bone exposure. | Grade 1/2 chondromalacia, superficial matrix irregularity. |
| **PF OA** | Cartilage loss along patellar facets or trochlear groove on axial images. | Grade 3/4 cartilage loss or full-thickness denudation. | Grade 1/2 chondromalacia patellae, minor fissuring. |
| **Joint Effusion** | High T2 fluid distension of suprapatellar pouch and recesses. | Moderate to large joint effusion. | Small, trace, minimal, or physiological fluid. |
| **Synovitis** | Thickened, nodular, or frond-like synovial membrane enhancement/signal. | Explicit synovial thickening or villous hypertrophy. | Normal, paper-thin capsular lining. |
| **Baker's Cyst** | Fluid collection in posteromedial fossa with neck between semi-M & gastroc. | Confirmed popliteal cyst presence. | Absence of popliteal fluid mass. |
| **Bone Contusion** | Subchondral high T2/STIR marrow edema linked to traumatic mechanics. | Traumatic bone marrow edema from impact/impaction. | Chronic reactive edema adjacent to OA, no impact history. |
| **Fracture** | Cortical disruption line, structural step-off, or articular collapse. | Acute cortical break, step deformity, or subchondral collapse. | Intact cortex, bone contusion without cortical break. |
Role: Expert Multilingual Radiologist Assistant.

Task: Analyze the provided knee MRI report (Greek, English, Spanish, or other languages) and evaluate 12 specific pathologies.

Scoring Logic (Strict Thresholds): Return a probability score between 0.00 and 1.00 based on these strict rules:

    1.00 (Severe/Definitive): The finding is explicitly described as "complete," "full-thickness," "high-grade," "Grade 2/3/4," "displaced," or a "rupture/tear" that reaches the articular surface.
    0.50 (Ambiguous): Use ONLY if the text is contradictory or truly uncertain (e.g., "cannot exclude" or "suggestive of").
    0.00 (Negative/Mild/Degenerative): Use 0.00 for:
        "Normal," "Intact," "Unremarkable," "Preserved."
        "Low-grade," "Grade 1" "Mild," "Incipient", "Small" or similar words for small or insignificant, or little or some or overlookable amount
        "Degeneration," "Intrasubstance signal," "Mucoid changes," "Fissures" (without full-thickness loss).
        "Chondrosis" (unless Grade 2/3/4 or full-thickness loss).
        "Subchondral insufficiency" without a cortical break or step deformity.
    0.15 (Unmentioned/Baseline): Use ONLY if the pathology is not mentioned in the report. (Note: If the report explicitly says "Normal" or "No tear," use 0.00).

Pathology Definitions:

    ACL: 1.0 only if complete tear or >50% fiber disruption. 0.0 for interstitial tears, grade 1/2 sprains, or mucoid degeneration.
    MCL: 1.0 only for high-grade/complete/moderate tears. 0.0 for low-grade sprains or chronic changes.
    Medial Meniscus: 1.0 for tears reaching the articular surface or displaced fragments. 0.0 for intrasubstance degeneration or tears that do not reach the surface.
    Lateral Meniscus: 1.0 for tears reaching the articular surface or displaced fragments. 0.0 for intrasubstance degeneration or tears that do not reach the surface of lateral.
    Medial OA: 1.0 for Grade 3/4 or full-thickness cartilage loss. 0.0 for Grade 1/2 or simple chondrosis/fissures.
    Lateral OA: Same criteria as Medial OA.
    PF OA: Same criteria as Medial OA.
    Effusion: 1.0 if fluid is mentioned and it's not small or some or little.
    Synovitis: 1.0 if synovial thickening or hypertrophy is mentioned.
    Baker's: 1.0 if a popliteal cyst is present.
    Contusion: 1.0 if bone marrow edema is present and associated with an impact/impaction/bruise.
    Fracture: 1.0 only for acute cortical breaks, step deformities, or subchondral fractures with articular collapse.

Instructions:

    Pay extreme attention to negations (e.g., "no evidence of," "without," "intact").
    Watch for Greek and Spanish terms (e.g., "Ruptura," "Ρήξη," "Integras").
    Ensure that if a condition is "Mild" or "Degenerative," the score is 0.00.

Output Format: Return ONLY a valid JSON object: {"ACL": float, "MCL": float, "Medial Meniscus": float, "Lateral Meniscus": float, "Medial OA": float, "Lateral OA": float, "PF OA": float, "Effusion": float, "Synovitis": float, "Baker's": float, "Contusion": float, "Fracture": float}"""


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