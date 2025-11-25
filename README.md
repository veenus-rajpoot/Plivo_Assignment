🚀 PII Entity Recognition for Noisy STT Transcripts
IIT Bombay – Plivo Coding Assignment
This repository implements a token-level Named Entity Recognition (NER) system designed to detect PII (Personally Identifiable Information) from noisy Speech-to-Text (STT) transcripts.
The system is optimized for:
High PII precision
Low-latency (< 20ms p95 on CPU)
Noisy / imperfect ASR inputs
Real-world deployment feasibility
📁 Project Structure
pii_ner_assignment_IITB/
│
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── stress.jsonl
│   └── test.jsonl
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── eval_span_f1.py
│   ├── measure_latency.py
│   ├── labels.py
│   ├── generate_dataset.py
│   └── utils.py (if needed)
│
├── out/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   └── predictions…
│
├── README.md
└── requirements.txt
🎯 Objective
Build a learned sequence labeling model that:
Detects 7 entity types:
CREDIT_CARD
PHONE
EMAIL
PERSON_NAME
DATE
CITY
LOCATION
Flags PII entities
Produces character-level spans
Handles noisy STT:
spelled-out digits (“eight seven five…”)
no punctuation
“dot”, “at”, “underscore”, etc.
Meets required performance:
PII precision ≥ 0.80
CPU latency p95 ≤ 20 ms
🧠 Model Architecture
Base Model
DistilBERT encoder (fast and lightweight)
Custom Classification Head
Dropout
Linear
ReLU
Linear
Why a Custom Model?
The default HuggingFace classification head wasn’t robust for noisy patterns.
The custom model:
Improves boundary detection
Reduces overfitting
Handles synthetic noisy inputs better
🧪 Dataset Summary
✔ Synthetic Noisy Data
Generated using generate_dataset.py:
Split	Samples	Notes
train	1000	Noisy STT patterns
dev	200	Same distribution
stress	100	Very noisy, realistic ASR errors
Noisy Patterns Included
Spelled-out digits
“dot” / “at” emails
“underscore”, “dash”, “space”
Typos + random fillers
STT-style numbers & dates
🏋️ Training Details
Hyperparameter	Value
Model	DistilBERT + Custom Head
Epochs	3
Batch Size	8
LR	5e-5
Max Length	256
Device	CPU
Training Command
python src/train.py \
--model_name distilbert-base-uncased \
--train data/train.jsonl \
--dev data/dev.jsonl \
--out_dir out \
--batch_size 8 \
--epochs 3 \
--lr 5e-5 \
--max_length 256 \
--device cpu
📈 Evaluation Results
✅ Dev Set (200 samples)
Model performs perfectly on in-distribution data.
CITY            P=1.000 R=1.000 F1=1.000
CREDIT_CARD     P=1.000 R=1.000 F1=1.000
DATE            P=1.000 R=1.000 F1=1.000
EMAIL           P=1.000 R=1.000 F1=1.000
LOCATION        P=1.000 R=1.000 F1=1.000
PERSON_NAME     P=1.000 R=1.000 F1=1.000
PHONE           P=1.000 R=1.000 F1=1.000

Macro-F1: 1.000  
PII Precision: **1.00**  
PII Recall:    **1.00**  
PII F1:        **1.00**
⚠️ Stress Set (100 samples) – Hard Noisy Test
CITY            P=0.649 R=0.600 F1=0.623
CREDIT_CARD     P=0.056 R=0.025 F1=0.034
DATE            P=0.000 R=0.000 F1=0.000
EMAIL           P=0.000 R=0.000 F1=0.000
PERSON_NAME     P=0.238 R=1.000 F1=0.385
PHONE           P=0.300 R=0.675 F1=0.415

Macro-F1: 0.243

PII-only metrics:
Precision = **0.258**  
Recall    = **0.495**  
F1        = **0.339**
Interpretation
Dev = Good (in-distribution)
Stress = Hard (very noisy STT)
Improvements possible with:
more aggressive synthetic noise
CRF layer
regex/normalization post-processing
⚡ Latency Benchmark
Measured using:
python src/measure_latency.py \
--model_dir out \
--input data/dev.jsonl \
--runs 50
Result:
p50: 13.29 ms  
p95: 14.55 ms
✔ Passes p95 ≤ 20ms requirement.
🔮 Future Improvements
Add CRF layer for tighter BIO decoding
Add regex-based validation:
email must contain “@”, “.”
phone ≥ 10 digits
credit card digits count (13–19)
Add stronger noise augmentations:
homophones
deletion noise
merging words
