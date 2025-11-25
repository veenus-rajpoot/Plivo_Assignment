import json
import argparse
import torch
import os
from transformers import AutoTokenizer
from model import create_model
from labels import ID2LABEL, label_is_pii


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)

        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end

        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def load_custom_model(model_dir, device):
    # Load config.json
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    model_name = config["model_name"]

    # Create same model architecture as training
    model = create_model(model_name)

    # Load PyTorch weights
    state_path = os.path.join(model_dir, "pytorch_model.bin")
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------------------------
    # Load custom model + tokenizer
    # ---------------------------
    model, model_name = load_custom_model(args.model_dir, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            )

            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = out["logits"][0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)

            ents = []
            for s, e, lab in spans:
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })

            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
