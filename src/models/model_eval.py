import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch
from transformers import CLIPProcessor, BlipProcessor, ViltProcessor
from train import ClassificationVQADataset, VQAClassifier, get_collate_fn

# Define constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
QUESTIONS = [
    "What limb is injured?",
    "Is the patient intubated?",
    "Where is the catheter inserted?",
    "Is there bleeding?",
    "Has the bleeding stopped?",
    "Is the patient moving?",
    "Is the patient breathing?",
    "Is there a tourniquet?",
    "Is there a chest tube?",
    "Are the patient and instruments secured?",
    "If a limb is missing which one?",
    "Is there mechanical ventilation?",
    "What is the position of the injury?"
]
CLASSES = [
    ['no limb is injured', 'left leg', 'left arm', 'right leg', 'right arm'],
    ["can't identify", 'no', 'yes'],
    ['no catheter is used', 'lower limb'],
    ['no', 'yes'],
    ['there is no bleeding', 'no', 'yes'],
    ["can't identify", 'yes', 'no'],
    ["can't identify", 'no', 'yes'],
    ['no', 'yes'],
    ['no', 'yes'],
    ['no', 'yes', "can't identify"],
    ['none', 'left arm', 'left leg', 'right leg'],
    ["can't identify", 'no', 'yes'],
    ['thorax', 'throat', "can't identify", 'lower limb', 'abdomen', 'upper limb']
]

# Helper to load model and processor
MODEL_CONFIGS = {
    "clip": ("openai/clip-vit-base-patch32", "../pt_model/clip_classifier.pt"),
    "blip": ("Salesforce/blip-vqa-base", "../pt_model/blip_classifier.pt"),
    "vilt": ("dandelin/vilt-b32-finetuned-vqa", "../pt_model/vilt_classifier.pt")
}


def load_model_and_processor(model_name):
    if model_name == "clip":
        processor = CLIPProcessor.from_pretrained(MODEL_CONFIGS[model_name][0])
        from transformers import CLIPModel
        base_model = CLIPModel.from_pretrained(MODEL_CONFIGS[model_name][0])
        hidden_dim = base_model.config.projection_dim
    elif model_name == "blip":
        processor = BlipProcessor.from_pretrained(MODEL_CONFIGS[model_name][0])
        from transformers import BlipForQuestionAnswering
        base_model = BlipForQuestionAnswering.from_pretrained(MODEL_CONFIGS[model_name][0])
        hidden_dim = base_model.config.text_config.hidden_size
    elif model_name == "vilt":
        processor = ViltProcessor.from_pretrained(MODEL_CONFIGS[model_name][0])
        from transformers import ViltModel
        base_model = ViltModel.from_pretrained(MODEL_CONFIGS[model_name][0])
        hidden_dim = base_model.config.hidden_size
    else:
        raise ValueError("Unsupported model")

    model = VQAClassifier(model_name, base_model, hidden_dim, [len(cls) for cls in CLASSES])
    # model.load_state_dict(torch.load(MODEL_CONFIGS[model_name][1]))
    model.load_state_dict(torch.load(MODEL_CONFIGS[model_name][1], map_location=torch.device(DEVICE)))
    model.to(DEVICE)
    model.eval()

    return model, processor


def evaluate_model(model_name):
    model, processor = load_model_and_processor(model_name)

    test_df = pd.read_csv("../data_csv/test_data.csv")
    collate_fn = get_collate_fn(processor)
    dataset = ClassificationVQADataset(test_df, image_dir="../frames", processor=processor, classes=CLASSES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    label_encoders = [LabelEncoder().fit(cls) for cls in CLASSES]

    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            question_idxs = batch["question_idxs"]

            for i in range(len(input_ids)):
                q_idx = question_idxs[i]
                logits = model(
                    input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0),
                    pixel_values[i].unsqueeze(0),
                    question_idx=q_idx
                )
                pred_idx = logits.argmax(dim=1).item()
                pred_label = label_encoders[q_idx].inverse_transform([pred_idx])[0]
                true_label = label_encoders[q_idx].inverse_transform([labels[i].item()])[0]

                frame_row_idx = (batch_idx * loader.batch_size + i) // len(QUESTIONS)
                row = dataset.data.iloc[frame_row_idx]

                results.append({
                    "video_id": row["video_id"],
                    "frame": row["frame"],
                    "question": QUESTIONS[q_idx],
                    "true_answer": true_label,
                    "predicted_answer": pred_label
                })

    df_results = pd.DataFrame(results)
    output_path = f"../predictions/{model_name}_predictions.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}\n")
    return df_results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VQA model on medical simulation data")
    parser.add_argument("--model", choices=["clip", "blip", "vilt"], required=True, help="Which model to evaluate")
    args = parser.parse_args()
    df_result = evaluate_model(args.model)