import os
import argparse
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import (
    CLIPModel, CLIPProcessor,
    BlipForQuestionAnswering, BlipProcessor,
    ViltProcessor, ViltModel
)
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Questions and answer classes
questions = [
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

classes = [
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

class ClassificationVQADataset(Dataset):
    def __init__(self, dataframe, image_dir, processor, classes):
        self.data = dataframe
        self.image_dir = image_dir
        self.processor = processor
        self.qa_columns = dataframe.columns[2:]
        self.label_encoders = [LabelEncoder().fit(cls) for cls in classes]

    def __len__(self):
        return len(self.data) * len(self.qa_columns)

    def __getitem__(self, idx):
        row_idx = idx // len(self.qa_columns)
        q_idx = idx % len(self.qa_columns)
        row = self.data.iloc[row_idx]
        question = self.qa_columns[q_idx]
        answer = row[question]
        if pd.isna(answer):
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)
        image_path = os.path.join(self.image_dir, row['video_id'], f"{row['video_id']}_frame{row['frame']}.jpg")
        image = Image.open(image_path).convert("RGB")
        label = self.label_encoders[q_idx].transform([str(answer).strip()])[0]
        return {
            "text": question.strip(),
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "question_idx": q_idx
        }

def get_collate_fn(processor):
    def classification_collate(batch):
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]
        labels = torch.stack([item["label"] for item in batch])
        question_idxs = [item["question_idx"] for item in batch]
        processed = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True)
        return {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "pixel_values": processed["pixel_values"],
            "labels": labels,
            "question_idxs": question_idxs
        }
    return classification_collate

class VQAClassifier(nn.Module):
    def __init__(self, model_name, base_model, hidden_dim, num_classes_per_question):
        super().__init__()
        self.name = model_name
        self.base = base_model
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes) for num_classes in num_classes_per_question
        ])

    def forward(self, input_ids, attention_mask, pixel_values, question_idx):
        if self.name == "blip":
            vision_outputs = self.base.vision_model(pixel_values=pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            text_inputs = self.base.text_encoder.embeddings(input_ids=input_ids)
            text_outputs = self.base.text_encoder.encoder(
                hidden_states=text_inputs,
                attention_mask=attention_mask
            )
            pooled = text_outputs.last_hidden_state[:, 0, :]
        elif self.name == "clip":
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            pooled = outputs.text_embeds + outputs.image_embeds
        else:
            outputs = self.base(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            pooled = outputs.last_hidden_state[:, 0, :]
        logits = self.classifiers[question_idx](pooled)
        return logits


def evaluate(model, dataloader, criterion):
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            question_idxs = batch["question_idxs"]

            for i in range(len(input_ids)):
                logits = model(
                    input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0),
                    pixel_values[i].unsqueeze(0),
                    question_idx=question_idxs[i]
                )
                loss = criterion(logits, labels[i].unsqueeze(0))
                val_loss += loss.item()

                preds = logits.argmax(dim=1).cpu().item()
                all_preds.append(preds)
                all_labels.append(labels[i].cpu().item())

    avg_loss = val_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    return avg_loss, f1

def plot_training(history, model_name, best_epoch=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.plot(epochs, history["val_f1"], label="Val F1 Score")

    if best_epoch:
        plt.axvline(best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch})')

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title(f"Training History - {model_name.upper()}")
    plt.legend()
    plt.tight_layout()
    os.makedirs("pt", exist_ok=True)
    plt.savefig(f"../pt_img/{model_name}_training_plot.png")
    plt.show()



def main():
    
    no_improve_count = 0

    parser = argparse.ArgumentParser()
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--model", type=str, choices=["clip", "blip", "vilt"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    
    args = parser.parse_args()

    patience = args.patience 

    model_configs = {
        "clip": {
            "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
            "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            "hidden_dim": 512
        },
        "blip": {
            "model": BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base"),
            "processor": BlipProcessor.from_pretrained("Salesforce/blip-vqa-base"),
            "hidden_dim": 768
        },
        "vilt": {
            "model": ViltModel.from_pretrained("dandelin/vilt-b32-mlm"),
            "processor": ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm"),
            "hidden_dim": 768
        }
    }

    config = model_configs[args.model]
    processor = config["processor"]
    collate_fn = get_collate_fn(processor)

    df = pd.read_csv("../data_csv/train_data.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_ds = ClassificationVQADataset(train_df, "../frames", processor, classes)
    val_ds = ClassificationVQADataset(val_df, "../frames", processor, classes)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    base_model = config["model"].to(device)
    model = VQAClassifier(args.model, base_model, config["hidden_dim"], [len(c) for c in classes]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1 = 0
    best_epoch = -1
    best_model_state = None


    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            question_idxs = batch["question_idxs"]

            losses = []
            for i in range(len(input_ids)):
                logits = model(
                    input_ids[i].unsqueeze(0),
                    attention_mask[i].unsqueeze(0),
                    pixel_values[i].unsqueeze(0),
                    question_idx=question_idxs[i]
                )
                loss = criterion(logits, labels[i].unsqueeze(0))
                losses.append(loss)

            batch_loss = sum(losses) / len(losses)
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += batch_loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_f1 = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            best_model_state = model.state_dict()
            no_improve_count = 0  # reset counter
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    print(f"\nBest Val F1: {best_f1:.4f} at Epoch {best_epoch}")
    torch.save(best_model_state, f"../pt_model/{args.model}_classifier.pt")
    print(f"Model saved to pt_model/{args.model}_classifier.pt")

    plot_training(history, args.model, best_epoch=best_epoch)

if __name__ == "__main__":
    main()


