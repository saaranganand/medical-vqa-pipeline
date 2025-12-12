import os
import re
import random
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    BlipProcessor, BlipForQuestionAnswering,
    CLIPProcessor, CLIPModel,
    AutoProcessor, Qwen2_5_VLForConditionalGeneration
)
import argparse

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Prompt templates

# prompts = {
#     "prompt1": lambda q, a: f"{q}",
#     "prompt2": lambda q, a: f"{q} Choose from: {', '.join(a)}. Respond with one or 'NA: Cannot be determined'.",
#     "prompt3": lambda q, a: f"The image shows a medical simulation. {q} Options: {', '.join(a)}. Choose one or 'NA: Cannot be determined'.",
#     "prompt4": lambda q, a: f"Analyze this clinical simulation image. {q} Select from: {', '.join(a)} or say 'NA: Cannot be determined'.",
#     "prompt5": lambda q, a: f"From the egocentric medical view, answer: {q} Available: {', '.join(a)}. Reply with one or 'NA: Cannot be determined'."
# }

prompts = {
    "prompt1": lambda q, a: f"Carefully examine the image and answer this medical question based solely on what is visually observable. Respond with the most likely answer based on the scene: {q}",
    "prompt2": lambda q, a: f"Observe the image and answer the medical question: {q}. Choose *only one* answer from these options: {', '.join(a)}. If no answer is visually inferable, reply 'NA: Cannot be determined'.""",
    "prompt3": lambda q, a: f"You are analyzing an emergency trauma scene image captured from a video recorded in a high-stress environment (e.g., mannequin simulation, white sheet background, first responder POV). Given this visual context, answer the question: {q}. Choose *only one* answer from the options below: {', '.join(a)}. If the image lacks sufficient evidence to determine an answer, say 'NA: Cannot be determined'",
    "prompt4": lambda q, a: f"You are a medical AI assistant helping triage trauma patients using scene images from training simulations. These images show patients with or without injuries in settings with mannequins, white sheets, and trauma instruments. Analyze the given image and answer this question based solely on what is visible: {q}. Select one answer from the following: {', '.join(a)}. Answer with *exactly* one choice. If it’s not possible to determine the correct option from the image alone, say: 'NA: Cannot be determined'."""
}

# Load all models once
models = {
    "vilt": {
        "type": "transformers",
        "model": ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device).eval(),
        "processor": ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    },
    "blip": {
        "type": "transformers",
        "model": BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval(),
        "processor": BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    },
    "clip": {
        "type": "clip",
        "model": CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval(),
        "processor": CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    }
}

# VQA logic

def process_image(img_path, img_name, prompt_id, model_id):
    print(f"Processing {img_name} with {model_id}/{prompt_id}")
    image = Image.open(img_path).convert("RGB")
    row = {"Image": img_name, "Model": model_id, "Prompt": prompt_id}
    match = re.search(r'_frame(\d+)', img_name)
    row["Frame"] = int(match.group(1)) if match else None

    model_info = models[model_id]
    model_type = model_info["type"]

    for idx, question in enumerate(questions):
        answers = classes[idx]
        prompt = prompts[prompt_id](question, answers)

        if model_id == "blip":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device).eval()
            inputs = processor(image, prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs)
            predicted_answer = processor.tokenizer.decode(out[0], skip_special_tokens=True)

        elif model_id == "vilt":
            processor = model_info["processor"]
            model = model_info["model"]
            encoding = processor(image, prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=40).to(device)
            with torch.no_grad():
                outputs = model(**encoding)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            predicted_answer = model.config.id2label.get(predicted_idx, str(predicted_idx))

        elif model_id == "clip":
            processor = model_info["processor"]
            model = model_info["model"]
            inputs = processor(text=answers, images=image, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            predicted_idx = probs.argmax(-1).item()
            predicted_answer = answers[predicted_idx]

        else:
            predicted_answer = "Unsupported model"

        row[question] = predicted_answer

    return row

# Sampling and parallel processing for all prompts

def process_sample_all_prompts(frames_root, model_id, output_dir, sample_size=100):
    image_paths = [(os.path.join(subdir, file), file)
                   for subdir, _, files in os.walk(frames_root)
                   for file in files if file.lower().endswith((".jpg", ".jpeg", ".png"))]

    # sample = random.sample(image_paths, min(sample_size, len(image_paths)))
    all_tasks = [(img_path, img_name, prompt_id, model_id)
                for prompt_id in prompts
                for (img_path, img_name) in image_paths]

    # with ThreadPoolExecutor() as executor:
    #     results = list(executor.map(lambda args: process_image(*args), all_tasks))
    results = [process_image(*args) for args in all_tasks]

    df = pd.DataFrame(results)
    df = df[["Image", "Frame", "Model", "Prompt"] + questions]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_id}_all_prompts_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} results to {output_file}")

# Example usage
def main():
    parser = argparse.ArgumentParser(description="Zero-shot VQA on trauma images using vision-language models.")
    parser.add_argument("--model", type=str, choices=["vilt", "blip", "clip"], required=True, help="Model to use.")

    args = parser.parse_args()


    # csv_path = "../analysis/data/combined-df.csv"
    csv_path = "../data_csv/test_data.csv"

    frames_folder = "frames"
    model_choice = args.model # "vilt", "blip", "clip"

    # output_dir = "outputs_prompt_set1"
    output_dir = "outputs_prompt_set2"

    # Load test.csv and create the list of relevant image paths
    df_test = pd.read_csv(csv_path)
    image_paths = []

    for _, row in df_test.iterrows():
        video_id = row["video_id"]
        frame = row["frame"]
        img_name = f"{video_id}_frame{frame}.jpg"
        img_path = os.path.join(frames_folder, video_id, img_name)
        if os.path.exists(img_path):
            image_paths.append((img_path, img_name))
        else:
            print(f"Warning: Missing image at {img_path}")

    # Run VQA on these images for all prompts
    all_tasks = [(img_path, img_name, prompt_id, model_choice)
                 for prompt_id in prompts
                 for (img_path, img_name) in image_paths]
    
    print(all_tasks)

    results = [process_image(*args) for args in all_tasks]

    df = pd.DataFrame(results)
    df = df[["Image", "Frame", "Model", "Prompt"] + questions]
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_choice}_all_prompts_results.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} results to {output_file}")

if __name__ == "__main__":
    main()
