
import os
import json
import statistics
import pandas as pd


def read_json_file(file_path):
    """Read and return data from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def write_statistics_report(output_path, frames_data, source_label):
    """
    Writes statistics about frame counts to a text file.
    """
    frame_counts = list(frames_data.values())
    num_videos = len(frames_data)

    with open(output_path, 'w') as output_file:
        output_file.write(f"Number of videos: {num_videos}\n")
        output_file.write("\n")
        for video, count in frames_data.items():
            output_file.write(f"{video} has {count} frames\n")

        if num_videos > 0:
            output_file.write("\nFrame Statistics:\n")
            output_file.write(f"Total frames: {sum(frame_counts)}\n")
            output_file.write(f"Minimum frames: {min(frame_counts)}\n")
            output_file.write(f"Maximum frames: {max(frame_counts)}\n")
            output_file.write(f"Mean frames: {statistics.mean(frame_counts):.2f}\n")
            output_file.write(f"Std frames: {statistics.stdev(frame_counts):.2f}\n" if num_videos > 1 else "Std frames: 0\n")

    print(f"[{source_label}] Results written to {output_path}")


def analyze_single_json_file(file_path, output_path):
    """Analyze a single JSON file and write frame stats."""
    data = read_json_file(file_path)
    frames_count = {}

    for video, content in data.items():
        num_frames = len(content["frames"])
        frames_count[video] = num_frames

    write_statistics_report(output_path, frames_count, "TEST")


def analyze_json_folder(folder_path, output_path):
    """Analyze all JSON files in a folder and write frame stats."""
    frames_data = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                data = read_json_file(file_path)
                frames_data[filename] = len(data.get("frames", {}))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
    write_statistics_report(output_path, frames_data, "TRAIN")


def extract_question_fields(json_dir, question_fields, output_csv):
    """Extract specified question fields from all JSON files in a folder and save as CSV."""
    all_data = []
    for filename in os.listdir(json_dir):
        if not filename.endswith(".json"):
            continue

        file_path = os.path.join(json_dir, filename)
        try:
            data = read_json_file(file_path)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {filename}")
            continue

        video_id = filename.replace(".json", "")
        frames_data = data.get("frames", {})

        if not frames_data:
            print(f"No frames found in {filename}, skipping...")
            continue

        for frame, frame_data in frames_data.items():
            frame_info = {"video_id": video_id, "frame": frame}
            for question in question_fields:
                answers = frame_data.get(question)
                frame_info[question] = answers[1] if isinstance(answers, list) and len(answers) > 1 else None
            all_data.append(frame_info)

    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Combined question field data saved to {output_csv}")


def main():
    # Part 1: Analyze single test JSON file
    test_file = 'Test/Annotations/test_questions.json'
    analyze_single_json_file(test_file, 'analysis-texts/output-test.txt')

    # Part 2: Analyze train folder
    train_folder = 'Train/Annotations'
    analyze_json_folder(train_folder, 'analysis-texts/output-train.txt')

    # Part 3: Extract question fields
    question_fields = [
        "What limb is injured?", "Is the patient intubated?", "Where is the catheter inserted?", "Is there bleeding?",
        "Has the bleeding stopped?", "Is the patient moving?", "Is the patient breathing?", "Is there a tourniquet?",
        "Is there a chest tube?", "Are the patient and instruments secured?", "If a limb is missing which one?",
        "Is there mechanical ventilation?", "What is the position of the injury?"
    ]
    extract_question_fields(train_folder, question_fields, 'data/combined-df.csv')


if __name__ == '__main__':
    main()
