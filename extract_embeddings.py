import json
import os 
import numpy as np
import pandas as pd
import torch
from models import BaseClassifier
from utils import get_subconfig


# Configuration
AUDIO_FOLDER = "data/BSD10k-v1.2/features/clap_audio_embeddings"
TEXT_FOLDER = "data/BSD10k-v1.2/features/clap_text_embeddings"

MODE = "both"  # "audio", "text", or "both"
FOLD = 2  # fold number to process (0-4)
MODEL_WEIGHTS = f"model_BSD10k-v1.2/{MODE}/fold_{FOLD}/best_model.pth"
OUTPUT_FOLDER = f"model_BSD10k-v1.2/{MODE}/fold_{FOLD}/embeddings"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

SAVE_PREDICTIONS = True
PREDICTIONS_PATH = os.path.join(OUTPUT_FOLDER, "predictions.csv")

# Load cat names
cat_dict_json = os.path.join(get_subconfig("output_path"), get_subconfig("class_dict_json"))
with open(cat_dict_json, 'r') as f:
    cat_dict = json.load(f)
id_to_cat = {v: k for k, v in cat_dict.items()}
cat_names = [id_to_cat.get(i, str(i)) for i in range(len(cat_dict))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
checkpoint = torch.load(MODEL_WEIGHTS, weights_only=True)
config = checkpoint["config"]

model = BaseClassifier(
    hidden_size=config["hidden_size"],
    num_classes=config["num_classes"],
    emb_size_audio=config["emb_size_audio"],
    emb_size_text=config["emb_size_text"],
    mode=config["mode"]
)
model.load_state_dict(checkpoint["model_state"])
model.to(device).eval()


def get_file_list(mode):
    def _npy(folder):
        return set(f for f in os.listdir(folder) if f.lower().endswith(".npy"))

    if mode == "audio":
        return sorted(_npy(AUDIO_FOLDER))
    elif mode == "text":
        return sorted(_npy(TEXT_FOLDER))
    else:
        return sorted(_npy(AUDIO_FOLDER) & _npy(TEXT_FOLDER))

def run_inference(filename):
    audio_tensor = text_tensor = None

    if MODE in ("audio", "both"):
        audio_emb = np.load(os.path.join(AUDIO_FOLDER, filename)).astype(np.float32)
        audio_tensor = torch.tensor(audio_emb).unsqueeze(0).to(device)
    if MODE in ("text", "both"):
        text_emb = np.load(os.path.join(TEXT_FOLDER, filename)).astype(np.float32)
        text_tensor = torch.tensor(text_emb).unsqueeze(0).to(device)

    with torch.no_grad():
        z, logits, _ = model(audio_emb=audio_tensor, text_emb=text_tensor)

    return z.cpu().numpy()[0], logits

def get_prediction(filename, logits):
    pred_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]
    score = torch.softmax(logits, dim=1).cpu().numpy()[0].max()
    sound_id = int(filename.split(".")[0])
    return {
        "sound_id": sound_id,
        "prediction_cat": cat_names[pred_idx],
        "prediction_idx": int(pred_idx),
        "prediction_score": float(score)
    }

def process_files(files):
    all_predictions = [] if SAVE_PREDICTIONS else None
    for filename in files:
        embedding_np, logits = run_inference(filename)
        np.save(os.path.join(OUTPUT_FOLDER, filename), embedding_np)
        if SAVE_PREDICTIONS:
            all_predictions.append(get_prediction(filename, logits))
    return all_predictions

def save_predictions(predictions):
    pd.DataFrame(predictions).to_csv(PREDICTIONS_PATH, index=False)


if __name__ == "__main__":
    files = get_file_list(MODE)
    predictions = process_files(files)
    print("Saved embeddings to:", OUTPUT_FOLDER)

    if SAVE_PREDICTIONS:
        save_predictions(predictions)
        print("Saved predictions to:", PREDICTIONS_PATH)

