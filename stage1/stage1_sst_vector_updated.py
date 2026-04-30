# ============================================================
# STAGE 1B: Emotion Feature Encoding using GoEmotions
# Updated for nego-data-final.csv schema:
# conv_id, turn_id, role, word, speaker_id, duration_min
# ============================================================

import sys
import subprocess

# Install dependencies when running in a fresh Colab environment.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "torch", "tqdm"], check=False)

import os
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "/content/optimal-nego/data/nego-data-final.csv"
OUTPUT_DIR = "/content/optimal-nego/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_PATH = f"{OUTPUT_DIR}/stage1_goemotions_features.csv"

# ----------------------------
# Column configuration
# ----------------------------
# The updated CSV uses `word` as the utterance text column.
# This helper also keeps the script backward-compatible with older files using `text`.
TEXT_COL_CANDIDATES = ["clean_word", "word", "text"]

def pick_text_col(df):
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No text column found. Expected one of {TEXT_COL_CANDIDATES}, "
        f"but found columns: {list(df.columns)}"
    )

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

TEXT_COL = pick_text_col(df)
print(f"Using text column: {TEXT_COL}")

# Preserve conversation order for turn-level modeling
sort_cols = [c for c in ["conv_id", "conversation_id", "turn_id"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# Optional convenience alias for downstream scripts expecting `text`
if "text" not in df.columns:
    df["text"] = df[TEXT_COL]

# ----------------------------
# Load GoEmotions model
# ----------------------------
MODEL_NAME = "SamLowe/roberta-base-go_emotions"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

labels = model.config.id2label
emotion_labels = [labels[i] for i in range(len(labels))]

# Keep all model labels. SamLowe/roberta-base-go_emotions returns 28 labels:
# the 27 GoEmotions emotions plus neutral.
print(f"Loaded GoEmotions model with {len(emotion_labels)} labels:")
print(emotion_labels)

# ----------------------------
# Batch prediction function
# ----------------------------
def predict_goemotions(texts, batch_size=32, max_length=256):
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs)

    return np.vstack(all_probs)

# ----------------------------
# Generate emotion probabilities
# ----------------------------
texts = df[TEXT_COL].tolist()
goemo_probs = predict_goemotions(texts, batch_size=32)

goemo_cols = [f"goemo_{label}" for label in emotion_labels]
goemo_df = pd.DataFrame(goemo_probs, columns=goemo_cols)

# ----------------------------
# Add dominant emotion features
# ----------------------------
goemo_df["goemo_dominant_emotion"] = goemo_df[goemo_cols].idxmax(axis=1).str.replace(
    "goemo_", "", regex=False
)
goemo_df["goemo_dominant_score"] = goemo_df[goemo_cols].max(axis=1)

# Emotion intensity as total probability mass across all 28 labels.
# If you want intensity excluding neutral, replace goemo_cols with
# [c for c in goemo_cols if c != "goemo_neutral"].
goemo_df["goemo_emotion_intensity"] = goemo_df[goemo_cols].sum(axis=1)

# ----------------------------
# Merge with original data
# ----------------------------
stage1_goemo = pd.concat(
    [df.reset_index(drop=True), goemo_df.reset_index(drop=True)],
    axis=1
)

# ----------------------------
# Save output
# ----------------------------
stage1_goemo.to_csv(OUTPUT_PATH, index=False)

print(f"Saved GoEmotions Stage 1 features to: {OUTPUT_PATH}")
stage1_goemo.head()
