# ============================================================
# STAGE 1B: Emotion Feature Encoding using GoEmotions
# Model-based alternative to lexicon-based emotion features
# ============================================================

!pip install -q transformers torch tqdm

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
# Load data
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Change this if your text column has a different name
TEXT_COL = "text"

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

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

print(f"Loaded GoEmotions model with {len(emotion_labels)} labels:")
print(emotion_labels)

# ----------------------------
# Batch prediction function
# ----------------------------
def predict_goemotions(texts, batch_size=32, max_length=256):
    all_probs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

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

goemo_df = pd.DataFrame(
    goemo_probs,
    columns=[f"goemo_{label}" for label in emotion_labels]
)

# ----------------------------
# Add dominant emotion features
# ----------------------------
goemo_df["goemo_dominant_emotion"] = goemo_df.idxmax(axis=1).str.replace("goemo_", "", regex=False)
goemo_df["goemo_dominant_score"] = goemo_df[[f"goemo_{label}" for label in emotion_labels]].max(axis=1)

# Optional: emotion intensity as total probability mass
goemo_df["goemo_emotion_intensity"] = goemo_df[[f"goemo_{label}" for label in emotion_labels]].sum(axis=1)

# ----------------------------
# Merge with original data
# ----------------------------
stage1_goemo = pd.concat([df.reset_index(drop=True), goemo_df.reset_index(drop=True)], axis=1)

# ----------------------------
# Save output
# ----------------------------
stage1_goemo.to_csv(OUTPUT_PATH, index=False)

print(f"Saved GoEmotions Stage 1 features to: {OUTPUT_PATH}")
stage1_goemo.head()
