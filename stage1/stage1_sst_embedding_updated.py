"""
Stage 1: Semantic Space Theory Embedding Pipeline
==================================================
Updated for nego-data-final.csv schema:
  conv_id, turn_id, role, word, speaker_id, duration_min

Outputs:
  stage1_turns_embedded.csv   — one row per substantive turn, retained emotion dims
  stage1_sst_dimensions.csv   — retained SST dimensions after split-half reliability
  stage1_backchannels.csv     — filtered backchannel turns
  stage1_diagnostics.txt      — filter counts, reliability coefficients
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings, re, os
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "/content/optimal-nego/data/nego-data-final.csv"
OUT_DIR = "/content/optimal-nego/outputs/"
os.makedirs(OUT_DIR, exist_ok=True)

SHR_THRESHOLD = 0.05

TEXT_COL_CANDIDATES = ["clean_word", "word", "text"]
CONV_COL_CANDIDATES = ["conv_id", "conversation_id"]

EMOTION_DIMS = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise"
]

EMOTION_SEEDS = {
    "admiration": ["impressive","remarkable","excellent","great job","well done","respect","appreciate","acknowledge","admire","outstanding"],
    "amusement": ["funny","laugh","haha","joke","amusing","hilarious","lol","humor","witty","lighthearted"],
    "anger": ["unacceptable","ridiculous","furious","angry","outrageous","frustrated","upset","annoyed","infuriating","unfair"],
    "annoyance": ["annoying","irritating","bother","ugh","come on","seriously","really","again","stop","enough"],
    "approval": ["agree","yes","okay","sure","absolutely","definitely","correct","right","sounds good","fair enough"],
    "caring": ["help","support","understand","concern","welfare","need","assist","accommodate","consider","sensitive"],
    "confusion": ["confused","unclear","not sure","what do you mean","lost","confusing","misunderstand","clarify","explain","huh"],
    "curiosity": ["wonder","curious","tell me","what about","how does","why","interested","want to know","question","explore"],
    "desire": ["want","wish","hope","looking for","need","prefer","ideally","dream","goal","aspire"],
    "disappointment": ["disappointed","expected more","unfortunately","hoped","let down","not what","thought","below","missed","fell short"],
    "disapproval": ["no","disagree","not acceptable","reject","refuse","against","oppose","object","cannot accept","not okay"],
    "disgust": ["disgusting","awful","terrible","horrible","appalling","unacceptable","offensive","revolting","bad","worst"],
    "embarrassment": ["embarrassed","awkward","sorry about","mistake","blunder","apologies","oops","my fault","red","regret"],
    "excitement": ["excited","great","wonderful","fantastic","amazing","thrilled","cant wait","perfect","love it","yes"],
    "fear": ["worried","afraid","concern","risk","scared","nervous","uncertain","threat","dangerous","hesitant"],
    "gratitude": ["thank","grateful","appreciate","thankful","thanks","generous","kind","gracious","indebted","owe"],
    "grief": ["sad","loss","miss","gone","unfortunate","devastated","heartbroken","painful","difficult","hard"],
    "joy": ["happy","pleased","delighted","glad","love","wonderful","great","enjoy","pleasure","satisfied"],
    "love": ["love","adore","cherish","treasure","dear","fond","affection","devoted","care deeply","attached"],
    "nervousness": ["nervous","anxious","uneasy","tense","stress","worry","uncertain","hesitate","afraid","on edge"],
    "optimism": ["hope","confident","positive","will","can do","forward","better","improve","optimistic","believe"],
    "pride": ["proud","achievement","accomplished","deserve","earned","stood for","value","principle","maintain","position"],
    "realization": ["realize","understand now","see","makes sense","ah","now I see","get it","point taken","acknowledged","noted"],
    "relief": ["relief","glad that","fortunate","luckily","phew","thankfully","at least","reassured","better now","resolved"],
    "remorse": ["sorry","regret","apologize","my fault","should not have","mistake","wrong","blame myself","wish I had","if only"],
    "sadness": ["sad","unhappy","unfortunate","too bad","shame","pity","regret","melancholy","disappointed","down"],
    "surprise": ["surprised","unexpected","didnt expect","wow","really","shocking","didnt know","astonished","startled","remarkable"],
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def pick_col(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No {label} column found. Expected one of {candidates}; found {list(df.columns)}")

def standardize_schema(df):
    """Create backward-compatible aliases while preserving original columns."""
    text_col = pick_col(df, TEXT_COL_CANDIDATES, "text")
    conv_col = pick_col(df, CONV_COL_CANDIDATES, "conversation id")

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str).str.strip()

    if "text" not in df.columns:
        df["text"] = df[text_col]
    if "conversation_id" not in df.columns:
        df["conversation_id"] = df[conv_col]

    # Preserve correct order for updated CSV
    sort_cols = [c for c in ["conversation_id", "turn_id"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, text_col, conv_col

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Load & Backchannel Filter
# ─────────────────────────────────────────────────────────────────────────────

def load_and_filter(path):
    df = pd.read_csv(path)
    df, text_col, conv_col = standardize_schema(df)

    df["text_len"] = df["text"].str.len()

    backchannel_pattern = re.compile(
        r"^(yeah|mhm|mm|um|uh|okay|ok|yes|no|hi|hello|so|right|"
        r"sure|huh|ah|oh|i|you|hmm|yep|nope|alright|well|hey)$",
        re.IGNORECASE
    )

    def is_backchannel(text):
        text_clean = str(text).strip().lower()
        if len(text_clean) <= 3:
            return True
        if backchannel_pattern.match(text_clean):
            return True
        return False

    df["is_backchannel"] = df["text"].apply(is_backchannel)

    substantive = df[~df["is_backchannel"]].copy().reset_index(drop=True)
    backchannels = df[df["is_backchannel"]].copy().reset_index(drop=True)

    return df, substantive, backchannels, text_col, conv_col

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Speaker Role Assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_roles(df):
    """
    Use existing `role` column when present.
    Otherwise infer buyer/seller using first price/offer mention.
    """
    df = df.copy()

    if "role" in df.columns:
        df["role"] = df["role"].astype(str).str.lower().str.strip()
        df["role"] = df["role"].replace({"seller": "seller", "dealer": "seller", "d": "seller",
                                         "buyer": "buyer", "b": "buyer"})
        return df, None

    if "speaker_id" not in df.columns:
        df["role"] = "unknown"
        return df, None

    price_pattern = re.compile(r"\$?\d[\d,\.]+|\bprice\b|\boffer\b|\bbid\b", re.IGNORECASE)
    role_map = {}

    for conv_id, group in df.groupby("conversation_id"):
        sort_cols = [c for c in ["turn_id"] if c in group.columns]
        group_sorted = group.sort_values(sort_cols) if sort_cols else group
        buyer_spk = None

        for _, row in group_sorted.iterrows():
            if price_pattern.search(str(row["text"])):
                buyer_spk = row["speaker_id"]
                break

        if buyer_spk is None:
            buyer_spk = 0

        speakers = group["speaker_id"].unique()
        seller_spk = [s for s in speakers if s != buyer_spk]
        seller_spk = seller_spk[0] if seller_spk else (1 - buyer_spk)
        role_map[conv_id] = {buyer_spk: "buyer", seller_spk: "seller"}

    df["role"] = df.apply(
        lambda row: role_map.get(row["conversation_id"], {}).get(row["speaker_id"], "unknown"),
        axis=1
    )
    return df, role_map

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Emotion Projection — GoEmotions 27-dim lexical SST approximation
# ─────────────────────────────────────────────────────────────────────────────

def build_seed_matrix(corpus_turns):
    texts = corpus_turns["text"].tolist()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=15000,
        strip_accents="unicode",
        lowercase=True
    )
    X = vectorizer.fit_transform(texts)
    vocab = vectorizer.get_feature_names_out()
    vocab_index = {w: i for i, w in enumerate(vocab)}

    seed_vectors = {}
    for emotion, seeds in EMOTION_SEEDS.items():
        indices = [vocab_index[w] for w in seeds if w in vocab_index]
        if indices:
            seed_vec = np.asarray(X[:, indices].mean(axis=1)).flatten()
        else:
            seed_vec = np.zeros(X.shape[0])
        seed_vectors[emotion] = seed_vec

    E = np.column_stack([seed_vectors[em] for em in EMOTION_DIMS])
    E_scaled = MinMaxScaler().fit_transform(E)

    emotion_df = pd.DataFrame(E_scaled, columns=EMOTION_DIMS, index=corpus_turns.index)
    result = pd.concat([corpus_turns.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)
    return result, vectorizer

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Split-Half Reliability
# ─────────────────────────────────────────────────────────────────────────────

def split_half_cca(emotion_matrix, n_components=27, n_splits=50):
    E = emotion_matrix.copy()
    n = len(E)
    n_comp = min(n_components, E.shape[1], max(1, n // 4))

    correlations = np.zeros((n_splits, n_comp))

    from sklearn.decomposition import TruncatedSVD
    from scipy.linalg import orthogonal_procrustes

    for i in range(n_splits):
        idx = np.random.permutation(n)
        half1_idx = idx[:n // 2]
        half2_idx = idx[n // 2: 2 * (n // 2)]

        E1 = E[half1_idx]
        E2 = E[half2_idx]

        svd1 = TruncatedSVD(n_components=n_comp, random_state=i)
        svd2 = TruncatedSVD(n_components=n_comp, random_state=i)

        C1 = svd1.fit_transform(E1)
        C2 = svd2.fit_transform(E2)

        min_rows = min(C1.shape[0], C2.shape[0])
        C1_trim = C1[:min_rows]
        C2_trim = C2[:min_rows]

        R, _ = orthogonal_procrustes(C2_trim, C1_trim)
        C2_aligned = C2_trim @ R

        for d in range(n_comp):
            r = np.corrcoef(C1_trim[:, d], C2_aligned[:, d])[0, 1]
            correlations[i, d] = abs(r) if not np.isnan(r) else 0.0

    mean_r = correlations.mean(axis=0)
    retained = np.where(mean_r > SHR_THRESHOLD)[0]
    return mean_r, len(retained), retained

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    diag_lines = []
    def log(s):
        print(s)
        diag_lines.append(str(s))

    log("=" * 60)
    log("STAGE 1: SST EMBEDDING PIPELINE")
    log("=" * 60)

    log("\n[Step 1] Loading data and filtering backchannels...")
    df_full, df_sub, df_bc, text_col, conv_col = load_and_filter(DATA_PATH)

    log(f"  Input text column  : {text_col}")
    log(f"  Conversation column: {conv_col}")
    log(f"  Total turns        : {len(df_full):,}")
    log(f"  Substantive turns  : {len(df_sub):,} ({len(df_sub)/len(df_full)*100:.1f}%)")
    log(f"  Backchannels       : {len(df_bc):,} ({len(df_bc)/len(df_full)*100:.1f}%)")
    log(f"  Conversations      : {df_sub['conversation_id'].nunique()}")

    df_bc.to_csv(OUT_DIR + "stage1_backchannels.csv", index=False)

    log("\n[Step 2] Assigning buyer/seller roles...")
    df_sub, role_map = assign_roles(df_sub)

    role_counts = df_sub["role"].value_counts()
    log(f"  Buyer turns  : {role_counts.get('buyer', 0):,}")
    log(f"  Seller turns : {role_counts.get('seller', 0):,}")
    log(f"  Unknown      : {role_counts.get('unknown', 0):,}")

    n_convs = df_sub["conversation_id"].nunique()
    n_convs_with_both = sum(
        1 for _, grp in df_sub.groupby("conversation_id")
        if {"buyer", "seller"}.issubset(set(grp["role"].unique()))
    )
    log(f"  Conversations with both roles: {n_convs_with_both}/{n_convs}")

    log("\n[Step 3] Building 27-dim GoEmotions lexical semantic space...")
    df_embed, vectorizer = build_seed_matrix(df_sub)

    log(f"  Embedding matrix shape: {len(df_embed)} x 27")
    log(f"  Vocab size (TF-IDF)   : {len(vectorizer.get_feature_names_out()):,}")

    log("\n  Mean activation per emotion dimension:")
    emotion_means = df_embed[EMOTION_DIMS].mean().sort_values(ascending=False)
    for em, val in emotion_means.items():
        bar = "█" * int(val * 40)
        log(f"    {em:<18} {val:.4f}  {bar}")

    log("\n  Buyer vs Seller mean activation (top 5 differences):")
    buyer_means = df_embed[df_embed["role"] == "buyer"][EMOTION_DIMS].mean()
    seller_means = df_embed[df_embed["role"] == "seller"][EMOTION_DIMS].mean()
    diff = (buyer_means - seller_means).abs().sort_values(ascending=False)
    for em in diff.head(5).index:
        log(f"    {em:<18} buyer={buyer_means[em]:.4f} seller={seller_means[em]:.4f} Δ={diff[em]:.4f}")

    log("\n[Step 4] Split-half reliability discovery (n_splits=50)...")
    E_matrix = df_embed[EMOTION_DIMS].values
    mean_r, n_retained, retained_idx = split_half_cca(E_matrix, n_splits=50)

    log(f"\n  Split-half reliability per dimension (threshold r > {SHR_THRESHOLD}):")
    dim_results = []
    for i, (r, em) in enumerate(zip(mean_r, EMOTION_DIMS)):
        retained = i in retained_idx
        marker = "✓" if retained else "✗"
        log(f"    [{marker}] Dim {i+1:2d} ({em:<18}) r = {r:.4f}")
        dim_results.append({
            "dimension": i + 1,
            "emotion": em,
            "split_half_r": round(float(r), 4),
            "retained": bool(retained)
        })

    log(f"\n  Retained dimensions : {n_retained} / {len(EMOTION_DIMS)}")

    dim_df = pd.DataFrame(dim_results)
    dim_df.to_csv(OUT_DIR + "stage1_sst_dimensions.csv", index=False)

    # Keep available original metadata; no longer assumes start_time/end_time/outcome.
    retained_emotions = [EMOTION_DIMS[i] for i in retained_idx]
    metadata_candidates = [
        "conv_id", "conversation_id", "turn_id", "speaker_id", "role",
        "duration_min", "word", "clean_word", "text", "text_len"
    ]
    output_cols = []
    for c in metadata_candidates:
        if c in df_embed.columns and c not in output_cols:
            output_cols.append(c)

    output_cols += retained_emotions
    df_out = df_embed[output_cols].copy()
    df_out.to_csv(OUT_DIR + "stage1_turns_embedded.csv", index=False)

    log(f"\n[Step 5] Output files written to {OUT_DIR}")
    log(f"  stage1_turns_embedded.csv — {len(df_out):,} turns x {len(retained_emotions)} retained emotion dims")
    log(f"  stage1_sst_dimensions.csv — {len(dim_df)} dims + reliability")
    log(f"  stage1_backchannels.csv   — {len(df_bc):,} backchannel turns")

    with open(OUT_DIR + "stage1_diagnostics.txt", "w") as f:
        f.write("\n".join(diag_lines))

    log("\n" + "=" * 60)
    log("STAGE 1 COMPLETE")
    log("=" * 60)

    return df_out, dim_df

if __name__ == "__main__":
    df_embedded, dim_table = main()
