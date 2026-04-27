"""
Study Validation Pipeline
=========================
Optimal Talking — Robustness and Design Selection

Covers all items in study-validation-list.txt:
  1. Data and role validation
  2. Text preprocessing variants
  3. Emotion measure alternatives
  4. SSM parameter grid
  5. Bargaining act annotation validation
  6. Outcome prediction model grid
  7. Robustness checks
  8. Interpretation safeguards
  9. Final decision rule

Usage:
  python validation/validation_pipeline.py \
    --data data/nego-data-final.csv \
    --outputs outputs/ \
    [--sections 1,2,3,4,5,6,7]   # run all by default

Each section writes results to:
  outputs/validation/section_{N}_{name}.csv
  outputs/validation/validation_summary.csv   (final decision table)
"""

import numpy as np
import pandas as pd
import re, os, json, argparse, warnings
from itertools import product
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss)
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_DIMS_27 = [
    'admiration','amusement','anger','annoyance','approval','caring',
    'confusion','curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy','love',
    'nervousness','optimism','pride','realization','relief','remorse',
    'sadness','surprise'
]

EMOTION_SEEDS = {
    'admiration':     ['impressive','remarkable','excellent','outstanding','admire'],
    'amusement':      ['funny','laugh','joke','amusing','humor','witty'],
    'anger':          ['unacceptable','ridiculous','furious','angry','frustrated','unfair'],
    'annoyance':      ['come on','seriously','stop','enough','listen','fine'],
    'approval':       ['agree','okay','sure','absolutely','correct','right','fair','deal'],
    'caring':         ['help','support','understand','concern','assist','important'],
    'confusion':      ['confused','not sure','unclear','clarify','explain','follow'],
    'curiosity':      ['wonder','curious','how','why','interested','question'],
    'desire':         ['want','wish','hope','need','prefer','would like','goal'],
    'disappointment': ['disappointed','unfortunately','expected','hoped','shame','missed'],
    'disapproval':    ['disagree','reject','refuse','against','oppose','not okay'],
    'disgust':        ['awful','terrible','horrible','appalling','worst'],
    'embarrassment':  ['sorry','mistake','apologies','wrong','regret','excuse'],
    'excitement':     ['excited','wonderful','fantastic','amazing','thrilled','excellent'],
    'fear':           ['worried','afraid','concern','risk','nervous','uncertain'],
    'gratitude':      ['thank','grateful','appreciate','thankful','thanks'],
    'grief':          ['sad','loss','miss','unfortunate','difficult','painful'],
    'joy':            ['happy','pleased','delighted','glad','enjoy','satisfied'],
    'love':           ['love','adore','dear','devoted','cherish','fond'],
    'nervousness':    ['nervous','anxious','uneasy','tense','stress','worry'],
    'optimism':       ['hope','confident','positive','better','improve','believe'],
    'pride':          ['deserve','earned','value','maintain','position','worth'],
    'realization':    ['realize','makes sense','understand','get it','noted','recognize'],
    'relief':         ['relief','glad that','fortunate','luckily','reassured'],
    'remorse':        ['sorry','regret','apologize','my fault','wrong','if only'],
    'sadness':        ['sad','unfortunate','shame','pity','regret','down'],
    'surprise':       ['surprised','unexpected','wow','shocking','astonished'],
}

EMOTION_PATTERNS = {
    em: re.compile(r'\b(' + '|'.join(re.escape(s) for s in seeds) + r')\b',
                   re.IGNORECASE)
    for em, seeds in EMOTION_SEEDS.items()
}

BC_RE_STRICT = re.compile(
    r'^\s*(yeah|mhm|mm|um|uh|okay|ok|yes|no|hi|hello|so|right|'
    r'sure|huh|ah|oh|hmm|yep|nope|alright|well|hey)\s*$', re.IGNORECASE)

BC_RE_LOOSE = re.compile(
    r'^\s*(yeah|mhm|mm|um|uh|okay|ok|yes|no|hi|hello|so|right|sure|huh|'
    r'ah|oh|hmm|yep|nope|alright|well|hey|right|got it|uh huh|'
    r'i see|go on|and|but|or)\s*$', re.IGNORECASE)

PRICE_RE    = re.compile(r'\$?\d[\d,\.]+|\bprice\b|\boffer\b|\bbid\b', re.IGNORECASE)
NUM_300_RE  = re.compile(r'\b(\d{3})\b')
NON_PRICE   = {'213','233','239','240','235','225','150','200','300',
               '000','1846','1715','1875','1920'}


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def assign_roles(df):
    role_map = {}
    for cid, grp in df.groupby('conversation_id'):
        buyer = None
        for _, row in grp.sort_values('start_time').iterrows():
            if PRICE_RE.search(str(row['text'])):
                buyer = row['speaker_id']; break
        if buyer is None: buyer = 0
        spks   = grp['speaker_id'].unique()
        seller = [s for s in spks if s != buyer]
        seller = seller[0] if seller else (1 - buyer)
        role_map[cid] = {buyer: 'buyer', seller: 'seller'}
    df = df.copy()
    df['role'] = df.apply(
        lambda r: role_map.get(r['conversation_id'], {}).get(r['speaker_id'], 'buyer'),
        axis=1)
    return df, role_map


def extract_prices(text):
    prices = set()
    for m in NUM_300_RE.finditer(str(text)):
        if m.group(1) in NON_PRICE: continue
        v = int(m.group(1))
        if 150 <= v <= 300:
            prices.add(v)
    return prices


def compute_emotion_features(df, window=3, dims=None):
    """Binary seed + rolling window emotion projection."""
    if dims is None:
        dims = list(EMOTION_PATTERNS.keys())
    df = df.copy()
    for em in dims:
        pat = EMOTION_PATTERNS[em]
        df[em] = df['text'].apply(lambda t: 1.0 if pat.search(str(t)) else 0.0)
    df = df.sort_values(['conversation_id','start_time']).reset_index(drop=True)
    for em in dims:
        df[em] = df.groupby('conversation_id')[em].transform(
            lambda x: x.rolling(window, min_periods=1, center=True).mean())
    return df


def compute_vader_sentiment(df):
    """Simple VADER-style proxy: positive/negative word counts."""
    pos_words = re.compile(
        r'\b(good|great|excellent|perfect|happy|pleased|wonderful|agree|deal|'
        r'fair|reasonable|positive|yes|absolutely|definitely)\b', re.IGNORECASE)
    neg_words = re.compile(
        r'\b(no|not|never|bad|terrible|awful|disagree|refuse|reject|'
        r'unfortunately|disappointed|problem|issue|concern|worry)\b', re.IGNORECASE)
    df = df.copy()
    df['vader_pos'] = df['text'].apply(lambda t: len(pos_words.findall(str(t))))
    df['vader_neg'] = df['text'].apply(lambda t: len(neg_words.findall(str(t))))
    df['vader_compound'] = (df['vader_pos'] - df['vader_neg']) / \
                           (df['vader_pos'] + df['vader_neg'] + 1)
    return df


def compute_liwc_proxies(df):
    """Approximation of key LIWC categories relevant to negotiation."""
    cats = {
        'interrog':    r'\b(what|when|where|who|why|how|which|could you|would you|can you)\b',
        'negemo':      r'\b(unfortunate|problem|sorry|bad|wrong|difficult|hard|worried|afraid)\b',
        'posemo':      r'\b(great|good|happy|please|love|wonderful|excellent|glad)\b',
        'money':       r'\b(price|cost|pay|afford|budget|dollar|thousand|value|worth)\b',
        'space':       r'\b(area|location|neighborhood|room|square|floor|space|nearby)\b',
        'tentat':      r'\b(maybe|perhaps|might|could|possibly|I think|I believe|not sure)\b',
        'certain':     r'\b(definitely|absolutely|certainly|always|never|must|will)\b',
        'social':      r'\b(we|us|our|together|both|mutual|each other|partner)\b',
        'motion':      r'\b(come|go|move|walk|bring|take|leave|stay)\b',
        'power':       r'\b(need|must|have to|require|demand|insist|firm|final)\b',
    }
    df = df.copy()
    for cat, pattern in cats.items():
        pat = re.compile(pattern, re.IGNORECASE)
        df[f'liwc_{cat}'] = df['text'].apply(
            lambda t: len(pat.findall(str(t))) / max(len(str(t).split()), 1))
    return df


def run_kalman_em(sequences, k, p=46, max_iter=60, tol=0.5):
    """Kalman EM SSM from Stage 2."""
    eps = 1e-6
    A   = np.eye(k)*0.9 + np.random.randn(k,k)*0.01
    C   = np.random.randn(p,k)*0.05
    Q   = np.eye(k)*0.1
    R   = np.eye(p)*0.3
    mu0 = np.zeros(k)
    V0  = np.eye(k)
    prev_ll = -np.inf

    for itr in range(max_iter):
        T_tot=0; s_zz=np.zeros((k,k)); s_zt_zt1=np.zeros((k,k))
        s_zt1_zt1=np.zeros((k,k)); s_xz=np.zeros((p,k))
        s_xx=np.zeros((p,p)); mu0_sum=np.zeros(k); V0_sum=np.zeros((k,k))
        total_ll=0.0; all_sm=[]

        for X in sequences:
            T=len(X)
            mu_f=np.zeros((T,k)); V_f=np.zeros((T,k,k))
            mu_p=np.zeros((T,k)); V_p=np.zeros((T,k,k))

            for t in range(T):
                mp = mu0 if t==0 else A@mu_f[t-1]
                Vp = V0  if t==0 else A@V_f[t-1]@A.T+Q
                mu_p[t]=mp; V_p[t]=Vp
                S  = C@Vp@C.T+R
                Si = np.linalg.pinv(S)
                K  = Vp@C.T@Si
                inn= X[t]-C@mp
                mu_f[t]=(np.eye(k)-K@C)@mp+K@X[t]
                V_f[t] =(np.eye(k)-K@C)@Vp
                sd,ld=np.linalg.slogdet(S)
                if sd>0: total_ll-=0.5*(ld+inn@Si@inn+p*np.log(2*np.pi))

            mu_s=mu_f.copy(); V_s=V_f.copy(); Vt_s=np.zeros((T-1,k,k))
            for t in range(T-2,-1,-1):
                G=V_f[t]@A.T@np.linalg.pinv(V_p[t+1])
                mu_s[t]=mu_f[t]+G@(mu_s[t+1]-mu_p[t+1])
                V_s[t] =V_f[t] +G@(V_s[t+1]-V_p[t+1])@G.T
                Vt_s[t]=V_s[t+1]@G.T
            all_sm.append(mu_s)

            T_tot+=T; mu0_sum+=mu_s[0]; V0_sum+=V_s[0]+np.outer(mu_s[0],mu_s[0])
            for t in range(T):
                ez=mu_s[t]; ezz=V_s[t]+np.outer(ez,ez)
                s_zz+=ezz; s_xz+=np.outer(X[t],ez); s_xx+=np.outer(X[t],X[t])
            for t in range(T-1):
                s_zt_zt1 +=Vt_s[t]+np.outer(mu_s[t+1],mu_s[t])
                s_zt1_zt1+=V_s[t] +np.outer(mu_s[t],mu_s[t])

        n=len(sequences)
        mu0=mu0_sum/n; V0=V0_sum/n-np.outer(mu0,mu0)+eps*np.eye(k)
        A  =s_zt_zt1@np.linalg.pinv(s_zt1_zt1+eps*np.eye(k))
        Q  =(s_zz-A@s_zt_zt1.T)/T_tot; Q=0.5*(Q+Q.T)+eps*np.eye(k)
        C  =s_xz@np.linalg.pinv(s_zz+eps*np.eye(k))
        Rf =((s_xx-C@s_xz.T)/T_tot); R=np.diag(np.maximum(np.diag(Rf),eps))*np.eye(p)

        delta=total_ll-prev_ll
        if itr>0 and abs(delta)<tol: break
        prev_ll=total_ll

    T_tot_all=sum(len(X) for X in sequences)
    n_params=k*k+p*k+k*(k+1)//2+p
    bic=-2*total_ll+n_params*np.log(T_tot_all)
    aic=-2*total_ll+2*n_params
    sr=max(abs(np.linalg.eigvals(A)))
    return dict(A=A,C=C,Q=Q,R=R,mu0=mu0,V0=V0,
                ll=total_ll,bic=bic,aic=aic,sr=sr,smoothed=all_sm,k=k)


def evaluate_classifier(F, y, clf=None, n_splits=5):
    """Full evaluation: AUC, accuracy, precision, recall, F1, brier."""
    if clf is None:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
    scaler = StandardScaler()
    F_sc   = scaler.fit_transform(np.nan_to_num(F))
    cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    aucs, accs, precs, recs, f1s, briers = [], [], [], [], [], []
    preds = np.zeros(len(y))

    for tr, te in cv.split(F_sc, y):
        clf.fit(F_sc[tr], y[tr])
        prob      = clf.predict_proba(F_sc[te])[:, 1]
        pred_cls  = (prob >= 0.5).astype(int)
        preds[te] = prob
        aucs.append(roc_auc_score(y[te], prob))
        accs.append(accuracy_score(y[te], pred_cls))
        precs.append(precision_score(y[te], pred_cls, zero_division=0))
        recs.append(recall_score(y[te], pred_cls, zero_division=0))
        f1s.append(f1_score(y[te], pred_cls, zero_division=0))
        briers.append(brier_score_loss(y[te], prob))

    return {
        'mean_auc':    round(np.mean(aucs), 4),
        'std_auc':     round(np.std(aucs), 4),
        'ci95_low':    round(np.mean(aucs) - 1.96*np.std(aucs)/np.sqrt(n_splits), 4),
        'ci95_high':   round(np.mean(aucs) + 1.96*np.std(aucs)/np.sqrt(n_splits), 4),
        'mean_acc':    round(np.mean(accs), 4),
        'mean_prec':   round(np.mean(precs), 4),
        'mean_rec':    round(np.mean(recs), 4),
        'mean_f1':     round(np.mean(f1s), 4),
        'mean_brier':  round(np.mean(briers), 4),
        'overall_auc': round(roc_auc_score(y, preds), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA AND ROLE VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def section1_data_validation(df, out_dir):
    print("\n[Section 1] Data and role validation...")
    results = []

    # 1a. Role assignment audit — who speaks first
    df2, role_map = assign_roles(df)
    first_speaker_is_buyer = sum(
        1 for cid, grp in df2.groupby('conversation_id')
        if grp.sort_values('start_time').iloc[0]['role'] == 'buyer'
    )
    results.append({'check': 'buyer_speaks_first',
                    'n': first_speaker_is_buyer,
                    'pct': round(first_speaker_is_buyer/178*100, 1)})

    # 1b. Cases where dealer/seller mentions price first
    dealer_anchors = sum(
        1 for cid, grp in df2.groupby('conversation_id')
        for _, row in grp.sort_values('start_time').iterrows()
        if PRICE_RE.search(str(row['text'])) and row['role'] == 'seller'
        and True  # first price mention by seller
    )

    # Check conversations where seller mentions price before buyer
    seller_first_price = 0
    for cid, grp in df2.groupby('conversation_id'):
        grp_s = grp.sort_values('start_time')
        for _, row in grp_s.iterrows():
            if PRICE_RE.search(str(row['text'])):
                if row['role'] == 'seller':
                    seller_first_price += 1
                break
    results.append({'check': 'seller_anchors_price_first',
                    'n': seller_first_price,
                    'pct': round(seller_first_price/178*100, 1)})

    # 1c. Sale vs no-sale counts
    outcome_counts = df2.groupby('conversation_id')['outcome'].first().value_counts()
    for outcome, count in outcome_counts.items():
        results.append({'check': f'outcome_{outcome}',
                        'n': count,
                        'pct': round(count/178*100, 1)})

    # 1d. Very short conversations (<5 substantive turns)
    df2['is_bc'] = df2['text'].apply(
        lambda t: len(str(t).strip()) <= 3 or bool(BC_RE_STRICT.match(str(t))))
    sub_counts = df2[~df2['is_bc']].groupby('conversation_id').size()
    short_convs = (sub_counts < 5).sum()
    results.append({'check': 'very_short_convs_lt5_turns',
                    'n': int(short_convs),
                    'pct': round(short_convs/178*100, 1)})

    # 1e. Effect of excluding very short calls on outcome rate
    valid_convs  = sub_counts[sub_counts >= 5].index
    df_filtered  = df2[df2['conversation_id'].isin(valid_convs)]
    sale_rate_full = (df2.groupby('conversation_id')['outcome'].first()=='sale').mean()
    sale_rate_filt = (df_filtered.groupby('conversation_id')['outcome'].first()=='sale').mean()
    results.append({'check': 'sale_rate_all', 'n': round(sale_rate_full, 3), 'pct': None})
    results.append({'check': 'sale_rate_excl_short', 'n': round(sale_rate_filt, 3), 'pct': None})

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section1_data_validation.csv', index=False)
    for _, r in df_out.iterrows():
        print(f"  {r['check']:<40}: n={r['n']}  pct={r['pct']}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TEXT PREPROCESSING VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def section2_preprocessing(df, out_dir):
    print("\n[Section 2] Text preprocessing variants...")
    df2, _ = assign_roles(df)
    results = []

    configs = {
        'strict_bc':    BC_RE_STRICT,
        'loose_bc':     BC_RE_LOOSE,
        'len3_only':    None,   # only remove turns <= 3 chars
        'no_filter':    None,   # keep all turns
    }

    for name, bc_re in configs.items():
        if name == 'no_filter':
            sub = df2.copy()
        elif name == 'len3_only':
            sub = df2[df2['text'].str.len() > 3].copy()
        else:
            sub = df2[~df2['text'].apply(
                lambda t: len(str(t).strip()) <= 3 or bool(bc_re.match(str(t))))
            ].copy()

        # Check: does "sounds good", "okay deal", "I can do that" survive?
        preserved_kw = re.compile(
            r'\b(sounds good|okay deal|I can do that|deal|agreed)\b', re.IGNORECASE)
        preserved = sub['text'].apply(lambda t: bool(preserved_kw.search(str(t)))).sum()

        # Sparsity of emotion signal
        em_df = compute_emotion_features(sub, window=3,
                                          dims=['approval','anger','fear','curiosity'])
        sparsity = (em_df[['approval','anger','fear','curiosity']].values == 0).mean()

        results.append({
            'config':       name,
            'n_turns':      len(sub),
            'pct_retained': round(len(sub)/len(df2)*100, 1),
            'preserved_kw': int(preserved),
            'emotion_sparsity': round(sparsity*100, 1),
        })
        print(f"  {name:<15}: {len(sub):,} turns ({round(len(sub)/len(df2)*100,1)}%)"
              f"  preserved_kw={preserved}  sparsity={round(sparsity*100,1)}%")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section2_preprocessing.csv', index=False)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EMOTION MEASURE ALTERNATIVES
# ─────────────────────────────────────────────────────────────────────────────

def section3_emotion_measures(df, out_dir):
    print("\n[Section 3] Emotion measure alternatives...")
    df2, _ = assign_roles(df)
    df2['is_bc'] = df2['text'].apply(
        lambda t: len(str(t).strip()) <= 3 or bool(BC_RE_STRICT.match(str(t))))
    sub = df2[~df2['is_bc']].copy()

    results = []

    # Measure 1: Seed lexicon (our approach) — 27 dims
    em27 = compute_emotion_features(sub, window=3, dims=EMOTION_DIMS_27)
    E27  = em27[EMOTION_DIMS_27].values
    results.append({
        'measure': 'seed_lexicon_27dim',
        'n_dims': 27,
        'sparsity_pct': round((E27==0).mean()*100, 1),
        'active_per_turn': round((E27>0).sum(axis=1).mean(), 2),
        'note': 'Current approach. Reproducible, no GPU.'
    })

    # Measure 2: VADER proxy (3 dims)
    vader_df = compute_vader_sentiment(sub)
    V = vader_df[['vader_pos','vader_neg','vader_compound']].values
    results.append({
        'measure': 'vader_proxy_3dim',
        'n_dims': 3,
        'sparsity_pct': round((V==0).mean()*100, 1),
        'active_per_turn': round((V>0).sum(axis=1).mean(), 2),
        'note': 'Simple baseline. Low-dim, loses emotion specificity.'
    })

    # Measure 3: LIWC proxies (10 dims)
    liwc_df = compute_liwc_proxies(sub)
    liwc_cols = [c for c in liwc_df.columns if c.startswith('liwc_')]
    L = liwc_df[liwc_cols].values
    results.append({
        'measure': 'liwc_proxies_10dim',
        'n_dims': len(liwc_cols),
        'sparsity_pct': round((L==0).mean()*100, 1),
        'active_per_turn': round((L>0).sum(axis=1).mean(), 2),
        'note': 'Approximates Heddaya et al. LIWC results. Interpretable.'
    })

    # Measure 4: Negotiation-specific signals (8 dims)
    nego_signals = {
        'firmness':    r'\b(firm|final offer|bottom line|will not|cannot|stand by)\b',
        'hesitation':  r'\b(maybe|perhaps|not sure|I think|I guess|possibly|hmm)\b',
        'urgency':     r'\b(need to|must|have to|right now|today|asap|deadline)\b',
        'resistance':  r'\b(no|not|never|refuse|reject|cannot accept|won\'t)\b',
        'agreement':   r'\b(agree|deal|yes|okay|sure|absolutely|sounds good|correct)\b',
        'concession':  r'\b(willing to|could do|drop to|come down|go up|meet you)\b',
        'inquiry':     r'\b(what|how|when|where|tell me|could you|would you|why)\b',
        'rapport':     r'\b(great|nice|lovely|wonderful|appreciate|thank you|pleased)\b',
    }
    for sig, pattern in nego_signals.items():
        pat = re.compile(pattern, re.IGNORECASE)
        sub[sig] = sub['text'].apply(lambda t: 1.0 if pat.search(str(t)) else 0.0)

    N = sub[list(nego_signals.keys())].values
    results.append({
        'measure': 'nego_specific_8dim',
        'n_dims': 8,
        'sparsity_pct': round((N==0).mean()*100, 1),
        'active_per_turn': round((N>0).sum(axis=1).mean(), 2),
        'note': 'Theoretically targeted. High face validity for RQ3.'
    })

    # Buyer vs seller profile comparison for seed lexicon
    buyer_em  = em27[em27['role']=='buyer'][EMOTION_DIMS_27].mean()
    seller_em = em27[em27['role']=='seller'][EMOTION_DIMS_27].mean()
    diff      = (buyer_em - seller_em).abs().sort_values(ascending=False)
    top5_diff = diff.head(5).to_dict()

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section3_emotion_measures.csv', index=False)
    pd.DataFrame([top5_diff]).to_csv(out_dir + 'section3_buyer_seller_top5_diff.csv')

    for _, r in df_out.iterrows():
        print(f"  {r['measure']:<30}: dims={r['n_dims']:2d}  "
              f"sparsity={r['sparsity_pct']:5.1f}%  "
              f"active/turn={r['active_per_turn']:.2f}")

    print(f"\n  Top 5 buyer-seller emotion differences:")
    for em, val in top5_diff.items():
        print(f"    {em:<20}: Δ={val:.4f}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SSM PARAMETER GRID
# ─────────────────────────────────────────────────────────────────────────────

def section4_ssm_grid(df, out_dir):
    print("\n[Section 4] SSM parameter grid (k=2..6, window=1,3,5)...")
    df2, _ = assign_roles(df)
    df2['is_bc'] = df2['text'].apply(
        lambda t: len(str(t).strip()) <= 3 or bool(BC_RE_STRICT.match(str(t))))

    # Use 23 retained dims from Stage 1
    DIMS_23 = [d for d in EMOTION_DIMS_27
               if d not in ['admiration','relief','remorse','surprise']]

    results = []

    for window, k in product([1, 3, 5], range(2, 7)):
        print(f"  window={window}, k={k}...", flush=True)

        # Build trajectories
        sub = df2[~df2['is_bc']].copy()
        em  = compute_emotion_features(sub, window=window, dims=DIMS_23)
        em  = em.sort_values(['conversation_id','start_time']).reset_index(drop=True)
        em['turn'] = em.groupby('conversation_id').cumcount()

        # Build coupled observations
        conv_ids = sorted(em['conversation_id'].unique())
        sequences = []
        outcomes  = []
        for cid in conv_ids:
            grp = em[em['conversation_id']==cid].sort_values('turn')
            T   = len(grp)
            bE  = np.zeros((T,23)); sE=np.zeros((T,23))
            lb  = np.zeros(23);    ls=np.zeros(23)
            for _, row in grp.iterrows():
                ev = row[DIMS_23].values.astype(float)
                if row['role']=='buyer': lb=ev
                else: ls=ev
                bE[int(row['turn'])]=lb; sE[int(row['turn'])]=ls
            sequences.append(np.hstack([bE,sE]))
            outcomes.append(1 if grp['outcome'].iloc[0]=='sale' else 0)

        # Fit SSM
        try:
            res = run_kalman_em(sequences, k=k, p=46, max_iter=50, tol=1.0)

            # Outcome prediction from latent states
            feats = []
            for z_seq in res['smoothed']:
                T   = len(z_seq)
                zm  = z_seq.mean(axis=0)
                zs  = z_seq.std(axis=0)+1e-8
                zl  = z_seq[max(0,2*T//3):].mean(axis=0)
                feats.append(np.concatenate([zm,zs,zl]))

            F = np.array(feats)
            y = np.array(outcomes)
            eval_res = evaluate_classifier(F, y)

            results.append({
                'window':       window,
                'k':            k,
                'bic':          round(res['bic'], 1),
                'aic':          round(res['aic'], 1),
                'log_lik':      round(res['ll'], 1),
                'spectral_r':   round(float(res['sr']), 4),
                'mean_auc':     eval_res['mean_auc'],
                'std_auc':      eval_res['std_auc'],
                'overall_auc':  eval_res['overall_auc'],
            })
            print(f"    k={k} w={window}: BIC={res['bic']:.0f}  "
                  f"sr={res['sr']:.3f}  AUC={eval_res['mean_auc']:.4f}")
        except Exception as e:
            print(f"    k={k} w={window}: FAILED — {e}")
            results.append({'window':window,'k':k,'error':str(e)})

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section4_ssm_grid.csv', index=False)

    # Best by BIC
    best = df_out.dropna(subset=['bic']).loc[df_out['bic'].idxmin()]
    print(f"\n  Best by BIC: k={best['k']}, window={best['window']}, "
          f"BIC={best['bic']}, AUC={best['mean_auc']}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — BARGAINING ACT ANNOTATION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def section5_annotation_validation(out_dir):
    print("\n[Section 5] Bargaining act annotation validation...")

    # Load Stage 3 annotated output
    try:
        ann = pd.read_csv(out_dir.replace('validation/', '') + 'stage3_annotated.csv')
    except:
        print("  stage3_annotated.csv not found — skipping section 5")
        return None

    sub   = ann[~ann['is_bc']].copy()
    total = len(sub)

    results = []

    # Act frequency and rate by outcome
    for col, name in [('act_push','push'), ('act_comparison','comparison'),
                      ('act_new_offer','new_offer'), ('act_allowance','allowance'),
                      ('act_end','end')]:
        rate_sale   = sub[sub['outcome']=='sale'][col].mean()
        rate_nosale = sub[sub['outcome']!='sale'][col].mean()
        results.append({
            'act':          name,
            'total_turns':  int(sub[col].sum()),
            'pct_all':      round(sub[col].mean()*100, 1),
            'rate_sale':    round(rate_sale*100, 1),
            'rate_nosale':  round(rate_nosale*100, 1),
            'delta_pp':     round((rate_sale - rate_nosale)*100, 2),
        })

    # Layer 2 sub-type breakdown
    push_turns = sub[sub['act_push']==1]
    for sub_t in ['push_constraint','push_disparagement','push_neutral']:
        n = (push_turns['push_subtype']==sub_t).sum()
        results.append({
            'act': sub_t,
            'total_turns': int(n),
            'pct_all': round(n/total*100, 2),
            'rate_sale': round((sub[sub['outcome']=='sale']['push_subtype']==sub_t).mean()*100, 2),
            'rate_nosale': round((sub[sub['outcome']!='sale']['push_subtype']==sub_t).mean()*100, 2),
            'delta_pp': round(((sub[sub['outcome']=='sale']['push_subtype']==sub_t).mean() -
                               (sub[sub['outcome']!='sale']['push_subtype']==sub_t).mean())*100, 2),
        })

    # End act: check proportion that occur at last turn (true closing) vs. mid-call
    end_turns = sub[sub['act_end']==1].copy()
    end_turns['turn_idx'] = end_turns.groupby('conversation_id').cumcount()
    conv_lens   = sub.groupby('conversation_id').size()
    end_turns['conv_len'] = end_turns['conversation_id'].map(conv_lens)
    end_turns['turn_pct'] = end_turns['turn_idx'] / end_turns['conv_len']
    true_end = (end_turns['turn_pct'] >= 0.8).mean()
    results.append({
        'act': 'end_at_last20pct',
        'total_turns': int(len(end_turns)),
        'pct_all': round(true_end*100, 1),
        'rate_sale': None, 'rate_nosale': None, 'delta_pp': None
    })

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section5_annotation_validation.csv', index=False)
    for _, r in df_out.iterrows():
        print(f"  {r['act']:<28}: n={r['total_turns']:4d}  "
              f"Δ={r['delta_pp'] if r['delta_pp'] is not None else 'n/a'}")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — OUTCOME PREDICTION MODEL GRID
# ─────────────────────────────────────────────────────────────────────────────

def section6_outcome_prediction(df, out_dir):
    print("\n[Section 6] Outcome prediction model grid...")
    df2, _ = assign_roles(df)
    df2['is_bc'] = df2['text'].apply(
        lambda t: len(str(t).strip()) <= 3 or bool(BC_RE_STRICT.match(str(t))))
    sub = df2[~df2['is_bc']].copy()
    sub = compute_emotion_features(sub, window=3)
    sub = compute_vader_sentiment(sub)
    sub = compute_liwc_proxies(sub)

    DIMS_23 = [d for d in EMOTION_DIMS_27
               if d not in ['admiration','relief','remorse','surprise']]
    liwc_cols = [c for c in sub.columns if c.startswith('liwc_')]

    # Load pre-computed SSM and tactic features if available
    try:
        latent = pd.read_csv(out_dir.replace('validation/','') + 'stage2_latent_states.csv')
        ann    = pd.read_csv(out_dir.replace('validation/','') + 'stage3_annotated.csv')
        has_precomputed = True
    except:
        has_precomputed = False
        print("  Stage 2/3 outputs not found — using raw emotion features as SSM proxy")

    # Build conversation-level feature sets
    conv_ids = sorted(sub['conversation_id'].unique())
    y = np.array([1 if sub[sub['conversation_id']==cid]['outcome'].iloc[0]=='sale'
                  else 0 for cid in conv_ids])

    def conv_features(cols):
        rows = []
        for cid in conv_ids:
            grp = sub[sub['conversation_id']==cid]
            T   = len(grp)
            vals = grp[cols].values
            zm   = vals.mean(axis=0)
            zs   = vals.std(axis=0)+1e-8
            zl   = vals[max(0,2*T//3):].mean(axis=0)
            rows.append(np.concatenate([zm,zs,zl]))
        return np.array(rows)

    # Price features
    price_rows = []
    for cid in conv_ids:
        grp    = sub[sub['conversation_id']==cid]
        prices = []
        for _, row in grp.iterrows():
            ps = extract_prices(row['text'])
            if ps: prices.extend(ps)
        last_p    = prices[-1] if prices else 230
        first_p   = prices[0]  if prices else 230
        n_offers  = len(prices)
        movement  = abs(last_p - first_p) if len(prices) > 1 else 0
        price_rows.append([last_p, first_p, movement, n_offers,
                           grp['start_time'].max() - grp['start_time'].min()])
    F_price = np.array(price_rows)

    # Duration + turn count baseline
    dur_rows = []
    for cid in conv_ids:
        grp = sub[sub['conversation_id']==cid]
        dur_rows.append([
            grp['end_time'].max() - grp['start_time'].min(),
            len(grp),
            (grp['role']=='buyer').mean(),
        ])
    F_dur = np.array(dur_rows)

    # Majority class baseline
    dummy = DummyClassifier(strategy='most_frequent')
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dummy_aucs = []
    for tr, te in cv.split(F_price, y):
        dummy.fit(F_price[tr], y[tr])
        prob = dummy.predict_proba(F_price[te])[:,1]
        try: dummy_aucs.append(roc_auc_score(y[te], prob))
        except: dummy_aucs.append(0.5)

    models = {
        'majority_class':     (F_dur,                         'Baseline'),
        'duration_turns':     (F_dur,                         'Control'),
        'price_only':         (F_price,                       'M1'),
        'emotion_23dim':      (conv_features(DIMS_23),        'M2a'),
        'vader_sentiment':    (conv_features(['vader_compound','vader_pos','vader_neg']), 'M2b'),
        'liwc_proxies':       (conv_features(liwc_cols),      'M2c'),
        'price_emotion':      (np.hstack([F_price, conv_features(DIMS_23)]), 'M3'),
        'price_liwc':         (np.hstack([F_price, conv_features(liwc_cols)]), 'M3b'),
    }

    if has_precomputed:
        # Add SSM and tactic features from pre-computed outputs
        sub_ann = ann[~ann['is_bc']].copy()
        sub_ann = sub_ann.sort_values(['conversation_id','start_time']).reset_index(drop=True)
        sub_ann['turn'] = sub_ann.groupby('conversation_id').cumcount()
        merged = sub_ann.merge(latent, on=['conversation_id','turn'], how='inner')
        z_cols = ['z_1','z_2','z_3','z_4']

        ssm_rows = []
        for cid in conv_ids:
            grp = merged[merged['conversation_id']==cid]
            if len(grp) == 0:
                ssm_rows.append(np.zeros(len(z_cols)*3))
                continue
            Z  = grp[z_cols].values
            T  = len(Z)
            ssm_rows.append(np.concatenate([
                Z.mean(axis=0), Z.std(axis=0)+1e-8, Z[max(0,2*T//3):].mean(axis=0)
            ]))
        F_ssm = np.array(ssm_rows)

        tact_rows = []
        act_cols_tact = ['act_push','act_comparison','act_allowance','act_new_offer']
        for cid in conv_ids:
            grp = sub_ann[sub_ann['conversation_id']==cid]
            T   = len(grp)
            row = [grp[c].sum()/T for c in act_cols_tact]
            row += [(grp['push_subtype']==st).sum()/T
                    for st in ['push_constraint','push_disparagement']]
            row += [(grp['comp_subtype']==st).sum()/T
                    for st in ['comparison_price','comparison_quality','comparison_mixed']]
            tact_rows.append(row)
        F_tact = np.array(tact_rows)

        models['ssm_latent']         = (F_ssm, 'M_SSM')
        models['price_ssm']          = (np.hstack([F_price, F_ssm]), 'M3_SSM')
        models['price_ssm_tactics']  = (np.hstack([F_price, F_ssm, F_tact]), 'M4_full')
        models['ssm_tactics']        = (np.hstack([F_ssm, F_tact]), 'M_ssm+tact')

    results = []
    for model_name, (F, label) in models.items():
        print(f"  Evaluating {model_name}...", flush=True)
        try:
            ev = evaluate_classifier(F, y)
            ev['model']  = model_name
            ev['label']  = label
            ev['n_feats'] = F.shape[1]
            results.append(ev)
            print(f"    AUC={ev['mean_auc']:.4f}±{ev['std_auc']:.4f}  "
                  f"CI=[{ev['ci95_low']:.4f},{ev['ci95_high']:.4f}]  "
                  f"F1={ev['mean_f1']:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section6_outcome_prediction.csv', index=False)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ROBUSTNESS CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def section7_robustness(df, out_dir):
    print("\n[Section 7] Robustness checks...")
    df2, _ = assign_roles(df)
    df2['is_bc'] = df2['text'].apply(
        lambda t: len(str(t).strip()) <= 3 or bool(BC_RE_STRICT.match(str(t))))
    sub = df2[~df2['is_bc']].copy()
    sub = compute_emotion_features(sub, window=3)

    DIMS_23 = [d for d in EMOTION_DIMS_27
               if d not in ['admiration','relief','remorse','surprise']]

    conv_meta = sub.groupby('conversation_id').agg(
        n_turns  = ('text','count'),
        duration = ('end_time','max'),
        outcome  = ('outcome','first'),
        buyer_pct= ('role', lambda x: (x=='buyer').mean()),
    ).reset_index()
    conv_meta['sale'] = (conv_meta['outcome']=='sale').astype(int)
    conv_meta['dur_tertile'] = pd.qcut(conv_meta['duration'], 3,
                                        labels=['short','medium','long'])

    results = []

    # 7a. By call length tertile
    for tertile in ['short','medium','long']:
        ids = conv_meta[conv_meta['dur_tertile']==tertile]['conversation_id']
        sub_t = sub[sub['conversation_id'].isin(ids)]
        y_t   = conv_meta[conv_meta['dur_tertile']==tertile]['sale'].values
        if len(y_t) < 10 or y_t.sum() < 3: continue

        rows = []
        for cid in ids:
            grp = sub_t[sub_t['conversation_id']==cid]
            rows.append(np.concatenate([
                grp[DIMS_23].mean().values,
                grp[DIMS_23].std().values+1e-8
            ]))
        F = np.array(rows)
        ev = evaluate_classifier(F, y_t, n_splits=min(5, y_t.sum()))
        results.append({
            'check': f'dur_tertile_{tertile}',
            'n_convs': len(ids),
            'sale_rate': round(y_t.mean()*100,1),
            **{k:v for k,v in ev.items()}
        })
        print(f"  duration={tertile}: n={len(ids)}  AUC={ev['mean_auc']:.4f}")

    # 7b. Control for duration and turn count
    dur_rows = []
    conv_ids = sorted(conv_meta['conversation_id'])
    for cid in conv_ids:
        row = conv_meta[conv_meta['conversation_id']==cid]
        grp = sub[sub['conversation_id']==cid]
        emotion_mean = grp[DIMS_23].mean().values
        controls     = np.array([float(row['duration'].iloc[0]), float(row['n_turns'].iloc[0]),
                                  float(row['buyer_pct'].iloc[0])])
        dur_rows.append(np.concatenate([emotion_mean, controls]))

    y_all = conv_meta.set_index('conversation_id').loc[conv_ids, 'sale'].values
    F_ctrl = np.array(dur_rows)
    ev_ctrl = evaluate_classifier(F_ctrl, y_all)
    results.append({
        'check': 'emotion_plus_duration_control',
        'n_convs': len(conv_ids),
        'sale_rate': round(y_all.mean()*100,1),
        **{k:v for k,v in ev_ctrl.items()}
    })
    print(f"  emotion+duration control: AUC={ev_ctrl['mean_auc']:.4f}")

    # 7c. Remove outlier conversations (>2 std from mean duration)
    dur_mean  = conv_meta['duration'].mean()
    dur_std   = conv_meta['duration'].std()
    no_outlier = conv_meta[
        (conv_meta['duration'] >= dur_mean - 2*dur_std) &
        (conv_meta['duration'] <= dur_mean + 2*dur_std)
    ]['conversation_id']
    sub_no = sub[sub['conversation_id'].isin(no_outlier)]
    y_no   = conv_meta[conv_meta['conversation_id'].isin(no_outlier)]['sale'].values
    rows_no = []
    for cid in no_outlier:
        grp = sub_no[sub_no['conversation_id']==cid]
        rows_no.append(grp[DIMS_23].mean().values)
    F_no = np.array(rows_no)
    ev_no = evaluate_classifier(F_no, y_no, n_splits=5)
    results.append({
        'check': 'excl_outlier_duration',
        'n_convs': int(len(no_outlier)),
        'sale_rate': round(y_no.mean()*100,1),
        **{k:v for k,v in ev_no.items()}
    })
    print(f"  excl outlier duration: n={len(no_outlier)}  AUC={ev_no['mean_auc']:.4f}")

    # 7d. Approval-token check — are results driven by "okay/sure/agree"?
    approval_words = re.compile(r'^\s*(okay|sure|right|yes|agree|deal)\s*$', re.IGNORECASE)
    sub_no_approval = sub[~sub['text'].apply(lambda t: bool(approval_words.match(str(t))))]
    dims_no_approval = [d for d in DIMS_23 if d != 'approval']
    rows_noa = []
    for cid in conv_ids:
        grp = sub_no_approval[sub_no_approval['conversation_id']==cid]
        if len(grp) == 0:
            rows_noa.append(np.zeros(len(dims_no_approval)))
        else:
            rows_noa.append(grp[dims_no_approval].mean().values)
    F_noa = np.array(rows_noa)
    ev_noa = evaluate_classifier(F_noa, y_all)
    results.append({
        'check': 'excl_approval_tokens',
        'n_convs': len(conv_ids),
        'sale_rate': round(y_all.mean()*100,1),
        **{k:v for k,v in ev_noa.items()}
    })
    print(f"  excl approval tokens: AUC={ev_noa['mean_auc']:.4f}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(out_dir + 'section7_robustness.csv', index=False)
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 + 9 — DECISION RULE: RETAIN / DISCARD
# ─────────────────────────────────────────────────────────────────────────────

def section89_decision_rule(out_dir):
    """
    Apply final decision rule from study-validation-list.txt Section 9:
    Retain a measure/model only if it satisfies:
      - Clear theoretical relevance
      - Acceptable reliability (AUC CI does not include 0.5)
      - Interpretable direction
      - Robustness across specifications
      - Incremental predictive value over price and duration baselines
      - Stable performance across folds (std_auc < 0.10)
    """
    print("\n[Section 8+9] Final decision rule...")

    try:
        s6 = pd.read_csv(out_dir + 'section6_outcome_prediction.csv')
    except:
        print("  Section 6 results not found — cannot generate decision table")
        return None

    # Price baseline
    price_auc = s6[s6['model']=='price_only']['mean_auc'].values
    price_auc = price_auc[0] if len(price_auc) > 0 else 0.539

    decisions = []
    for _, row in s6.iterrows():
        auc       = row.get('mean_auc', 0)
        ci_low    = row.get('ci95_low', 0)
        std_auc   = row.get('std_auc', 1)
        above_05  = ci_low > 0.50
        above_price = auc > price_auc
        stable    = std_auc < 0.10

        # Causal language check — flag for Section 8
        causal_risk = auc < 0.60

        verdict = 'RETAIN' if (above_05 and stable) else \
                  'EXPLORATORY' if above_05 else 'DISCARD'

        decisions.append({
            'model':           row.get('model',''),
            'mean_auc':        round(auc, 4),
            'ci95_low':        round(ci_low, 4),
            'std_auc':         round(std_auc, 4),
            'ci_above_0.5':    above_05,
            'beats_price':     above_price,
            'stable_folds':    stable,
            'causal_flag':     causal_risk,
            'verdict':         verdict,
            'interpretation':  'Treat as exploratory if AUC < 0.60' if causal_risk else 'Reportable',
        })

    df_out = pd.DataFrame(decisions)
    df_out.to_csv(out_dir + 'section89_decision_table.csv', index=False)

    print(f"\n  {'Model':<30} {'AUC':>6} {'CI_low':>7} {'Stable':>7} {'Verdict':>12}")
    print("  " + "─"*65)
    for _, r in df_out.iterrows():
        print(f"  {r['model']:<30} {r['mean_auc']:>6.4f} {r['ci95_low']:>7.4f} "
              f"{'Yes' if r['stable_folds'] else 'No':>7} {r['verdict']:>12}")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Study Validation Pipeline')
    parser.add_argument('--data',     default='data/nego-data-final.csv')
    parser.add_argument('--outputs',  default='outputs/')
    parser.add_argument('--sections', default='1,2,3,4,5,6,7,8',
                        help='Comma-separated section numbers to run')
    args = parser.parse_args()

    out_dir = args.outputs.rstrip('/') + '/validation/'
    os.makedirs(out_dir, exist_ok=True)

    sections = [int(s) for s in args.sections.split(',')]

    print("=" * 65)
    print("STUDY VALIDATION PIPELINE")
    print(f"Data   : {args.data}")
    print(f"Out    : {out_dir}")
    print(f"Running sections: {sections}")
    print("=" * 65)

    df = pd.read_csv(args.data)
    df['text'] = df['text'].fillna('').astype(str)

    all_results = {}

    if 1 in sections:
        all_results['s1'] = section1_data_validation(df, out_dir)
    if 2 in sections:
        all_results['s2'] = section2_preprocessing(df, out_dir)
    if 3 in sections:
        all_results['s3'] = section3_emotion_measures(df, out_dir)
    if 4 in sections:
        all_results['s4'] = section4_ssm_grid(df, out_dir)
    if 5 in sections:
        all_results['s5'] = section5_annotation_validation(args.outputs)
    if 6 in sections:
        all_results['s6'] = section6_outcome_prediction(df, out_dir)
    if 7 in sections:
        all_results['s7'] = section7_robustness(df, out_dir)
    if 8 in sections:
        all_results['s89'] = section89_decision_rule(out_dir)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print(f"Results in: {out_dir}")
    for name, df_r in all_results.items():
        if df_r is not None:
            print(f"  {name}: {df_r.shape}")
    print("=" * 65)


if __name__ == '__main__':
    main()
