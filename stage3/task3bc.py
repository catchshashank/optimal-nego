"""
Stage 3B + 3C: Tactic × Latent State Overlay and Horse Race Regression
========================================================================

Task 3B — Tactic → Latent State Shift
  For each bargaining act (and sub-type), compute:
  - Mean z_t at the turn where the act occurs
  - Mean z_{t+1} (counterpart's NEXT turn) latent state
  - Shift = z_{t+1} - z_t  (the emotional response the act induces)
  This maps Heddaya et al.'s acts onto the SSM emotional space and
  identifies which acts cause the largest dyadic emotional shifts.

Task 3C — Horse Race Regression
  Nested logistic regression predicting sale outcome (binary):
  Model 1: Price features only
    - last_price_buyer, last_price_seller, price_gap,
      n_new_offers, concession_size
  Model 2: SSM latent features only
    - z_mean, z_std, z_last_third, z_slope (from Stage 2)
  Model 3: Price + SSM
  Model 4: Price + SSM + Tactic features
    - push_rate, comparison_rate, allowance_rate,
      push_constraint_rate, push_disparagement_rate,
      comparison_price_rate, comparison_quality_rate
  Test: Does Model 4 beat Model 1 on AUC?
  This is the test of the independent linguistic channel.

Theoretical grounding:
  De Dreu et al. (2000): emotional trajectories mediate tactic → outcome
  Lee & Ames (2017): constraint vs disparagement have opposite effects
  Heddaya et al. (2024): Push × Comparison sequences predict outcome
  Keltner et al. (2023): z_t captures dynamic emotional semantic space
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings, re, os
warnings.filterwarnings('ignore')

np.random.seed(42)
OUT_DIR = '/mnt/user-data/outputs/'

# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND MERGE
# ─────────────────────────────────────────────────────────────────────────────

def load_merged():
    ann    = pd.read_csv(OUT_DIR + 'stage3_annotated.csv')
    latent = pd.read_csv(OUT_DIR + 'stage2_latent_states.csv')

    # Substantive turns only
    ann_sub = ann[~ann['is_bc']].copy()
    ann_sub = ann_sub.sort_values(
        ['conversation_id','start_time']).reset_index(drop=True)
    ann_sub['turn'] = ann_sub.groupby('conversation_id').cumcount()

    merged = ann_sub.merge(latent, on=['conversation_id','turn'], how='inner')
    merged = merged.rename(columns={'outcome_x':'outcome'})
    merged['sale'] = (merged['outcome'] == 'sale').astype(int)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3B — TACTIC × LATENT STATE OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def task_3b(merged, z_cols):
    """
    For each bargaining act:
      1. Find all turns where act == 1
      2. Get z_t at that turn
      3. Get z_{t+1} — the NEXT turn in the same conversation
         (regardless of speaker — the dyadic response)
      4. Compute shift = z_{t+1} - z_t
      5. Report mean z_t, mean z_{t+1}, mean shift, and which z-dim shifts most

    Returns a DataFrame with one row per act × z-dimension.
    """
    results = []

    act_cols = {
        'new_offer':            'act_new_offer',
        'repeat_offer':         'act_repeat',
        'push':                 'act_push',
        'push_constraint':      None,   # from push_subtype
        'push_disparagement':   None,
        'push_neutral':         None,
        'comparison':           'act_comparison',
        'comparison_price':     None,
        'comparison_quality':   None,
        'comparison_mixed':     None,
        'allowance':            'act_allowance',
        'end':                  'act_end',
    }

    for act_name, col in act_cols.items():
        # Select turns with this act
        if col is not None:
            act_turns = merged[merged[col] == 1].copy()
        elif act_name.startswith('push_'):
            act_turns = merged[merged['push_subtype'] == act_name].copy()
        elif act_name.startswith('comparison_'):
            act_turns = merged[merged['comp_subtype'] == act_name].copy()
        else:
            continue

        if len(act_turns) < 10:
            continue

        # Get z_{t+1} for each act turn
        # Build lookup: (conv_id, turn) → z vector
        z_lookup = merged.set_index(['conversation_id','turn'])[z_cols]

        shifts = []
        z_t_vals  = []
        z_t1_vals = []

        for _, row in act_turns.iterrows():
            cid   = row['conversation_id']
            t     = row['turn']
            t1    = t + 1
            key_t  = (cid, t)
            key_t1 = (cid, t1)

            if key_t1 in z_lookup.index:
                z_t  = z_lookup.loc[key_t].values
                z_t1 = z_lookup.loc[key_t1].values
                shifts.append(z_t1 - z_t)
                z_t_vals.append(z_t)
                z_t1_vals.append(z_t1)

        if not shifts:
            continue

        shifts   = np.array(shifts)
        z_t_arr  = np.array(z_t_vals)
        z_t1_arr = np.array(z_t1_vals)

        mean_shift = shifts.mean(axis=0)
        dominant_dim = np.argmax(np.abs(mean_shift)) + 1

        for di, zc in enumerate(z_cols):
            results.append({
                'act':              act_name,
                'n_turns':          len(shifts),
                'z_dim':            zc,
                'mean_z_t':         round(z_t_arr[:, di].mean(), 5),
                'mean_z_t1':        round(z_t1_arr[:, di].mean(), 5),
                'mean_shift':       round(mean_shift[di], 5),
                'abs_shift':        round(abs(mean_shift[di]), 5),
                'dominant_dim':     f'z_{dominant_dim}',
            })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3C — HORSE RACE REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def extract_price_features(merged):
    """
    Extract conversation-level price features.
    Price is embedded in turn text — use annotated new_offer turns
    and reconstruct trajectory.
    """
    PRICE_RE = re.compile(r'\b(\d{3})\b')
    NON_PRICE = {'213','233','239','240','235','225','150','200','300',
                 '000','1846','1715','1875','1920'}

    def get_prices(text):
        prices = set()
        for m in PRICE_RE.finditer(str(text)):
            v = m.group(1)
            if v in NON_PRICE: continue
            iv = int(v)
            if 150 <= iv <= 300:
                prices.add(iv)
        return prices

    records = []
    for cid, grp in merged.groupby('conversation_id'):
        grp = grp.sort_values('turn')
        outcome = grp['sale'].iloc[0]

        buyer_prices  = []
        seller_prices = []

        for _, row in grp.iterrows():
            if row['act_new_offer'] == 1 or row['act_allowance'] == 1:
                ps = get_prices(row['text'])
                if ps:
                    if row['role'] == 'buyer':
                        buyer_prices.extend(ps)
                    else:
                        seller_prices.extend(ps)

        last_buyer  = buyer_prices[-1]  if buyer_prices  else 230
        last_seller = seller_prices[-1] if seller_prices else 230
        first_buyer = buyer_prices[0]   if buyer_prices  else 230
        first_seller= seller_prices[0]  if seller_prices else 240

        price_gap       = abs(last_seller - last_buyer)
        initial_gap     = abs(first_seller - first_buyer)
        buyer_movement  = abs(last_buyer  - first_buyer)  if len(buyer_prices)  > 1 else 0
        seller_movement = abs(last_seller - first_seller) if len(seller_prices) > 1 else 0
        n_offers        = grp['act_new_offer'].sum()
        n_allowances    = grp['act_allowance'].sum()

        records.append({
            'conversation_id': cid,
            'sale':            outcome,
            'last_buyer':      last_buyer,
            'last_seller':     last_seller,
            'price_gap':       price_gap,
            'initial_gap':     initial_gap,
            'buyer_movement':  buyer_movement,
            'seller_movement': seller_movement,
            'n_offers':        n_offers,
            'n_allowances':    n_allowances,
            'midpoint':        (last_buyer + last_seller) / 2,
        })

    return pd.DataFrame(records)


def extract_ssm_features(merged, z_cols):
    """Conversation-level SSM latent trajectory features (from Stage 2)."""
    records = []
    for cid, grp in merged.groupby('conversation_id'):
        grp = grp.sort_values('turn')
        Z   = grp[z_cols].values
        T   = len(Z)
        outcome = grp['sale'].iloc[0]

        z_mean      = Z.mean(axis=0)
        z_std       = Z.std(axis=0) + 1e-8
        z_last      = Z[max(0, 2*T//3):].mean(axis=0)
        z_slope     = np.array([
            np.polyfit(np.arange(T), Z[:, d], 1)[0] for d in range(len(z_cols))
        ])
        # Buyer-seller divergence (mean abs diff in obs space)
        buyer_z  = grp[grp['role']=='buyer'][z_cols].values
        seller_z = grp[grp['role']=='seller'][z_cols].values
        min_len  = min(len(buyer_z), len(seller_z))
        divergence = np.abs(
            buyer_z[:min_len] - seller_z[:min_len]
        ).mean(axis=0) if min_len > 0 else np.zeros(len(z_cols))

        rec = {'conversation_id': cid, 'sale': outcome}
        for i, zc in enumerate(z_cols):
            rec[f'{zc}_mean']      = z_mean[i]
            rec[f'{zc}_std']       = z_std[i]
            rec[f'{zc}_last']      = z_last[i]
            rec[f'{zc}_slope']     = z_slope[i]
            rec[f'{zc}_diverge']   = divergence[i]
        records.append(rec)

    return pd.DataFrame(records)


def extract_tactic_features(merged):
    """Conversation-level tactic rate features from Stage 3 annotation."""
    act_cols = ['act_new_offer','act_repeat','act_push',
                'act_comparison','act_allowance']
    records = []

    for cid, grp in merged.groupby('conversation_id'):
        T = len(grp)
        outcome = grp['sale'].iloc[0]

        rec = {'conversation_id': cid, 'sale': outcome}

        # Overall act rates
        for col in act_cols:
            rec[col.replace('act_','')] = grp[col].sum() / T

        # Layer 2 sub-type rates (as fraction of ALL turns)
        for sub_t in ['push_constraint','push_disparagement','push_neutral']:
            rec[sub_t] = (grp['push_subtype'] == sub_t).sum() / T
        for sub_t in ['comparison_price','comparison_quality','comparison_mixed']:
            rec[sub_t] = (grp['comp_subtype'] == sub_t).sum() / T

        # Buyer vs seller tactic asymmetry
        buyer  = grp[grp['role']=='buyer']
        seller = grp[grp['role']=='seller']
        if len(buyer) > 0:
            rec['buyer_push_rate']       = buyer['act_push'].mean()
            rec['buyer_comparison_rate'] = buyer['act_comparison'].mean()
            rec['buyer_allowance_rate']  = buyer['act_allowance'].mean()
        else:
            rec['buyer_push_rate'] = rec['buyer_comparison_rate'] = \
            rec['buyer_allowance_rate'] = 0.0

        if len(seller) > 0:
            rec['seller_push_rate']       = seller['act_push'].mean()
            rec['seller_comparison_rate'] = seller['act_comparison'].mean()
            rec['seller_allowance_rate']  = seller['act_allowance'].mean()
        else:
            rec['seller_push_rate'] = rec['seller_comparison_rate'] = \
            rec['seller_allowance_rate'] = 0.0

        # Push × Comparison sequence: n turns with both acts
        rec['push_then_comp'] = (
            (grp['act_push']==1) & (grp['act_comparison']==1)
        ).sum() / T

        records.append(rec)

    return pd.DataFrame(records)


def horse_race(price_df, ssm_df, tactic_df, n_splits=5):
    """
    4-model nested AUC comparison:
      M1: Price only
      M2: SSM only
      M3: Price + SSM
      M4: Price + SSM + Tactics
    """
    # Merge all feature sets
    base = price_df[['conversation_id','sale']].copy()
    df   = base.merge(price_df.drop('sale',axis=1), on='conversation_id')
    df   = df.merge(ssm_df.drop('sale',axis=1),    on='conversation_id')
    df   = df.merge(tactic_df.drop('sale',axis=1), on='conversation_id')

    price_feats  = [c for c in price_df.columns
                    if c not in ['conversation_id','sale']]
    ssm_feats    = [c for c in ssm_df.columns
                    if c not in ['conversation_id','sale']]
    tactic_feats = [c for c in tactic_df.columns
                    if c not in ['conversation_id','sale']]

    y = df['sale'].values

    models = {
        'M1_price':       price_feats,
        'M2_ssm':         ssm_feats,
        'M3_price_ssm':   price_feats + ssm_feats,
        'M4_full':        price_feats + ssm_feats + tactic_feats,
    }

    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)

    results = {}
    for model_name, feats in models.items():
        F    = df[feats].fillna(0).values
        F_sc = StandardScaler().fit_transform(F)
        aucs = []
        preds = np.zeros(len(y))
        for tr, te in cv.split(F_sc, y):
            clf.fit(F_sc[tr], y[tr])
            prob = clf.predict_proba(F_sc[te])[:, 1]
            preds[te] = prob
            aucs.append(roc_auc_score(y[te], prob))
        results[model_name] = {
            'mean_cv_auc':   round(np.mean(aucs), 4),
            'std_cv_auc':    round(np.std(aucs), 4),
            'overall_auc':   round(roc_auc_score(y, preds), 4),
            'n_features':    len(feats),
        }

    return results, df, models


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    diag = []
    log  = lambda s: (print(s), diag.append(str(s)))

    log("=" * 65)
    log("TASK 3B: TACTIC × LATENT STATE OVERLAY")
    log("TASK 3C: HORSE RACE REGRESSION")
    log("=" * 65)

    merged = load_merged()
    z_cols = ['z_1','z_2','z_3','z_4']

    log(f"\nMerged dataset: {merged.shape}")
    log(f"Conversations: {merged['conversation_id'].nunique()}")
    log(f"Sale rate: {merged.groupby('conversation_id')['sale'].first().mean():.1%}")

    # ── TASK 3B ──────────────────────────────────────────────────────────
    log("\n" + "─"*65)
    log("TASK 3B — Tactic → Latent State Shift (z_t → z_{t+1})")
    log("─"*65)

    shift_df = task_3b(merged, z_cols)
    shift_df.to_csv(OUT_DIR + 'task3b_tactic_shifts.csv', index=False)

    # Summary: dominant shift per act
    log("\n  Mean shift in dominant z-dimension per act:")
    log(f"  {'Act':<25} {'N':>5}  {'Dom dim':>7}  "
        f"{'Mean shift':>10}  {'z_t':>8}  {'z_t+1':>8}")
    log("  " + "-"*68)

    summary = shift_df.loc[
        shift_df.groupby('act')['abs_shift'].idxmax()
    ].reset_index(drop=True)

    for _, row in summary.sort_values('abs_shift', ascending=False).iterrows():
        direction = "↑" if row['mean_shift'] > 0 else "↓"
        log(f"  {row['act']:<25} {row['n_turns']:>5}  "
            f"{row['z_dim']:>7}  "
            f"{row['mean_shift']:>+9.5f}{direction}  "
            f"{row['mean_z_t']:>8.5f}  "
            f"{row['mean_z_t1']:>8.5f}")

    # Key finding: which act causes the largest z-shift toward sale signature?
    # Sale conversations have lower z_1 and z_3 (from Stage 2 analysis)
    # So acts that DECREASE z_1 or z_3 are moving toward sale
    log("\n  Acts moving toward SALE signature (decreasing z_1/z_3):")
    sale_acts = shift_df[
        (shift_df['z_dim'].isin(['z_1','z_3'])) &
        (shift_df['mean_shift'] < 0)
    ].sort_values('mean_shift')
    for _, row in sale_acts.head(5).iterrows():
        log(f"    {row['act']:<25} {row['z_dim']}  shift={row['mean_shift']:+.5f}")

    # ── TASK 3C ──────────────────────────────────────────────────────────
    log("\n" + "─"*65)
    log("TASK 3C — Horse Race Regression (4 nested models)")
    log("─"*65)

    log("\n  Extracting feature sets...")
    price_df  = extract_price_features(merged)
    ssm_df    = extract_ssm_features(merged, z_cols)
    tactic_df = extract_tactic_features(merged)
    log(f"  Price features  : {len([c for c in price_df.columns if c not in ['conversation_id','sale']])}")
    log(f"  SSM features    : {len([c for c in ssm_df.columns if c not in ['conversation_id','sale']])}")
    log(f"  Tactic features : {len([c for c in tactic_df.columns if c not in ['conversation_id','sale']])}")

    log("\n  Running 5-fold stratified CV...")
    horse_results, feat_df, models = horse_race(price_df, ssm_df, tactic_df)

    log("\n  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    log("  │ Model               │ N feats  │ Mean AUC │  Std AUC │ Ovrl AUC │")
    log("  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    for mname, res in horse_results.items():
        log(f"  │ {mname:<19} │ {res['n_features']:>8} │ "
            f"{res['mean_cv_auc']:>8.4f} │ "
            f"{res['std_cv_auc']:>8.4f} │ "
            f"{res['overall_auc']:>8.4f} │")
    log("  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘")

    # Key test: M4 vs M1
    m1_auc = horse_results['M1_price']['mean_cv_auc']
    m4_auc = horse_results['M4_full']['mean_cv_auc']
    gain   = m4_auc - m1_auc
    m2_auc = horse_results['M2_ssm']['mean_cv_auc']
    m3_auc = horse_results['M3_price_ssm']['mean_cv_auc']

    log(f"\n  KEY RESULTS:")
    log(f"  SSM adds to price          : M3 - M1 = {m3_auc - m1_auc:+.4f}")
    log(f"  Tactics add to price+SSM   : M4 - M3 = {m4_auc - m3_auc:+.4f}")
    log(f"  Full linguistic channel    : M4 - M1 = {gain:+.4f}")
    log(f"  Emotion alone vs price     : M2 - M1 = {m2_auc - m1_auc:+.4f}")

    if gain > 0:
        log(f"\n  ✓ Linguistic channel (SSM + tactics) adds {gain:+.4f} AUC above")
        log(f"    price features alone. The language channel is non-trivial.")
    else:
        log(f"\n  ✗ M4 does not beat M1. Price dominates in this corpus.")
        log(f"    Consider: corpus is single-item, price ZOPA is tight ($225-235k).")

    # Save outputs
    shift_df.to_csv(OUT_DIR + 'task3b_tactic_shifts.csv', index=False)
    pd.DataFrame(horse_results).T.to_csv(
        OUT_DIR + 'task3c_horse_race.csv')
    feat_df.to_csv(OUT_DIR + 'task3c_features.csv', index=False)

    log(f"\n  Files saved to {OUT_DIR}")
    log("  task3b_tactic_shifts.csv")
    log("  task3c_horse_race.csv")
    log("  task3c_features.csv")

    with open(OUT_DIR + 'task3bc_diagnostics.txt', 'w') as f:
        f.write('\n'.join(diag))

    log("\n" + "=" * 65)
    log("TASKS 3B + 3C COMPLETE")
    log("=" * 65)

    return shift_df, horse_results, feat_df


if __name__ == '__main__':
    shift_df, horse_results, feat_df = main()
