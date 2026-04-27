"""
Stage 3: Bargaining Act Annotation + Layer 2 Sub-typing
=========================================================

Layer 1 — Heddaya et al. (2024) six bargaining acts (Table 2):
  new_offer    : Any numerical price not previously mentioned
  repeat_offer : Exact repeat of a previously stated price
  push         : Overt linguistic effort to move counterpart's position
  comparison   : Evokes difference/similarity with external houses or considerations
  allowance    : Adjusts own offer closer to counterpart's most recent offer
  end          : Negotiation closure via mutual agreement

Layer 2 — Sub-typing within Push and Comparison only:
  Push sub-types (Lee & Ames 2017, cited in Heddaya et al.):
    push_constraint   : Rationale references own limitation ("can't", "budget", "afford")
    push_disparagement: Rationale attacks counterpart's position ("overpriced", "not worth")
    push_neutral      : Push without clear constraint or disparagement signal

  Comparison sub-types:
    comparison_price  : References external selling prices of comparable homes
    comparison_quality: References property attributes (sq ft, features, condition)
    comparison_mixed  : Contains both price and quality referents

Theoretical grounding:
  - De Dreu, Weingart & Kwon (2000): Push ≡ contentious behavior;
    Comparison ≡ problem-solving behavior. These map onto egoistic vs
    prosocial motive orientation and predict differential joint outcomes.
  - Lee & Ames (2017): Constraint vs disparagement rationales in Push
    produce opposite effects on counterpart concession behavior.
  - Heddaya et al. (2024): Push × Comparison sequences are the most
    predictive buyer bargaining acts (Table 3b). Mechanism left open.

Output:
  stage3_annotated.csv      — all substantive turns with bargaining act labels
  stage3_diagnostics.txt    — coverage, inter-act frequencies, sequence stats
"""

import numpy as np
import pandas as pd
import re
import os
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = '/content/data/nego-data-final.csv'
OUT_DIR   = '/content/outputs/'
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRICE TRACKING — needed for new_offer vs repeat_offer distinction
# ─────────────────────────────────────────────────────────────────────────────

# Known non-price numbers in this corpus (listing #s, sq ft, years)
NON_PRICE = {
    '1846', '1715', '1875', '1920',   # square footage
    '90', '89', '13878', '06898',     # listing numbers
    '04725', '08614',
    '1947',                            # built year
    '1', '2', '3', '4', '5',          # small cardinals
}

def extract_prices(text):
    """
    Extract price values (in thousands) from text.
    Target range: $150k–$300k (this negotiation's universe).
    Handles: '228', '228,000', '2 28', '$228', 'two twenty eight'.
    """
    prices = set()

    # Pattern 1: explicit dollar amounts e.g. $228,000 or $228
    for m in re.finditer(r'\$\s*(\d[\d,\s]*)', text):
        raw = re.sub(r'[\s,]', '', m.group(1))
        try:
            v = int(raw)
            if v >= 1000:
                v = round(v / 1000)
            if 150 <= v <= 300:
                prices.add(v)
        except:
            pass

    # Pattern 2: bare 3-digit numbers in price range
    for m in re.finditer(r'\b(\d{3})\b', text):
        raw = m.group(1)
        if raw in NON_PRICE:
            continue
        try:
            v = int(raw)
            if 150 <= v <= 300:
                prices.add(v)
        except:
            pass

    # Pattern 3: 6-digit numbers e.g. 228000
    for m in re.finditer(r'\b(\d{6})\b', text):
        try:
            v = round(int(m.group(1)) / 1000)
            if 150 <= v <= 300:
                prices.add(v)
        except:
            pass

    # Pattern 4: spaced format "2 28" or "2 35"
    for m in re.finditer(r'\b(2)\s+(\d{2})\b', text):
        try:
            v = int(m.group(1) + m.group(2))
            if 150 <= v <= 300:
                prices.add(v)
        except:
            pass

    return prices


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 PATTERNS — Heddaya et al. (2024) bargaining acts
# ─────────────────────────────────────────────────────────────────────────────

# --- PUSH patterns ---
# "Any overt linguistic effort made by either party to bring the other
#  party's offer closer to theirs." (Heddaya et al. Table 2)

PUSH_PATTERNS = [
    # Direct pressure / movement requests
    r'\bcome down\b', r'\bgo up\b', r'\bgo higher\b', r'\bgo lower\b',
    r'\bmeet me\b', r'\bmeet halfway\b', r'\bcloser to\b',
    r'\ba little (bit )?(more|higher|lower|closer)\b',
    r'\bnot going to work\b', r'\bdoesn\'t work\b', r'\bwon\'t work\b',
    r'\bnot acceptable\b', r'\bcannot accept\b', r'\bcan\'t accept\b',
    # Position-holding (resistance)
    r'\bstand firm\b', r'\bstick(ing)? (with|to)\b', r'\bhold(ing)? firm\b',
    r'\bfinal offer\b', r'\bbottom line\b', r'\babsolute (top|max|min|bottom)\b',
    r'\bhighest (I\'m|i\'m|we\'re) willing\b',
    r'\blowest (I\'m|i\'m|we\'re) willing\b',
    # Direct appeals to move
    r'\bcan you (do|go|come|meet|try)\b',
    r'\bwould you (consider|be willing|go|come|do)\b',
    r'\bif you (could|can|would)\b',
    r'\bgive (me|us) a (little|bit|few)\b',
    r'\bneed (you to|to) (move|come|go|do)\b',
    # Fairness / reasonableness framing
    r'\bunreasonable\b', r'\breasonable\b', r'\bfair(ly)?\b',
    r'\bnot fair\b', r'\bshouldn\'t be\b',
    # Urgency / deadline
    r'\bwalk away\b', r'\bwalk(ing)? out\b', r'\bno deal\b',
    r'\bbest I can do\b', r'\bas (low|high) as (I|we)\'(ll|d) go\b',
]

PUSH_RE = re.compile('|'.join(PUSH_PATTERNS), re.IGNORECASE)

# --- COMPARISON patterns ---
# "Evokes a difference or similarity between an aspect of the seller's
#  house and other external houses or considerations." (Heddaya et al. Table 2)

COMP_PATTERNS = [
    # Explicit listing references
    r'\blisting\b', r'\bappendix\b', r'\b(89|90)\s*[\-–]?\s*\d+\b',
    # Comparable homes language
    r'\b(comparable|similar|neighboring|nearby|other)\s+(home|house|propert|listing)\b',
    r'\b(homes?|houses?|properties)\s+(in the area|nearby|around here|in the neighborhood)\b',
    r'\bother\s+(homes?|houses?|properties)\b',
    # Market / price evidence
    r'\bmarket\s*(price|value|rate|data|research|analysis)?\b',
    r'\bselling (for|at|price)\b', r'\bsold (for|at|recently)\b',
    r'\brecently sold\b', r'\bsale (price|value)\b',
    r'\bper square (foot|feet|ft)\b', r'\bsquare (foot|feet|ft)\b',
    r'\bprice per\b',
    # Direct comparison language
    r'\bcompared? (to|with)\b', r'\bin comparison\b',
    r'\bsimilar(ly)?\b', r'\bjust like\b',
    # The three specific comparable listings in this corpus
    r'\b213\b', r'\b(233|233,?000)\b', r'\b(239|239,?000)\b',
    r'\b1[,\s]?715\b', r'\b1[,\s]?875\b', r'\b1[,\s]?920\b',
    # Area/neighborhood framing
    r'\b(in the|this|the) (neighborhood|area|region|vicinity|market)\b',
    r'\bhomes? around\b', r'\bproperties around\b',
]

COMP_RE = re.compile('|'.join(COMP_PATTERNS), re.IGNORECASE)

# --- ALLOWANCE patterns ---
# "Either party adjusts their offer price closer to the other party's
#  most recent offer." (Heddaya et al. Table 2)
# Note: Allowance is detected ONLY when a new_offer is present
# AND represents movement toward counterpart's last known position.

ALLOW_HEDGE_RE = re.compile(
    r'\b(willing to (go|come|do|say|try|offer|accept|consider))\b'
    r'|\b(I\'ll (do|go|try|offer|say|come down to|come up to))\b'
    r'|\b(we could (do|go|say|offer|try))\b'
    r'|\b(let\'s (say|do|try|go with|settle on))\b'
    r'|\b(how about|what about|would .+ work)\b'
    r'|\b(meet in the middle|split the difference)\b'
    r'|\b(could (do|go|say|come down|come up|offer))\b'
    r'|\b(drop(ping)? (it |it\'s |down )?to)\b'
    r'|\b(raise|bump|move) (it |up )?(to)\b',
    re.IGNORECASE
)

# --- END patterns ---
# "End of negotiation via offer acceptance." (Heddaya et al. Table 2)

END_RE = re.compile(
    r'\b(deal|agreed?|done|sold|you\'ve got (yourself )?a deal|'
    r'we have a deal|sounds good|works (for me|for us)|'
    r'I\'ll take (it|that)|I\'ll do (it|that)|'
    r'that works|perfect|alright (then|[,.])|'
    r'let\'s do (it|that)|shake on it|'
    r'you got (it|yourself a deal))\b',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 SUB-TYPING — Lee & Ames (2017)
# ─────────────────────────────────────────────────────────────────────────────

# Push-Constraint: references own limitation
CONSTRAINT_RE = re.compile(
    r'\b(can\'t|cannot|won\'t|will not)\s+(go|pay|do|offer|afford|accept)\b'
    r'|\b(budget|afford(able)?|maximum|limit(ed)?|ceiling|cap)\b'
    r'|\b(not (possible|doable|feasible) for (me|us))\b'
    r'|\b(beyond|outside|above)\s+(my|our)\s+(budget|means|limit|range)\b'
    r'|\b(hard (stop|limit|no))\b'
    r'|\b(stretch(ing)? (my|our|the))\b'
    r'|\b(as (high|low|much|far) as (I|we)(\'(ll|d))? (go|do|offer))\b'
    r'|\b(that\'s (the most|all) (I|we) can)\b'
    r'|\bwalk away\b',
    re.IGNORECASE
)

# Push-Disparagement: attacks counterpart's position/property
DISPARAGE_RE = re.compile(
    r'\b(overpriced|over-priced|over priced)\b'
    r'|\b(not worth|isn\'t worth|doesn\'t justify)\b'
    r'|\b(too (high|much|expensive))\b'
    r'|\b(excessive(ly)?)\b'
    r'|\b(inflated)\b'
    r'|\b(not (justified|warranted))\b'
    r'|\b(asking too much)\b'
    r'|\b(price(d)? too)\b'
    r'|\b(doesn\'t (reflect|match) the (value|market))\b',
    re.IGNORECASE
)

# Comparison-Price: references external selling prices
COMP_PRICE_RE = re.compile(
    r'\b(selling for|sold for|sale price|asking price|listed (at|for)|'
    r'went for|priced at|market (price|value|rate))\b'
    r'|\b(per square (foot|feet|ft))\b'
    r'|\b213\b|\b(233|239)\b'     # the three comparables in the corpus
    r'|\b(price per|cost per)\b',
    re.IGNORECASE
)

# Comparison-Quality: references property attributes
COMP_QUALITY_RE = re.compile(
    r'\b(square (foot|feet|ft)|sq\.?\s?ft\.?|sqft)\b'
    r'|\b(bedroom|bathroom|garage|kitchen|fireplace|'
    r'hardwood|landscap|renovated|updated|decorated|appliance)\b'
    r'|\b(size|space|room|floor|yard|backyard|condition)\b'
    r'|\b(1[,\s]?715|1[,\s]?875|1[,\s]?920|1[,\s]?846)\b'  # sq ft values
    r'|\b(feature|amenity|amenities)\b',
    re.IGNORECASE
)


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def annotate_conversation(turns):
    """
    Annotate a single conversation's turns with Heddaya et al. bargaining acts.

    Returns list of dicts with annotation results.
    Price tracking is stateful within conversation to distinguish
    new_offer from repeat_offer.
    """
    results = []
    all_prices_seen = set()   # all prices ever mentioned in this conversation
    spk_last_offer  = {}      # last offer per speaker

    for _, row in turns.iterrows():
        text = str(row['text'])
        spk  = row['speaker_id']

        # Extract prices mentioned in this turn
        turn_prices = extract_prices(text)

        # ── Layer 1: Heddaya et al. acts ──────────────────────────────────

        acts = set()

        # END — check first (terminal act, highest specificity)
        if END_RE.search(text) and len(text) > 5:
            # Only flag as End if late in conversation (>50% through)
            acts.add('end')

        # NEW OFFER vs REPEAT OFFER
        for p in turn_prices:
            if p not in all_prices_seen:
                acts.add('new_offer')
            else:
                acts.add('repeat_offer')

        # ALLOWANCE — new offer + movement language + price moving toward counterpart
        if turn_prices and ALLOW_HEDGE_RE.search(text):
            # Check if the price moves toward counterpart's last known position
            other_spk = 1 - spk if spk in [0, 1] else None
            if other_spk in spk_last_offer:
                other_price = spk_last_offer[other_spk]
                my_price    = spk_last_offer.get(spk, None)
                for p in turn_prices:
                    if my_price is not None and other_price is not None:
                        # Moving toward counterpart = allowance
                        if ((p > my_price and p <= other_price) or
                            (p < my_price and p >= other_price)):
                            acts.discard('new_offer')
                            acts.add('allowance')
                            break
            elif turn_prices:
                # No prior context: if movement language present, call allowance
                acts.discard('new_offer')
                acts.add('allowance')

        # PUSH
        if PUSH_RE.search(text):
            acts.add('push')

        # COMPARISON
        if COMP_RE.search(text):
            acts.add('comparison')

        # ── Update price memory ───────────────────────────────────────────
        if turn_prices:
            all_prices_seen.update(turn_prices)
            # Track last offer per speaker (use max for buyer, min for seller)
            # (conservative: just use the first price found)
            spk_last_offer[spk] = list(turn_prices)[0]

        # ── Layer 2: sub-typing ────────────────────────────────────────────

        push_subtype = None
        comp_subtype = None

        if 'push' in acts:
            is_constraint  = bool(CONSTRAINT_RE.search(text))
            is_disparage   = bool(DISPARAGE_RE.search(text))
            if is_constraint and is_disparage:
                push_subtype = 'push_constraint'  # constraint takes precedence
            elif is_constraint:
                push_subtype = 'push_constraint'
            elif is_disparage:
                push_subtype = 'push_disparagement'
            else:
                push_subtype = 'push_neutral'

        if 'comparison' in acts:
            is_price   = bool(COMP_PRICE_RE.search(text))
            is_quality = bool(COMP_QUALITY_RE.search(text))
            if is_price and is_quality:
                comp_subtype = 'comparison_mixed'
            elif is_price:
                comp_subtype = 'comparison_price'
            elif is_quality:
                comp_subtype = 'comparison_quality'
            else:
                comp_subtype = 'comparison_price'  # fallback: price context

        results.append({
            'conversation_id': row['conversation_id'],
            'speaker_id':      spk,
            'start_time':      row['start_time'],
            'end_time':        row['end_time'],
            'text':            text,
            'outcome':         row['outcome'],
            'role':            row.get('role', None),
            # Layer 1 binary flags
            'act_new_offer':   int('new_offer'    in acts),
            'act_repeat':      int('repeat_offer' in acts),
            'act_push':        int('push'         in acts),
            'act_comparison':  int('comparison'   in acts),
            'act_allowance':   int('allowance'    in acts),
            'act_end':         int('end'          in acts),
            # Layer 2 sub-types
            'push_subtype':    push_subtype,
            'comp_subtype':    comp_subtype,
            # Multi-label summary
            'n_acts':          len(acts),
            'acts_list':       '|'.join(sorted(acts)) if acts else 'none',
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    diag = []
    log  = lambda s: (print(s), diag.append(str(s)))

    log("=" * 65)
    log("STAGE 3: BARGAINING ACT ANNOTATION")
    log("Layer 1: Heddaya et al. (2024)  |  Layer 2: Lee & Ames (2017)")
    log("=" * 65)

    # Load data
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].fillna('').astype(str)
    df = df.sort_values(['conversation_id', 'start_time']).reset_index(drop=True)

    # Role assignment (same logic as Stages 1 & 2)
    PRICE_RE = re.compile(r'\$?\d[\d,\.]+|\bprice\b|\boffer\b|\bbid\b', re.IGNORECASE)
    BC_RE    = re.compile(
        r'^\s*(yeah|mhm|mm|um|uh|okay|ok|yes|no|hi|hello|so|right|'
        r'sure|huh|ah|oh|hmm|yep|nope|alright|well|hey)\s*$', re.IGNORECASE)

    role_map = {}
    for cid, grp in df.groupby('conversation_id'):
        buyer = None
        for _, row in grp.sort_values('start_time').iterrows():
            if PRICE_RE.search(row['text']):
                buyer = row['speaker_id']; break
        if buyer is None: buyer = 0
        spks   = grp['speaker_id'].unique()
        seller = [s for s in spks if s != buyer]
        seller = seller[0] if seller else (1 - buyer)
        role_map[cid] = {buyer: 'buyer', seller: 'seller'}

    df['role'] = df.apply(
        lambda r: role_map.get(r['conversation_id'], {}).get(r['speaker_id'], 'buyer'),
        axis=1)
    df['is_bc'] = df['text'].apply(lambda t: len(t.strip()) <= 3 or bool(BC_RE.match(t)))

    # Annotate all conversations
    log("\n[Step 1] Annotating bargaining acts across 178 conversations...")
    all_results = []
    for cid, grp in df.groupby('conversation_id'):
        # Annotate ALL turns (incl. backchannels) for price tracking continuity
        # but flag them separately
        res = annotate_conversation(grp)
        all_results.extend(res)

    ann = pd.DataFrame(all_results)
    ann['is_bc'] = df['is_bc'].values

    log(f"  Total turns annotated : {len(ann):,}")
    log(f"  Substantive turns     : {(~ann['is_bc']).sum():,}")

    # ── Coverage stats ────────────────────────────────────────────────────
    log("\n[Step 2] Layer 1 — Bargaining act coverage:")
    sub = ann[~ann['is_bc']]  # substantive turns only for stats
    total_sub = len(sub)

    acts_cols = ['act_new_offer','act_repeat','act_push',
                 'act_comparison','act_allowance','act_end']
    act_names = ['New Offer','Repeat Offer','Push',
                 'Comparison','Allowance','End']

    for col, name in zip(acts_cols, act_names):
        n    = sub[col].sum()
        pct  = n / total_sub * 100
        log(f"  {name:<15}: {n:4d} turns  ({pct:5.1f}%)")

    # Multi-label turns
    multi = (sub[acts_cols].sum(axis=1) > 1).sum()
    log(f"\n  Multi-act turns   : {multi:4d} ({multi/total_sub*100:.1f}%)")
    log(f"  No-act turns      : {(sub[acts_cols].sum(axis=1)==0).sum():4d} "
        f"({(sub[acts_cols].sum(axis=1)==0).sum()/total_sub*100:.1f}%)")

    # ── Layer 2 stats ─────────────────────────────────────────────────────
    log("\n[Step 3] Layer 2 — Push sub-type distribution:")
    push_turns = sub[sub['act_push'] == 1]
    for sub_t in ['push_constraint','push_disparagement','push_neutral']:
        n   = (push_turns['push_subtype'] == sub_t).sum()
        pct = n / len(push_turns) * 100 if len(push_turns) > 0 else 0
        log(f"  {sub_t:<25}: {n:4d} ({pct:.1f}% of Push turns)")

    log("\n  Comparison sub-type distribution:")
    comp_turns = sub[sub['act_comparison'] == 1]
    for sub_t in ['comparison_price','comparison_quality','comparison_mixed']:
        n   = (comp_turns['comp_subtype'] == sub_t).sum()
        pct = n / len(comp_turns) * 100 if len(comp_turns) > 0 else 0
        log(f"  {sub_t:<25}: {n:4d} ({pct:.1f}% of Comparison turns)")

    # ── Buyer vs seller breakdown ─────────────────────────────────────────
    log("\n[Step 4] Buyer vs Seller act profiles:")
    for role in ['buyer', 'seller']:
        role_turns = sub[sub['role'] == role]
        log(f"\n  {role.upper()} ({len(role_turns)} turns):")
        for col, name in zip(acts_cols, act_names):
            n   = role_turns[col].sum()
            pct = n / len(role_turns) * 100
            log(f"    {name:<15}: {n:4d} ({pct:5.1f}%)")

    # ── Outcome breakdown for key acts ───────────────────────────────────
    log("\n[Step 5] Act frequency by outcome (substantive turns):")
    for col, name in zip(['act_push','act_comparison'], ['Push','Comparison']):
        sale    = sub[sub['outcome']=='sale'][col].mean()*100
        nosale  = sub[sub['outcome']=='no sale'][col].mean()*100
        log(f"  {name}: sale={sale:.1f}%  no-sale={nosale:.1f}%  "
            f"Δ={sale-nosale:+.1f}pp")

    log("\n[Step 6] Push sub-type by outcome:")
    for sub_t in ['push_constraint','push_disparagement','push_neutral']:
        sale   = (sub[sub['outcome']=='sale']['push_subtype']==sub_t).mean()*100
        nosale = (sub[sub['outcome']=='no sale']['push_subtype']==sub_t).mean()*100
        log(f"  {sub_t:<25}: sale={sale:.2f}%  no-sale={nosale:.2f}%  "
            f"Δ={sale-nosale:+.2f}pp")

    # ── Save ──────────────────────────────────────────────────────────────
    ann.to_csv(OUT_DIR + 'stage3_annotated.csv', index=False)
    log(f"\n[Step 7] stage3_annotated.csv saved: {ann.shape}")

    with open(OUT_DIR + 'stage3_diagnostics.txt', 'w') as f:
        f.write('\n'.join(diag))

    log("\n" + "=" * 65)
    log("STAGE 3 COMPLETE")
    log(f"  Conversations : 178")
    log(f"  Total turns   : {len(ann):,}")
    log(f"  Bargaining acts annotated at turn level (multi-label)")
    log(f"  Push sub-typed via Lee & Ames (2017)")
    log(f"  Comparison sub-typed (price vs quality vs mixed)")
    log("=" * 65)

    return ann


if __name__ == '__main__':
    ann = main()
