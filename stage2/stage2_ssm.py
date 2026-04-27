import numpy as np, pandas as pd, re, warnings
from scipy import linalg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
np.random.seed(42)

OUT = '/content/outputs/'

EMOTION_DIMS = ['amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust',
    'embarrassment','excitement','fear','gratitude','grief','joy',
    'love','nervousness','optimism','pride','realization','sadness']

EMOTION_SEEDS = {
    'amusement':     ['funny','laugh','joke','amusing','hilarious','humor','witty'],
    'anger':         ['unacceptable','ridiculous','furious','angry','frustrated','upset','unfair'],
    'annoyance':     ['come on','seriously','stop','enough','honestly','listen','fine'],
    'approval':      ['agree','okay','sure','absolutely','definitely','correct','right','fair','deal'],
    'caring':        ['help','support','understand','concern','assist','accommodate','important'],
    'confusion':     ['confused','not sure','unclear','lost','clarify','explain','follow'],
    'curiosity':     ['wonder','curious','how','why','interested','question','asking'],
    'desire':        ['want','wish','hope','need','prefer','would like','seeking','goal'],
    'disappointment':['disappointed','unfortunately','expected','hoped','shame','missed'],
    'disapproval':   ['disagree','reject','refuse','against','oppose','not okay','problem'],
    'disgust':       ['awful','terrible','horrible','appalling','offensive','worst'],
    'embarrassment': ['sorry','mistake','apologies','wrong','awkward','regret','excuse'],
    'excitement':    ['excited','wonderful','fantastic','amazing','thrilled','excellent'],
    'fear':          ['worried','afraid','concern','risk','nervous','uncertain','hesitant'],
    'gratitude':     ['thank','grateful','appreciate','thankful','thanks','kind'],
    'grief':         ['sad','loss','miss','unfortunate','difficult','hard','painful'],
    'joy':           ['happy','pleased','delighted','glad','enjoy','pleasure','satisfied'],
    'love':          ['love','adore','dear','devoted','cherish','fond','treasure'],
    'nervousness':   ['nervous','anxious','uneasy','tense','stress','worry','hesitate'],
    'optimism':      ['hope','confident','positive','better','improve','optimistic','believe'],
    'pride':         ['deserve','earned','value','maintain','position','principle','worth'],
    'realization':   ['realize','makes sense','understand','get it','noted','acknowledge','recognize'],
    'sadness':       ['sad','unfortunate','shame','pity','regret','down','difficult'],
}

# Compile patterns - safe escaping
def make_pattern(seeds):
    parts = [re.escape(s) for s in seeds]
    return re.compile(r'\b(' + '|'.join(parts) + r')\b', re.IGNORECASE)

PATTERNS = {em: make_pattern(seeds) for em, seeds in EMOTION_SEEDS.items()}
BC_RE    = re.compile(r'^\s*(yeah|mhm|mm|um|uh|okay|ok|yes|no|hi|hello|so|right|sure|huh|ah|oh|hmm|yep|nope|alright|well|hey)\s*$', re.IGNORECASE)
PRICE_RE = re.compile(r'\$?\d[\d,\.]+|\bprice\b|\boffer\b|\bbid\b', re.IGNORECASE)

print('='*60)
print('STAGE 2: COUPLED SSM ON SST EMOTIONAL TRAJECTORIES')
print('='*60)

print('\n[Step 1] Building dense emotion trajectories...')
df = pd.read_csv('/mnt/user-data/uploads/nego-data-final.csv')
df['text'] = df['text'].fillna('').astype(str)
df['is_bc'] = df['text'].apply(lambda t: len(t.strip())<=3 or bool(BC_RE.match(t)))

# Role assignment
role_map = {}
for cid, grp in df.groupby('conversation_id'):
    buyer = None
    for _, row in grp.sort_values('start_time').iterrows():
        if PRICE_RE.search(row['text']):
            buyer = row['speaker_id']; break
    if buyer is None: buyer = 0
    spks   = grp['speaker_id'].unique()
    seller = [s for s in spks if s != buyer]
    seller = seller[0] if seller else (1-buyer)
    role_map[cid] = {buyer:'buyer', seller:'seller'}

df['role'] = df.apply(
    lambda r: role_map.get(r['conversation_id'],{}).get(r['speaker_id'],'buyer'), axis=1)

# Binary detection + rolling smooth
for em, pat in PATTERNS.items():
    df[em] = df['text'].apply(lambda t: 1.0 if pat.search(t) else 0.0)

df = df.sort_values(['conversation_id','start_time']).reset_index(drop=True)
for em in EMOTION_DIMS:
    df[em] = df.groupby('conversation_id')[em].transform(
        lambda x: x.rolling(3, min_periods=1, center=True).mean())

df_sub = df[~df['is_bc']].copy().reset_index(drop=True)
E_all  = df_sub[EMOTION_DIMS].values
sparsity = (E_all==0).mean()*100
active   = (E_all>0).sum(axis=1).mean()
print(f'  Sparsity: {sparsity:.1f}%  (Stage 1 was 94.9%)')
print(f'  Active dims/turn: {active:.2f}  (Stage 1 was 1.17)')
df_sub.to_csv(OUT+'stage2_trajectories.csv', index=False)

print('\n[Step 2] Building coupled buyer-seller observations...')
conv_ids = sorted(df_sub['conversation_id'].unique())
conv_data = {}
for cid in conv_ids:
    grp = df_sub[df_sub['conversation_id']==cid].sort_values('start_time').reset_index(drop=True)
    outcome = grp['outcome'].iloc[0]
    T = len(grp)
    bE = np.zeros((T,23)); sE = np.zeros((T,23))
    lb = np.zeros(23);     ls = np.zeros(23)
    for t, (_, row) in enumerate(grp.iterrows()):
        ev = row[EMOTION_DIMS].values.astype(float)
        if row['role']=='buyer': lb=ev
        else:                    ls=ev
        bE[t]=lb; sE[t]=ls
    X = np.hstack([bE, sE])
    conv_data[cid] = {'X':X,'outcome':outcome,'T':T}

sequences = [conv_data[cid]['X'] for cid in conv_ids]
print(f'  {len(sequences)} conversations, obs_dim=46, avg_T={np.mean([len(s) for s in sequences]):.1f}')

# ── Kalman EM ──────────────────────────────────────────────────────────────
def run_em(sequences, k, p=46, max_iter=60, tol=0.5):
    eps = 1e-6
    A   = np.eye(k)*0.9 + np.random.randn(k,k)*0.01
    C   = np.random.randn(p,k)*0.05
    Q   = np.eye(k)*0.1
    R   = np.eye(p)*0.3
    mu0 = np.zeros(k)
    V0  = np.eye(k)
    prev_ll = -np.inf

    for itr in range(max_iter):
        T_tot=0; s_z=np.zeros(k); s_zz=np.zeros((k,k))
        s_zt_zt1=np.zeros((k,k)); s_zt1_zt1=np.zeros((k,k))
        s_xz=np.zeros((p,k)); s_xx=np.zeros((p,p))
        mu0_sum=np.zeros(k); V0_sum=np.zeros((k,k))
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
                mu_f[t]=(np.eye(k)-K@C)@mp+K@X[t]  # numerically stable
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

            T_tot+=T; mu0_sum+=mu_s[0]
            V0_sum+=V_s[0]+np.outer(mu_s[0],mu_s[0])
            for t in range(T):
                ez=mu_s[t]; ezz=V_s[t]+np.outer(ez,ez)
                s_zz+=ezz; s_xz+=np.outer(X[t],ez); s_xx+=np.outer(X[t],X[t])
            for t in range(T-1):
                s_zt_zt1 +=Vt_s[t]+np.outer(mu_s[t+1],mu_s[t])
                s_zt1_zt1+=V_s[t] +np.outer(mu_s[t],mu_s[t])

        n=len(sequences)
        mu0=mu0_sum/n
        V0 =V0_sum/n-np.outer(mu0,mu0)+eps*np.eye(k)
        A  =s_zt_zt1@np.linalg.pinv(s_zt1_zt1+eps*np.eye(k))
        Q  =(s_zz-A@s_zt_zt1.T)/T_tot; Q=0.5*(Q+Q.T)+eps*np.eye(k)
        C  =s_xz@np.linalg.pinv(s_zz+eps*np.eye(k))
        Rf =((s_xx-C@s_xz.T)/T_tot); R=np.diag(np.maximum(np.diag(Rf),eps))*np.eye(p)

        delta=total_ll-prev_ll
        if itr%10==0 or itr<3:
            print(f'    iter {itr+1:3d}: ll={total_ll:.1f}  Δ={delta:.1f}')
        if itr>0 and abs(delta)<tol:
            print(f'    Converged at iter {itr+1}')
            break
        prev_ll=total_ll

    T_tot_all=sum(len(X) for X in sequences)
    n_params=k*k+p*k+k*(k+1)//2+p
    bic=-2*total_ll+n_params*np.log(T_tot_all)
    sr=max(abs(np.linalg.eigvals(A)))
    return dict(A=A,C=C,Q=Q,R=R,mu0=mu0,V0=V0,ll=total_ll,bic=bic,sr=sr,smoothed=all_sm)

print('\n[Step 3] BIC-based latent dimension selection (k=2,3,4)...')
bic_results = {}
model_cache  = {}
for k in [2,3,4]:
    print(f'\n  --- k={k} ---')
    res = run_em(sequences, k=k, max_iter=60, tol=0.5)
    bic_results[k] = res['bic']
    model_cache[k] = res
    print(f'  k={k}: ll={res["ll"]:.1f}  BIC={res["bic"]:.1f}  spectral_r={res["sr"]:.4f}')

best_k = min(bic_results, key=bic_results.get)
print(f'\n  BIC scores: {bic_results}')
print(f'  Best k = {best_k}')

best = model_cache[best_k]
print(f'\n[Step 4] Final model (k={best_k}): spectral_r={best["sr"]:.4f}')

# Save SSM params
np.savez(OUT+'stage2_ssm_params.npz',
         A=best['A'],C=best['C'],Q=best['Q'],R=best['R'],
         mu0=best['mu0'],V0=best['V0'],best_k=best_k)

# Save latent states
print('\n[Step 5] Saving latent state trajectories...')
rows=[]
for cid,z_seq in zip(conv_ids, best['smoothed']):
    outcome=conv_data[cid]['outcome']
    for t,z in enumerate(z_seq):
        row={'conversation_id':cid,'turn':t,'outcome':outcome}
        for d in range(best_k): row[f'z_{d+1}']=round(float(z[d]),6)
        rows.append(row)
latent_df=pd.DataFrame(rows)
latent_df.to_csv(OUT+'stage2_latent_states.csv', index=False)
print(f'  stage2_latent_states.csv: {latent_df.shape}')

# Outcome prediction
print('\n[Step 6] Outcome prediction from latent trajectories...')
outcomes=[1 if conv_data[cid]['outcome']=='sale' else 0 for cid in conv_ids]
feats=[]
for cid,z_seq in zip(conv_ids, best['smoothed']):
    T=len(z_seq)
    zm=z_seq.mean(axis=0); zs=z_seq.std(axis=0)+1e-8
    zl=z_seq[max(0,2*T//3):].mean(axis=0)
    sl=np.array([np.polyfit(np.arange(T),z_seq[:,d],1)[0] for d in range(best_k)])
    X_obs=conv_data[cid]['X']
    div=np.abs(X_obs[:,:23]-X_obs[:,23:]).mean(axis=0)
    feats.append(np.concatenate([zm,zs,zl,sl,div]))

F=np.array(feats); y=np.array(outcomes)
F_sc=StandardScaler().fit_transform(F)
cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf=LogisticRegression(max_iter=1000, class_weight='balanced', C=0.1)
aucs=[]; preds=np.zeros(len(y))
for fold,(tr,te) in enumerate(cv.split(F_sc,y)):
    clf.fit(F_sc[tr],y[tr])
    prob=clf.predict_proba(F_sc[te])[:,1]
    preds[te]=prob; aucs.append(roc_auc_score(y[te],prob))
    print(f'  Fold {fold+1}: AUC={aucs[-1]:.4f}')

mean_auc=np.mean(aucs); overall_auc=roc_auc_score(y,preds)
result_df=pd.DataFrame({'conversation_id':conv_ids,'outcome':outcomes,'pred_prob':preds})
result_df.to_csv(OUT+'stage2_outcome_prediction.csv', index=False)

# Diagnostics
diag=[
    'STAGE 2 DIAGNOSTICS',
    '='*50,
    f'Dense trajectory sparsity  : {sparsity:.1f}%  (Stage 1: 94.9%)',
    f'Active dims per turn       : {active:.2f}  (Stage 1: 1.17)',
    '',
    'BIC model selection:',
] + [f'  k={k}: BIC={v:.1f}' + (' <- best' if k==best_k else '') for k,v in sorted(bic_results.items())] + [
    '',
    f'Best k                     : {best_k}',
    f'Final log-likelihood       : {best["ll"]:.2f}',
    f'Spectral radius A          : {best["sr"]:.4f}',
    f'Transition noise Q (mean)  : {np.diag(best["Q"]).mean():.4f}',
    f'Emission noise R (mean)    : {np.diag(best["R"]).mean():.4f}',
    '',
    'Outcome prediction (logistic, 5-fold CV):',
    f'  Mean CV AUC              : {mean_auc:.4f}',
    f'  Overall AUC              : {overall_auc:.4f}',
    '',
    'Baselines:',
    '  Manzoor et al. AUC t=60s : 0.9400 (fine-tuned GPT-4.1)',
    '  NL replication AUC t=90s : 0.7530 (zero-shot GPT-4o)',
    f'  Stage 2 SSM emotion AUC  : {overall_auc:.4f} (no price info)',
]
with open(OUT+'stage2_diagnostics.txt','w') as f:
    f.write('\n'.join(diag))

print()
print('='*60)
print('STAGE 2 COMPLETE')
print(f'  Sparsity fix     : {sparsity:.1f}% vs 94.9% (Stage 1)')
print(f'  Active dims/turn : {active:.2f} vs 1.17 (Stage 1)')
print(f'  Best SSM k       : {best_k}  (BIC = {bic_results[best_k]:.1f})')
print(f'  Spectral radius  : {best["sr"]:.4f}')
print(f'  Mean CV AUC      : {mean_auc:.4f}')
print(f'  Overall AUC      : {overall_auc:.4f}')
print('  Files saved to   :', OUT)
print('='*60)
