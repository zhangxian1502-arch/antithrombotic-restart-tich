# -*- coding: utf-8 -*-
"""Build pooled T3-v3 dataset: sicdb from v3 (fixed units), others from v2.
Save pool CSV + recompute Table-1 AT distribution and Firth ext models."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

BASE = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data'
OUT_POOL = f'{BASE}/pool_t3_v3.csv'

frames = []
for db in ['mimic','eicu','nwicu','umcdb']:
    df = pd.read_csv(f'{BASE}/{db}_t3_v2.csv', low_memory=False)
    df['db'] = db
    frames.append(df)
sic = pd.read_csv(f'{BASE}/sicdb_t3_v3.csv', low_memory=False)
sic['db'] = 'sicdb'
frames.append(sic)
pool = pd.concat(frames, ignore_index=True)

for c in ['age','gcs_total','gcs_motor_only','charlson_age','hr1','sbp1','spo2_1','temp_c1',
          'l_wbc','l_cr','l_glu','l_na','l_k','l_lac','l_alb','l_tbili','vaso_flag','mv_flag',
          'neuro_surgery','admit_to_icu_h','cc_cvd','cc_renal','cc_dm','cc_cancer','hosp_death',
          'hb_drop2','tx_any','thrombosis_any','resume_h']:
    if c in pool.columns: pool[c] = pd.to_numeric(pool[c], errors='coerce')
for c in ['baseline_at','ever_at','reversal','resume_cls']:
    pool[c] = pool[c].where(pd.notna(pool[c]), None)
pool['at_exposed'] = pool['baseline_at'].notna().astype(int)
pool['ever_exposed'] = pool['ever_at'].notna().astype(int)
pool['reversal_any'] = pool['reversal'].notna().astype(int)
pool['restarted'] = pool['resume_cls'].notna().astype(int)
pool['gcs_any'] = pool['gcs_total'].where(pool['gcs_total'].notna(), pool['gcs_motor_only'])
pool['male'] = (pool['sex'].astype(str).str.lower().str.startswith('m')).astype(int)
pool.to_csv(OUT_POOL, index=False)

at = pool[pool['at_exposed']==1].copy()
print('POOL n =', len(pool))
print('per db:', pool['db'].value_counts().to_dict())
print('AT n =', len(at))
print('AT per db:', at['db'].value_counts().to_dict())
print('AT class labels:', at['baseline_at'].value_counts().to_dict())
print('restart n (AT):', int(at['restarted'].sum()))
print('reversal n (AT):', int(at['reversal_any'].sum()))
print('ever n =', int(pool['ever_exposed'].sum()))

# ---- Firth ext models on new pool ----
EXT = ['hr1','sbp1','spo2_1','temp_c1','l_wbc','l_cr','l_glu','l_na','l_k','l_lac','l_alb','l_tbili',
       'vaso_flag','mv_flag','neuro_surgery','admit_to_icu_h','cc_cvd','cc_renal','cc_dm','cc_cancer']
DBS = ['mimic','eicu','nwicu','sicdb','umcdb']

def firth_fit(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, p = X.shape
    beta = np.zeros(p)
    for it in range(300):
        eta = X @ beta
        mu = 1/(1+np.exp(-np.clip(eta, -40, 40)))
        w = mu*(1-mu)
        XWX = X.T @ (X * w[:, None])
        try:
            XWXI = np.linalg.inv(XWX)
        except np.linalg.LinAlgError:
            XWX = XWX + np.eye(p)*1e-8; XWXI = np.linalg.inv(XWX)
        h = np.sum((X @ XWXI) * (X * w[:, None]), axis=1)
        score = X.T @ (y - mu + h*(0.5 - mu))
        try:
            delta = np.linalg.solve(XWX, score)
        except np.linalg.LinAlgError:
            delta = np.linalg.solve(XWX + np.eye(p)*1e-8, score)
        beta = beta + delta
        if np.max(np.abs(delta)) < 1e-8: break
    eta = X @ beta; mu = 1/(1+np.exp(-np.clip(eta,-40,40))); w = mu*(1-mu)
    XWX = X.T @ (X * w[:, None])
    try:
        cov = np.linalg.inv(XWX)
    except np.linalg.LinAlgError:
        cov = np.linalg.inv(XWX + np.eye(p)*1e-8)
    se = np.sqrt(np.diag(cov))
    return beta, se

def firth_run(outcome, exposure, data):
    d = data[data[outcome].notna()].copy()
    if d[exposure].nunique() < 2 or d[outcome].sum() < 5:
        return {'error': 'insufficient', 'n': int(len(d))}
    Xd = pd.DataFrame(index=d.index)
    Xd['age'] = d['age']; Xd['male'] = d['male']; Xd['gcs'] = d['gcs_any']; Xd['cc_age'] = d['charlson_age']
    for db in DBS[1:]:
        Xd[f'db_{db}'] = (d['db']==db).astype(int)
    for c in EXT:
        if c in d.columns: Xd[c] = d[c]
    for c in Xd.columns:
        if Xd[c].dtype.kind in 'fiu': Xd[c] = Xd[c].fillna(Xd[c].median())
    Xd = Xd[[c for c in Xd.columns if Xd[c].nunique() > 1]]
    Xd[exposure] = d[exposure].values
    y = d[outcome].astype(float)
    Xm = Xd.values.astype(float)
    try:
        beta, se = firth_fit(y, np.column_stack([np.ones(len(d)), Xm]))
    except Exception as e:
        return {'error': str(e)[:120], 'n': int(len(d))}
    idx = Xd.columns.get_loc(exposure) + 1
    b = beta[idx]; s = se[idx]
    from scipy.stats import norm
    return {'or': float(np.exp(b)), 'lo': float(np.exp(b-1.96*s)), 'hi': float(np.exp(b+1.96*s)),
            'p': float(2*(1-norm.cdf(abs(b/s)))), 'n': int(len(d))}

res = {'pool_n': int(len(pool)), 'at_n': int(len(at)), 'at_per_db': at['db'].value_counts().to_dict(),
       'firth_ext': {}}
for block, outs, expo in [('reversal', ['hosp_death','hb_drop2','tx_any','thrombosis_any'], 'reversal_any'),
                          ('restart', ['hosp_death','thrombosis_any'], 'restarted')]:
    res['firth_ext'][block] = {}
    for out in outs:
        res['firth_ext'][block][out] = firth_run(out, expo, at)
        print(f"FIRTH {block} {out}: {res['firth_ext'][block][out]}")

# per-db forest data for reversal->tx (primary signal)
res['perdb'] = {}
for db in ['mimic','eicu','nwicu','sicdb','umcdb']:
    d = at[at['db']==db].copy()
    row = {}
    for out in ['hosp_death','tx_any','hb_drop2','thrombosis_any']:
        row[out] = firth_run(out, 'reversal_any', d)
    res['perdb'][db] = row
    print('perdb', db, row)

# restart tertiles on new pool
rh = at.loc[at['restarted']==1, 'resume_h']
res['restart_h_summary'] = {'p25': float(rh.quantile(.25)), 'p50': float(rh.median()), 'p75': float(rh.quantile(.75)),
                            'n': int(len(rh)), 'min': float(rh.min()), 'max': float(rh.max())}
print('restart_h summary:', res['restart_h_summary'])

json.dump(res, open('F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/analysis_results_v3.json','w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('saved analysis_results_v3.json')
print('DONE')
