# -*- coding: utf-8 -*-
"""Sensitivity: Firth models (exact original implementation) additionally
adjusted for antithrombotic class, admission INR, admission MAP and lowest Hb.
v3 pool, data unchanged."""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')

BASE = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data'
pool = pd.read_csv(f'{BASE}/pool_t3_v3.csv', low_memory=False)

pool['at_exposed'] = pool['baseline_at'].notna().astype(int)
pool['reversal_any'] = pool['reversal'].notna().astype(int)
pool['restarted'] = pool['resume_cls'].notna().astype(int)
pool['gcs_any'] = pool['gcs_total'].where(pool['gcs_total'].notna(), pool['gcs_motor_only'])
pool['male'] = (pool['sex'].astype(str).str.lower().str.startswith('m')).astype(int)
at = pool[pool['at_exposed']==1].copy()

cls = at['baseline_at'].astype(str)
at['at_warfarin'] = (cls == 'warfarin').astype(int)
at['at_doac'] = (cls == 'doac').astype(int)
at['at_ap_warfarin'] = (cls == 'antiplatelet|warfarin').astype(int)
at['at_ap_doac'] = (cls == 'antiplatelet|doac').astype(int)
print('class dummies sums:', at[['at_warfarin','at_doac','at_ap_warfarin','at_ap_doac']].sum().to_dict())

DBS = ['mimic','eicu','nwicu','sicdb','umcdb']
EXT = ['hr1','sbp1','spo2_1','temp_c1','l_wbc','l_cr','l_glu','l_na','l_k','l_lac','l_alb','l_tbili',
       'vaso_flag','mv_flag','neuro_surgery','admit_to_icu_h','cc_cvd','cc_renal','cc_dm','cc_cancer']
NEW = ['at_warfarin','at_doac','at_ap_warfarin','at_ap_doac','admit_inr','admit_map','hgb_min72']
print('new covariate availability:', {c: (c in at.columns) for c in NEW})

# ---- exact original Firth implementation (from analyze_t3_v3_pool.py) ----
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

def firth_run(outcome, exposure, d, extra):
    d = d[d[outcome].notna()].copy()
    if d[exposure].nunique() < 2 or d[outcome].sum() < 5:
        return {'error': 'insufficient', 'n': int(len(d))}
    Xd = pd.DataFrame(index=d.index)
    Xd['age'] = d['age']; Xd['male'] = d['male']; Xd['gcs'] = d['gcs_any']; Xd['cc_age'] = d['charlson_age']
    for db in DBS[1:]:
        Xd[f'db_{db}'] = (d['db']==db).astype(int)
    for c in EXT + extra:
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

res = {'sens_new_covars': {}, 'orig_ext': {}}
for block, outs, expo in [('reversal', ['hosp_death','hb_drop2','tx_any','thrombosis_any'], 'reversal_any'),
                          ('restart', ['hosp_death','thrombosis_any'], 'restarted')]:
    res['orig_ext'][block] = {}
    res['sens_new_covars'][block] = {}
    for out in outs:
        res['orig_ext'][block][out] = firth_run(out, expo, at, [])
        res['sens_new_covars'][block][out] = firth_run(out, expo, at, NEW)
        print(f"FIRTH {block}/{out}")
        print('  orig:', res['orig_ext'][block][out])
        print('  sens:', res['sens_new_covars'][block][out])

json.dump(res, open('F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/firth_sensitivity_covariates.json','w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print('saved firth_sensitivity_covariates.json')
