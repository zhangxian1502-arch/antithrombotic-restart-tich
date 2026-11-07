# -*- coding: utf-8 -*-
"""T3-competing: time-varying exposure competing-risks models on MIMIC AT subset (n=149).
Exposures (time-varying): reversal(t) and restart(t) (0->1 at documented hour).
Outcomes: death (precise deathtime) vs thrombosis (discharge diagnosis; proxy time = frac x LOS).
Models: (1) CSH via lifelines counting-process; (2) Fine-Gray subdistribution hazard (own partial-likelihood);
(3) Aalen-Johansen CIF by restart group.
Writes results/competing_risk_results.json."""
import pandas as pd, numpy as np, json
from lifelines import CoxTimeVaryingFitter
from scipy.optimize import minimize
from scipy.stats import norm

DATA = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data'
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/results/competing_risk_results.json'
rng = np.random.default_rng(42)

df = pd.read_csv(f'{DATA}/mimic_t3_v2.csv', low_memory=False)
at = df[df['baseline_at'].notna()].copy()
tms = pd.read_csv(f'{DATA}/mimic_at_admissions_times.csv')
at = at.merge(tms, on='hadm_id', how='left')
at['icu_intime'] = pd.to_datetime(at['icu_intime'])
at['deathtime'] = pd.to_datetime(at['deathtime'])
at['dischtime'] = pd.to_datetime(at['dischtime'])
at['death_h'] = np.where(at['deathtime'].isna(), np.nan, (at['deathtime']-at['icu_intime']).dt.total_seconds()/3600)
at['los_h'] = np.where(at['dischtime'].isna(), np.nan, (at['dischtime']-at['icu_intime']).dt.total_seconds()/3600)
at['restart_h'] = pd.to_numeric(at['resume_h'], errors='coerce')
at['reversal_h'] = pd.to_numeric(at['reversal_h_first'], errors='coerce')
at['restarted'] = at['resume_cls'].notna().astype(int)
at['reversed'] = at['reversal'].notna().astype(int)
at['gcs'] = at['gcs_total'].where(at['gcs_total'].notna(), at['gcs_motor_only'])
at['male'] = (at['sex'].astype(str).str.lower().str.startswith('m')).astype(int)

# ---- event times ----
LOS = at['los_h'].clip(lower=1)
EV = {}
for frac in [0.5, 0.75, 1.0]:
    T = np.where(at['hosp_death']==1, at['death_h'].fillna(LOS), np.where(at['thrombosis_any']==1, LOS*frac, LOS))
    E = np.where(at['hosp_death']==1, 1, np.where(at['thrombosis_any']==1, 2, 0)).astype(int)
    T = T + rng.uniform(0, 0.05, len(at))
    EV[frac] = (T, E)
E_death_bin = at['hosp_death'].astype(int).values
E_thr_bin = at['thrombosis_any'].astype(int).values
print('AT n=', len(at), '| deaths=', int((at['hosp_death']==1).sum()), '| thrombo=', int((at['thrombosis_any']==1).sum()),
      '| restarted=', int(at['restarted'].sum()), '| reversed=', int(at['reversed'].sum()))

def build_cp(d, T, E):
    rows = []
    for idx, r in d.iterrows():
        t_end = float(T[idx]); e = int(E[idx])
        rv = float(r['reversal_h']) if pd.notna(r['reversal_h']) else np.inf
        rs = float(r['restart_h']) if pd.notna(r['restart_h']) else np.inf
        cuts = sorted([c for c in [rv, rs] if 0 < c < t_end])
        if cuts:
            segs = [(0.0, cuts[0])] + [(a, b) for a, b in zip(cuts, cuts[1:])] + [(cuts[-1], t_end)]
        else:
            segs = [(0.0, t_end)]
        n = len(segs)
        for i, (a, b) in enumerate(segs):
            if b <= a: continue
            rev = 1.0 if (rv <= b) else 0.0
            rst = 1.0 if (rs <= b) else 0.0
            ev = e if i == n-1 else 0
            rows.append({'id': int(r['hadm_id']), 'start': a, 'stop': b, 'rev': rev, 'rst': rst,
                         'age': float(r['age']) if pd.notna(r['age']) else np.nan,
                         'male': float(r['male']), 'gcs': float(r['gcs']) if pd.notna(r['gcs']) else np.nan,
                         'cc_age': float(r['charlson_age']) if pd.notna(r['charlson_age']) else np.nan,
                         'event': ev})
    cp = pd.DataFrame(rows)
    for c in ['age','gcs','cc_age']:
        cp[c] = cp[c].fillna(cp[c].median())
    return cp

# ============ 1. CSH (CoxTimeVaryingFitter counting process) ============
def run_csh(T, E, label, covs=['age','male','gcs','cc_age','rev','rst']):
    cp = build_cp(at, T, E)
    if cp['event'].sum() < 5:
        return {'error': 'too few events', 'n': int(cp['event'].sum())}
    ctv = CoxTimeVaryingFitter()
    try:
        ctv.fit(cp, id_col='id', start_col='start', stop_col='stop', event_col='event', show_progress=False)
        out = {}
        for c in covs:
            if c in ctv.summary.index:
                ci = ctv.confidence_intervals_.loc[c]
                out[c] = {'hr': float(np.exp(ctv.summary.loc[c, 'coef'])),
                          'lo': float(np.exp(ci[0])), 'hi': float(np.exp(ci[1])),
                          'p': float(ctv.summary.loc[c, 'p'])}
        out['n_events'] = int(cp['event'].sum())
        return out
    except Exception as ex:
        return {'error': str(ex)[:200]}

csh_death = run_csh(at['death_h'].fillna(at['los_h']), E_death_bin, 'CSH death')
csh_thr = {}
for frac in [0.5, 0.75, 1.0]:
    T, E = EV[frac]
    csh_thr[str(frac)] = run_csh(T, E_thr_bin, f'CSH thromb ({frac}xLOS)')
print('CSH death:', json.dumps(csh_death, default=str))
print('CSH thr 0.75:', json.dumps(csh_thr['0.75'], default=str))

# ============ 2. Fine-Gray (own partial likelihood, time-varying Z) ============
def km_survival(T_all, C_all):
    """KM survival of censoring: event=C (censoring indicator)."""
    order = np.argsort(T_all)
    T = T_all[order]; C = C_all[order]
    ut = np.unique(T)
    # risk set at each unique time (before time), events = censoring
    S = 1.0
    out = {}
    for t in ut:
        at_risk = (T >= t).sum()
        n_events = ((T == t) & (C == 1)).sum()
        if at_risk > 0:
            S = S * (1 - n_events/at_risk)
        out[t] = S
    return out  # G(t) = P(censoring > t)

def fg_fit(d, T, E, target, covs, time_varying=False, ridge=0.0):
    """Fine-Gray subdistribution hazard. target: event type 1 or 2.
    time_varying=False: Z = baseline (restarted/reversed fixed). time_varying=True: Z(t) = rev(t)/rst(t) with ridge penalty."""
    d = d.copy()
    d['T'] = T; d['E'] = E
    d['cc_age'] = d['charlson_age']
    cens = (d['E'] == 0).astype(int)
    G = km_survival(d['T'].values, cens.values)
    def Gt(t):
        ts = np.array(sorted(G.keys()))
        idx = np.searchsorted(ts, t, side='right') - 1
        return G[ts[max(idx,0)]] if idx >= 0 else 1.0
    ev_times = sorted(d.loc[d['E']==target, 'T'].unique())
    ev_times = [t for t in ev_times if t > 0]
    if time_varying:
        def z_at(r, t):
            rv = r['reversal_h'] if pd.notna(r['reversal_h']) else np.inf
            rs = r['restart_h'] if pd.notna(r['restart_h']) else np.inf
            base = [r['age'] if pd.notna(r['age']) else d['age'].median(),
                    r['male'], r['gcs'] if pd.notna(r['gcs']) else d['gcs'].median(),
                    r['cc_age'] if pd.notna(r['cc_age']) else d['cc_age'].median()]
            tv = [1.0 if t >= rv else 0.0, 1.0 if t >= rs else 0.0]
            return dict(zip(covs, base + tv))
    else:
        def z_at(r, t):
            base = [r['age'] if pd.notna(r['age']) else d['age'].median(),
                    r['male'], r['gcs'] if pd.notna(r['gcs']) else d['gcs'].median(),
                    r['cc_age'] if pd.notna(r['cc_age']) else d['cc_age'].median()]
            tv = [float(r['reversed']), float(r['restarted'])]
            return dict(zip(covs, base + tv))
    def negll(beta):
        b = dict(zip(covs, beta))
        ll = 0.0
        for t in ev_times:
            comp = (d['E'] != target) & (d['E'] != 0) & (d['T'] < t)
            keep = (d['T'] >= t) | comp
            sub = d[keep]
            if len(sub) == 0: continue
            comp_sub = comp[keep]
            Z = sub.apply(lambda r: z_at(r, t), axis=1, result_type='expand')
            lin = np.array([sum(b[c]*Z.loc[i, c] for c in covs) for i in sub.index])
            mx = lin.max()
            w = np.ones(len(sub))
            if comp_sub.any():
                compT = sub.loc[comp_sub, 'T'].values
                w[comp_sub.values] = np.array([Gt(t)/max(Gt(cT), 1e-9) for cT in compT])
            denom = np.log(np.sum(w * np.exp(lin - mx))) + mx
            tar = (d['E']==target) & (d['T']==t)
            for i in sub.index:
                if tar[i]:
                    ll += (lin[list(sub.index).index(i)] - denom)
        pen = 0.5 * ridge * np.dot(beta, beta)
        return -(ll - pen)
    beta0 = np.zeros(len(covs))
    res = minimize(negll, beta0, method='BFGS', options={'maxiter': 500})
    beta = res.x
    if res.hess_inv is not None:
        se = np.sqrt(np.clip(np.diag(res.hess_inv), 1e-12, None))
    else:
        se = np.full(len(covs), np.nan)
    out = {}
    for c, b, s in zip(covs, beta, se):
        out[c] = {'sHR': float(np.exp(np.clip(b, -20, 20))),
                  'lo': float(np.exp(np.clip(b-1.96*s, -20, 20))),
                  'hi': float(np.exp(np.clip(b+1.96*s, -20, 20))),
                  'p': float(2*(1-norm.cdf(abs(b/s))))}
    out['converged'] = bool(res.success)
    out['n_events'] = int((d['E']==target).sum())
    return out

covs_full = ['age','male','gcs','cc_age','rev','rst']
fg_thr = {}
for frac in [0.5, 0.75, 1.0]:
    T, E = EV[frac]
    fg_thr[str(frac)] = {'base': fg_fit(at, T, E, target=2, covs=covs_full, time_varying=False),
                         'tv_ridge': fg_fit(at, T, E, target=2, covs=covs_full, time_varying=True, ridge=0.1)}
covs_red = ['age','gcs','rev','rst']
T1, E1 = EV[0.75]
fg_death = {'base': fg_fit(at, T1, E1, target=1, covs=covs_red, time_varying=False),
            'tv_ridge': fg_fit(at, T1, E1, target=1, covs=covs_red, time_varying=True, ridge=0.1)}
print('FG thr 0.75 base rst:', json.dumps(fg_thr['0.75']['base'].get('rst'), default=str))
print('FG thr 0.75 tv_ridge rst:', json.dumps(fg_thr['0.75']['tv_ridge'].get('rst'), default=str))
print('FG death base rst:', json.dumps(fg_death['base'].get('rst'), default=str))

# ============ 3. Aalen-Johansen CIF by restart group ============
def aj_cif(T, E, grid):
    """Aalen-Johansen CIF for event types 1,2."""
    n = len(T)
    cif1 = np.zeros(len(grid)); cif2 = np.zeros(len(grid))
    surv = 1.0
    prev_t = 0.0
    prev1 = 0.0; prev2 = 0.0
    for k, t in enumerate(grid):
        at_risk = (T > prev_t) | ((T == prev_t) & (E == 0))
        # recompute step by step over intervals
        seg = (T > prev_t) & (T <= t)
        if seg.any() and at_risk.sum() > 0:
            d1 = int(((E == 1) & seg).sum()); d2 = int(((E == 2) & seg).sum())
            risk = int(((T >= prev_t) & ((E == 0) | (T >= prev_t))).sum())
            risk = int((T >= prev_t).sum())
            if risk > 0:
                surv = surv * (1 - (d1+d2)/risk)
                prev1 += surv * d1/risk
                prev2 += surv * d2/risk
        cif1[k] = prev1; cif2[k] = prev2
        prev_t = t
    return cif1, cif2

grid = np.arange(0, 24*30+1, 12)  # 0..30 days, 12h steps
cif = {'restart_yes': {}, 'restart_no': {}}
for grp, mask in [('restart_yes', at['restarted']==1), ('restart_no', at['restarted']==0)]:
    T, E = EV[0.75]
    c1, c2 = aj_cif(T[mask], E[mask], grid)
    cif[grp] = {'hours': grid.tolist(), 'cif_death': c1.tolist(), 'cif_thrombosis': c2.tolist()}

res = {'at_n': int(len(at)), 'n_death': int(at['hosp_death'].sum()), 'n_thrombosis': int(at['thrombosis_any'].sum()),
       'n_restarted': int(at['restarted'].sum()), 'n_reversed': int(at['reversed'].sum()),
       'csh_death': csh_death, 'csh_thrombosis': csh_thr,
       'fine_gray_thrombosis': fg_thr, 'fine_gray_death': fg_death,
       'cif': cif, 'proxy_note': 'thrombosis time = frac x ICU/hosp LOS (discharge-diagnosis, no timestamp)'}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=2, default=str)
print('saved', OUT)
print('DONE')
