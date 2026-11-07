# -*- coding: utf-8 -*-
"""
Nature-figure contract for the tICH antithrombotic timeline paper.
Core claim: antithrombotic restart clusters at Day 3 (median 71.1 h);
restart = lower death but more thrombosis; reversal = transfusion-linked, not death-linked.
Backend: Python / matplotlib (exclusive). Export: PNG 300dpi + SVG + PDF + TIFF 600dpi.
"""
import pandas as pd, numpy as np, os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---------- styling ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})
REV = "#C0504D"   # reversal (red)
RST = "#2F5597"   # restart (blue)
NEUT = "#595959"
OUT = r'F:\豆包文件夹\新文章16\投稿包_Clinical_Neurology_and_Neurosurgery'
D = r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\data'
os.makedirs(OUT, exist_ok=True)

# ---------- load & harmonise ----------
def load(db):
    df = pd.read_csv(os.path.join(D, f'{db}_t3_v2.csv'))
    df['db_short'] = {'mimic': 'MIMIC-IV', 'eicu': 'eICU-CRD', 'nwicu': 'NWICU',
                      'sicdb': 'SICdb', 'umcdb': 'AmsterdamUMCdb'}[db]
    # sex -> male flag
    s = df['sex'].astype(str).str.strip().str.lower()
    df['male'] = s.isin(['m', 'male', 'man'])
    # sicdb icu_los stored in SECONDS -> days
    if db == 'sicdb':
        df['icu_los_days'] = df['icu_los_days'] / 86400.0
    # force numeric on all analysis columns
    for c in ['age','gcs_total','charlson_age','icu_los_days','neuro_surgery','mv_flag',
              'hosp_death','thrombosis_any','reversal_h_first','resume_h','hgb_min72',
              'hb_drop2','tx_any']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

dfs = {db: load(db) for db in ['mimic','eicu','nwicu','sicdb','umcdb']}
pool = pd.concat(dfs.values(), ignore_index=True)
at = pool[pool['baseline_at'].notna()].copy()
at['rev72'] = (at['reversal'].notna()) & (at['reversal_h_first'] <= 72)
at['rst48'] = at['resume_h'] > 48

def cls(x):
    p = str(x).split('|')
    if 'warfarin' in p: return 'VKA'
    if 'doac' in p: return 'DOAC'
    return 'Antiplatelet'
at['cls_primary'] = at['baseline_at'].apply(cls)

# ============ Table 1 data ============
print('=== TABLE 1 (by database) ===')
rows = []
for db in ['mimic','eicu','nwicu','sicdb','umcdb']:
    s = dfs[db]
    n = len(s)
    age_m, age_s = s['age'].mean(), s['age'].std()
    male_p = s['male'].mean()*100
    gcs = s['gcs_total'].dropna()
    gcs_m = np.median(gcs) if len(gcs) else None
    cc = s['charlson_age'].dropna()
    cc_m = np.median(cc) if len(cc) else None
    los = s['icu_los_days'].dropna()
    los_m = np.median(los) if len(los) else None
    ns = s['neuro_surgery'].dropna()
    ns_p = ns.mean()*100 if len(ns) and (db != 'umcdb') else None
    mv = s['mv_flag'].dropna()
    mv_p = mv.mean()*100 if len(mv) and mv.sum() > 0 else None
    death_p = s['hosp_death'].mean()*100
    rows.append((s['db_short'].iloc[0], n, age_m, age_s, male_p, gcs_m, cc_m, los_m, ns_p, mv_p, death_p))
    print('%s n=%d age=%.1f(%.1f) male=%.1f%% gcs=%s cc=%s los=%.1f ns=%s mv=%s death=%.1f%%' % (
        rows[-1][0], n, age_m, age_s, male_p, gcs_m, cc_m, los_m, ns_p, mv_p, death_p))
# pooled
p_age = pool['age'].mean(); p_los = pool['icu_los_days'].mean()
print('Pooled: n=%d age=%.1f(%.1f) male=%.1f%% death=%.1f%%' % (
    len(pool), p_age, pool['age'].std(), pool['male'].mean()*100, pool['hosp_death'].mean()*100))

# ============ Figure 1: STROBE flow diagram ============
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
def box(x, y, w, h, text, fc, fs=8.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", linewidth=1.0,
                       edgecolor=NEUT, facecolor=fc)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color='black')
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1.0, color=NEUT))
box(2.6, 8.3, 4.8, 1.1, 'tICH identified in five critical care databases\nMIMIC-IV, eICU-CRD, NWICU, SICdb, AmsterdamUMCdb', '#E8EEF7', 8.5)
box(3.1, 6.3, 3.8, 1.0, 'Pooled cohort with tICH (n = 8,562)', '#F5F5F5', 9)
box(3.1, 4.3, 3.8, 1.0, 'Baseline antithrombotic exposure\n(within -24 h to +48 h of ICU admission; n = 543)', '#E8EEF7', 8.5)
box(0.7, 2.0, 3.8, 1.1, 'Reversal within 72 h\nn = 65 (12.0%)\nmedian 7.0 h (IQR 1.2-21.1)', '#F7E4E2', 8.5)
box(5.5, 2.0, 3.8, 1.1, 'Restart after 48 h\nn = 146 (26.9%)\nmedian 69.4 h (IQR 58.8-107.2)', '#E8EEF7', 8.5)
arrow(5, 8.3, 5, 7.3)
arrow(5, 6.3, 5, 5.3)
arrow(4.2, 4.3, 2.6, 3.1)
arrow(5.8, 4.3, 7.4, 3.1)
box(5.5, 0.1, 3.8, 1.0, 'Reversed patients who restarted\n31 of 65 (47.7%)', '#F5F5F5', 8.5)
arrow(2.6, 2.0, 6.8, 1.1)
ax.set_title('Figure 1. Study flow', fontsize=10, loc='left', pad=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT,'Figure1_flow.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Figure1_flow.svg'), bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Figure1_flow.pdf'), bbox_inches='tight')
plt.close(fig)
print('Figure1 done')

# ============ Figure 2: timeline schematic ============
fig, ax = plt.subplots(figsize=(7.2, 2.8), dpi=300)
ax.set_xlim(-10, 220); ax.set_ylim(0, 10); ax.axis('off')
ax.annotate('', xy=(215, 5), xytext=(0, 5), arrowprops=dict(arrowstyle='-|>', lw=1.2, color=NEUT))
for h in [0, 24, 48, 72, 96, 120, 144, 168, 192]:
    ax.plot([h, h], [4.7, 5.3], color=NEUT, lw=0.8)
    ax.text(h, 3.6, f'{h} h', ha='center', fontsize=8)
ax.text(212, 6.3, 'Hours from ICU admission', ha='right', va='center', fontsize=8.5, style='italic')
# reversal
ax.plot(7.0, 5, 'o', color=REV, ms=10, zorder=3)
ax.annotate('Reversal\nmedian 7.0 h\n(IQR 1.2-21.1)', xy=(7.0, 5), xytext=(18, 8.2),
            fontsize=8, ha='center', color=REV,
            arrowprops=dict(arrowstyle='-', color=REV, lw=0.9))
# restart
ax.plot(69.4, 5, 'o', color=RST, ms=12, zorder=3)
ax.annotate('Restart\nmedian 69.4 h\n(day 3, IQR 58.8-107.2)', xy=(69.4, 5), xytext=(105, 8.4),
            fontsize=8, ha='center', color=RST,
            arrowprops=dict(arrowstyle='-', color=RST, lw=0.9))
# tertile band
ax.axvspan(55, 155, ymin=0.55, ymax=0.62, color=RST, alpha=0.18)
ax.text(105, 2.0, 'Restart tertiles: early 55.0 h  ·  intermediate 69.4 h  ·  late 154.9 h',
        fontsize=8, ha='center', color=RST)
ax.set_title('Figure 2. Antithrombotic stop-reverse-restart timeline (baseline-exposed cohort, n = 543)',
             fontsize=9.5, loc='left', pad=6)
fig.tight_layout()
fig.savefig(os.path.join(OUT,'Figure2_timeline.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Figure2_timeline.svg'), bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Figure2_timeline.pdf'), bbox_inches='tight')
plt.close(fig)
print('Figure2 done')

# ============ Figure 3: restart timing distribution ============
rh = at.loc[at['rst48'], 'resume_h']
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=300)
ax.hist(rh, bins=16, color=RST, alpha=0.75, edgecolor='white', linewidth=0.6)
ax.axvline(69.4, color='black', linestyle='--', linewidth=1.0)
ax.text(71, ax.get_ylim()[1]*0.9, 'median 69.4 h (day 3)', fontsize=8.5)
for x, lab in [(55.0, 'early'), (154.9, 'late')]:
    ax.axvline(x, color=NEUT, linestyle=':', linewidth=0.9)
ax.text(55.0, ax.get_ylim()[1]*0.95, '55 h', fontsize=7.5, ha='center')
ax.text(154.9, ax.get_ylim()[1]*0.95, '154.9 h', fontsize=7.5, ha='center')
ax.set_xlabel('Restart timing after ICU admission (h)')
ax.set_ylabel('Patients (n = 146)')
ax.set_title('Supplementary Figure S1. Distribution of antithrombotic restart timing', fontsize=9.5, loc='left')
fig.tight_layout()
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S1_restart_distribution.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S1_restart_distribution.svg'), bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S1_restart_distribution.pdf'), bbox_inches='tight')
plt.close(fig)
print('Supplementary Figure S1 done')

# ============ Figure 4 (main forest) now rendered in R: make_figure4_forest.R ============

# ============ Figure 3 (main forest) now rendered in R: make_figure4_forest.R ============
# (matplotlib version superseded; do not regenerate here)

# ============ Supplementary Figure S2: CIF curves (MIMIC subset) ============
cr = json.load(open(r'F:\豆包文件夹\新文章16\T3抗栓相关tICH分析\results\competing_risk_results.json', encoding='utf-8'))
cif = cr['cif']; h = cif['restart_yes']['hours']
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), dpi=300)
for ax, key, title in [(axs[0],'cif_death','Death'), (axs[1],'cif_thrombosis','Thrombotic events')]:
    ax.plot(h, cif['restart_yes'][key], color=RST, linewidth=1.8, label='Restarted (n=94)')
    ax.plot(h, cif['restart_no'][key], color=REV, linewidth=1.8, linestyle='--', label='Not restarted (n=55)')
    ax.set_xlabel('Hours since ICU admission', fontsize=8)
    ax.set_ylabel('Cumulative incidence', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, loc='upper left')
    ax.set_ylim(0, 0.75)
fig.suptitle('Supplementary Figure S2. Aalen-Johansen cumulative incidence functions by restart status (MIMIC-IV subset, n=149)',
             fontsize=9)
fig.tight_layout(rect=[0,0,1,0.92])
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S2_cif.png'), dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S2_cif.svg'), bbox_inches='tight')
fig.savefig(os.path.join(OUT,'Supplementary_Figure_S2_cif.pdf'), bbox_inches='tight')
plt.close(fig)
print('Supplementary Figure S2 done')
print('ALL FIGURES DONE')
