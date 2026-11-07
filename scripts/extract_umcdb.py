# -*- coding: utf-8 -*-
"""T3 extraction: AmsterdamUMCdb 1.0.2. Cohort = traumatic intracranial injury via APACHE diagnosis items.
No ICD; long-term death via dateofdeath; labs via numericitems; meds via drugitems (start datetime)."""
import duckdb, pandas as pd, numpy as np

con = duckdb.connect('F:/UMCdb/umcdb.duckdb', read_only=True)
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/umcdb_t3.csv'

coh = con.execute("""
SELECT DISTINCT li.admissionid FROM listitems li
WHERE li.item IN ('NICE APACHEII diagnosen','NICE APACHEIV diagnosen')
 AND (lower(li.value) LIKE '%hoofdtrauma%' OR lower(li.value) LIKE '%hoofd trauma%'
   OR lower(li.value) LIKE '%intracerebr, subduraal, subarachnoidaal haematoom%'
   OR lower(li.value) LIKE '%hemorrhage/hematoma-intracranial%')
 AND lower(li.value) NOT LIKE '%aneurysm%' AND lower(li.value) NOT LIKE '%cva%' AND lower(li.value) NOT LIKE '%stroke%'""").fetchdf()
ids = tuple(coh['admissionid'].tolist())
ids_s = ','.join(str(i) for i in ids)

cohort = con.execute(f"""
SELECT admissionid, patientid, gender, agegroup, specialty, admittedat, dischargedat, lengthofstay, dateofdeath
FROM admissions WHERE admissionid IN ({ids_s})""").fetchdf()
cohort['admittedat'] = pd.to_datetime(cohort['admittedat'])
cohort['dischargedat'] = pd.to_datetime(cohort['dischargedat'])
cohort['dateofdeath'] = pd.to_datetime(cohort['dateofdeath'])
print('UMCdb cohort:', len(cohort))

# subtype from APACHE diagnosis values
diag = con.execute(f"""
SELECT admissionid, value FROM listitems
WHERE item IN ('NICE APACHEII diagnosen','NICE APACHEIV diagnosen') AND admissionid IN ({ids_s})""").fetchdf()
def sub(v):
    v = (v or '').lower()
    if 'subduraal' in v: return 'sdh'
    if 'epiduraal' in v: return 'edh'
    if 'subarachnoidaal' in v: return 'tsah'
    if 'intracerebr' in v: return 'tich'
    if 'hoofd' in v or 'head' in v: return 'unspecified_head'
    return 'unspecified'
diag['sub'] = diag['value'].apply(sub)
subsum = diag.groupby('admissionid')['sub'].agg(lambda s:'|'.join(sorted(set(s)))).rename('s06_subtypes').reset_index()
cohort = cohort.merge(subsum, on='admissionid', how='left')
# neuro_surgery: operative/head surgery diagnosis
ns = diag[diag['value'].str.contains('Operatief|surgery for', case=False, na=False)]
cohort['neuro_surgery'] = cohort['admissionid'].isin(ns['admissionid']).astype(int)

# ---- drug mapping ----
AT = {'carbasalaatcalcium':'antiplatelet','ascal':'antiplatelet','acetylsalicylzuur':'antiplatelet',
      'clopidogrel':'antiplatelet','ticagrelor':'antiplatelet','prasugrel':'antiplatelet',
      'acenocoumarol':'warfarin','fenprocoumon':'warfarin','warfarine':'warfarin',
      'apixaban':'doac','rivaroxaban':'doac','dabigatran':'doac','edoxaban':'doac',
      'nadroparine':'heparin','heparine':'heparin','enoxaparine':'heparin','dalteparine':'heparin','fondaparinux':'heparin'}
REV = {'fytomenadion':'vitk','konakion':'vitk','vitamine k':'vitk',
       'fresh frozen plasma':'ffp','vers plasma':'ffp',
       'protrombinecomplex':'pcc','beriplex':'pcc','cofact':'pcc','octaplex':'pcc',
       'tranexaminezuur':'txa','cyklokapron':'txa','tranexamic':'txa',
       'protamine':'protamine',
       'idarucizumab':'idarucizumab','praxbind':'idarucizumab','andexanet':'andexanet',
       'trombocytenconcentraat':'platelets','thrombocytenconcentraat':'platelets'}
TX = {'erytrocytenconcentraat':'rbc','trombocytenconcentraat':'plt','thrombocytenconcentraat':'plt',
      'fresh frozen plasma':'ffp','vers plasma':'ffp'}

drug = con.execute(f"""
SELECT admissionid, item, start, administered FROM drugitems
WHERE admissionid IN ({ids_s})""").fetchdf()
drug['start'] = pd.to_datetime(drug['start'])
drug['h'] = (drug['start'] - cohort.set_index('admissionid')['admittedat'].reindex(drug['admissionid']).values).dt.total_seconds()/3600.0

def match(drugs, name):
    nl = name.lower()
    for k, v in drugs.items():
        if k in nl: return v
    return None
drug['cls'] = drug['item'].apply(lambda x: match(AT, x))
drug['rev'] = drug['item'].apply(lambda x: match(REV, x))
drug['tx'] = drug['item'].apply(lambda x: match(TX, x))

base = drug[(drug['h']>=-24)&(drug['h']<=48)&(drug['cls'].isin(['warfarin','doac','antiplatelet']))]
base_g = base.groupby('admissionid')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('baseline_at').reset_index()
rev = drug[(drug['h']>=0)&(drug['h']<=72)&(drug['rev'].notna())]
rev_g = rev.groupby('admissionid')['rev'].agg(lambda s:'|'.join(sorted(set(s)))).rename('reversal').reset_index()
rev_first = rev.sort_values('h').groupby('admissionid').first()['h'].rename('reversal_h_first').reset_index()
ever = drug[(drug['h']>=-24)&(drug['h']<=30*24)&(drug['cls'].isin(['warfarin','doac','antiplatelet']))]
ever_g = ever.groupby('admissionid')['cls'].agg(lambda s:'|'.join(sorted(set(s)))).rename('ever_at').reset_index()
res = drug[(drug['h']>48)&(drug['cls'].isin(['warfarin','doac','antiplatelet']))].sort_values('h').groupby('admissionid').first().reset_index()
res_g = res[['admissionid','cls','h']].rename(columns={'cls':'resume_cls','h':'resume_h'})
cohort = cohort.merge(base_g, on='admissionid', how='left')
cohort = cohort.merge(rev_g, on='admissionid', how='left')
cohort = cohort.merge(rev_first, on='admissionid', how='left')
cohort = cohort.merge(ever_g, on='admissionid', how='left')
cohort = cohort.merge(res_g, on='admissionid', how='left')
cohort['last_anticoag_stop_h'] = np.nan

tx = drug[drug['tx'].notna()]
txg = tx.groupby('admissionid').agg(
    tx_rbc=('tx', lambda s:int((s=='rbc').sum())),
    tx_ffp=('tx', lambda s:int((s=='ffp').sum())),
    tx_plt=('tx', lambda s:int((s=='plt').sum())),
    tx_any=('tx', lambda s:int(len(s)>0)),
    tx_first_h=('h','min')).reset_index()
cohort = cohort.merge(txg, on='admissionid', how='left')
for c in ['tx_rbc','tx_ffp','tx_plt','tx_any']:
    cohort[c] = cohort[c].fillna(0).astype(int)

# ---- labs (numericitems) ----
LABMAP = {'prothrombinetijd':'inr','aptt':'ptt','fibrinogeen':'fib','hb (bloed)':'hgb'}
lab = con.execute(f"""
SELECT admissionid, item, value, measuredat FROM numericitems
WHERE admissionid IN ({ids_s})
 AND (lower(item) LIKE '%prothrombinetijd%' OR lower(item) LIKE '%aptt%' OR lower(item) LIKE '%fibrinogeen%'
   OR lower(item) = 'hb (bloed)')""").fetchdf()
lab['measuredat'] = pd.to_datetime(lab['measuredat'])
lab['h'] = (lab['measuredat'] - cohort.set_index('admissionid')['admittedat'].reindex(lab['admissionid']).values).dt.total_seconds()/3600.0
lab['val'] = pd.to_numeric(lab['value'], errors='coerce')
lab['lab'] = lab['item'].str.lower().map(LABMAP)
lab = lab[lab['lab'].notna() & lab['val'].notna()]
first24 = lab[(lab['h']>=-24)&(lab['h']<=24)].sort_values('h').groupby(['admissionid','lab']).first().reset_index()
coag = first24.pivot_table(index='admissionid', columns='lab', values='val', aggfunc='first').reset_index()
coag.columns = ['admissionid'] + ['admit_'+c for c in coag.columns[1:]]
cohort = cohort.merge(coag, on='admissionid', how='left')
for _c in ['admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb']:
    if _c not in cohort.columns:
        cohort[_c] = np.nan
hgb72 = lab[(lab['lab']=='hgb')&(lab['h']>=-24)&(lab['h']<=72)].groupby('admissionid')['val'].min().rename('hgb_min72').reset_index()
cohort = cohort.merge(hgb72, on='admissionid', how='left')
cohort['hb_drop2'] = ((cohort['admit_hgb']-cohort['hgb_min72'])>=2).astype(float)
cohort['hb_drop2'] = cohort['hb_drop2'].where(cohort['admit_hgb'].notna() & cohort['hgb_min72'].notna())

# ---- GCS motor (listitems) ----
g = con.execute(f"""
SELECT admissionid, item, value, measuredat FROM listitems
WHERE item='Beste motore reactie van de armen' AND admissionid IN ({ids_s})""").fetchdf()
g['measuredat'] = pd.to_datetime(g['measuredat'])
g['h'] = (g['measuredat'] - cohort.set_index('admissionid')['admittedat'].reindex(g['admissionid']).values).dt.total_seconds()/3600.0
g = g[(g['h']>=-24)&(g['h']<=24)]
g_first = g.sort_values('h').groupby('admissionid').first()['value'].rename('gcs_motor_only').reset_index()
cohort = cohort.merge(g_first, on='admissionid', how='left')

# ---- tidy ----
cohort['db'] = 'umcdb'
cohort['gcs_total'] = np.nan
cohort['admit_map'] = np.nan
cohort['mv_flag'] = np.nan
cohort['charlson'] = np.nan
cohort['charlson_age'] = np.nan
cohort['sex'] = cohort['gender']
cohort['age'] = cohort['agegroup'].astype(str).str.replace('>89','90').str.replace('80-89','85').str.replace('70-79','75').str.replace('60-69','65').str.replace('50-59','55').str.replace('40-49','45').str.replace('30-39','35').str.replace('18-29','24').str.replace('<18','15')
cohort['age'] = pd.to_numeric(cohort['age'], errors='coerce')
cohort['icu_los_days'] = cohort['lengthofstay']/24.0
cohort['hosp_los_days'] = (cohort['dischargedat']-cohort['admittedat']).dt.total_seconds()/86400.0
cohort['hosp_death'] = cohort['dateofdeath'].notna().astype(int)
for c in ['pe','mi','is','dvt','arte','thrombosis_any']:
    cohort[c] = np.nan
cohort = cohort.rename(columns={'admissionid':'stay_id','patientid':'subject_id'})

keep = ['db','subject_id','stay_id','age','sex','icu_los_days','hosp_los_days','s06_subtypes',
        'gcs_total','gcs_motor_only','admit_map','admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb','hgb_min72','hb_drop2',
        'charlson','charlson_age','neuro_surgery','mv_flag',
        'baseline_at','ever_at','reversal','reversal_h_first','resume_cls','resume_h','last_anticoag_stop_h',
        'tx_rbc','tx_ffp','tx_plt','tx_any','tx_first_h',
        'pe','mi','is','dvt','arte','thrombosis_any',
        'hosp_death']
cohort[keep].to_csv(OUT, index=False)
print('saved', OUT, len(cohort))
print('baseline_at:', cohort['baseline_at'].value_counts(dropna=False).to_dict())
print('reversal:', cohort['reversal'].value_counts(dropna=False).head(8).to_dict())
print('hosp_death:', cohort['hosp_death'].sum())
print('gcs motor notna:', cohort['gcs_motor_only'].notna().sum())
