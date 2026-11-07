# -*- coding: utf-8 -*-
"""T3 extraction: MIMIC-IV. Cohort = S06 ICU stays with baseline antithrombotic + reversal + outcomes + covariates."""
import duckdb, pandas as pd, numpy as np, sys, json

DB = 'F:/MIMIC3.1/mimiciv.duckdb'
con = duckdb.connect(DB, read_only=True)
OUT = 'F:/豆包文件夹/新文章16/T3抗栓相关tICH分析/data/mimic_t3.csv'

# ---------- 1. Cohort: ICU stays with S06 (ICD-10) ----------
q = """
SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime,
       a.admittime, a.dischtime, a.hospital_expire_flag, a.discharge_location, a.deathtime,
       p.gender, p.anchor_age
FROM icu.icustays i
JOIN hosp.admissions a USING (hadm_id)
JOIN hosp.patients p ON i.subject_id = p.subject_id
WHERE i.stay_id IN (
    SELECT DISTINCT i2.stay_id FROM icu.icustays i2
    JOIN hosp.diagnoses_icd d USING (hadm_id)
    WHERE d.icd_code LIKE 'S06%' AND d.icd_version=10)
"""
cohort = con.execute(q).fetchdf()
print('cohort ICU stays:', len(cohort))
sys.stdout.flush()

# S06 codes per stay -> subtype flags
d = con.execute("""
SELECT d.hadm_id, d.icd_code FROM hosp.diagnoses_icd d
WHERE d.icd_code LIKE 'S06%' AND d.icd_version=10 AND d.hadm_id IN (
  SELECT DISTINCT hadm_id FROM icu.icustays WHERE stay_id IN (SELECT stay_id FROM cohort))""").fetchdf()
d['sub'] = np.where(d.icd_code.str.startswith('S061'),'edema',
          np.where(d.icd_code.str.startswith('S062'),'diffuse',
          np.where(d.icd_code.str.startswith('S063'),'contusion',
          np.where(d.icd_code.str.startswith('S064'),'edh',
          np.where(d.icd_code.str.startswith('S065'),'sdh',
          np.where(d.icd_code.str.startswith('S066'),'tsah',
          np.where(d.icd_code.str.startswith('S068'),'other','unspecified')))))))
subsum = d.groupby('hadm_id')['sub'].agg(lambda s: '|'.join(sorted(set(s)))).rename('s06_subtypes').reset_index()
cohort = cohort.merge(subsum, on='hadm_id', how='left')

# ---------- 2. Pharmacy: antithrombotic baseline (0-48h), reversal (0-72h), resumption (>48h) ----------
def med_class(drug):
    if drug is None or (isinstance(drug, float) and pd.isna(drug)): return None
    dl = drug.lower()
    if any(x in dl for x in ['warfarin']): return 'warfarin'
    if any(x in dl for x in ['apixaban','rivaroxaban','dabigatran','edoxaban','eliquis','xarelto','pradaxa']): return 'doac'
    if any(x in dl for x in ['aspirin','clopidogrel','ticagrelor','prasugrel','dipyridamole','cilostazol','plavix','brilinta']): return 'antiplatelet'
    if any(x in dl for x in ['enoxaparin','lovenox','heparin','dalteparin','fondaparinux']): return 'heparin'
    return None

def reversal_class(drug):
    if drug is None or (isinstance(drug, float) and pd.isna(drug)): return None
    dl = drug.lower()
    if any(x in dl for x in ['phytonadione','vitamin k','mephyton']): return 'vitk'
    if any(x in dl for x in ['kcentra','prothrombin complex','pcc','prothrombinex','4-factor','four factor']): return 'pcc'
    if any(x in dl for x in ['fresh frozen','ffp','plasma']): return 'ffp'
    if any(x in dl for x in ['platelet','platelets']): return 'platelets'
    if any(x in dl for x in ['idarucizumab','praxbind']): return 'idarucizumab'
    if any(x in dl for x in ['andexanet','andexxa']): return 'andexanet'
    if any(x in dl for x in ['protamine']): return 'protamine'
    if any(x in dl for x in ['tranexamic','txa']): return 'txa'
    return None

ph = con.execute("""
SELECT hadm_id, starttime, stoptime, medication FROM hosp.pharmacy
WHERE CAST(hadm_id AS BIGINT) IN (SELECT DISTINCT hadm_id FROM cohort)""").fetchdf()
print('pharmacy rows:', len(ph))
ph['hadm_id'] = pd.to_numeric(ph['hadm_id'], errors='coerce').astype('Int64')
ph['start'] = pd.to_datetime(ph['starttime'])
cohort['admittime'] = pd.to_datetime(cohort['admittime'])
admit = cohort[['hadm_id','intime']].rename(columns={'intime':'anchor'}).drop_duplicates()

def hours_since_admit(row):
    return (row['start'] - row['admittime']).total_seconds()/3600.0

ph = ph.merge(admit, on='hadm_id', how='left')
ph['h'] = ph.apply(lambda r: (r['start']-r['anchor']).total_seconds()/3600.0 if pd.notna(r['start']) and pd.notna(r['anchor']) else np.nan, axis=1)
ph['cls'] = ph['medication'].apply(med_class)
ph['rev'] = ph['medication'].apply(reversal_class)

# baseline antithrombotic: any order starting within [0,48h] of hospital admission
base = ph[(ph['h']>=-24)&(ph['h']<=48)&(ph['cls'].isin(['warfarin','doac','antiplatelet']))]
base_g = base.groupby('hadm_id')['cls'].agg(lambda s: '|'.join(sorted(set(s)))).rename('baseline_at').reset_index()
# ever exposed: any antithrombotic order during ICU stay (sensitivity exposure)
ever = ph[(ph['h']>=-24)&(ph['h']<=30*24)&(ph['cls'].isin(['warfarin','doac','antiplatelet']))]
ever_g = ever.groupby('hadm_id')['cls'].agg(lambda s: '|'.join(sorted(set(s)))).rename('ever_at').reset_index()
cohort = cohort.merge(ever_g, on='hadm_id', how='left')
# reversal: any reversal order within [0,72h]
rev = ph[(ph['h']>=0)&(ph['h']<=72)&(ph['rev'].notna())]
rev_g = rev.groupby('hadm_id')['rev'].agg(lambda s: '|'.join(sorted(set(s)))).rename('reversal').reset_index()
rev_first = rev[rev['h']>=0].sort_values('h').groupby('hadm_id').first()['h'].rename('reversal_h_first').reset_index()
# resumption: first antithrombotic (warfarin/doac/antiplatelet) order with start >48h after admission
res = ph[(ph['h']>48)&(ph['cls'].isin(['warfarin','doac','antiplatelet']))].sort_values('h').groupby('hadm_id').first().reset_index()
res_g = res[['hadm_id','cls','h']].rename(columns={'cls':'resume_cls','h':'resume_h'})
# exposure duration of baseline anticoagulants (warfarin/doac) in first 48h: hours covered (proxy for "停药" timing)
wd = base[base['cls'].isin(['warfarin','doac'])].copy()
def stop_h_fn(r):
    try:
        st = pd.to_datetime(r['stoptime'])
        if pd.isna(st): return 48.0
        h = (st-r['anchor']).total_seconds()/3600.0
        return h if h>0 else 48.0
    except Exception:
        return 48.0
wd['stop_h'] = wd.apply(stop_h_fn, axis=1)
wd_last = wd.groupby('hadm_id')['stop_h'].max().rename('last_anticoag_stop_h').reset_index()

cohort = cohort.merge(base_g, on='hadm_id', how='left')
cohort = cohort.merge(rev_g, on='hadm_id', how='left')
cohort = cohort.merge(rev_first, on='hadm_id', how='left')
cohort = cohort.merge(res_g, on='hadm_id', how='left')
cohort = cohort.merge(wd_last, on='hadm_id', how='left')

# ---------- 3. Transfusions (inputevents) ----------
items = con.execute("SELECT itemid, label FROM icu.d_items WHERE lower(label) LIKE '%red blood%' OR lower(label) LIKE '%plasma%' OR lower(label) LIKE '%platelet%' OR lower(label) LIKE '%cryoprecipitate%'").fetchdf()
print('blood items:', items.to_dict('records'))
itemmap = {}
for _, r in items.iterrows():
    l = r['label'].lower()
    if 'red blood' in l or l=='rbc': itemmap[r['itemid']]='rbc'
    elif 'fresh frozen' in l or 'plasma' in l: itemmap[r['itemid']]='ffp'
    elif 'platelet' in l: itemmap[r['itemid']]='plt'
    elif 'cryo' in l: itemmap[r['itemid']]='cryo'
inp = con.execute("""
SELECT stay_id, itemid, starttime, amount FROM icu.inputevents
WHERE itemid IN ({}) AND stay_id IN (SELECT stay_id FROM cohort)""".format(','.join(map(str,itemmap.keys())))).fetchdf()
inp['cls'] = inp['itemid'].map(itemmap)
inp['start'] = pd.to_datetime(inp['starttime'])
icu_time = cohort[['stay_id','intime']].rename(columns={'intime':'int'})
inp = inp.merge(icu_time, on='stay_id', how='left')
inp['h_icu'] = inp.apply(lambda r: (r['start']-r['int']).total_seconds()/3600.0 if pd.notna(r['start']) and pd.notna(r['int']) else np.nan, axis=1)
tx_g = inp.groupby('stay_id').agg(
    tx_rbc=('cls', lambda s: int((s=='rbc').sum())),
    tx_ffp=('cls', lambda s: int((s=='ffp').sum())),
    tx_plt=('cls', lambda s: int((s=='plt').sum())),
    tx_any=('cls', lambda s: int(len(s)>0)),
    tx_first_h=('h_icu', 'min')).reset_index()
cohort = cohort.merge(tx_g, on='stay_id', how='left')
cohort['tx_any'] = cohort['tx_any'].fillna(0).astype(int)

# ---------- 4. Labs: coag first value in 24h; Hgb admission vs min 72h ----------
labs = con.execute("""
SELECT l.subject_id, l.hadm_id, l.itemid, l.charttime, l.valuenum
FROM hosp.labevents l JOIN hosp.d_labitems i USING (itemid)
WHERE l.hadm_id IN (SELECT DISTINCT hadm_id FROM cohort)
  AND (lower(i.label) LIKE '%inr%' OR lower(i.label) LIKE '%ptt%' OR lower(i.label) LIKE '%platelet count%'
       OR lower(i.label) LIKE '%hemoglobin%' OR lower(i.label) LIKE '%fibrinogen%')
""").fetchdf()
labs['charttime'] = pd.to_datetime(labs['charttime'])
labitems = con.execute("SELECT itemid, label FROM hosp.d_labitems WHERE lower(label) LIKE '%inr%' OR lower(label) LIKE '%ptt%' OR lower(label) LIKE '%platelet count%' OR lower(label) LIKE '%hemoglobin%' OR lower(label) LIKE '%fibrinogen%'").fetchdf()
labmap = {}
for _, r in labitems.iterrows():
    l = r['label'].lower()
    if 'inr' in l: labmap[r['itemid']]='inr'
    elif l.startswith('ptt') and 'control' not in l and 'la' not in l and 'mea' not in l: labmap[r['itemid']]='ptt'
    elif 'platelet count' in l: labmap[r['itemid']]='plt'
    elif l == 'hemoglobin': labmap[r['itemid']]='hgb'
    elif 'fibrinogen' in l and 'immunologic' not in l: labmap[r['itemid']]='fib'
labs['lab'] = labs['itemid'].map(labmap)
labs = labs[labs['lab'].notna() & labs['valuenum'].notna()]
admit_lab = cohort[['hadm_id','admittime']].drop_duplicates()
labs = labs.merge(admit_lab, on='hadm_id', how='left')
labs['h'] = labs.apply(lambda r: (r['charttime']-r['admittime']).total_seconds()/3600.0 if pd.notna(r['charttime']) and pd.notna(r['admittime']) else np.nan, axis=1)
# first value in [0,24h]
first24 = labs[(labs['h']>=0)&(labs['h']<=24)].sort_values('h').groupby(['hadm_id','lab']).first().reset_index()
coag = first24.pivot_table(index='hadm_id', columns='lab', values='valuenum', aggfunc='first').reset_index()
coag.columns = ['hadm_id'] + ['admit_'+c for c in coag.columns[1:]]
cohort = cohort.merge(coag, on='hadm_id', how='left')
# min Hgb in 0-72h
hgb72 = labs[(labs['lab']=='hgb')&(labs['h']>=0)&(labs['h']<=72)].groupby('hadm_id')['valuenum'].min().rename('hgb_min72').reset_index()
cohort = cohort.merge(hgb72, on='hadm_id', how='left')
cohort['hb_drop2'] = ((cohort['admit_hgb'] - cohort['hgb_min72']) >= 2).astype(float)
cohort['hb_drop2'] = cohort['hb_drop2'].where(cohort['admit_hgb'].notna() & cohort['hgb_min72'].notna())

# ---------- 5. Chartevents: GCS, MAP, vent ----------
stay_list = list(cohort['stay_id'].unique())
gcs = con.execute("""
SELECT stay_id, itemid, charttime, valuenum FROM icu.chartevents
WHERE itemid IN (220739,223900,223901) AND stay_id IN (SELECT stay_id FROM cohort)""").fetchdf()
gcs['charttime'] = pd.to_datetime(gcs['charttime'])
gcs = gcs.merge(icu_time, on='stay_id', how='left')
gcs['h'] = (gcs['charttime']-gcs['int']).dt.total_seconds()/3600.0
gcs = gcs[(gcs['h']>=0)&(gcs['h']<=24)&gcs['valuenum'].notna()]
gcs_first = gcs.sort_values('h').groupby(['stay_id','itemid']).first().reset_index()
gcs_wide = gcs_first.pivot_table(index='stay_id', columns='itemid', values='valuenum', aggfunc='first').reset_index()
gcs_wide.columns = ['stay_id'] + ['gcs_'+str(c) for c in gcs_wide.columns[1:]]
cohort = cohort.merge(gcs_wide, on='stay_id', how='left')
cohort['gcs_total'] = cohort[['gcs_220739','gcs_223900','gcs_223901']].sum(axis=1, min_count=1)
cohort['gcs_motor_only'] = cohort['gcs_223901']

map_ = con.execute("""
SELECT stay_id, itemid, charttime, valuenum FROM icu.chartevents
WHERE itemid IN (220181,220052) AND stay_id IN (SELECT stay_id FROM cohort)""").fetchdf()
map_['charttime'] = pd.to_datetime(map_['charttime'])
map_ = map_.merge(icu_time, on='stay_id', how='left')
map_['h'] = (map_['charttime']-map_['int']).dt.total_seconds()/3600.0
map_ = map_[(map_['h']>=0)&(map_['h']<=24)&map_['valuenum'].notna()]
map_first = map_.sort_values('h').groupby(['stay_id','itemid']).first().reset_index()
map_wide = map_first.pivot_table(index='stay_id', columns='itemid', values='valuenum', aggfunc='first').reset_index()
cohort['admit_map'] = np.nan
for _, r in map_wide.iterrows():
    sid = r['stay_id']
    v = r.get('220181', np.nan)
    if pd.isna(v): v = r.get('220052', np.nan)
    cohort.loc[cohort['stay_id']==sid, 'admit_map'] = v

vent = con.execute("""
SELECT DISTINCT stay_id FROM icu.chartevents WHERE itemid=223849 AND stay_id IN (SELECT stay_id FROM cohort)""").fetchdf()
cohort['mv_flag'] = cohort['stay_id'].isin(vent['stay_id']).astype(int)

# ---------- 6. Procedures: cranial surgery ----------
proc = con.execute("""
SELECT hadm_id, icd_code, icd_version FROM hosp.procedures_icd
WHERE CAST(hadm_id AS BIGINT) IN (SELECT DISTINCT hadm_id FROM cohort)
  AND ((icd_code LIKE '009U%') OR (icd_code LIKE '01.2%') OR (icd_code LIKE '01.3%'))""").fetchdf()
cohort['neuro_surgery'] = cohort['hadm_id'].isin(proc['hadm_id']).astype(int)

# ---------- 7. All diagnoses: thrombotic events + Charlson ----------
all_dx = con.execute("""
SELECT hadm_id, icd_code, icd_version FROM hosp.diagnoses_icd
WHERE CAST(hadm_id AS BIGINT) IN (SELECT DISTINCT hadm_id FROM cohort)""").fetchdf()

def thrombo(icd, ver):
    icd = icd.replace('.','')
    if ver==9:
        return {'pe': icd.startswith('4151'),
                'mi': icd.startswith('410'),
                'is': icd.startswith('433') or icd.startswith('434'),
                'dvt': icd.startswith('4534') or icd.startswith('4538'),
                'arte': icd.startswith('444')}
    else:
        return {'pe': icd.startswith('I26'),
                'mi': icd.startswith('I21') or icd.startswith('I22'),
                'is': icd.startswith('I63'),
                'dvt': icd.startswith('I82'),
                'arte': icd.startswith('I74')}
tlist = []
for _, r in all_dx.iterrows():
    tlist.append((r['hadm_id'],) + tuple(thrombo(r['icd_code'].upper(), int(r['icd_version'])).values()))
tdf = pd.DataFrame(tlist, columns=['hadm_id','pe','mi','is','dvt','arte'])
tcols = ['pe','mi','is','dvt','arte']
tg = tdf.groupby('hadm_id')[tcols].max().reset_index()
tg['thrombosis_any'] = tg[tcols].max(axis=1)
cohort = cohort.merge(tg, on='hadm_id', how='left')
for c in tcols+['thrombosis_any']:
    cohort[c] = cohort[c].fillna(0).astype(int)

# Charlson (Quan ICD-9/10)
def charlson_cats(icd, ver):
    code = icd.replace('.','').upper()
    cats = set()
    if ver==9:
        if code.startswith(('410','412')): cats.add('mi')
        if code.startswith(('39891','40201','40211','40291','40401','40403','40411','40413','40491','40493','4254','4255','4256','4257','4258','4259','428')): cats.add('chf')
        if code.startswith(('440','441','4431','4432','4433','4434','4435','4436','4437','4438','4439','4471','5571','5579','V434')): cats.add('pvd')
        if code.startswith(('36234','430','431','432','433','434','435','436','437','438')): cats.add('cvd')
        if code.startswith(('290','2941','3312')): cats.add('dementia')
        if code.startswith(('4168','4169','490','491','492','493','494','495','496','497','498','499','500','501','502','503','504','505','5064','5081','5088')): cats.add('copd')
        if code.startswith(('4465','4466','7100','7101','7102','7103','7104','7140','7141','7142','7148','725')): cats.add('rheum')
        if code.startswith(('531','532','533','534')): cats.add('pud')
        if code.startswith(('07022','07023','07032','07033','07044','07054','0706','0709','570','571','5733','5734','5738','5739','V427')): cats.add('liver_mild')
        if code.startswith(('2500','2501','2502','2503','2508','2509')): cats.add('dm_mild')
        if code.startswith(('2504','2505','2506','2507')): cats.add('dm_sev')
        if code.startswith(('3341','342','343','3440','3441','3442','3443','3444','3445','3446','3449')): cats.add('hemi')
        if code.startswith(('40301','40311','40391','40402','40403','40412','40413','40492','40493','582','5830','5831','5832','5833','5834','5835','5836','5837','585','586','5880','V420','V451','V56')): cats.add('renal')
        if code.startswith(('140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','200','201','202','203','204','205','206','207','208','2386')): cats.add('cancer')
        if code.startswith(('196','197','198','199')): cats.add('mets')
        if code.startswith(('042','043','044')): cats.add('aids')
        if code.startswith(('4560','4561','4562','5722','5723','5724','5725','5726','5727','5728')): cats.add('liver_sev')
    else:
        if code.startswith(('I21','I22','I252')): cats.add('mi')
        if code.startswith(('I099','I110','I130','I132','I255','I420','I425','I426','I427','I428','I429','I43','I50','P290')): cats.add('chf')
        if code.startswith(('I70','I71','I731','I738','I739','I771','I790','I792','K551','K558','K559','Z958','Z959')): cats.add('pvd')
        if code.startswith(('G45','G46','H340','I60','I61','I62','I63','I64','I65','I66','I67','I68','I69')): cats.add('cvd')
        if code.startswith(('F00','F01','F02','F03','F051','G30','G311')): cats.add('dementia')
        if code.startswith(('I278','I279','J40','J41','J42','J43','J44','J45','J46','J47','J60','J61','J62','J63','J64','J65','J66','J67','J684','J701','J703')): cats.add('copd')
        if code.startswith(('M05','M06','M315','M32','M33','M34','M351','M353','M360')): cats.add('rheum')
        if code.startswith(('K25','K26','K27','K28')): cats.add('pud')
        if code.startswith(('B18','K700','K701','K702','K703','K709','K713','K714','K715','K717','K73','K74','K760','K762','K763','K764','K768','K769','Z944')): cats.add('liver_mild')
        if code.startswith(('E100','E101','E109','E110','E111','E119','E120','E121','E129','E130','E131','E139','E140','E141','E149')): cats.add('dm_mild')
        if code.startswith(('E102','E103','E104','E105','E106','E107','E108','E112','E113','E114','E115','E116','E117','E118','E122','E123','E124','E125','E126','E127','E128','E132','E133','E134','E135','E136','E137','E138','E142','E143','E144','E145','E146','E147','E148')): cats.add('dm_sev')
        if code.startswith(('G041','G114','G801','G802','G81','G82','G830','G831','G832','G833','G834','G839')): cats.add('hemi')
        if code.startswith(('I120','I131','N032','N033','N034','N035','N036','N037','N052','N053','N054','N055','N056','N057','N18','N19','N250','Z490','Z491','Z492','Z940','Z992')): cats.add('renal')
        if code.startswith(('C00','C01','C02','C03','C04','C05','C06','C07','C08','C09','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C30','C31','C32','C33','C34','C37','C38','C39','C40','C41','C43','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C60','C61','C62','C63','C64','C65','C66','C67','C68','C69','C70','C71','C72','C73','C74','C75','C76','C81','C82','C83','C84','C85','C88','C90','C91','C92','C93','C94','C95','C96','C97')): cats.add('cancer')
        if code.startswith(('C77','C78','C79','C80')): cats.add('mets')
        if code.startswith(('B20','B21','B22','B24')): cats.add('aids')
        if code.startswith(('I850','I859','I864','I982','K704','K711','K721','K729','K765','K766','K767')): cats.add('liver_sev')
    return cats

weights = {'mi':1,'chf':1,'pvd':1,'cvd':1,'dementia':1,'copd':1,'rheum':1,'pud':1,'liver_mild':1,'dm_mild':1,
           'dm_sev':2,'hemi':2,'renal':2,'cancer':2,'liver_sev':3,'mets':6,'aids':6}
cat_rows = []
for _, r in all_dx.iterrows():
    cats = charlson_cats(r['icd_code'], int(r['icd_version']))
    cat_rows.append([r['hadm_id']] + [1.0 if c in cats else 0.0 for c in weights.keys()])
cat_df = pd.DataFrame(cat_rows, columns=['hadm_id']+list(weights.keys())).astype({'hadm_id':'Int64'})
ch = cat_df.groupby('hadm_id')[list(weights.keys())].max().reset_index()
ch['charlson'] = sum(ch[c]*w for c,w in weights.items())
cohort = cohort.merge(ch[['hadm_id','charlson']], on='hadm_id', how='left')
cohort['charlson_age'] = cohort['charlson'] + np.where(cohort['anchor_age']>=40, (cohort['anchor_age']-40)//10, 0)

# ---------- 8. Outcomes & final tidy ----------
cohort['hosp_death'] = cohort['hospital_expire_flag'].fillna(0).astype(int)
cohort['icu_los_days'] = (pd.to_datetime(cohort['outtime']) - pd.to_datetime(cohort['intime'])).dt.total_seconds()/86400.0
cohort['hosp_los_days'] = (pd.to_datetime(cohort['dischtime']) - pd.to_datetime(cohort['admittime'])).dt.total_seconds()/86400.0
cohort['db'] = 'mimic'
cohort = cohort.rename(columns={'intime':'icu_intime','outtime':'icu_outtime','anchor_age':'age','gender':'sex'})

keep = ['db','subject_id','hadm_id','stay_id','age','sex','icu_los_days','hosp_los_days','s06_subtypes',
        'gcs_total','gcs_motor_only','admit_map','admit_inr','admit_ptt','admit_plt','admit_fib','admit_hgb','hgb_min72','hb_drop2',
        'charlson','charlson_age','neuro_surgery','mv_flag',
        'baseline_at','ever_at','reversal','reversal_h_first','resume_cls','resume_h','last_anticoag_stop_h',
        'tx_rbc','tx_ffp','tx_plt','tx_any','tx_first_h',
        'pe','mi','is','dvt','arte','thrombosis_any',
        'hosp_death','icu_intime','icu_outtime','admittime']
cohort[keep].to_csv(OUT, index=False)
print('saved', OUT, 'rows:', len(cohort))
print('baseline_at counts:'); print(cohort['baseline_at'].value_counts(dropna=False).to_string())
print('reversal counts:'); print(cohort['reversal'].value_counts(dropna=False).head(15).to_string())
print('resumption counts:', cohort['resume_cls'].notna().sum(), 'of', len(cohort))
print('hosp_death:', cohort['hosp_death'].sum(), '/', len(cohort))
