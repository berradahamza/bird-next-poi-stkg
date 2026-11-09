"""Verify per-artwork enrichment source against the flattened semantic CSV."""
import csv,hashlib,json,zipfile
from pathlib import Path
from stkg_cll_poi.preprocess import put_json,sha256
ROOT=Path(__file__).resolve().parent.parent
raw=ROOT/'data/raw'
sem={r['image_id']:r for r in csv.DictReader(open(raw/'oeuvres_semantique.csv',encoding='utf-8-sig',newline=''))}
source={}
with zipfile.ZipFile(raw/'resultats_source.zip') as z:
    for name in z.namelist():
        if not name.endswith('.json'):continue
        data=json.loads(z.read(name))
        if data['image_id'] in source:raise ValueError('Duplicate result ID')
        source[data['image_id']]=data
if set(source)!=set(sem):raise ValueError('Source JSON and semantic CSV artwork IDs differ')
for poi,v in source.items():
    row=sem[poi]
    for field in ['sujet_principal','description_visuelle','style_visuel_probable']:
        if (v.get(field) or '').strip()!=(row.get(field) or '').strip():
            raise ValueError(f'Mismatch in {poi}:{field}')
report={'format':'stkg.cll.annotation_lineage.v1',
        'source_archive_sha256':sha256(raw/'resultats_source.zip'),
        'semantic_csv_sha256':sha256(raw/'oeuvres_semantique.csv'),
        'matched_ids_and_main_subject_description_style':len(source),
        'entries':{poi:{'source_status':v['statut'],'model':v.get('model'),
                        'prompt_version':v.get('prompt_version'),
                        'fingerprint':v.get('fingerprint'),
                        'sha256_download':v.get('sha256_download'),
                        'identity_control':v.get('controle_identite'),
                        'result_json_name':f'resultats/{poi}.json'} for poi,v in sorted(source.items())}}
put_json(ROOT/'data/processed/annotation_lineage.json',report)
print('Verified annotation lineage:',len(source),'artworks')
