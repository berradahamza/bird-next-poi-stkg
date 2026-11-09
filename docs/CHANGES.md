# Fichiers ajoutés ou modifiés — v2

Comparaison par SHA-256 avec le ZIP fourni. Aucun fichier initial supprimé ; les runs et checkpoints historiques sont conservés. Les seuls fichiers exclus de l’archive sont les caches Python produits à l’exécution.

## Code, données et documentation

| Fichier | Statut |
| --- | --- |
| `README.md` | Modifié |
| `data/processed/data_audit.json` | Modifié |
| `data/processed/graph_summary.json` | Ajouté |
| `data/processed/heterogeneous_graph.json` | Ajouté |
| `data/processed/spatial_validation.json` | Ajouté |
| `data/raw/bird_adjacency.json` | Ajouté |
| `docs/HETEROGENEOUS.md` | Ajouté |
| `docs/METHODS.md` | Modifié |
| `docs/MODEL_CARD.md` | Modifié |
| `docs/README_LEGACY.md` | Ajouté |
| `docs/REPRODUCIBILITY.md` | Modifié |
| `docs/ROOMS_COMPARISON.md` | Modifié |
| `pyproject.toml` | Modifié |
| `requirements.txt` | Modifié |
| `scripts/run_heterogeneous_ablations.py` | Ajouté |
| `stkg_cll_poi/__init__.py` | Modifié |
| `stkg_cll_poi/__main__.py` | Modifié |
| `stkg_cll_poi/experiment.py` | Modifié |
| `stkg_cll_poi/features.py` | Modifié |
| `stkg_cll_poi/hetero_model.py` | Ajouté |
| `stkg_cll_poi/heterogeneous.py` | Ajouté |
| `stkg_cll_poi/model.py` | Modifié |
| `stkg_cll_poi/spatial.py` | Ajouté |
| `tests/test_heterogeneous.py` | Ajouté |

## Résultats de vérification ajoutés

| Fichier | Statut |
| --- | --- |
| `outputs/legacy_prediction.json` | Ajouté |
| `outputs/prepare.log` | Ajouté |
| `outputs/smoke_prediction.json` | Ajouté |
| `outputs/smoke_test.log` | Ajouté |
| `outputs/smoke_train.log` | Ajouté |
| `outputs/tests.log` | Ajouté |
| `outputs/verification.json` | Ajouté |
| `runs/hetero_smoke_seed2022/bird_adjacency.json` | Ajouté |
| `runs/hetero_smoke_seed2022/config.json` | Ajouté |
| `runs/hetero_smoke_seed2022/feature_normalizer.json` | Ajouté |
| `runs/hetero_smoke_seed2022/graph_summary.json` | Ajouté |
| `runs/hetero_smoke_seed2022/graph_train.json` | Ajouté |
| `runs/hetero_smoke_seed2022/metrics.json` | Ajouté |
| `runs/hetero_smoke_seed2022/model.pt` | Ajouté |
| `runs/hetero_smoke_seed2022/split.json` | Ajouté |
| `runs/hetero_smoke_seed2022/test_metrics.json` | Ajouté |
| `runs/hetero_smoke_seed2022/test_predictions.csv` | Ajouté |
| `docs/CHANGES.md` | Ajouté : cet inventaire |
| `outputs/change_manifest.json` | Ajouté : inventaire et SHA-256 |

Le manifeste exclut son propre hachage et celui de ce document pour éviter une référence circulaire. Il contient leurs chemins dans `manifest_files`.

## Rôle des modifications principales

- `spatial.py` : validation du plan, normalisation statique, création des salles et des adjacences.
- `heterogeneous.py` : nœuds et relations explicites, ablations, validation des extrémités et résumés JSON.
- `hetero_model.py` : deux couches relationnelles compactes, propagation ROOM/SUBJECT/POI et temporalité des transitions.
- `model.py` : sélection du nouvel encodeur, maintien de l’ancien et du GRU/CLL.
- `features.py` : assertions de split et vérification du total des transitions train ; statistiques et exemples inchangés.
- `experiment.py` : branchement, normalisation train contrôlée, graphes figés/hachés et rechargement vérifié.
- `__main__.py` : validation spatiale, inspection du graphe et options des nouvelles variantes.
- `run_heterogeneous_ablations.py` : comparaisons validation à protocole identique.
- `test_heterogeneous.py` : 17 tests nouveaux ; les 8 tests historiques sont inchangés.
- `data_audit.json` : régénération ajoutant les raisons des trois fins non appariées, déjà prévues par le code source initial.
- Documents historiques : bandeau indiquant leur périmètre v1 ; nouveau README et audit d’implémentation v2.
