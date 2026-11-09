# STKG_CLL_POI - BIRD

Package indépendant et reproductible pour prédire **la prochaine œuvre dont l'observation commence** dans une visite du musée BIRD. Sources BIRD, annotations sémantiques, deux JSON canoniques, audit, code, tests et exemples de résultats sont inclus. Les résultats du dossier `runs/` sont des contrôles exploratoires d'une seule initialisation : **aucune amélioration du contraste ou de la sémantique n'est revendiquée sans la matrice d'ablations multi-initialisations**.

## Périmètre fixé

- 359 œuvres candidates issues du catalogue ; 51 journaux de visite ; séparation stricte de 35/8/8 visiteurs avec seed de split 2022. Elle reproduit la liste des visiteurs du module Rooms après normalisation du préfixe `Visitor_` → `items_`, mais **la cible et les dénominateurs POI diffèrent**.
- Prévision faite au **début de l'observation courante**. Le modèle voit l'identité de l'œuvre courante, les anciens débuts, les durées antérieures connues ou imputables causalement ; il ne voit ni la fin courante ni le début de l'œuvre cible.
- Le graphe comprend `Artwork → Room`, `Artwork → SubjectMain`, `Artwork → SubjectSecondary`, `Artwork → Artwork` transition orientée avec décompte et statistiques de délais/durée causale agrégées **uniquement sur train**. La prédiction compare l'état GRU aux représentations des 359 œuvres candidates.
- `style_raw` et `visual_description` sont conservés comme archives pour le futur module PLM mais **aucun modèle de ce paquet ne les lit**. Aucune image ou PLM n'est requise à l'entraînement.
- Par défaut, aucun masque de disponibilité : une entrée BIRD du 20 février 2019 observe `0_2100` alors que le catalogue lui donne un accrochage au 21 février. Le masque éventuel nécessite d'abord une correction justifiée de la source et une nouvelle préparation.

## Installation et exécution

Python >=3.10 et PyTorch CPU/GPU. Sous Linux/macOS, depuis la racine du ZIP :

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python -m stkg_cll_poi prepare
python -m stkg_cll_poi baselines
python -m stkg_cll_poi train --output runs/full_seed2022
python -m stkg_cll_poi predict --run-dir runs/full_seed2022 --visit-id items_201902181020 --event-index 10
```

Sous Windows PowerShell, remplacer l'activation par `.venv\\Scripts\\Activate.ps1` et `python` par `py` si nécessaire. Les commandes ont été exécutées en CPU avec Python 3.12.14, NumPy 2.3.5 et PyTorch 2.8.0+cpu. `pip install -e .` utilise `pyproject.toml`. Les données brutes sont sous `data/raw/` ; `prepare` écrase les JSON dérivés de manière déterministe et enregistre leurs hachages.

La préparation écrit dans `data/processed/` :

| Sortie | Rôle |
| --- | --- |
| `trajectories_poi.json` | 7 204 débuts préservés, appariements start/end, fins observées et durées imputées **hors ligne**, clairement étiquetées. |
| `artworks_semantics.json` | Catalogue, salles et sujets normalisés ; description/style stockés à part et exclus des features. |
| `data_audit.json` | Pertes, ambiguïtés, ancrage source SHA-256, dates et cas incohérents. |
| `annotation_lineage.json` | Correspondance contrôlée des 359 résultats source avec CSV, modèle/prompt/checksum image ; généré par `scripts/build_annotation_lineage.py`. |
| `split.json`, `graph_train.json` | Visiteurs figés et relations comportementales train seulement ; caractéristiques temporelles des arêtes. |

`train` crée `config.json`, `split.json`, `graph_train.json`, `feature_normalizer.json`, `model.pt`, `metrics.json`. Il sélectionne le meilleur checkpoint sur le **MRR validation** avec arrêt anticipé. Le fichier `metrics.json` marque le test « non évalué ». Après avoir figé la méthode et le checkpoint :

```bash
python -m stkg_cll_poi evaluate-test --run-dir runs/full_seed2022
```

Cette commande crée `test_metrics.json` et `test_predictions.csv` et refuse une deuxième évaluation dans le même dossier. Elle vérifie les SHA-256 des données préparées. Le fait de connaître déjà des résultats exploratoires sur le split 2022 limite toute revendication confirmatoire sur ce split ; voir `docs/METHODS.md`.

## Refaire les comparaisons

Le script `scripts/run_validation_matrix.py` fixe les **mêmes visiteurs** (`--split-seed 2022`), varie uniquement les initialisations (`--seeds 2022 2023 2024`) et n'ouvre pas le test des modèles neuraux :

```bash
python scripts/run_validation_matrix.py --seeds 2022 2023 2024
```

Il entraîne les 15 bras préannoncés dans le script : POI-ID GRU, transitions, full supervision, augmentation seule, contraste séquentiel seul, contraste graphe seul, combinaison, ablations sujet principal/secondaire, salle/temps/comportement, permutation des sujets entre œuvres (distribution préservée), `sasrec_style`, et Transformer causal temporel. `sasrec_style` signifie **implémentation de référence inspirée de SASRec**, pas reproduction officielle ; le Transformer temporel utilise les features temporelles du préfixe et n'est **pas** TiSASRec. Tous les bras partagent le catalogue candidat et le split de visiteurs, mais leurs capacités restent déclarées dans les configurations. Le script génère un CSV des scores validation ; sélectionner sur validation et rapporter tous les bras déclarés avant de toucher au test.

Contrôles ponctuels utiles :

```bash
python -m stkg_cll_poi train --variant full --seq-weight 0 --graph-weight 0 --no-augmentation --output runs/full_supervised
python -m stkg_cll_poi train --variant full --seq-weight 0 --graph-weight 0 --output runs/full_augmentation_only
python -m stkg_cll_poi train --variant no_time --output runs/no_time
python -m stkg_cll_poi train --architecture sasrec_style --variant poi_id --seq-weight 0 --graph-weight 0 --no-augmentation --output runs/sasrec_style
```

`--seed` est la graine **du modèle** ; `--split-seed` détermine exclusivement le split. Ne pas mélanger les deux sources de variabilité dans un même tableau.

## Interprétation honnête

Les visiteurs sont peu nombreux ; les milliers de préfixes ne sont pas des individus indépendants. Les métriques incluent Acc@1, Hit@5/10, MRR, macro par visiteur et IC bootstrap par visiteur ; la précision est limitée par huit visiteurs test pour le split principal. Les sujets générés par IA exigent une revue humaine ciblée ; `statut=ok` ne certifie pas leur vérité. Les identifiants et annotations statiques sont connus pour toutes les œuvres (cadre transductif), tandis que les transitions et leurs statistiques sont calculées sur les seuls visiteurs train. Pour les œuvres jamais cibles dans train, les embeddings ID reçoivent encore des gradients par la softmax de tous les candidats : ne pas présenter le modèle comme une méthode zero-shot démontrée.

**Documentation :** `docs/METHODS.md` (définitions mathématiques, fuites, pertes et évaluations), `docs/DATA_AUDIT.md` (constats chiffrés), `docs/ROOMS_COMPARISON.md` (traçabilité de la version salle), `docs/MODEL_CARD.md`, `docs/REPRODUCIBILITY.md` (procédure d'article), `SOURCE_LICENSES.md`.
