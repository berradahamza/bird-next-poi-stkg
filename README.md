# STKG_CLL_POI — graphe hétérogène BIRD, v2

Prédiction du prochain POI avec **359 POI, 41 ROOM et 9 SUBJECT** dans les données fournies. Les salles et sujets sont désormais des nœuds qui reçoivent et transmettent des messages. Le GRU, les features temporelles causales, les pertes contrastives et le protocole d'évaluation restent ceux du projet initial.

**Le fichier `bird_adjacency.json` est déjà intégré dans `data/raw/bird_adjacency.json`.** Aucun CSV supplémentaire n'est nécessaire. Les anciens checkpoints et leurs variantes (`full`, `no_time`, etc.) utilisent toujours l'ancien encodeur. Le nouveau mode par défaut est `poi_room_subject`.

## Installation et lancement

Depuis le dossier `STKG_CLL_POI` extrait du ZIP, avec Python >= 3.10 :

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m unittest discover -s tests -v
python -m stkg_cll_poi validate-spatial --strict
python -m stkg_cll_poi prepare
```

Les données préparées sont incluses : `prepare` permet de les reconstruire depuis les bruts. Le programme d'entraînement actuel s'exécute sur CPU, comme celui livré initialement. Il n'ajoute aucune dépendance de graphe telle que PyG ou DGL. Les commandes sont aussi utilisables sous PowerShell ; remplacer `python` par `py` si nécessaire.

Entraînement complet, sélection par **MRR micro validation** :

```bash
python -m stkg_cll_poi train --variant poi_room_subject --seed 2022 --split-seed 2022 --history 32 --hidden 64 --epochs 30 --patience 5 --output runs/hetero_full_seed2022
```

Évaluation test du checkpoint choisi, après avoir fixé la configuration :

```bash
python -m stkg_cll_poi evaluate-test --run-dir runs/hetero_full_seed2022
```

Prédiction depuis un préfixe :

```bash
python -m stkg_cll_poi predict --run-dir runs/hetero_full_seed2022 --visit-id items_201902181020 --event-index 10 --topk 5
```

`train` ne calcule pas de scores test. `evaluate-test` crée `test_metrics.json` et `test_predictions.csv`, une seule fois par run. Un répertoire de run déjà entraîné n'est pas écrasé : choisir un autre `--output` pour relancer.

## Architecture

1. Identités POI ; ROOM = identité + embedding catégoriel d'étage + surface standardisée et indicateur de disponibilité ; identités SUBJECT.
2. Deux couches de messages hétérogènes : moyenne par relation, projection partagée par type source et modulation vectorielle propre à chaque relation. Les transitions POI conservent leurs projections entrante/sortante, `log1p(count)` et leur porte temporelle.
3. Les embeddings POI obtenus alimentent la séquence, avec les cinq features temporelles causales existantes, puis le GRU.
4. L'état du GRU est comparé par score cosinus appris aux embeddings de tous les POI candidats. Les deux pertes InfoNCE existantes, sur la séquence et sur les POI, sont conservées.

Avec les réglages par défaut, le modèle complet contient **127 301 paramètres**. Deux sauts permettent POI → ROOM/SUBJECT → POI et ROOM voisine → ROOM → POI. Les POI de la salle voisine demanderaient trois sauts ; cette première version reste limitée à deux.

Titre et artiste sont conservés comme attributs explicites des POI dans le graphe exporté. Ils ne deviennent pas des nœuds et ne sont pas encodés en vecteurs textuels. Les sujets secondaires sont des arêtes distinctes vers des nœuds partagés, avec une relation inverse ; leur agrégation se fait pendant les messages, sans embedding moyen de secondaires préconstruit.

## Relations effectivement construites

Comptages calculés sur le fichier fourni et le split 2022, sans constante de taille dans le constructeur :

| Relation | Arêtes dirigées |
| --- | ---: |
| POI → next_poi → POI | 2 218 |
| POI → located_in → ROOM | 359 |
| ROOM → contains → POI | 359 |
| POI → has_main_subject → SUBJECT | 357 |
| SUBJECT → main_subject_of → POI | 357 |
| POI → has_secondary_subject → SUBJECT | 776 |
| SUBJECT → secondary_subject_of → POI | 776 |
| ROOM → adjacent_border → ROOM | 96 |
| ROOM → adjacent_door → ROOM | 4 |
| **Total** | **5 302** |

Les 2 218 paires comportementales distinctes agrègent 5 082 transitions train. Leur traitement entrant/sortant reprend l'ancien encodeur ; cela ne transforme pas la relation comportementale en adjacence physique non orientée.

Le plan contient 16 salles à l'étage 0, 20 à l'étage 1 et 5 à l'étage 2. Ses 50 paires donnent 100 arêtes, sans salle isolée et avec trois composantes. Aucune connexion verticale n'est ajoutée. Les 41 salles sont conservées, y compris celles sans POI du catalogue.

## Cinq ablations comparables

Toutes les commandes suivantes gardent le même split, la même seed d'initialisation, les features temporelles, le GRU/CLL, deux couches et les mêmes réglages d'entraînement.

| Variante | POI ↔ ROOM | Étage/surface | ROOM ↔ ROOM | POI ↔ SUBJECT |
| --- | :---: | :---: | :---: | :---: |
| `poi_only` | — | — | — | — |
| `poi_room` | Oui | Oui | — | — |
| `poi_room_spatial` | Oui | Oui | Oui | — |
| `poi_subject` | — | — | — | Oui |
| `poi_room_subject` | Oui | Oui | Oui | Oui |

```bash
python -m stkg_cll_poi train --variant poi_only --seed 2022 --split-seed 2022 --output runs/ablation_poi_only
python -m stkg_cll_poi train --variant poi_room --seed 2022 --split-seed 2022 --output runs/ablation_poi_room
python -m stkg_cll_poi train --variant poi_room_spatial --seed 2022 --split-seed 2022 --output runs/ablation_poi_room_spatial
python -m stkg_cll_poi train --variant poi_subject --seed 2022 --split-seed 2022 --output runs/ablation_poi_subject
python -m stkg_cll_poi train --variant poi_room_subject --seed 2022 --split-seed 2022 --output runs/ablation_poi_room_subject
```

Pour toutes les variantes avec trois initialisations, sans ouvrir le test :

```bash
python scripts/run_heterogeneous_ablations.py --seeds 2022 2023 2024 --split-seed 2022 --epochs 30
```

Le script contrôle l'égalité des listes de visiteurs, des exemples, des données, des transitions et du normaliseur. Il écrit `runs/heterogeneous_ablations/validation_matrix.csv` avec Acc@1, Hit@5, Hit@10 et MRR micro/macro. Les seeds de modèle varient ensemble pour chaque variante ; le split reste à 2022.

Pour distinguer connaissance de la salle, caractéristiques, topologie et géométrie :

```bash
python -m stkg_cll_poi train --variant poi_room --room-features identity --seed 2022 --split-seed 2022 --output runs/room_identity_only
python -m stkg_cll_poi train --variant poi_room_spatial --midpoint --seed 2022 --split-seed 2022 --output runs/spatial_with_midpoint
```

Comparer `room_identity_only` à `ablation_poi_room`, puis `ablation_poi_room` à `ablation_poi_room_spatial`, puis ce dernier à `spatial_with_midpoint`. Le midpoint est désactivé par défaut. Il est standardisé séparément par étage et accompagné d'un indicateur de disponibilité ; aucune distance inter-étage n'est calculée.

L'option `--gnn-layers 1` est disponible pour les variantes sans adjacency. Les variantes spatiales exigent deux couches pour que l'adjacence atteigne les POI. `poi_only --gnn-layers 1` reprend la formule comportementale de l'ancien encodeur, temps activé ; un test vérifie l'équivalence à poids identiques. La comparaison principale utilise deux couches pour les cinq variantes.

L'ancienne architecture complète reste accessible sans modification de sa formule :

```bash
python -m stkg_cll_poi train --variant full --seed 2022 --split-seed 2022 --output runs/legacy_full_comparison
python scripts/compare_runs.py runs/hetero_full_seed2022 runs/legacy_full_comparison --fold validation
```

Les poids contrastifs restent `--seq-weight 0.05`, `--graph-weight 0.02`, température `0.2`. L'edge dropout par défaut est `0.1` **uniquement sur `next_poi`** ; les arêtes ROOM/SUBJECT et les adjacences sont préservées. Le dropout de représentation reste `0.2`. Pour une comparaison sans contraste, employer les mêmes options `--seq-weight 0 --graph-weight 0` pour tous les bras ; ajouter `--no-augmentation` pour une seule passe supervisée.

## Validation spatiale et anomalies connues

`validate-spatial` contrôle références, symétrie, doublons, attributs, valeurs finies et géométrie. Les relations réciproques de `neighbors` sont la topologie de référence ; `adjacencies` donne leurs attributs. Chaque paire valide produit exactement deux arêtes. Les types sont distincts, sans codage ordinal `border=0, door=1` utilisé comme quantité.

En cas de fichier incohérent, des warnings nomment les problèmes. Les références inconnues, self-loops et déclarations unilatérales sont exclues. Une paire réciproque sans attributs fiables garde le type catégoriel `unknown`, sans coordonnées inventées. `--strict` sur le validateur, ou `--strict-spatial` sur `train`/`prepare`, fait échouer le chargement en cas de problème dans le JSON.

**Conflit entre catalogue et plan :** `1_5892` (Raphaël dessinant dans la campagne de Rome) est rattaché à `Room_2` mais déclaré à l'étage 1 dans le catalogue ; le JSON place `Room_2` à l'étage 0. Le code garde `room_id` et utilise l'étage de la salle issu du plan. Il émet un warning et écrit `poi_room_floor_conflicts` ; aucune correction de la source n'est inventée. Le JSON spatial lui-même passe la validation stricte.

Un POI (`0_0696`) n'a aucun sujet exploitable ; aucun n'est sans salle connue. L'absence de sujet signifie absence d'arête, sans catégorie sémantique artificielle.

## Sorties et reproductibilité

`prepare` préserve les trajectoires et écrit en plus :

- `data/processed/heterogeneous_graph.json` : graphe complet avec les attributs POI, ROOM, SUBJECT et les relations explicites ;
- `data/processed/graph_summary.json` : comptages, composants, anomalies et normalisations ;
- `data/processed/spatial_validation.json` : diagnostic du fichier spatial.

Pour inspecter une variante sans entraîner :

```bash
python -m stkg_cll_poi build-graph --variant poi_room_subject --output runs/graph_inspection
```

Chaque nouveau run hétérogène sauvegarde `graph_train.json`, `graph_summary.json`, une copie de `bird_adjacency.json`, `split.json`, `feature_normalizer.json`, `config.json`, `model.pt` et `metrics.json`. Les données et les entrées du run sont hachées. L'évaluation recharge ce graphe figé et vérifie sa provenance, même si le plan original a changé depuis. Pour utiliser un autre plan à l'entraînement : `--spatial-json CHEMIN/bird_adjacency.json`.

Le protocole reste **35/8/8 visiteurs**, **5 082/988/1 083 préfixes**, seed de split 2022, historique 32, catalogue de 359 candidats, répétitions conservées, mêmes règles de fins/imputation causale. Les données statiques sont connues pour tout le catalogue. Transitions, fréquences, statistiques temporelles et normalisations comportementales utilisent exclusivement TRAIN. Les métriques conservent micro, macro par visiteur et bootstrap des visiteurs.

## Vérifications livrées

La suite inclut les tests historiques et les tests de topologie, propagation à deux sauts, gradients par relation, ablations, causalité, absence de fuite et rechargement. Le smoke test `runs/hetero_smoke_seed2022/` utilise **deux époques seulement**, hidden=64 et historique=32 ; il a parcouru entraînement, sélection validation, sauvegarde/rechargement, évaluation test et prédiction. Il sert à vérifier le fonctionnement, pas à conclure à un gain de performance. Les résultats historiques restent dans leurs dossiers.

Détails d'implémentation : `docs/HETEROGENEOUS.md`. Vérifications et environnement : `outputs/verification.json`. Inventaire complet des fichiers ajoutés/modifiés : `docs/CHANGES.md` et `outputs/change_manifest.json`. L'ancien README et l'audit PDF concernent la version initiale ; ils restent disponibles pour comparaison.

## Licence et attribution

- **Code** : licence MIT, voir [`LICENSE`](LICENSE).
- **Données BIRD originales** : fournies par le projet MBANv2 / musée des Beaux-Arts de Nancy,
  sous licence **CC BY-NC-SA 4.0** (https://mbanv2.loria.fr/). Les fichiers de données dérivés
  de BIRD présents dans ce dépôt restent soumis à cette licence (pas d'usage commercial,
  partage dans les mêmes conditions).
- Si vous utilisez ces données, citez l'article original :
  > Alexanne Worm et al., *BIRD: A Museum Open Dataset Combining Behavior Patterns and
  > Identity Types to Better Model Visitors' Experience*. DOI : 10.1145/3708319.3733686
