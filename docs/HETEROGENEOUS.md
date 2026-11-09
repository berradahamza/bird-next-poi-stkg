# Audit d'implémentation — STKG hétérogène

## Analyse du projet avant modification

Le projet fourni sépare déjà correctement plusieurs responsabilités :

- `preprocess.py` charge catalogue/annotations, apparie les débuts et fins et conserve les imputations hors ligne dans un compartiment non utilisé par le modèle.
- `features.py` construit les préfixes causaux, le split par visiteur et les transitions train. `graph_manifest` agrège les fréquences et trois statistiques temporelles par paire dirigée.
- L'ancien `RelationalGraph` dans `model.py` apprend des tables POI/ROOM/SUBJECT, ajoute les embeddings statiques projetés à chaque POI, et agrège ses prédécesseurs/successeurs pondérés. Les secondaires sont moyennés avant projection. ROOM et SUBJECT ne reçoivent pas de messages des POI et il n'existe pas de topologie ROOM→ROOM.
- `STKGCLL` récupère les embeddings dans l'historique, ajoute les cinq features de temps, exécute un GRU et compare l'état final aux embeddings de tous les candidats. La supervision est une CE complète. Les projecteurs contrastifs sont distincts pour séquence et graphe ; la loss est une InfoNCE bidirectionnelle B×B, pas une NT-Xent SimCLR 2B×2B.
- `experiment.py` sépare seed du split et seed d'initialisation, ajuste le normaliseur sur les préfixes train, sélectionne sur MRR micro validation et réserve les scores test à `evaluate-test`.
- `evaluation.py` calcule Acc@1, Hit@5, Hit@10 et MRR, micro et macro par visiteur, avec bootstrap par visiteurs entiers.

Cette architecture a été conservée. Le changement principal remplace le bloc graphe pour les cinq variantes hétérogènes. L'ancien bloc et les variantes historiques restent disponibles, avec les mêmes noms de paramètres pour recharger les checkpoints livrés.

## Construction et topologie

`spatial.py` charge directement `data/raw/bird_adjacency.json`. Les nœuds ROOM viennent de **toutes** les clés de `rooms`. Le graphe spatial est l'ensemble des paires réciproques de `neighbors`, sans ajout de self-loop. `adjacencies` fournit le type, le drapeau de même étage et les données géométriques/debug. Les dimensions du vocabulaire sont calculées à partir des fichiers.

Le validateur vérifie salles référencées, symétrie, paires de taille deux, doublons, présence/validité de floor et area, type d'arête, cohérence de same_floor_rooms, midpoint et points finis. Les anomalies sont émises en warnings et conservées dans le diagnostic JSON. En mode strict, elles provoquent une erreur. Les relations unilatérales sont exclues plutôt que complétées. Les conflits d'attributs sur une paire réciproque entraînent un type `unknown` et la suppression de sa géométrie ambiguë. La géométrie n'est jamais utilisée pour inférer une nouvelle paire.

Les 50 paires fournies sont valides : 48 border et 2 door, donc 96 + 4 arêtes dirigées. Le sous-graphe ROOM comporte trois composantes, respectivement sur les trois étages. Il n'existe aucun chemin spatial entre elles. Les transitions comportementales peuvent traverser les étages sans créer d'adjacence physique. Une connexion verticale ne serait conservée que si elle était explicitement et réciproquement fournie ; ses coordonnées ne seraient pas comparées à celles d'un autre étage.

`heterogeneous.py` crée séparément les nœuds et relations POI/ROOM/SUBJECT. Les inverses `contains`, `main_subject_of` et `secondary_subject_of` sont de vrais jeux d'arêtes. Les labels secondaires sont dédupliqués et reliés individuellement. Les titres/artistes sont remontés depuis le compartiment d'archive historique vers des attributs explicites, sans modifier le JSON canonique d'origine ni leur donner un rôle dans les features apprises.

## Représentations et messages

Pour une salle r, l'initialisation est :

`h_ROOM(r) = E_ROOM(r) + E_FLOOR(floor(r)) + Linear([zscore(area(r)), area_available(r)])`.

L'étage est une catégorie, jamais une quantité ordonnée utilisée telle quelle. Le numéro de salle ne représente qu'un indice de vocabulaire. Les statistiques de surface sont ajustées sur les salles statiques connues du musée. Une surface manquante devient z=0 avec indicateur 0 ; l'étage manquant reçoit une catégorie explicite `unknown`. Ces cas produisent des warnings.

Les deux couches sont synchrones. Pour chaque relation statique r d'un type source s vers un type cible t :

`message_r(u) = W_s h_s(u) ⊙ scale_r`.

Les messages sont moyennés par destination **dans chaque relation**, puis ajoutés à la projection propre du nœud. Une LayerNorm, GELU et le dropout de représentation terminent la couche. Le partage de W_s et les vecteurs scale_r évitent une matrice complète pour chaque type d'arête. Les scalings catégoriels démarrent à 1 et se différencient par apprentissage. Le nombre de relations reste calculé à partir des données.

Pour les adjacences, border et door utilisent des relations distinctes et donc des scalings distincts. Un petit encodeur linéaire ajoute le drapeau same_floor_rooms et son indicateur de disponibilité au message. Avec `--midpoint`, il reçoit aussi x/y standardisés et le masque de disponibilité. Les statistiques sont distinctes pour chaque étage et utilisent une fois chaque paire non orientée. Les directions réciproques partagent les attributs de leur paire. `points` et `anchors` restent dans les exports de debug.

Pour les transitions POI, on conserve exactement les valeurs suivantes :

- count ;
- mean_log_start_gap ;
- mean_log_causal_dwell ;
- causal_dwell_available_rate.

Le poids est `log1p(count) × 2 × sigmoid(Linear(normalized_edge_time))`. La porte démarre à un facteur 1. Les moyennes entrantes et sortantes sont normalisées par la somme de ces poids, avec deux projections distinctes, comme dans le modèle d'origine. Aucune arête inverse comportementale n'est ajoutée au manifeste : l'agrégation des successeurs utilise la direction de la paire observée, comme auparavant.

Après deux couches, un POI peut intégrer les autres POI de sa salle et de ses sujets. Il reçoit également les caractéristiques des salles voisines via ROOM→ROOM→POI. Un chemin POI de la salle voisine→ROOM→ROOM→POI ferait trois sauts : cette portée supplémentaire n'est pas implémentée. Les variantes spatiales refusent une profondeur d'une couche, car elle ne permettrait pas à leur topologie d'atteindre les POI. La dernière couche ne calcule que les états POI ; les mises à jour finales ROOM/SUBJECT inutilisées et leurs paramètres sont omis.

## Intégration, contraste et ablations

Le tenseur retourné par `HeterogeneousGraph` conserve la forme `[nombre_POI, hidden]` et l'ordre de `poi_ids`. La suite de `STKGCLL.forward` est inchangée. Ce tenseur sert à la fois aux historiques et aux candidats du score final.

Le dropout d'arêtes ne touche que `next_poi`. Un même masque est conservé dans les deux couches d'une vue ; la seconde passe contrastive tire un nouveau masque. Les relations statiques, leurs inverses et les adjacences restent toutes présentes. Le dropout de représentation reste actif et indépendant ; la documentation distingue bien perturbation d'arêtes et perturbation de vecteurs.

Les cinq variantes sont déclarées dans `HETERO_VARIANTS`. Toutes gardent les features temporelles causales, y compris `poi_only`. Les variantes historiques `transitions` et `poi_id` avaient un temps désactivé : elles ne sont donc pas présentées comme des ablations identiques. Le contrôle `poi_only --gnn-layers 1` reproduit la formule comportementale legacy lorsque les poids sont copiés, ce que vérifie le test d'équivalence. La comparaison principale garde deux couches dans tous les bras.

`--room-features identity` conserve les nœuds et relations de salle tout en retirant floor et area. Le midpoint a son option propre et est désactivé par défaut. Cela sépare l'effet d'identité de salle, de métadonnées, de topologie et de géométrie.

## Absence de fuite et rechargement

Les assertions de split s'exécutent au moment de construire les exemples **et** le graphe, y compris hors de `prepare` : les trois listes doivent être non vides, disjointes, sans doublon et couvrir les visiteurs fournis. La somme des fréquences des transitions doit correspondre exactement aux transitions consécutives des seuls visiteurs train.

La construction hétérogène ne reçoit pas les trajectoires validation/test ; elle reçoit le manifeste comportemental train et les données statiques. Le normaliseur séquentiel exige une liste explicite de visiteurs train et refuse tout exemple issu d'un autre visiteur. Les paramètres de normalisation des arêtes sont exportés avec leur provenance train. Les normalisations des salles et de la géométrie utilisent les connaissances statiques du musée, ce qui est permis dans ce protocole transductif en catalogue.

Des tests modifient toutes les identités, les temps et les fins des visiteurs validation/test : les arêtes, leurs fréquences/statistiques et les deux normalisations comportementales restent invariantes. Les tests historiques contrôlent aussi l'invariance des features d'un préfixe à la modification de son futur.

Chaque run hétérogène fige son graphe, le plan, son split et son normaliseur, et enregistre leurs SHA-256 dans la configuration également stockée dans le checkpoint. Évaluation et prédiction vérifient ces hachages ainsi que l'identité de configuration/checkpoint, puis comparent les transitions figées à une reconstruction à partir du train. Elles ne dépendent pas d'un plan source qui pourrait avoir été édité entre-temps. Les checkpoints historiques conservent leur chemin de reconstruction d'origine.

## Diagnostic des données fournies

| Élément | Valeur |
| --- | ---: |
| POI / ROOM / SUBJECT | 359 / 41 / 9 |
| Transitions train distinctes / observées | 2 218 / 5 082 |
| Total des arêtes explicites | 5 302 |
| Paires spatiales / arêtes dirigées | 50 / 100 |
| Composantes ROOM / liaisons entre étages | 3 / 0 |
| POI sans salle connue | 0 |
| POI sans aucun sujet | 1 (`0_0696`) |
| POI sans sujet principal | 2, dont un possède des secondaires |
| Paramètres du modèle complet, hidden=64 | 127 301 |

Le conflit `1_5892` est signalé : `room_id=Room_2`, floor du catalogue=1, floor de la salle dans le JSON=0. Le rattachement fourni à la salle est maintenu et les métadonnées ROOM utilisent le JSON spatial. Le sujet et les journaux ne sont pas réécrits pour masquer cette anomalie.

La relance de `prepare` conserve à l'octet près `trajectories_poi.json`, `artworks_semantics.json`, `split.json` et `graph_train.json` d'origine. Le `data_audit.json` régénéré ajoute uniquement les motifs des trois fins non appariées : le code de préparation fourni produisait déjà ces motifs, mais l'ancien fichier exporté ne les contenait pas.

## Vérification et limites du livrable

Les résultats exécutés sont dans `outputs/verification.json` et les logs associés. La suite vérifie notamment :

- fidélité exacte à neighbors, absence de self-loop et de connexion verticale inventée ;
- traitement explicite des références invalides, doublons, valeurs non finies et catégories vides ;
- normalisation de la géométrie par étage, désactivation de la géométrie entre étages ;
- gradients dans les relations utiles, les embeddings ROOM/SUBJECT et la porte temporelle ;
- influence d'un POI pair à deux sauts, des types d'adjacence et du midpoint activé ;
- indépendance des résultats vis-à-vis des attributs textuels, des features de salle désactivées et de la géométrie désactivée ;
- conservation des arêtes statiques même avec suppression de toutes les transitions ;
- rejet d'un run modifié et compatibilité de prédiction avec un checkpoint historique.

Le smoke test complet utilise seulement deux époques, sur les vrais préfixes BIRD, avec les dimensions et l'historique par défaut. Il comprend une évaluation test technique d'un run explicitement marqué smoke. Ses scores ne démontrent pas l'efficacité du nouveau graphe. La matrice de 30 époques sur plusieurs seeds est fournie à exécuter ; aucun gain sur l'ancien CLL n'est revendiqué à ce stade.

Les limites déjà documentées sur les annotations sémantiques, l'imputation causale, le faible nombre de visiteurs et le caractère exploratoire du split 2022 restent applicables. L'audit PDF historique ne décrit pas les nouveaux nœuds spatiaux ; ce document fait foi pour la version v2.
