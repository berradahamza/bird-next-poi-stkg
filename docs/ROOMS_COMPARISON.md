> **Version initiale conservée pour traçabilité.** Les passages sur l’encodeur à un saut et les dimensions du graphe décrivent le mode historique `full`. Pour la version hétérogène v2, voir [HETEROGENEOUS.md](HETEROGENEOUS.md) et le [README actuel](../README.md). Les définitions des préfixes, pertes et métriques restent applicables.

# Traçabilité par rapport au module Rooms fourni

Source étudiée : `GSITM_CLL_Rooms(8).zip`, fichiers `README.md`, `data.py`, `model.py`, `run.py`, `data/BIRD`. Le code Rooms n'est pas requis à l'exécution de ce paquet ; les journaux BIRD bruts ont été extraits directement de sa sous-arborescence `data/BIRD/`.

| Composant | Rooms initial | STKG_CLL_POI |
| --- | --- | --- |
| Cible | Prochaine salle après agrégation d'observations | Prochaine œuvre observée depuis les débuts BIRD bruts, répétitions gardées |
| Visiteurs/split | 35/8/8 sur 51 visiteurs, graine 2022 | 35/8/8, même ensemble par pli après changement de préfixe ID ; `split_seed` séparé de graine d'initialisation |
| Candidats | 37 salles observées | 359 œuvres catalogue ; 356 vues dans les débuts |
| Graphe | Matrice salle-salle pondérée sur train | Relations typées œuvre/salle/sujet principal/sujets secondaires + transitions POI train et statistiques temporelles train |
| Entrée du GRU | Salle encodée par attention de graphe | Œuvre encodée relationnellement + features de temps causales du préfixe |
| Sortie | `Linear(hidden, n_rooms)` | Produit entre historique et embeddings STKG des œuvres candidates |
| Vues contrastives | Deux passages dropout, représentation historique | Deux passages dropout d'embeddings et éventuellement arêtes train, projecteurs séparés pour préfixes et POI |
| Loss | CE + InfoNCE bidirectionnelle B×B, coefficient effectif **1** dans run.py (README décrit 0,05) | CE + `lambda_seq * InfoNCE_B×B(prefix)` + `lambda_graph * InfoNCE_B×B(POI)`, poids explicites, CE et CL journalisées |
| Durées/sujets | Absents | Délais et durées strictement causales, sujets principaux/secondaires ; style/description réservés |

Les scores Rooms et POI ne sont **pas directement comparables**, car le nombre de candidats, les événements et la cible changent. Le code Rooms évaluait le test après chaque entraînement ; ici `train` se limite à validation et `evaluate-test` est une action séparée. Les divergences code/documentation du module Rooms sont documentées plutôt que recopiées.
