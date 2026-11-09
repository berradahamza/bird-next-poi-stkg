> **Version initiale conservée pour traçabilité.** Les passages sur l’encodeur à un saut et les dimensions du graphe décrivent le mode historique `full`. Pour la version hétérogène v2, voir [HETEROGENEOUS.md](HETEROGENEOUS.md) et le [README actuel](../README.md). Les définitions des préfixes, pertes et métriques restent applicables.

# Méthodes et interprétation scientifique

## 1. Unité d'observation et instant de décision

Pour une visite `v`, les événements `e_0,...,e_{n-1}` correspondent aux lignes d'ouverture d'observation d'œuvre. Pour `0 <= t < n-1`, l'historique contient `e_0,...,e_t` (tronqué aux 32 plus récents), la cible est l'identifiant POI de `e_{t+1}`. La prédiction a lieu au temps `start_t`. Aucun timestamp ou durée de l'événement cible, aucune fin actuelle et aucune moyenne complète de visite n'est un input. Le modèle classe **l'ensemble complet des 359 œuvres**, avec une softmax entière plutôt qu'un petit échantillon de négatifs. Les sorties sont ordonnées par score décroissant ; les ex æquo sont départagés de façon déterministe par ID trié.

La date locale est inférée du nom du fichier `items_YYYYMMDDHHMM` et sert seulement à l'audit de disponibilité ; sa signification/calendrier doivent être confirmés à la source BIRD. Une œuvre peut apparaître plusieurs fois dans une visite ; les répétitions ne sont pas fusionnées.

## 2. Appariement et imputation

Pour un début `s_i`, sélectionner la fin non encore utilisée de même POI la plus précoce telle que `s_i <= end <= s_{i+1}` ; le dernier événement n'a pas de borne supérieure. Une fin isolée ou un timestamp vide est compté dans l'audit. Toutes les lignes brutes restent accessibles. Durée observée `d_i = end_i − s_i` si le couple est fiable.

Pour une fin absente et un début suivant connu, la **reconstruction rétrospective demandée** est :

```
gap_i = s_(i+1) - s_i
mu_full_v = moyenne des durées observées fiables positives de la visite v
d_offline_i = min(gap_i, (gap_i + mu_full_v) / 2)
end_offline_i = s_i + d_offline_i
```

La borne évite `end_offline_i > s_(i+1)`. Sans début suivant ou moyenne mesurée disponible, aucune durée n'est inventée. `duration_origin` indique la provenance. L'égalité avec `gap_i` est conservée et quantifiée : 1 201 cas sur 1 563 imputations possibles dans l'instantané.

Pour la **feature causale** d'un événement *antérieur* `i < t`, l'observation est d'abord utilisée si sa fin est connue au début `s_t`. Sinon, si `s_(i+1)` est déjà connu, `mu_prefix` est la moyenne des seules fins fiables d'événements `j < i` dont les fins étaient déjà survenues à `s_(i+1)` ; `d_causal_i = min(gap_i,(gap_i+mu_prefix)/2)`. Sans cette moyenne, durée=0 et indicateur de disponibilité=0. À l'instant `s_t`, la durée de l'événement courant t vaut toujours zéro avec indicateurs à zéro, même si une fin apparaîtra plus tard dans le fichier. Les horodatages sont en secondes depuis l'origine du journal ; le modèle n'utilise pas la durée totale de visite.

`features.py` retourne pour chaque événement du préfixe : log(1+écart au début précédent), log(1+durée antérieure utilisable), indicateur observée, indicateur imputée causalement, log(1+temps depuis le premier début). Les trois variables continues sont standardisées sur **les préfixes du train seulement** ; les indicateurs restent 0/1. `no_time` désactive aussi les attributs temporels des arêtes, pas seulement les features séquentielles.

## 3. Graphe STKG et portée transductive

Nœuds : Artwork (359), Room (37), Subject (9 sujets disponibles). Relations typées : `in_room`, `main_subject`, `secondary_subject`, `transition_out`, `transition_in`. Les arêtes statiques peuvent concerner toutes les œuvres du catalogue ; les transitions et leurs attributs proviennent des **seuls 35 visiteurs train**. Les sujets sont vérifiés par identifiant, normalisés, et exclus lorsqu'ils sont `NA`. Les secondaires sont dédupliqués et distincts du principal. Les attributs textuels `visual_description`, `style_raw`, titre, artiste et provenance figurent dans un compartiment JSON réservé au PLM ; le constructeur du modèle ne les lit pas.

Chaque paire orientée `(a,b)` du train contient `count(a,b)`, moyenne de `log(1+start_b−start_a)`, moyenne de `log(1+d_causal_a)` lorsqu'une durée était connue au début b, et proportion de transitions à durée utilisable. Un temps manquant n'est pas remplacé sans indicateur. Les moyennes normalisantes des attributs d'arête sont calculées sur les **seules arêtes du train**. Les attributs temporels modulent un poids `log(1+count) * 2 sigmoid(MLP_edge(temporal_edge))` ; le MLP démarre au poids unité. L'agrégation orientée vers les successeurs et prédécesseurs pondérés est normalisée par la somme des poids du voisinage. Les représentations de salle, sujet principal, secondaires et POI ont des projections relationnelles distinctes avant GELU et LayerNorm. Il s'agit d'un **encodeur relationnel à un saut**, sans prétendre implémenter un R-GAT ou un STKG-PLM publié dans leur intégralité.

Les identifiants, pièces et attributs statiques sont connus pour tous les candidats ; **seuls les comportements sont hors validation/test**. C'est un cadre transductif en catalogue et inductif en visiteurs. Les paramètres d'embedding ID des candidats jamais positifs en train participent néanmoins au dénominateur de la softmax : les performances sur cibles absentes du train sont rapportées à part, sans les présenter comme généralisation zero-shot prouvée.

## 4. Score et objectif

L'encodeur produit `z_p` pour chaque POI ; les historiques d'embeddings et de features temporelles entrent dans un GRU à un seul étage. La représentation finale `h_t` est comparée à **tous** les `z_p` par `scale * cosine(W h_t, z_p)`. La loss principale est la cross-entropy exacte sur les 359 candidats. Ce lien à `z_p` empêche un classifieur final indépendant de contourner entièrement le graphe, sans garantir que la sémantique influence réellement la décision : mesurer `no_subject`, `main_only`, `shuffled_subjects` (permutation déterministe des annotations conjointes entre œuvres, seed 31415), et `no_behavior`.

Deux passes stochastiques du **même préfixe** appliquent dropout aux représentations, et une faible suppression indépendante d'arêtes de transition train, en préservant toutes les arêtes statiques. La supervision moyenne les deux CE même lorsque les poids contrastifs sont nuls : l'ablation « augmentation seule » partage ainsi la procédure de vues. `--no-augmentation` fait une seule passe et supprime la perturbation des arêtes, mais conserve le dropout ordinaire comme régularisation. Des têtes MLP distinctes projettent les représentations de séquence et d'œuvre seulement pour leurs pertes contrastives ; l'inférence utilise les vecteurs avant projection.

`L_seq` = InfoNCE bidirectionnelle **B×B**, positives = deux vues du même préfixe ; les copies exactes d'historiques et les préfixes ayant le même POI cible ne sont pas négatifs. Le masque, le nombre de paires autorisées et la température sont disponibles dans le code/historique. `L_graph` = InfoNCE bidirectionnelle entre deux embeddings de **même œuvre** sous deux réalisations indépendantes du dropout d'arêtes, parmi les œuvres présentes dans les vrais historiques du lot (padding exclu) ; les œuvres distinctes servent de négatifs, même lorsqu'elles partagent un sujet. Aucun terme n'utilise une fin future. La somme est `L = CE + lambda_seq * L_seq + lambda_graph * L_graph`, avec paramètres explicites par run. Ce n'est **pas** la NT-Xent SimCLR 2B×2B ; ne pas changer sa dénomination dans l'article. Le dropout, l'embedding ID partagé et la sélection des négatifs peuvent provoquer des raccourcis ; `L_graph` doit donc être interprétée à travers les ablations `graph_weight=0` et les contrôles de sujets.

## 5. Baselines et validation

Les baselines sont popularité train, Markov-1, Markov-2 (repli 1 puis popularité), GRU POI-ID supervisé, Transformer causal `sasrec_style` et Transformer causal avec features temporelles. Les deux Transformers sont des implémentations de référence sur la même tâche/full softmax : **ni reproduction officielle de SASRec ni de TiSASRec**. Pour des comparaisons formelles avec ces publications, importer leur implémentation publique, adapter le même candidat/temps/split et documenter les différences de loss. L'archive offre aussi `scripts/run_validation_matrix.py` avec 15 comparaisons préannoncées.

Sélection sur MRR micro validation, arrêt après 5 époques sans amélioration (maximum 30), puis évaluation test par commande séparée. Les métriques précisent nombre de préfixes et nombre de visiteurs, micro et macro par visiteur, intervalles percentile 95 % par bootstrap de **visiteurs entiers**. Les IC de huit visiteurs test peuvent être larges et les transitions de la même visite ne sont pas indépendantes. Les initialisations 2022/2023/2024 gardent le `split_seed=2022`; d'autres splits groupés formeraient une analyse distincte. Comparer les différences *appariées* par visiteur et conserver les mêmes exemples/candidats lors des ablations.

**Ne pas publier le test comme confirmation aveugle après consultation des baselines et essais exploratoires sur le split 2022.** Pour une affirmation confirmatoire, verrouiller méthode/hyperparamètres à partir des seules évaluations validation, puis employer soit un ensemble de visiteurs externe indépendant, soit des splits groupés pré-définis avec procédure d'estimation qui ne sélectionne pas sur leurs tests. Documenter toute utilisation préalable des résultats test.

## Références méthodologiques

- BIRD : Worm et al., DOI 10.1145/3708319.3733686 ; projet https://mbanv2.loria.fr/.
- SimCLR : Chen et al. (2020), https://proceedings.mlr.press/v119/chen20j.html (la présente loss est nommée InfoNCE bidirectionnelle et non NT-Xent canonique).
- HestGCL : Liu et al. (2025), https://www.sciopen.com/article/10.26599/TST.2023.9010148 (positionnement POI, pas reproduction).
- Zhou et al., WWW 2024, https://arxiv.org/abs/2403.11136 (importance de la comparaison augmentation seule / CL).
- SASRec : https://arxiv.org/abs/1808.09781 ; TiSASRec : https://dl.acm.org/doi/10.1145/3336191.3371786 (baselines de référence à adapter sans revendiquer un port exact).
