> **Version initiale conservée pour traçabilité.** Les passages sur l’encodeur à un saut et les dimensions du graphe décrivent le mode historique `full`. Pour la version hétérogène v2, voir [HETEROGENEOUS.md](HETEROGENEOUS.md) et le [README actuel](../README.md). Les définitions des préfixes, pertes et métriques restent applicables.

# Reproductibilité et dossier d'article

## Ordre opératoire

1. Archiver le ZIP tel quel et son SHA-256 ; ne pas modifier les bruts. Installer les versions de l'environnement indiquées dans `README.md` et relancer `python -m unittest discover -s tests -v`. Refaire aussi `python scripts/build_annotation_lineage.py` pour vérifier la provenance des 359 annotations.
2. Refaire `python -m stkg_cll_poi prepare`. Comparer `data_audit.json` et les SHA-256 ; vérifier les trois fins isolées, la fin sans timestamp et le conflit d'accrochage. Sans résolution documentée de ce dernier, laisser le masque désactivé.
3. Recalculer les baselines. Geler les 15 variantes et trois seeds du script d'ablations ainsi que leur budget de réglage ; archiver l'ordre de l'évaluation et tout écart de configuration.
4. Exécuter la matrice sur validation. Montrer score et dispersion par seed pour `full_supervised`, `full_aug_only`, `full_seq_cl`, `full_graph_cl`, `full_joint_cl`. Une amélioration de `full_joint_cl` sur `full_supervised` ne suffit pas à attribuer le gain à `L_graph` ni à la sémantique.
5. Après figement, évaluer les checkpoints choisis sur test avec `evaluate-test` une fois par run préannoncé. Regrouper les records de chaque run par visiteur pour IC/différences appariés. Conserver runs non concluants.
6. Inclure tableaux micro/macro, effectifs des cibles jamais positives en train, distribution des durées observées/imputées et vérification humaine des annotations principales/secondaires. Déclarer toute consultation exploratoire préalable du test 2022.

## Vérifications programmées

`tests/test_pipeline.py` contrôle la conservation des débuts, les fins à usage unique, les anomalies du dataset, la formule d'imputation, l'invariance des features de préfixe à une modification du futur, l'invariance du graphe à une modification des visiteurs test, l'absence de lecture de description/style par le modèle, la propagation de gradient dans les relations statiques et le temps des arêtes, le masquage des négatifs dupliqués, la désactivation du masque de disponibilité incohérent et l'invariance du Transformer aux tokens de padding.

## Contrôles humains indispensables

Le fichier sémantique a 359 lignes « ok » mais 282 contrôles d'image non validés humainement et 140 désaccords entre sujet initial et enrichi. Faire relire un échantillon stratifié et formaliser les règles d'arbitrage. Le biais de l'imputation plafonnée (1 201/1 563 cas) doit être étudié par faux masquage de fins mesurées et comparé à une variante « durées observées uniquement ». Aucun résultat de ces contrôles humains ou de l'ensemble complet des ablations n'est prétendu acquis dans le présent paquet.

Le script `scripts/compare_runs.py RUN_A RUN_B --fold validation` calcule une différence de MRR macro **appariée par visiteur** avec IC bootstrap de visiteurs entiers, après contrôle du split, des hachages des sources, de la longueur d'historique et de l'ensemble candidat. Utiliser `--fold test` uniquement après avoir figé les runs comparés et exécuté leurs commandes `evaluate-test` ; ne pas effectuer de sélection rétroactive sur ces intervalles.
