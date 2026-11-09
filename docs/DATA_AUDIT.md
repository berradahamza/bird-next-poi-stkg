# Audit de données - instantané du 23 septembre 2026

Le programme `prepare` recalcule et écrit toutes les valeurs dans `data/processed/data_audit.json`. Source : 51 fichiers `start_obs_artworks`, 51 fichiers `end_obs_artworks`, catalogue `artworks_dataset.csv`, annotations `oeuvres_semantique.csv` (avec archive de résultats source).

| Contrôle | Résultat de l'instantané fourni |
| --- | ---: |
| Visites / œuvres du catalogue / œuvres observées | 51 / 359 / 356 |
| Débuts / lignes de fin brutes | 7 204 / 5 635 |
| Couples début-fin retenus | 5 631 |
| Débuts sans fin retenue | 1 573 |
| Lignes de fin horodatées non appariées | 3 |
| Cas à plusieurs fins candidates pour un même début | 1 |
| Lignes de fin invalides (horodatage vide) | 1 |
| Durées hors ligne imputables (début suivant et moyenne disponible) | 1 563 |
| Imputations plafonnées au prochain début | 1 201 |
| Œuvres avec sujet principal NA | 2 |
| Contrôles d'image IA non vérifiés humainement / désaccords sujets | 282 / 140 |
| Accrochage incompatible avec observation | 1 (`0_2100` le 20/02 ; date catalogue 21/02) |
| Visiteurs train / validation / test | 35 / 8 / 8 |
| Arêtes POI distinctes du train | 2 218 |

La différence brute entre 7 204 et 5 635 n'est **pas** le nombre exact de fins manquantes : quatre fins ne forment pas un couple acceptable (trois valides isolées, une ligne sans heure). Les journaux restent entiers sous `data/raw`; aucune ligne de début n'est perdue. Le choix d'appariement exige même POI, fin unique, `start_i <= end <= start_{i+1}`. En cas de fins candidates multiples, la plus précoce est choisie et le cas entre dans `ambiguous_end_candidates`. Des observations d'une même œuvre répétées sont conservées comme événements distincts.

La moyenne hors ligne de toute une visite fuit potentiellement des fins futures : son imputation n'est jamais lue comme feature. Si une durée passée manque et que le début suivant est déjà connu, `features.causal_duration` la reconstruit seulement avec les fins fiables connues à ce moment-là. La durée courante reste masquée. Une revue des observations appariées, des trois fins isolées, des 1 201 plafonds et d'un échantillon des annotations discordantes demeure nécessaire avant l'article.

**Conflit de disponibilité.** Le catalogue `0_2100` (Room_5) annonce `hook date=2019-02-21`, mais le journal `items_201902201020.csv` l'observe le 20. Le masque temporel est désactivé par défaut ; l'option le réclamant échoue tant que le conflit reste dans l'audit. Ne pas corriger en regardant uniquement les cibles test sans documenter la source documentaire.

## Faux masquage des fins fiables

`scripts/audit_imputation.py` masque une fin mesurée à la fois et la reconstruit avec le prochain début et la moyenne hors ligne **hors mesure masquée** ; l'autre variante utilise seulement les fins antérieures causalement disponibles. Sur 5 590 événements mesurés ayant un début suivant, l'imputation hors ligne présente une MAE de 7,72 s (erreur absolue médiane 2,89 s) ; la variante causale est définie pour 5 539 et donne une MAE de 7,52 s (médiane 2,74 s). Cette expérience ne mesure que le comportement sur les fins **observées** : si les fins réellement absentes ne sont pas manquantes au hasard, ces chiffres ne sont pas une garantie de validité sur leurs 1 573 événements. Rapport détaillé : `runs/imputation_sensitivity.json`.
