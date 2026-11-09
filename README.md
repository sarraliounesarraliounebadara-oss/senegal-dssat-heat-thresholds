
# Analyse des sorties DSSAT : scripts à déposer

Article : « Statistical heat thresholds persist after neutralising selected heat functions and depend on the water regime: a synthetic-truth test in the Sahel ».

Ces scripts partent des rendements déjà simulés par DSSAT et produisent les tableaux et les figures de l'article. Les scripts qui écrivent les fichiers d'entrée de DSSAT et lancent le modèle ne sont pas dans ce dossier.

## Avant de lancer

1. Ouvrir `scripts/paths.py` et indiquer les quatre dossiers :
   - `AFM` : le projet sur les rendements observés ;
   - `OUTPUTS` : les masques et les rendements DAPSA ;
   - `DSSAT` : les sorties DSSAT ;
   - `DATA` : les grilles ANACIM.
2. Installer les bibliothèques de `requirements.txt`.
3. Lancer les scripts depuis le dossier `scripts/`, dans l'ordre des numéros.

## Scripts

| Script | Rôle | Résultats dans l'article |
|---|---|---|
| `paths.py` | Les quatre dossiers à adapter | |
| `climate_io.py`, `degree_days.py` | Lecture des grilles ANACIM ; degrés-jours par sinus simple | |
| `department_yields.py` | Agrégation des rendements des pixels par département (pondération MIRCA2000 × cos(lat)) | |
| `01_growing_season_exposures.py` | Expositions par département, culture et année : degrés-jours, pluie, jours de pluie, plus longue séquence sèche | Entrées de toute la chaîne |
| `02_threshold_panel_model.py` | Recherche du seuil (26–40 °C), modèle GDD + EDD, bootstrap par années entières, effet de +1 °C | Tableau 2 |
| `03_edd_by_threshold.py` | Coefficient EDD à chaque seuil candidat | Figure 3b,e |
| `04_search_adjusted_tests.py` | Tests ajustés pour la recherche du seuil (bootstrap sauvage restreint, 9 999 tirages) | Tableau 2, Tables S5–S6 |
| `05_threshold_stability.py` | Stabilité du seuil : bootstrap et retrait d'une année à la fois | Table S4 |
| `10_evaluate_dssat_against_dapsa.py` | Évaluation de DSSAT face à DAPSA (calage 1986–2000, validation 2001–2016) | Tableau 1, Figure 2 |
| `11_e1_build_simulated_yield_files.py` | Mise des rendements simulés au format de la chaîne (variantes : obs, sim, 2001–2016, sans 1997, fertilité ± 0,1) | Entrées de E1 |
| `12_e1_summary_observed_vs_simulated.py` | Synthèse observé / simulé et contrôles de robustesse | Tableau 2, Table S4 |
| `13_e1_figure3_threshold_profiles.py` | Profils de seuil, observé et simulé | Figure 3 |
| `20_e2_factorial_heat_water.py` | Chaîne appliquée à chaque traitement (fonctions de chaleur × régime hydrique) et décomposition | Tableau 2, Figure 4 |
| `30_x1_true_vs_inferred_warming.py` | Effet vrai de +1 °C et +2 °C contre l'effet inféré, avec la décomposition | Tableau 2, Figure 6, Table S8 |
| `40_x2x3_spei_and_pixel_exposures.py` | SPEI des départements et expositions des pixels | Entrées de X2 et X3 |
| `41_x2_water_controls.py` | Chaîne avec SPEI ou stress hydrique simulé comme contrôle | Table S5, Figure 7 |
| `42_x3_scale_of_analysis.py` | Panels pixel, mixte et département | Table S6, Figure 8a,b |
| `43_harmonise_bootstrap_draws.py` | Mêmes tirages bootstrap pour les lignes communes de X2, X3 et E2 | Tables S5 et S6 |
| `50_x4_noise_test.py` | Résidus DAPSA permutés et mis à l'échelle (k = 0,5–3 ; 1 000 panels par niveau) | Table S7, Figure 8c–f |
| `60_article_figures.py` | Figures 1 et 4 à 8 (1000 dpi) | Figures 1, 4–8 |
| `70_check_groundnut_minus3_nowaterstress.py` | Vérification du test ajusté pour l'arachide à −3 °C sans stress hydrique : t recalculé, p reproduit, seconde graine | §3.2 |
| `71_figure_S1_bootstrap_null.py` | Distribution sous l'hypothèse nulle de ce test | Figure S1 |

## Ordre d'exécution

1. **Chaîne statistique :** `01`, puis `02` à `05` pour chaque jeu de rendements. Les scripts `20`, `41` et `42` appellent eux-mêmes `02` et `04`.
2. **Évaluation de DSSAT :** `10`.
3. **E1 :** `11`, puis `02` à `05` dans chaque dossier `e1/<variante>`, puis `12` et `13`.
4. **E2 :** `20`.
5. **X1 :** `30`.
6. **X2 et X3 :** `40`, puis `41`, puis `42`, puis `43`.
7. **X4 :** `50`.
8. **Figures et contrôles :** `60`, puis `70` et `71`.

## Vérifications

- Les scripts `02`, `03`, `04`, `05`, `10`, `11`, `12`, `30`, `43`, `50` (mode `summary`), `70` et `71` ont été relancés depuis ce dossier.
  - Ils reproduisent les sorties de l'article à l'identique.
  - Pour `50` en mode `summary`, les écarts sont au plus de 0,0001, à cause de l'arrondi des réplicats stockés.
- Les scripts `13` et `60` produisent bien leurs figures.
- Les scripts `01`, `20`, `40`, `41` et `42` sont longs. Ils compilent et se chargent sans erreur.
