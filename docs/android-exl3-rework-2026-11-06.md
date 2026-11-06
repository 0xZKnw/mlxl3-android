# EXL3 Vulkan : diagnostic S24 et refonte des projections

Essais préenregistrés ANDROID-PERF-2026-11-06-11/12 dans `opti.md`.
Baseline publiée : code `4c07189b1676b6407a0db06f6038166b1b8f7985`,
checkout documentaire `75e87bd`. Modèle utilisateur LFM3-1.7B EXL3 4,49 bpw.

## Ce que le rapport physique établit

[Rapport brut utilisateur](measurements/s24-4c07189-user.json), 13 étapes de
decode : 2,197566 tok/s, 154 soumissions synchrones par étape.

| Compteur | Total decode | Par étape | Limite d'interprétation |
| --- | ---: | ---: | --- |
| Modèle | 5 899,33 ms | 453,79 ms | Latence host |
| Attente de fence | 5 027,09 ms | 386,70 ms | 85,21% du modèle, pas un timestamp GPU |
| Transformations CPU des projections | 423,98 ms | 32,61 ms | Hors attente GPU |
| Enregistrement et soumission | 210,29 ms | 16,18 ms | CPU/pilote |
| Head | 256,90 ms | 19,76 ms | Imbriqué dans modèle/GPU, ne pas additionner |
| Readback | 27,19 ms | 2,09 ms | Copie/validation côté host |

Le head représente 4,35% du modèle. L'attente moyenne de fence est 2,51 ms
par projection. Le prefill est également token par token : 13 tokens,
1 990 soumissions, 5,106 s. L'ancien kernel est bien `scalar_reference`.
Le rapport identifie le Xclipse 940 physique et la mémoire locale cohérente
flags7 : pas de fallback Mesa sur ce téléphone.

La fence seule ne sépare pas calcul, mémoire, pilote, réveil du CPU et
ordonnancement. Statut thermique1, mode économie désactivé, low_memoryfalse
et environ1,5Go disponibles n'établissent pas l'absence de throttling ou de
pression mémoire. Aucune cause exclusive du débit n'est prouvée.

Les 34tok/s rapportés sur S25Ultra concernent un autre SoC, probablement un
GGUF Q4_K_M, sans confirmation du backend ou protocole. Termux n'implique pas
Vulkan. Cette comparaison n'établit pas un plancher20tok/s pour le S24.

## Recherche : réalisations et limites

- [strix-llama.cpp PR204](https://github.com/halo-box/strix-llama.cpp/pull/204),
  Nathanw1014, head `bd64c24736eab68a006c1f44f99da3b994ba098c`, dépendance
  formatPR201 : vrai port EXL3/Vulkan avec essais StrixHalo/gfx1151,
  RADV Mesa26.2.4. L'auteur rapporte39,9tok/s pour Qwen3.5-9B EXL3 4bpw
  contre1,31 avec les projections CPU. Ce n'est ni un téléphone, ni LFM3,
  ni une comparaison avec notre ancien kernel GPU. Le chemin dot décode
  huit états voisins depuis une fenêtre de registres ; Hadamard et échelles
  restent surGPU. Le code exige wave32/FP16, le prefill coopmat a ses propres
  conditions matérielles. Shaders et licenceMIT lus ; attribution conservée.
- [vllm.cpp : backend Vulkan EXL3](https://github.com/localai-org/vllm.cpp/blob/main/.agents/specs/backend-vulkan-exl3.md)
  : chemin GPU sans fallback CPU et tests synthétiques llvmpipe ; pas de
  mesure publiée sur GPU physique ou modèle complet dans cette spécification.
- [infr issue6](https://github.com/KessokuNet/infr/issues/6) : discussion
  EXL3/RX580 et budgets ALU. Les commentaires consultés sont des propositions,
  pas un résultat EXL3 terminé. Le débit Q4_K cité ne valide pas EXL3.
- [trellis-serve](https://github.com/sybil-solutions/trellis-serve) : chemins
  EXL3 CUDA3090 et IntelXPU/B70 ; pas de backend Vulkan/Android démontré.

Aucun benchmark EXL3/Android ou S24 public vérifiable trouvé dans cette
recherche. Ce constat ne prouve pas qu'aucun essai n'existe.

## Changement retenu

1. Un groupe de128threads calcule16sorties, quatre partitions deK et huit
   poids par lane logique. La fenêtre trellis charge2à4uint32 pour ces huit
   codes, contre deux chargements par poids dans le scalaire. À4bits,
   exactement deux mots par lane. Ce sont des instructions de lecture
   logiques, pas une mesure de traficDRAM ou un facteur de débit promis.
   Réduction partagée unique après la boucleK ; aucun requis de subgroup,
   float16arithmétique, coopmat ou extension32/64lanes.
2. SU/Hadamard128 puis Hadamard128/SV exécutés surGPU, troisdispatches avec
   barrières de dépendance dans une soumission. Même normalisation
   f32sqrt128 et frontièresFP16 explicites que la référence CPU ; padding
   logique initialisé à+0. Pas de déquantification dense globale.
3. Descripteurs et commandes préparés une fois par matrice au chargement.
   Activations temporaires résidentes et réutilisées uniquement après fence.
   Maximum193matrices et707sets de quatrebindings ; validation des formes,
   échelles, propriétaires de ressources, tailles, groupes et budget existant.
4. TimestampsGPU optionnels TOP→BOTTOM pour chaque soumission si la file
   annonce une largeur valide et une période positive. Masquage des compteurs
   qui rebouclent, lecture après fence sans WAIT ; échec de lecture nonfatal.
   `device_samples` distingue absence de mesure et zéro. Intervalle incluant
   les passes/barrières, pas une mesure isolée des unités arithmétiques.

L'ordre de sommationFP32 change ; aucune égalité bit à bit générale annoncée.
Le vieux kernel scalaire et le HadamardCPU restent des oracles, l'autotuner et
le tuilage4 abandonné sont supprimés. Le microbenchmark LUT historique garde
son kernel scalaire ; il n'est pas relancé ni utilisé pour sélectionner ce code.

Le modèle reste hybride : normalisations, attention, convolutions et caches
surCPU. **154soumissions par token restent présentes**. Cette refonte vise le
calcul EXL3 et le travailCPU autour de chaque projection ; la fusion du graphe
et le prefill matriciel restent nonimplémentés, sans gainS24 présumé.

## Vérification

- 19testsCPU réussis (`cargo test --manifest-path engine/Cargo.toml --locked`).
  Les tests Vulkan ignorés par cette commande sont exécutés explicitement.
- 96projections brutes : 1..8bits, troiscodebooks, 16×32/128×128/256×384/
  2048×512 contre reconstructionCPU indépendante et accumulationf64.
  Tolérance préexistante `1e-4 + 2e-5 * abs(expected)`.
- 72matrices complètes : mêmes huitbitrates et troiscodebooks, formes
  logiques/paddées1/128×1/128,127/128×129/256,256×384. SU/SV signées FP16,
  entrées nonexactesFP16, commandes réutilisées, zéros, overflow, NaN,
  dimensions/échelles invalides, limite de matrices et propriétaireGPU.
  Tolérance préenregistrée `2e-4 + 2e-3 * abs(expected)` contre ancien
  chemin CPU/Hadamard/scalar ; nonfinitude rejetée.
- Parcours réel Engine→Linear→GPU sur fixture LFM3 nonnulle, huitétapes/
  128logits, head, attention complète/glissante, conv, reset, annulation,
  budgetcontexte et compteurs/timestamps. Tolérance fixture inchangée
  `5e-4 + 2e-3 * abs(expected)` ; pas de remplacement des logits de référence.
- Kani0.68/CBMC6.11 : onzeharnais réussis,1 005obligations,23covers ; trois
  obligations de formatstd internes inatteignables. Nouvelles propriétés :
  bornes des formes/buffers/groupes pour champsusize/u32 symboliques sans
  hypothèse,83obligations/3covers ; delta des timestampsu64 et largeuru32
  symboliques,46obligations/3covers. Groupe16sorties et contratsCPU existants
  conservés, permutation256 avecunwind260, mémoire4types/unwind7 et dense
  quatrechunks/unwind6. Ce model checking borné ne prouve ni shaders, ni
  Vulkan/pilote, ni FFI, ni inférence réelle. Ancien harnais flottant Kani
  incompatible reste nonvérifié et désactivé comme auparavant.
- Compilation shaders/validationSPIR-V Vulkan1.1, Clippystrict et format
  contrôlés. Vérification de sensibilité, CI du commit exact et réceptionAPK
  consignées ci-dessous une fois terminées. Logs temporaires `build/perf12-*`.

La mutation temporaire `total`→`-total` du nouveau shader fait échouer l'oracle
dès16×32/K1/codebookDefault : -0,22641754 au lieu de+0,22641754, exit101.
Source restaurée dans unfinally, shaders reconstruits, les huitcontrôles
fonctionnels Vulkan finaux passent. Le benchmark historique facultatif reste
ignoré. `cargo fmt`, `cargo clippy --locked --all-targets -- -D warnings`,
`cargo kani --lib` (après le harnais shape ciblé), `actionlint` et `git diff
--check` passent. Commandes GPU identiques à celles du jobVulkan CI : quatre
filtreslib (`codebooks`, `stream_qmv`, `prepared_projection`,
`persistent_scratch`), puis testsintegration `vulkan` et `flow`, tous avec
`--ignored --test-threads=1`. ICD local :
`/Users/justin/.codex/cache/android-sdk/emulator/lib64/vulkan/lvp_icd.json`.
La variableDYLD_LIBRARY_PATH est limitée auxbinaires de tests, jamais rustc.

Tous les contrôles Vulkan locaux utilisent Mesa logiciel, sans benchmark GPU
matériel ni chargement du vrai checkpoint en concurrence avec l'autre chat.
Le débit, la correction numérique du modèle complet4,49bpw et la stabilité
Xclipse940 restent à mesurer sur le S24 après installation de l'APK.

## Réception CI et APK

Source livrée : `bd9671479d01f3e251c10d106a08856c4191ea6e`.
[Run42259603754](https://github.com/0xZKnw/mlxl3-android/actions/runs/42259603754) :
les cinqjobs terminés avec succès, résultats et logs inspectés. 19testsCPU,
huitcontrôlesVulkan explicites, onzeharnaisKani/1 005obligations/23covers et
douzetestsUI exécutés (six par écran). TestsJVM inchangés restaurés du cache
Gradle, lint et buildsARM64/APK réussis. BenchmarkLUT facultatif sauté.
Le premier runcec814c/42259538294 est **annulé**, supersédé par la correction
de la ligneblancheEOF ; ce n'est pas une validation supplémentaire.

[Artefact13525355179](https://github.com/0xZKnw/mlxl3-android/actions/runs/42259603754/artifacts/13525355179)
:

- ZIP :13 243 993octets, digestGitHub vérifié
  `e1e859bdee96849fb2dd69997dc55f14214198274662b1dd648cee0b77c47a01`.
- `debug/app-debug.apk` :33 965 868octets, SHA-256 vérifié contre
  `SHA256SUMS` : `eb42fe3caf056f73cd6aa4ffe168e12196b09bbe1fe65cbdfe9fdcadf211b56e`.
- SignatureAPK v2 vérifiée avecapksigner36.0.0 ; zipalign16K réussi.
  Les deuxbibliothèques sontARM64 : moteur et AndroidXgraphicspath.
  Première vérification decontenu tropstricte attendait une seulebibliothèque
  et a rejeté AndroidX ; assertioncorrigée, contrôlerefait réussi. Aucun
  défaut de l'APK. `llvm-readelf` confirme touslessegmentsLOAD moteur à0x4000.
- Moteur embarqué SHA-256 :
  `23cdb597886becdabae4fcb3030cb1d807effbc6629cb18f9422f3ae76d42fec` ;
  SHA source et marqueurs `register_window_16` /
  `gpu_hadamard_prepared_commands` retrouvés dans le binaire.

APK local : `build/apk-bd96714/debug/app-debug.apk`. Logs CI conservés dans
`build/perf12-ci-{apk,proofs,vulkan,ui-small,ui-phone}.log` ; fichiers temporaires
nonversionnés. État : code public et APKCI, aucune release ni installation
téléphone. Recherche terminée ; correction fonctionnelle validée danslesdomaines
ci-dessus, performanceS24 **nonmesurée/nonconcluante**.
