# S24 : projections groupées et détail des temps GPU

Essai ANDROID-PERF-2026-11-06-13, préenregistré dans `opti.md`.
Baseline exécutable `bd9671479d01f3e251c10d106a08856c4191ea6e` ;
arbre documentaire précédent `2e41efa`.

## Diagnostic physique reçu

[JSON utilisateur intégral](measurements/s24-bd96714-user.json),
LFM3-1.7B EXL3 4,49bpw, GalaxyS24Exynos/Xclipse940, pilote104869891,
thermique0, économied'énergie désactivée, low_memoryfalse.
16tokensprompt,19étapesdecode,3,222052tok/s.

| Mesure decode | Par token |
| --- | ---: |
| Modèle host | 308,59 ms |
| Intervalles GPU | 221,88 ms (71,90% du modèle) |
| Attente host de fence | 263,09 ms |
| CPU projections | 7,58 ms |
| Enregistrement/soumission host | 13,61 ms |
| Head host, imbriqué | 17,50 ms |
| Readback host | 2,41 ms |

154soumissions/token,142matrices EXL3 préparées. Le gros coût est encore
côtéGPU ; les timestamps ne séparent pas ALU, accès mémoire, Hadamard et
barrières. GPU et attentehost se recouvrent : ne pas additionner ces colonnes.
Le GPU travaille pendant environ4,216s des5,863s du modèle decode ; les
fences seules n'expliquent pas tout.

Comparaison descriptive avec [le rapport4c07189](measurements/s24-4c07189-user.json)
:2,198→3,222tok/s (+46,6%), modèle453,79→308,59ms/token,
CPUprojections32,61→7,58ms/token. Prompts/générations13/13 contre16/19,
thermique1 contre0, mémoire disponible et contexte différents, pas de
répétitions : aucune attribution causale complète ni gain reproductible
déclaré. Pas de benchmark supplémentaire lancé localement.

## Changement

- Q/K/V partagent une soumission ; W1/W3 partagent une soumission.
  L'entrée commune est uploadée une seulefois, les mêmes kernels/commandes
  s'exécutent dans le même ordre numérique.24pairesFFN et11tripletsattention
  retirent46soumissions/token : cible mécanique154→108, pas un facteur de
  débit promis. Fixture3couches :22→15 constaté par test.
- Une sortieGPU persistante propre à chaque matrice conserve les résultats
  jusqu'au fence partagé. Aucun alias de commandes accepté dans un groupe,
  mêmes formes d'entrée et propriétaireGPU vérifiés avant soumission.
  BarrièreALL_COMMANDS entre commandes pour le scratch partagé, puis une
  seule barrièreSHADER_WRITE→HOST_READ en fin de groupe avant lectureCPU.
- La conversionCPU FP16 redondante des sorties EXL3 est supprimée : le
  postshader l'effectue déjà et le readback rejette les nonfinitudes. Les
  frontièresFP16 des projections denses et les kernels numériques restent
  inchangés. Aucun nouvel algorithme de déquantification sélectionné à l'aveugle.
- Intervalles GPU séparés : `device_exl3_input_ns`, `device_exl3_matvec_ns`,
  `device_exl3_output_ns`, `device_other_ns` (head/denses dans le moteur).
  `device_compute_ns` est leur somme. Quatre queries propres à chaque
  matrice, pool776slots ; collecte après fence sans WAIT. Le tempsCPU passé
  à lire les queries apparaît dans `query_read_ns`.
- `device_samples` compte les commandes de calcul mesurées,
  `device_exl3_samples` les matrices EXL3 ; `decode_submissions` compte les
  soumissions/fences. Ces nombres ne sont plus égaux après regroupement.
  Les intervallesmatrice **excluent la barrière finale de visibilitéhost** ;
  l'attentehost l'inclut. Ne pas interpréter une différence avec l'ancien
  périmètreGPU comme une accélération physique établie.
- Capacités subgroup et shaderFloat16/coopmat disponibles ajoutées au
  diagnostic, enlectureseule. Aucune feature ni nouveau backend activé.

Sorties supplémentaires : taille logique4×cols parmatrice, borne193×32768×4
=25 296 896octets, sous le budgetGPU3Gio existant. L'allocation effective
dépend du modèle ; ce plafond n'est ni la consommationprocessus ni le trafic
mémoire. Le modèle reste hybride et le prefill token par token.

La synchronisation suit les scopes de soumission décrits par
[Khronos](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html).
Les contrôles dynamiques ne constituent pas une preuve du piloteXclipse.

## Vérification locale et limites

- 19testsCPU réussis (`cargo test --manifest-path engine/Cargo.toml --locked`).
  Les testsGPU ignorés par défaut sont sélectionnés explicitement comme enCI.
- NeufcontrôlesVulkan explicites surMesa logiciel :196608valeurscodebook,
  96casQMV CPU/f64,72matrices avec échelles/padding/FP16/overflow/réutilisation,
  dense/head, mappings persistants, nouveau batch et fixtureEngine128logits.
  Oracles et tolérances du lot12 inchangés.
- Testbatch : groupes1/2/3, sortiesrectangulaires128/256/383, entréelogique129
  paddée256, bits4/5/8 et troiscodebooks, troisentrées successives, ordre
  inversé ; égalitébitàbit avec exécution séparée, une soumissionpargroupe,
  samples/plagesquery distinctes et somme des phases. Vide/4éléments/doublons/
  mauvaises dimensions/NaN/propriétaireGPU refusés sans soumission, puis
  nouvelle exécution avec zéros réussie.
- CheminEngine réel : même oracle128logits, cachesattention/glissants et
  convolution, reset/cancel/budgetcontexte.15soumissions/étapefixture,
  prefill `14 * prompt_tokens + 1`, decode `15 * decode_tokens` ; samples
  demeurent respectivement `21 * prompt_tokens + 1` et `22 * decode_tokens`.
- Mutation temporaire du compteur pour compter les matrices plutôt que les
  soumissions détectée au groupe2 :2 au lieu1, exit101. Source restaurée
  enfinally et neufcontrôlesVulkan finaux réussis. Aucun kernel numérique
  modifié pour cette mutation.
- Kani0.68/CBMC6.11 : cibléquery114obligations/3covers réussis sur les vrais
  helpers, indexusize et tailleusize symboliques complets sansassume ;
  acceptation1..3 et plages4alignées dans776queries pour193matrices.
  SuiteCPU12harnais :1119obligations/26covers, troisformatstdinatteignables.
  Le compteurGPU n'est pas dans la tranche des preuves ; tentative globale
  locale lancée en parallèle du contrôle de mutation, donc la preuveCI sur
  commitfinal sera la réception de provenance exacte. Cela ne prouve ni
  GLSL/Vulkan, ni synchronisation, ni FFI, ni checkpoint1.7B. Harnaisflottant
  Kani historique toujours incompatible/nonexécuté.
- Format, Clippystrict host/ARM64, actionlint et diffcheck réussis. Shaders
  inchangés du lot12 ; compilation/validationSPIR-V et APK revérifiées parCI.
  Logs temporaires `build/perf13-*`, dont les échecs de mutation, conservés.

CI étendue avec coucheKhronos et validation des dépendances pour batch et
Engine : vérification du chargement réel de la couche et échec surVUID ou
SYNC-HAZARD. Pas de dépendance runtimeAPK ajoutée. Cette validation, l'APK et
leurs résultats exacts sont à consigner après le run.

DébitS24 de cetteversion, pleincheckpoint/logits/états1.7B, stabilitéXclipse,
RAM physique/thermique nonmesurés ici. Aucune promesse20tok/s ; le calculGPU
reste le poste principal à qualifier avec le prochain diagnostic de phases.

## Réception CI et APK

Source `e554439c975f31f556ab86677252173140e10cf2`,
[run42287156602](https://github.com/0xZKnw/mlxl3-android/actions/runs/42287156602)
terminé : **cinqjobs réussis**, résultats et logs inspectés.19CPU, neufcontrôles
Vulkan usuels, batch+Engine répétés souscoucheKhronos, douzetestsUI (six par
écran). SuiteKani du commit exact :12harnais,1119obligations/26covers,
troisformatstdinatteignables ; ciblémémoire190 exécuté aussi en préalable,
pas compté comme un treizièmeharnais. TestsJVM inchangés restaurés du cache
Gradle, lint et compilationshost/ARM64/APK réussis. Benchmark facultatif sauté.

KhronosValidationLayers1.3.275.0-1 installé ; couche effectivement chargée
(marqueurloader contrôlé), validation de synchronisation activée par
`VK_LAYER_ENABLES=VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT`.
AucunVUID niSYNC-HAZARD dans les deuxparcours contrôlés. Log dédié dans
[l'artefact13538735512](https://github.com/0xZKnw/mlxl3-android/actions/runs/42287156602/artifacts/13538735512).
Ces tests dynamiques logiciels ne prouvent pas la synchronisation du piloteS24.

[APK artefact13538461133](https://github.com/0xZKnw/mlxl3-android/actions/runs/42287156602/artifacts/13538461133) :

- ZIP :13 256 586octets, digestGitHub vérifié
  `995381b0c059b212aabbbdcd7bfc603d0fa7d02a851db465085997ce375f04ea`.
- `debug/app-debug.apk` :33 998 148octets, SHA-256 contreSHA256SUMS
  `70b19b8122140d02a21ceddbbdaa8fcfa3f3eed88da24f9a6988152c66ffd2b5`.
- Signaturev2 vérifiée avecapksigner36.0.0, zipalign16K réussi, deuxlibsARM64
  (moteur/AndroidXgraphicspath), segmentsLOAD moteur alignés0x4000.
- MoteurSHA-256 :
  `945e18648c78f6bbab546b1a1b6d0634f5f2bd77bcc703ce4d1e5e00e4b9e2c8`.
  SHA source, marqueur `gpu_hadamard_prepared_grouped` et nouvellesphases
  retrouvés dans le binaire embarqué.

APK local `build/apk-e554439/debug/app-debug.apk`, logsCI
`build/perf13-ci-{apk,proofs,vulkan,ui-small,ui-phone}.log`, nonversionnés.
État **public/APKCI**, aucune release ni installationtéléphone. Essai clos
fonctionnellement dans les domaines décrits ; vitesseS24 **nonmesurée et
nonconcluante** jusqu'au prochain essai physique. Aucun essai local encore en cours.
