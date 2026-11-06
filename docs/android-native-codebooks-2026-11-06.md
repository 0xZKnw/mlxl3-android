# Matvec EXL3 natif et diagnostic par projection

Essais14/15 préenregistrés dans `opti.md`, baselineexécutable
`e554439c975f31f556ab86677252173140e10cf2`, arbredocumentairecd6b1a8.

## Diagnostic et recherche

[Rapport S24 brut](measurements/s24-e554439-user.json) :3,495986tok/s,
13tokensprompt/13decode, thermique1,108soumissions/token. MatvecEXL3
198,210ms/token, Hadamardentrée1,299ms etsortie0,636ms. Matvec92,28%des
intervallesGPU et69,74%du modèlehost284,21ms/token. Head16,96ms imbriqué,
lecturequeries0,139ms. Il faut cibler le matvec, pas optimiser davantage
Hadamard ou la collecte des timestamps. La latenceGPU inclut mémoire et
synchronisations ; ce rapport ne prouve pas une saturationALU ou DRAM.

Le téléphone annonce shaderFloat16 et des sousgroupesnatifs64 ; pas de
coopmatKHR/NV dans la liste sélectionnée. Les propriétés supplémentaires
permettront de vérifier les arrondis et les tailles imposables. Aucune
accélération liée au seul nom d'une feature n'est supposée.

Sourcesprimaires : [EXL3Vulkan PR204](https://github.com/halo-box/strix-llama.cpp/pull/204)
(headbd64c24736eab68a006c1f44f99da3b994ba098c), dont le décodageFP16 parpaires ;
[typesGLSL explicites](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)
pour les types et opérations16bits. Le [guideRDNA](https://gpuopen.com/learn/rdna-performance-guide/)
décrit les compromis de taille des waves ; wave32 n'est pas imposé sans
mesure. Les chiffresStrixHalo ne sont pas un benchmarkS24.

## Kernel livré

- Même fenêtretrellis, seizecolonnes/groupe128, accumulation/réductionFP32 et
  frontièresHadamardFP16. ReconstructionDefault/MCG par additionsf16vec2
  précises ; deuxcoefficients à lafois, quatreFMAsvec2FP32 dans le même ordre
  parsortie. Ces codebooks évitent les lecturesdispersées deLUT dans le matvec.
- HuitbinariesSPIR-V avecBITS1..8 littéral. Les indices/shifts sont pliés
  parglslc avant le pilote, sans dépendre uniquement de son optimisation
  des constantes de spécialisation. SPIR-Vinspecté : OpFAddv2half précis,
  quatreFmav2float, aucuneOpSpecConstant. Pas d'accumulationFP16.
- Mul1 reste sur la LUT exacte : le premier prototypeFMAFP16 échouait sur
  Mesa au codeMul1/1 (0,640625 au lieu0,6411133). Log
  `build/perf15-native.log` conservé. La révision intermédiaireFMAFP32/RTE
  était correcte mais ajoutait des calculs sans avantageFP16 ; elle n'est pas
  livrée. Le cheminMul1 conservé bénéficie seulement des bitslittéraux.
- Activation uniquement si shaderFloat16 et extensionAPI1.1 compatibles.
  Au chargement,393216comparaisonsGPU/CPU bitexact vérifient196608valeurs
  distinctes sur chacune des deuxcomposantes. Résultatsfinis et tailles
  vérifiés ; sinon repli vers le kernelLUT actuel. Échec de création du
  kernel rapide parbitrate également diagnostiqué et replié. TimeoutGPU
  reste une erreur imposant redémarrage ; pas de repli avec travailpendant.
- Ce contrôle est un test de correction, pas une calibration de vitesse.
  Ses douzesoumissions et sa durée sont séparées des compteurs d'inférence.
  Poids toujours compressés ; aucun checkpoint ou modèleGPU supplémentaire.

Ce sont des changements de calcul réels, mais le gainphysiqueS24 reste
**nonmesuré**. La validationlogicielle et les features ne garantissent pas
un débit20tok/s ni l'absence de régressionphysique.

## Nouveau diagnostic, schema_version2

Le menu de copie existant inclut automatiquement ces champs, sansnouvelleUI :

- `projection_groups` pardimensionslogiques/paddées, bitrate, codebook et
  kernel effectivement choisi ; nombre de matrices/samples, tempscumulés,
  minimum/maximum/moyenne du matvec et échecs de collecte.
- `slowest_projections` : huitmatricesnommées les plus coûteuses parphase,
  ordre déterministe en casd'égalité, volumescompressés et mêmes statistiques.
- `logical_weight_gb_per_second` : volumecompressélogique divisé par temps,
  **pas une mesure de bande passanteDRAM**. Valeurnull quand aucunéchantillon
  ou durée nulle ; aucune fausse vitesse issue de sortiesvides.
- TempsCPU distincts pour normalisation, attention/caches, convolution,
  RoPE, activations/résidus, embeddings ; restehost `host_unattributed_ns`.
  Head est imbriqué, GPU et attentehost se recouvrent : ne pas lesadditionner.
- Erreurs de lecturequeries et derniercode, modecodebooks/bitratessélectionnés,
  résultat et durée d'autocontrôle. Pas de succèsnatif annoncé aprèsrepli.
- Pilote nom/versioninfo, heapstailles et limitesGPU, float-controls16,
  taillesmin/max de sousgroupesimposables et extensionspertinentes.
- Dimensions du modèle, typesdecouches, contexte/position, taillelogiquedes
  cachesCPU et paramètres de génération. Aucun prompt, réponse ou cheminlocal.

Préfill/decode restent séparés ;108soumissions/token cible inchangé. Le
rapport est construit après le chronométrage de génération. Collecteurs
bornés193matrices, agrégats de forme et top8 plutôt qu'un journal parappel.
Les catégoriesCPU sont exclusives ; le reste inclut les portions nonmesurées
et les coûts du profiler. Aucun percentile ni capteur de fréquence n'est inventé.

## Vérification et limites

- 21testsCPU réussis : compteurs/groupe/top8/ordre/nullsanssample/saturation,
  erreurs/bornes et suiteexistante. Profils remis à zéro entre phases.
- DixcontrôlesVulkan explicites surMesa : contrôle393216valeurs du cheminnatif
  réellement activé, repli volontaire sansFloat16 utilisable ;96casQMV
  1..8bits×3codebooks×4formes CPU/f64 et bitexactauLUT,72projectionscomplètes,
  mappings/dense/head, groupes/réutilisation/erreurs et fixtureEngine128logits.
  Tolérancesexistantes inchangées ; aucun oracle de logits remplacé.
- L'échec initialMul1 a été détecté par le testnatif dédié, alors que les
  testsQMV génériques passaient grâce au repliLUT. Le contrôle ne confond
  donc pas fallback fonctionnel et validation du nouveau kernel.
- Fixture réelle Engine→Linear→GPU vérifie nomsdeprojection, sommesgroupées
  égales auxcompteursglobaux, top8, CPUcatégories, privacy etschéma2,
  sanschangerreset/cancel/budgetcontexte ni l'oracle128logits.
- Kani0.68/CBMC6.11 : treizeharnaisCPU ; cibléprofilage137obligations/2covers,
  vrais compteurs avec deuxéchantillonsu64[3] et succès/échecssymboliques sans
  assume. Counts, échecs, min/max et sommes saturées vérifiés ; comptageborné
  ne prouve niJSON complet, niFloat16, niVulkan/pilote, niFFI ou inférenceréelle.
  Ancien harnaisflottant incompatible demeure nonexécuté.
- Format/Clippystrict host+ARM64, actionlint, compilation etvalidationSPIR-V
  Vulkan1.1 réussis. VérificationCI du commit exact etAPKàconsigner aprèsrun.
  Logs locaux temporaires `build/perf14-*`, `build/perf15-*`,
  `build/perf14-15-*`, dontle prototype négatif, conservés.

Aucun benchmarkGPUmatérielMac ni vrai checkpoint concurrent. Parité complète
LFM3-1.7B4,49bpw, débit/stabilité/température/RAMS24 de cetAPK nonvérifiés ici.
