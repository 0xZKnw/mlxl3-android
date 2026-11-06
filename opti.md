### ANDROID-LFM3-2026-11-05-01 — moteur EXL3 Vulkan et APK — en cours

- Demande : implémenter LFM3-1.7B texte EXL3 sur Android, premier matériel Galaxy S24 Exynos, architecture GPU portable ; UI inspirée du Desktop et APK construit rapidement par GitHub Actions. Pas de modèle distant, autre architecture, vision ou accélération spéculative. Branche isolée `codex/android-lfm3-vulkan`, baseline `6030540b5573185a4bf8b13207011872d94d0045` ; checkout personnel modifié préservé.
- Historique : codec/checkpoint Rust, backend Metal direct et projections MLX relus ; aucune implémentation Android/Vulkan existante trouvée. Campagne QUANT-2026-11-05-LFM3-MODAL du checkout personnel consultée pour identifier config, format et exports, sans reprendre ses conclusions de qualité. Architecture officielle et convertisseur à confronter avant code.
- Hypothèse/changement : moteur Rust autonome limité à LFM3, EXL3 décompressé par tuiles dans les shaders Vulkan ; pas de matrice dense globale. Commencer par kernels conservateurs sans largeur de sous-groupe imposée ; pas de mesure de vitesse ni gain annoncé. Première UI charge un dossier choisi par l'utilisateur, génère/annule et affiche les erreurs.
- Protocole : vérifier format/codebooks, projections, opérations LFM3 et transitions/caches avec oracles indépendants, fixtures jetables et tests d'intégration ; compiler/valider SPIR-V, tenter Kani sur contrats de production, fmt/Clippy/tests/build du package Android, tests/lint/assemblage APK. CI PR/push main et branche Android avec versions figées, caches Rust/Gradle, ARM64 uniquement et sans téléchargement des poids. Inspecter le run GitHub exact si accès disponible. Validation S24 physique explicitement différée à l'utilisateur ; CPU/Vulkan logiciel ne prouvent pas Xclipse.
- Sources/checkpoint, commandes, empreintes et résultats : préparation initiale dans le worktree MLXL3 ; voir maintenant `docs/s24-validation.md` et les preuves locales `build/` du dépôt séparé. Statut initial conservé : prototype en cours, qualité/performance non mesurées, APK non construit, aucune installation ni release.

#### Transfert et contrôles du 5 novembre — CI encore en cours

- L'utilisateur a demandé un dépôt séparé puis confirmé sa visibilité publique : `https://github.com/0xZKnw/mlxl3-android`. Code transféré dans `/Users/justin/Documents/mlxl3-android`, sans modifier les sources du checkout personnel MLXL3. Codec réutilisé depuis MLXL3 `1ac910d0a40efcad049c6e5a8cf081d770723465`, artwork/palette Desktop depuis `6030540b5573185a4bf8b13207011872d94d0045`. Architecture officielle LiquidAI épinglée `0dc4268b561499c648c7ba7c9466bf3d8d01517a`, EXLlamaV3 1.6.6 `bd1f519172406a9560bd5c90e61000691bb69a07` consultés sans exécuter leur code distant.
- Implémentation : projections EXL3 tuilées Vulkan, 1..8 bits et trois codebooks, aucun poids dense global reconstruit ; projections BF16/F16 non quantifiées du head/gates découpées selon la limite de buffer GPU. Convolutions7, attention GQA complète/glissante, RoPE partiel à fréquence dimension128, gates, sandwich RMSNorm, SwiGLU, tokenizer et génération greedy. Opérations légères/attention et caches CPU pour cette première version. Mémoire GPU logique plafonnée3Gio, fichiers4Gio, contexte2048 ; ce sont des limites de validation, **pas des pics processus mesurés**. Un seul modèle/une tâche. Aucune mesure de débit modèle ni promesse de vitesse.
- Premiers échecs conservés : compilation Rust (ancien helper Metal du codec importé inutile, emprunt temporaire), compilation Android JNI (`TryLockError` non Send pour anyhow::Context), lint Rust (tailles des enums/chunks), lint Android (règles de backup, icône, versions volontairement figées et ABI mobile). Corrections ciblées ; codec portable repris à sa révision stable, erreurs JNI explicites, enums alloués et contrôles stricts rerun. Les versions figées et ChromeOS sont exclus des seules alertes de disponibilité/hors cible ; commit préférences synchrone sur worker conservé pour ne pas supprimer l'ancien modèle avant persistance. Qualificateur adaptatif `v26` conservé : sa variante non qualifiée est omise par AGP8.9.2, échecAAPT conservé dans `build/gradle-build-final.log`/`gradle-build-clean.log`.
- CPU : `cargo test --manifest-path engine/Cargo.toml --locked` : **12 réussis**,3 tests GPU explicitement ignorés dans cette commande. Régressions format/codebooks/128-Hadamard contre matrice Sylvester indépendante, RoPEpartiel, convolutioncausale, attention/gates, vocabulaire, config réelle, corruption/shapes/NaN/offsets/Safetensors couverts. `cargo fmt --check`, Clippy strict host et Android ARM64, compilation release ARM64/NDK27.2.12479018 réussis. Rust1.98.1, glslc2026.4, SPIR-V validé pour Vulkan1.1. Logs `build/{cpu-tests-final2,clippy-final,android-clippy,native-build-final2}.log`.
- GPU local : invocation explicite `VK_ICD_FILENAMES=<MoltenVK_icd.json> DYLD_LIBRARY_PATH=<vulkan-loader>:<homebrew>/lib cargo test --manifest-path engine/Cargo.toml --test vulkan -- --ignored --test-threads=1` : **2 réussis**,24 comparaisons EXL3 (K1..8 ×3codebooks),128 sorties par cellule, nonvides/finies ; oracle reconstructionCPU/f64, tolérance absolue1e-4+relative2e-5. Dense F16/BF16 et erreurs vérifiés. Matrices128×128 seulement, aucune inférence de grand modèle simultanée par cette campagne. `build/vulkan-tests.log`.
- Intégration : fixture LFM3 réduite non nulle (hidden128,3couches conv/full/sliding, EXL3réel, vocab16), chargeur→Vulkan→états→logits→tokenizer→génération/reset/annulation/budget. Premier essai **échoué** : lignes de head sans token tokenizer pouvaient être sélectionnées et produire du texte vide (`build/flow-tests.log`). Correctif sélection dans les IDs réellement présents, test détectant l'ancien cas, essai suivant **réussi** (`build/flow-tests-fixed.log`). Ce modèle réduit ne valide pas les poids1.7B ni la parité de leurs logits/états.
- Kani0.68.0/CBMC6.11.0 : harnais ciblé budgetcontext réussi puis **5harnais réussis** (bitrate, indices pack/unpack, permutation inverse256, budgetcontexteusize64). DomainesK1..8/logical<16K/bit<16/state<256, permutationunwind260 ; budget sans hypothèse restreignant les usize,2covers. Aucun stub. `build/kani-targeted.log`, `kani-suite.log`, puis `kani-final.log`. Tentative supplémentaire sur greedy4logitsf32/2IDs symboliques/unwind6 : **erreur interne du compilateur**, pas contre-exemple (`kani-vocabulary.log`). Une tentative équivalente avec contrôle d'exposantIEEE échoue aussi (`kani-vocabulary-retry.log`), changement retiré. Harnais conservé hors sélection par défaut, reproductible avec `RUSTFLAGS='--cfg kani_float_checks' cargo kani --manifest-path engine/Cargo.toml --lib --harness greedy_selection_stays_in_vocabulary` ; propriété **non vérifiée**. Kani ne prouve ni Vulkan/SPIR-V, JNI/Kotlin, concurrence, ni génération complète.
- Android local : `./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebug :app:assembleDebugAndroidTest`, JDK17/Gradle8.11.1/AGP8.9.2/Kotlin2.1.20/SDK35 : compilation et lint réussis ; **3tests JVM** import/limites/annulation/fournisseur silencieux. APK signé debug vérifié par apksigner, aucune inclusion de poids. Un contrôle de correspondance binaire a détecté un APK local antérieur au dernier rebuild natif lors de compilations parallèles ; reconditionnement séquentiel et contrôle ELF/APK demandés avant livraison (`build/apk-final-sequential.log`). Aucune installation S24.
- Publication source autorisée, premiers commits `ee8fa7f`/`25ff74d`/`859a6d8` puis `0ca45f21ec840132176de0c7442b14a35940cabb`. Première CI :5harnais Kani réussis, UI **échouée** car APK/tests limitésARM64 sur émulateurx86_64 ; correction `-PuiTest` pour la seule UI, moteur livréARM64. Téléchargement inutile de mise à jourMesa17,5Mo observé lent ; pipelineAPK séparé des qualificationsVulkan, paquets installés sans upgrade. Les anciens runs sont annulés ; run à inspecter `https://github.com/0xZKnw/mlxl3-android/actions/runs/42116269570`, statut **en attente/en cours**, pas déclaré vert. APKCI, testsémulateur/capture et qualificationMesa à compléter.
- État : **code public expérimental**, correction composant locale validée dans les domaines ci-dessus ; S24/poids1.7B/logits/cache physique/mémoire réelle/thermique/vitesse **non vérifiés**. Aucune release, remplacement d'app ou benchmark de vitesse lancé. Aucun gain revendiqué.

### ANDROID-CI-2026-11-05-02 — réutiliser les outils shaders du NDK — préenregistré

- Hypothèse : le NDK27.2.12479018 déjà nécessaire contient `shader-tools/<host>/glslc` et `spirv-val` (présence vérifiée localement). Les réutiliser retire l'installationAPT du cheminAPK ; les lenteurs du miroirAzure observées dans les anciens runs ne justifient pas une dépendance système supplémentaire. Aucun changement de kernel ni d'architecture.
- Baseline : source/CI `600fcbf` (ancienne révision80b6c78 pour les timestamps complets), téléchargementglslcAPT observé lent. Variante : SDK/NDK initialisés avant contrôles Rust, `GLSLC` explicite vers le NDK, validationSPIR-V par le NDK ; autres jobs inchangés. Préserver `GLSLC` utilisateur dans le script local.
- Protocole prévu avant modification : comparer compilationSPIR-V et testsVulkan3cas/fixture avec ce compilateur, fmt/Clippy/buildARM64/APK ; inspecter ensuite le runGitHub exact et sa durée de jobAPK. Ne pas présenter différence de durée comme gain reproductible si caches/réseau diffèrent ; but fonctionnel : aucune installationAPT dans le jobAPK. Statut : **en cours**, aucune mesure de vitesse modèle.

### ANDROID-NUM-2026-11-05-03 — arrondi FP16 portable — préenregistré

- Échec réel CI600fcbf conservé : Mesa llvmpipeLLVM20.1.2, K1/codebookDefault/sortie1, -2.2867622 au lieu de -2.2864646911621094 (tolérance1e-4+relative2e-5), alors que MoltenVK passait. Log local `build/ci-600-vulkan.log`. Ne pas élargir la tolérance pour masquer cette divergence.
- Hypothèse : `packHalf2x16` ne fournit pas une frontière d'arrondi suffisamment stable entre compilateurs/pilotes pour le contrat EXL3. Changement prévu : arrondi IEEE FP32→FP16 explicite par opérations entières ; partage du codebook entre projection réelle et shader de contrôle. Conserver opérations de réduction et tolérances.
- Protocole fixé avant code : comparer bit-à-bit les65536états de chacun des3codebooks sur le GPU avec le codecRustréel (196608valeurs), puis les24QMV, dense et parcours LFM3 réduit. Compiler avec les outils NDK27.2 et valider SPIR-V ; vérifier localement avant un seul push regroupé, puis confronter à MesaCI. Si la cause est différente ou l'arrondi échoue, conserver le diagnostic et corriger avant publication du lot. Aucun benchmark de vitesse.

#### Résultats locaux du lot approuvé — 5 novembre, avant publication

- Historique CI600fcbf inspecté : Kani et UI réussis, Vulkan échoué sur l’arrondi
  précité, APK échoué sur OldTargetApi35. Logs `build/ci-600-{vulkan,apk}.log`.
  Capture ancienne révélant texte noir sur fond noir conservée ; interface
  remplacée et approuvée par l’utilisateur (« parfait la »).
- ANDROID-NUM-2026-11-05-03 : **validé localement**, CI à confronter. Ancien
  shader reproduit le même contre-exemple sur MesaLLVM21.1.4 macOS ; nouvel
  arrondi passe196608valeurs bit-à-bit sur Mesa et MoltenVK1.4.2,24QMV avec
  tolérance inchangée, dense et fixture complète réduite. Aucun modèle1.7B
  chargé, aucune vitesse mesurée. Logs `build/lavapipe-{before-retry,after,
  codebooks-retry}.log`, `build/codebook-rte.log`. Première invocation du
  compilateur Rust avec bibliothèqueLLVMémulateur a crashé ; corrigée en
  séparant compilation normale et exécution du seul binairetest avec ICDMesa.
- ANDROID-CI-2026-11-05-02 : **validé localement**, intégré au lot ; aucune
  installationAPT dans le jobAPK. CompilationNDK/SPIR-V/ARM64 et APK réussies.
  BaselineAPTglslc du run80b6c78 :4min57 entre21:58:54 et22:03:51 ; durée
  nouvelleCI non mesurée à ce stade, aucun gain reproductible revendiqué.
- SDK migré37.0/buildtools36, AGP9.1.1/Gradle9.3.1/Kotlinintégré2.2.10,
  JDK17. PackageSDKexact `platforms;android-37.0` ; tentativeandroid-37
  inexistante échouée puis corrigée. Lintstrict conserve OldTargetApi actif.
- UI : Surface/couleurs explicites, accueil/header/bulles/composer sobres,
  IMEinsets et adjustResize. Régression contraste reproduite en restaurant
  temporairement l’ancienne UI puis source approuvée restaurée.5testsUIréels
  passent, dont saisie3lignes/claviervisible/bascomposer au-dessus du clavier.
 3testsJVM passent. Logs `build/ui-contrast-before.log`, `verified-app-final.log` ;
  captures `build/ui-approved/`. Aucune installation téléphone.
- Vérification finale :12testsCPUréussis,4testsGPUignorés par défaut puis
  explicitement réussis comme indiqué ; fmt, Clippystrict host/ARM64, buildnatif,
  SPIR-V, lint/assemblage/testsAndroid, actionlint/shellcheck réussis.5harnaisKani
  réussis (422obligations,8covers,1assertioninterneformatinatteignable) ; tentative
  greedyflottant resteICE/nonvérifiée. `build/{verified-cpu-tests,verified-clippy,
  verified-android-clippy,verified-native,kani-verified-final}.log`.
- APKlocal final34049240octets, SHA256
  `4b36f17ca2fbbb471dd04ccd71e8a236dccf781ed36faf392acc926cadbb860a` ;
  apksigner/zipalign16K et correspondancebinaire natif vérifiés. Résultats et
  limites détaillés dans `docs/verification-2026-11-05.md`.
- État : code local validé dans ces domaines, publication groupée autorisée
  et CIexacte à inspecter ; pas de release ni gain de vitesse revendiqué.
  S24/poids1.7B/pariténumériquecomplète/mémoire/thermique restent non vérifiés.

#### Publication7af1cf7 et résultatGitHub

- Source7af1cf7c61a9c9780f289d95cef217bd27834756 poussée conformément à la
  demande ; run https://github.com/0xZKnw/mlxl3-android/actions/runs/42122762917
  **4jobs réussis** : APK,Vulkan,UI,Kani. Les essaisANDROID-NUM-03 et
  ANDROID-CI-02 sont **validés et intégrés** dans ces domaines ; aucune attente
  shader ni divergenceMesa. JobAPK5min29 ; caches/réseaudifférents, pas de gain
  reproductible revendiqué. Log `build/ci-7af-complete.log`.
- APKCI téléchargé : checksum748f7681faa4789faefd82994566b0f679b62e02b859bb8ed0b586e8ee5c610f
  conforme au manifeste ; apksigner et zipalign16K réussis. Source et APK
  expérimentaux publics, aucune release/appS24installée.
- CaptureIMEGitHub320x640 : composer visible mais trop haut ; assertioninitiale
  seulement anti-chevauchement insuffisante. Test renforcé sur proximité<=12dp
  et headervisible, attentebornée d’événementsIME. Suite5tests locale320x640
  réussie ; pas de modification du design approuvé. Premier ajoutducontrôle
  header a sélectionné2labelsMLXL3, assertioncorrigée vers le premierheader
  (`build/ui-small-regression.log`, `ui-small-final.log`, `ui-small-suite.log`).

#### Arrêt des modifications demandé par l’utilisateur — lot3358834

- Run42123893204/source33588346dd262ba16ddf9dfc52d11f04bbd19083 terminé :
  APK,Vulkan,Kani **réussis**, UI **échouée** sur le nouveau contrôleproximitéIME
  (gap190px, composerbottom156, fenêtre640, IME294). Le succès antérieur des
 5tests avec oracleanti-chevauchement ne valide donc pas ce cas. Log
  `build/ci-335-ui.log`. Codeapplication/moteur identique à7af1cf7 ; seul le
  test et la documentation changent. Aucun correctif supplémentaire poussé.
- Investigation locale :320x640/mdpi, suite5testsavecproximité réussie avec
  animationsnormales ; test ciblé aussi réussi avec animationssystème0
  (`build/ui-small-strengthened.log`, `ui-normal-strengthened.log`,
  `ui-ime-zero-animation.log`). L’imageAOSPARM64 API35 a été téléchargée pour
  reproduire l’environnement, mais aucun test sur cette image n’a été lancé.
- À la demande « laisse déjà ça finir histoire que je puisse tester l’app »,
  investigation **interrompue** et aucun nouveau changement de code ;
  émulateurlocal arrêté. APKCI disponible, capture/proximité sur émulateurAOSP
  reste **non résolue**. S24 et modèlecomplet toujoursnonvérifiés.

### ANDROID-UI-2026-11-05-04 — reprise du correctif IME — en cours

- Autorisation de reprise : « ok c’est bon tu peux fix ». Baseline3358834,
  oracleUIstrict conservé (gap<=12dp et header visible) ; échecCI190px conservé.
- Périmètre : fenêtreActivity/insets/composer, hôte des tests et CIUI. Aucun
  changement moteur, benchmark ou gain de vitesse prévu. Reproduire sur AOSP35
  ARM64/mdpi320x640 avant correctif ; comparer au GoogleAPI35 déjà passé.
- Vérifier le parcours réel clavier/saisie multilignes/fermeture et réouverture,
  puis suiteUI/JVM/lint/APK ; vérifier le runGitHub exact après publication.
  Kotlin/Compose n’a pas de vérificateur source praticable installé ; aucune
  preuve formelle de dispositionAndroid revendiquée. Limites physiquesS24 inchangées.

- RepriseUI : contre-exemple190px reproduit sur AOSP35ARM64
  (`build/ui-aosp-before-retry.log`). Inversionordre fenêtre puis mode
  ADJUST_NOTHING essayés, mêmes190px, retirés (`ui-aosp-window-order.log`,
  `ui-aosp-fixed.log`). DiagnosticcontenuY=-182 avecsurface640 confirmepan.
  Remplacement du champvalue/callback par TextFieldState déjà disponible dans
  Compose1.8 : testclavier strict réussi (`ui-aosp-stateful-retry.log`).
  Premier essai utilisait une extensiondelete non importée, remplacée par le
  membre replace ; aucun ajout de dépendance. Testscomplets et cyclesIME à finir.
- Nouvelle demande lenteurprefill+decode sur S24. Relecture du moteur et du
  journal : poidsGPU déjà résidents, décodageprocéduralcodebook par poids.
  Piste tableprécalculée identifiée, **aucun prototype ni benchmark moteur lancé**.
  L’utilisateur signale un profilage dans un autrechat : mesuresmoteur différées
  pour ne pas interférer ; attendre ces résultats avant optimisation. Moteur
  inchangé. ÉmulateursUI utilisentSwiftShaderlogiciel, testsUI uniquement.

- Vérification supplémentaire : le champTextFieldState seul passe l’ouverture
  initiale mais échoue après fermeture/réouverture sur l’AVDAOSP sansprofil
  (mêmes190px, `build/ui-aosp-ci-final-retry.log`). Régression conservée et
  aucune conclusion de correction complète. TestcycleIME maintenu, brouillon
  conservé attendu. Essai complémentaire : libérer le focus à la fermetureIME.
- Hôte des tests : premierAVDPixel6 redimensionné320x640 avait une découpe128px
  héritée, retiré du protocole principal ; nouvelAVDsansprofil conformeCI.
  Premier lancement du secondAVDtransportADB absent, relancewipe-data sur5584
  fonctionnelle. Tous nos émulateurs arrêtés avant la capture de l’autrechat.
- Consultation du chat « Consulter l’issue75 » : profilageAR39Desktop/M5 en
  préparation. Aucun benchmarkAndroid ni modèleGPUsupplémentaire lancé.
  Vérifications lourdes suivantes déportées enCI pour laisser le Mac libre.

### ANDROID-PERF-2026-11-06-05 — table de codebooks EXL3 — préenregistré

- Demande : prefill et decode « ultra lents » sur S24. Aucun débit/timing
  téléphone fourni. HistoriqueAndroid relu : pas de campagneperf précédente,
  seulement testscomposants ; poidsGPU déjà résidents, head ignoré pendant les
  tokensintermédiaires du prefill. Ne pas prétendre corriger ces coûts déjà évités.
- Hypothèse : recalculer FP16/codebook pour chaque poids consomme des opérations
  répétées ; tableimmuable de3*65536float32 (786432octets, pas de poidsdense)
  calculée unefois au chargement, shaderlit la même valeur. Aucun changement
  format, ordreFMA/réduction, synchronisation, tokenizer, attention ou caches.
- Baseline moteur3358834/7af1cf7, shaderprocedural expliciteRTE conservé comme
  oracle de tests ; hashsha256qmv=a3fae6f6c7023ebc51f4500fd9cc2ed1044b6fd96375d3c890ba9585e0f3f732,
  hashsha256codebook=7a18468ca457e73fca85d5b0395e0c4f97033c78eb8a0e0faf87acdeeadb29d2.
- Protocole AVANT prototype : vérifier196608sorties lookup/procédural/codecCPU
  bit-à-bit,24QMV contreoracleCPU et pipelineprocedural, fixtures nonnulles
  LFM3/reinit/annulation/budgets. Kaniindex table complet(u16state,u32codebook),
  fmt/Clippy/testCPU/SPIR-V/buildARM64. Aucune preuveGPU complète revendiquée.
- Mesure conditionnelle APRÈS correction seulement : microbenchmarkCI Mesa,
  formes2048x2048,2048x6144,2048x7680 et7680x2048,4bit,troiscodebooks,
  buffersidentiques/nonvides, warmup2 et7paires alternées procédural/lookup,
  commandes terminées par fence, tempshôte ms/matvec, médianes par variante,
  égalitébit-à-bit préalable et aprèssérie. Pas de compilation pendantmesure,
  unGPU/un modèlefixture à la fois. Source/outils/device/conditions/empreintes
  et résultatsbruts conservés. Échec ou divergence interdit gain/validation.
- Aucun benchmarklocal/MoltenVK/S24 pendant le profilageAR39Desktop de l’autre
  chat ; CIlogicielle ne démontre aucun gain physiqueAndroid. Décisionintégration
  conditionnée à correction et absence de régressionlogicielle ; gainS24 reste
  nonmesuré. Mémoiretable allouée annoncée séparément du picprocessus.

#### Révision candidats du 6 novembre — vérificationsCI à lancer

- UIed5152b/run42126372473 : nouveauchamp+libérationfocus seul **échoue**
  encore au second cycleIME, gap190px (`build/ci-ed5-ui.log`), effet retiré.
  Essai suivant : champstateful et ajustementfenêtre ADJUST_NOTHING Android11+,
  ADJUST_RESIZE conservé Android9/10 ; configuration partagée production/tests
  pour éviter qu’un testvalide une fenêtre différente. Les essais précédents
  du mode seul sur ancienchamp n’étaient pas suffisants ; combinaison nonencore
  validée. Assertiondiagnostique conserve modefenêtre et tolérance12dp inchangée.
- PERF05 : table786432octets et shaderlookup intégrés **prototype local**,
  oracleprocedural antérieur conservé sousmacrotest et inclus seulement dans
  les binairesdetest. HarnaisKaniindex + testCPUtable + GPU196608valeurs et
  benchmarkalterné12cas prêts. Aucun code moteur compilé ni benchmark local
  lancé ; fmt/actionlint/diffcheck légers réussis. CIcorrectness AVANT mesure,
  rapportmicro séparé. Pas de déclaration de gain S24 à ce stade.

#### Résultat353e7e8 et révision de vérification

- Run42127003850 : Vulkan196608valeurs lookup/procédural/CPUbitexact,
  24QMVoracleCPU, dense, fixtureLFM3 et microbenchmark12cas **réussis** ;
  Kani6harnais **réussis**. APK **échoué** uniquement lintClippy dans le nouveau
  testtable(chunks_exact→as_chunks), corrigé sans changer le moteur/shader.
- UI **échouée** encore190px au secondcycle avecmode48 confirmé ; ADJUST_NOTHING
  inutile, retiré. Champstateful conservé pour le chemin de saisie moderne ;
  nouveaucorrectif au nœud de visibilité racine : propager le viewport déjà
  ajusté par lesinsets au lieu de laisser Androidpan toute la conversation.
  Scrolls internes texte/liste restent propriétaires du contenu. Pas d’offset
  fixe ni toléranceassouplie. TestcycleIME inchangé à vérifierenCI.
- PERF05correction : sortiesQMVbitexact sur12formes/codebooks avant chaque
  chronométrage et après chaque appel ; 2warmups/7pairesalternées. Données et
  dérives dans `docs/measurements/android-codebook-lookup-20261106/mesa-353e7e8.json`.
  Cases2048x7680 dériventfortement, donc **pas de gain global établi** ; autres
  caslogiciels descriptifs uniquement. Aucun résultatS24/macro1.7B revendiqué.
- RépétitionCIcorrectness justifiée par linttest et correctifUI ; kernel/table
  inchangés. Benchmarks désormais seulement sur workflow_dispatch explicite,
  aucune nouvellemesure automatique nécessaire pour un changementUI. Provenance
  rustc/glslc/SHAshaders etSPIR-V sera conservée lors d’un nouvelessaimanuel.
  ProfilageDesktop n’a subi aucune mesure/compilation moteur locale de notrepart.

- UI6abcc46/run42127962806 : nœudviewport **échoue** encore au secondcycleIME
  (190px,mode16), retiré sans l’intégrer dans main. Remplacé par synchronisation
  requestLayout de la View Android lors des changementsIME/brouillon, pour
  recalculer le rectanglefocus de ViewRoot après les relayouts internesCompose.
  Aucun ajustementpixel, mode standardadjustResize conservé. Testdiagnostique
  étendu au rectanglefocus ; CIUI sur320x640/mdpi ET1080x2400/dpi420 pour couvrir
  petit et grandécran. Vérifications lourdes toujours uniquementGitHub.

#### Vérification639a0e8 — correctionlayout validée, ancragecurseur à finir

- RejetsGitcommit_refs3fois et erreurGraphQL lors du refupdate, sans corruption
  locale(fsckréussi). Créationtree/commitAPI : arbre68faf8d173be5a0110564f119ba4b4f9b9d26599
  strictement égal au59e5fe9local ; updateRESTnonforcé vers639a0e8 réussi.
  Anciencommitlocal59e5fe9 conservé dans la branchelocale.
- Run42129073097 **5jobs réussis** : APK,Vulkan,6Kani,UI5casx2résolutions
  (320x640/mdpi et1080x2400/dpi420). Testouverture/3lignes/fermeture/réouverture
  passe sans changer tolérance12dp.13testsCPU et5GPUignorés par défaut dont
 4exécutés explicitement enCI ; benchmarkoptionnel déjà archivé, nonrépété.
- APK639 téléchargé : sha2561be48c46bcf1aec11c126a537fe31556beacef98241151ae2a01b5d9eb2685fc,
  conforme au manifeste, apksigner/zipalign16K réussis. Sourceprototype seulement,
  main encore3358834. Pas de remplacement de l’appS24.
- Inspectiondes2captures : zonebienau-dessus du clavier etheaderstable mais
  popupcurseurmalancré près du header aprèsréouverture. **Limite détectée malgré
  assertionspassées**, non déclaré correctifcomplet. Requête rectanglefocus
  immédiate après layout pour éviter l’animationnativepan restante ; nouveau
  contrôle du centre réel des popups dans lescreenrect du composer, à validerCI.
- ContrôleKani désormais cibléindex puis suitecomplète dans le workflow.
  Aucun programmeGPU/compilation/prover local lancé pendantprofilageDesktop.
- Utilisateur précise quantificationS24 **4,49bpw exactement** ; ne pas assimiler
  au benchmarksynthetique4bits. Tousbitratessont couvertsnumériquement mais
  aucun débit ni paritécomplète de cecheckpoint physique n’est mesuré.

- UIa66373e/run42130625343 : testsrenforcés **détectent** le popupmalancré sur
 2résolutions (petitécran centre139,98 vscomposer30,215..298,330), donc oracle
  géométrique crédible. doOnNextLayout+requestRectangleimmédiat ne résout pas
  l’ancrage, retiré. Moteur/format/kernels inchangés.
- Rechercheamont : source officielleCompose1.8 getFocusedRect renvoie le focus
  interne ; source1.12.1 renvoie la vuecomplète pour contenuCompose et réserve
  le rectanglelocal aux vuesembarquées, évitant le panViewRoot parasite. Maven
  Google publie BOM2026.09.00→UI/Foundation/Runtime1.12.1/Material3 1.4.0,
  Kotlinstdlib2.1.20 compatible compilateur2.2.10 existant. Comparaisonsources
  `build/AndroidComposeView{,-new}.kt` ; aucun codeamont copié dans l’app.
  MiseàjourBOMstable et suppression de TOUSles contournementslayout/focus/root,
  TextFieldState et testsréels conservés. À confronterenCI, pas encore validé.

- UI6631d86/run42131287988 : ComposeBOMnouveau **ne corrige pas** le pan,
  gap168px surpetitécran, rectanglefocus138,236..140,260 ; testéchoué. L’hypothèse
  sur getFocusedRectamont ne suffisaitpas (la fonctionembedded résout aussi du
  focusCompose). MiseàjourBOM retirée pour conserver le style/dépendancesvalidés.
- Nouveaucorrectiffrontière hôteAndroid API30+ : contentFrame applique unefois
  lebottommax(IME,navbars), sonComposeView a réellement la hauteurvisible ;
  Composeconsomme cesinsets déjàappliqués au lieu d’ajouter un secondpadding.
  Rectanglesdefocus/popup et layoutpartagent lemêmeviewport. Android9/10 garde
  l’ancienpaddingCompose/adjustResize pour l’émulationd’insetslegacy. Testshôtes
  utilisent exactement la configurationproduction, oraclesgéométriques inchangés.
  Hypothèse à valider avant intégrationmain ; aucun gain moteur supplémentaire.

#### Bilan final ee39235 — 6 novembre

- ANDROID-UI-04 **validé enCI**, codequalifiéee39235a7b39646e74a4bf54d775aefe4c3f419c,
  run https://github.com/0xZKnw/mlxl3-android/actions/runs/42132302512 :5jobsréussis.
  5casUIx2écrans, vraiIME/3lignes/fermeture/réouverture, gap<=12dp et popup
  effectivementdans le composer ; captures2résolutions inspectéesvisuellement.
  Source58b7454 intermédiaire nonmesurée annulée après ajoutdu modepartagé16,
  run42132173495 obsolète ; seulsrésultatsfinauxee39235 qualifient l’intégration.
- ANDROID-PERF05 **validé numériquement**, performanceglobale **non concluante**
  surMesaetS24 **non mesuré** ; table786432octets retenue commeoptimisation
  expérimentale du calculrépété.196608codesbitexact,24casCPUoracle,4testsGPU
  explicites etfixtureLFM3réussis ; aucune modification desopérateurs/caches.
  Le microbenchmark utiliseK4synthetique, pas lecheckpoint4,49bpwduS24 ;
  2048x7680 dériveforte, pas d’agrégatboost. Asymétriehostdocumentée (validation
  productionincluse lookup, référencecompute directe), pas de tempsGPUisolé.
- 13testsCPUréussis/5ignoréspar défaut (4GPUexplicitementexécutés, benchmark
  optionnel nonrépété),3JVM,10exécutionsUI ; fmt/Clippyhost+ARM64/lint/SPIR-V/
  buildreleaseARM64/APK réussis. Kaniindexciblé27obligations+3covers puis suite
 6harnais/449obligations+11covers,1assertionformatinatteignable. Ancienharnais
  greedyflottantresteICE/nonvérifié ; aucunetheorieGPU/JNI/Compose complète.
- APKee39235 téléchargé,33975708octets, checksumdedcc748615ae83122df524a30585e399feb7601dd712f297eefc533a03ac833
  conformeau manifeste, signaturedebug etalignement16Kvalidés ; aucune donnée
  modèleembarquée. Résultats/commandes/bornes dans `docs/verification-2026-11-06.md`.
- Décision : sourcepubliquedansbranchequalifiée, publicationmain autorisée,
  sans release ni appS24remplacée. ProfilageDesktop respecté : aucunecompilation
  moteur/preuve/modèle/mesurelocale lancée. Tousessaisdece lot sont clos dans
  les statutsci-dessus ; vitessephysique/fullcheckpoint/Android9-10 restent
  explicitement à qualifier. La publication regroupe ces sourcesetleursrapports.

#### Intégrationeffective main et reçuAPK

- Sources9ae3b94b0ec530ff7de700ceb17feb09de31a511 publiées surmain ; diffexécutable
  nul avec ee39235qualifié. Run https://github.com/0xZKnw/mlxl3-android/actions/runs/42133293698
  **5jobs réussis** inspectés. Publicationeffective **code+APKCIexpérimentaux**,
  pas de release ni appinstallée. Reçu documentaire ultérieur sans changement
  exécutable, donc pas de nouvelleCI exigée par le filtrepaths.
- APK33975708octets, sha25628b7de1a77102f7f343704f341d82d7e32331c937622b33d128a451fcedaded8,
  manifestes/signaturedebug/zipalign16K vérifiés après téléchargement. Natif
  b4c6dfc13345ddf2485bd43f9231a58fa7a09adc2337e448559dc234c97a1509.
  Artefact13463758703, détails dans `docs/verification-2026-11-06.md`.
- Aucun essai encore en cours : UIvalidée2résolutions, LUTnumériquementvalidée,
  performanceglobaleMesa nonconcluante, S24/4,49bpw physique nonmesuré. Aucun
  benchmarklocal durant le profilageDesktop ; ces limites ne sont pas transformées
  en validation de débit/qualitécomplètedu modèle.

### ANDROID-PERF-2026-11-06-06 — issue1, head groupé et débit visible — validé fonctionnellement, performance non mesurée

- Issue1 lue, aucun commentaire ; baseline544b0c3, moteur identique9ae3b94.
  Utilisateur demande explicitement implémentation/vérificationrapide/pushAPK,
  sans grossecampagnedetests, plus tok/s souschaqueréponse. S24/4,49bpw non
  connectéADB ; profilagephysiqueP0 impossible ici, pas de dominantcoût mesuré
  ni promesse de gain. Pas de nouveau benchmarkMac pendantprofilageDesktop.
- Hypothèse ciblée : head dense découpé attend chaqueprojectionindépendante ;
  conserver chaquechunk/réduction/arrondi mais enregistrer ses dispatches avec
  des descripteurs/sorties distincts, un uploadinput et une soumission/fence.
  Cas128000x2048/64Mio :8soumissions→1 calculées, débitS24 nonmesuré.
- Avantcode : régressionciblée bit-à-bit batch vs séquentiel F16/BF16 sur8chunks,
  compteur réel des soumissions, entréesinvalides et répétition aprèséchec.
  Vérifier bornes128chunks/128000sorties, tenter Kani ciblé sur le compteur de
  sorties, fmt/Clippy/buildARM64 et CIexistante. Pas de balayageperformance.
- Débit : mesurer séparément les étapesdecode réellement évaluées et leurtemps,
  exclureprefill ; valeurindéfinie si aucuneétapedecode. UI petitecaption sous
  assistantuniquement ; conserverstyle, envoyer/annuler et texte inchangés.

- Implémentationciblée : headbatch avec descripteurs/sortiesdistincts, inputcopié
  unefois et fencecommune ; cas1chunk conservecheminnormal. Compteursubmit réel
  ajouté, deltaprefill/decode dans done ; débitdecode calculé sur les forwards
  réellement évalués, pas sur le tempsprefill. Captionassistant après done.
- Vérificationrapide : fmt/actionlint/diffcheck et Clippystrictalltargets réussis
  (2,51s) ;12testsCPUlib passent (0,03s),2GPUignorés. TestGPU ciblélogicielMesa :
  initialassertdriver car128layouts passés au pipeline au lieu d’unlayout,
  corrigé en séparant allocation128sets/layoutunique ; essairelancé **réussi**
  (0,15s), F16/BF16bitexact sur8chunks,1submit, erreurs et reprise couvertes.
  Logs `build/issue1-{clippy,cpu,dense-batch,dense-batch-fixed}.log`.
- Aucun profilage/S24 ni gain de débit mesuré ; simple contrepoint fonctionnel
  sur lavapipe. Compteur8→1 observé dans le test, pas une accélération8x.
  Kani borné4chunksymboliques/unwind6 et contrôlesAndroid délégués à CI ; pas
  de grossecampagne ajoutée conformément à la demandeexplicite de rapidité.

#### Réception CI de l'essai 06

- Source publique `38f2d17331ba52420f788b2f3481e3fd9afc0733` ;
  [run 42211210916](https://github.com/0xZKnw/mlxl3-android/actions/runs/42211210916)
  inspecté : **cinq jobs réussis**, APK, Vulkan, preuves et deux tailles UI.
  15 tests CPU, cinq contrôles Vulkan, sept harnais Kani et dix tests UI passent.
  APK disponible dans l'artefact `13499091839`, sans installation S24 ni release.
  Code intégré/public, correction vérifiée dans ces domaines ; gain S24 **non
  mesuré**, performance réelle **non concluante**. Logs `build/issue1-ci38f.log`.

### ANDROID-PERF-2026-11-06-07 — recherche et projections indépendantes groupées — validé fonctionnellement, puis rejeté comme optimisation

- Reprise demandée : utilisateur rapporte2,7tok/s sur S24/4,49bpw et veut des
  pistesplusfortes, rechercheInternet autorisée. APKexact de la mesure non
  confirmé ; ne pas convertir cechiffre en comparaison contrôlée.
- Historique relu : LUT05 numériquevalidée/perfglobaleinconcluante ; head06
  groupé8→1 testé mais pasdegainS24mesuré. CI38f2d17/run42211210916 inspectée :
 5jobsréussis, état06 codepublic/APKCI ; réceptiondocumentaire à compléter.
- Rechercheprévue : guides officielsVulkan/ARM et moteursVulkanouverts pour
  réduireattentes, garderactivationssurGPU, kernelsQMV et prefill. Pas de
  gainchiffré extrapolé des autresmoteurs. Pas de benchmarkMac concurrent.
- Pisteà vérifierdanslecode : W1/W3 et Q/K/V/gate partagent une activationCPU
  et sont indépendants ; transformationsSU propresàchaquepoids à conserver.
  Préparer leursentréesCPU commeavant, descripteurs/inputs/outputs distincts,
  commandesgroupées/fencecommune. Réduire soumissionssans changer shader,
  réduction, arrondi, ordrecausal des caches ni contexte. ÉtapesGPUdépendantes
  restent synchronisées, ne pas simplement supprimerlesfences.
- Protocole AVANT prototype : bitexact chaqueprojectiongroupée vs séquentielle,
  casbits/codebooks/densemixtes, réutilisation/erreurs ; compterlesqueuesubmit
  réelles. VérificationfocusedClippy/tests, fixtureEngine et Kani bornesbatch
  déjàpratiques enCI. Pas de campagnephysique sans appareil ; garder les
  écarts entre compteursdérivés, microtests et gainS24 mesuré explicites.

#### Résultats de l'essai 07 avant push

- Implémentation : FFN W1/W3 et attention Q/K/V/gate groupés, inputs transformés
  indépendants, kernels et arrondis conservés. Baseline source `38f2d17`.
  Décode théorique 154→97 soumissions sur 13 conv/11 attention ; fixture réelle
  trois couches 22→13. Pas de conversion de ces compteurs en gain de débit.
- Vérifications ciblées terminées : fmt, Clippy strict, actionlint, diff-check,
  **15 tests CPU** ; sept GPU/benchmark ignorés par défaut, puis **six contrôles
  Vulkan fonctionnels** exécutés explicitement avec succès sur Mesa logiciel.
  Parité bit à bit des groupes (trois codebooks, 3/4/5 bits, F16/BF16, padding,
  SU/SV différentes), erreurs sans soumission et reprise ; mutation entrée0
  détectée puis restaurée. Fixture Engine : 128 logits/8 forwards bit à bit
  contre les sources `38f2d17` ; oracle permanent et compteurs bridge vérifiés.
- Kani bornes réelles des groupes : domaine 0..4 éléments/usize symboliques,
  unwind6, **131 obligations/0 échec/2 covers**, une obligation interne format
  inatteignable. Aucune preuve du pilote ou du calcul GPU. Commandes, diagnostic
  du premier parseur de logs, sources Internet et limites dans
  [le rapport](docs/android-speed-2026-11-06.md). Logs temporaires disponibles
  `build/independent-*`, oracle permanent `engine/tests/fixtures/lfm3-reduced-logits.json`.
- État : correction **validée localement**, code local prêt à publier sous
  autorisation existante ; gain S24 **non mesuré/non concluant**. Aucun essai
  physique ni campagne GPU concurrente. CI exacte/APK à inspecter après push.

#### Publication et réception de l'essai 07

- Source `d7116bdc584e29fc8a2c0bafb90f5e4a4d2aa6ec` poussée sur main ;
  [run 42239998740](https://github.com/0xZKnw/mlxl3-android/actions/runs/42239998740)
  terminé **avec cinq jobs réussis** : APK, Vulkan, preuves et UI deux tailles.
  15 tests CPU passent ; six tests fonctionnels Vulkan passent explicitement ;
  cinq tests UI passent sur chacune des deux tailles d'émulateur (dix au total).
  Benchmark facultatif volontairement sauté, aucun chiffre de vitesse ajouté.
  Sept harnais Kani passent : 581 obligations, 13 covers, deux obligations
  internes format inatteignables. Le harnais ciblé131 est exécuté séparément
  puis dans cette suite ; ne pas le compter comme une huitième preuve.
- Job APK **2min43**, conditions de cache/réseau propres à ce run ; aucune
  accélération de CI reproductible inférée. Artefact
  [13513837166](https://github.com/0xZKnw/mlxl3-android/actions/runs/42239998740/artifacts/13513837166),
  fichier `debug/app-debug.apk`, **34 012 388 octets**, SHA-256
  `dd8d5e23741470eece78896e66814012f1768d2723e10730c5ca4d51f75e5df4`
  conforme au manifeste CI. Signature APK et alignement16K vérifiés ; moteur
  ARM64 embarqué SHA-256
  `a1b7207bb570ae08d04fbabd3dfd276036ce1186781b59ed13bb77a036260235`.
  Premier contrôle d'archive trop strict rejetant aussi la bibliothèque
  AndroidX de Compose ; corrigé vers la présence du moteur et l'ABI ARM64 de
  toutes les bibliothèques. Contrôles relancés avec succès, aucun défaut APK.
- État final : **code public/APK CI**, vérification fonctionnelle validée dans
  les domaines ci-dessus ; gain de débit **non mesuré/non concluant** sur S24.
  Pas de release, installation téléphone ni essai encore en cours. Sources et
  priorités suivantes dans [le rapport](docs/android-speed-2026-11-06.md).
  Logs temporaires `build/independent-ci-{apk,proofs,vulkan}.log`, APK local
  `build/apk-d7116bd/debug/app-debug.apk` ; oracle permanent conservé.

#### Révision utilisateur de l'essai 07 — résultat négatif

- Retour : **1,9 tok/s** avec le nouvel APK `d7116bd`, même prompt et même
  checkpoint4,49bpw que les **2,7 tok/s** précédents. Thermique, contexte exact,
  alimentation et répétitions non relevés ; comparaison utilisateur négative,
  causalité exclusive du regroupement non démontrée. La réduction des attentes
  n'a pas validé une accélération. Décision : **rejeté comme optimisation par
  défaut**, revenir aux projections indépendantes séquentielles de `38f2d17`.
  Les anciennes preuves fonctionnelles restent valides dans leur domaine.

### ANDROID-PERF-2026-11-06-08 — EXL3 spécialisé/tuilé et sélection sur l'appareil — validé fonctionnellement en CI, gain S24 non mesuré

- Baseline source `7939437`/moteur `d7116bd` ; chemin séquentiel qualifié
  `38f2d17` à restaurer (head groupé conservé). Historique05..07 et rapports
  relus. Pas de répétition LUT/procédural ni de campagne GPU Mac concurrente.
- Hypothèse : QMV une sortie/groupe disperse les lectures et répète le calcul
  des adresses ; bits dynamiques et une réduction par sortie coûtent cher.
  Prototype prévu : quatre sorties voisines/groupe128, accumulateurs vec4,
  même ordre FMA/réduction128 pour chaque sortie, adresses invariantes hors
  boucle, bits spécialisés1..8. Aucune déquantification dense globale, aucune
  modification du fichier modèle, des arrondis, du contexte ou des caches.
- Éviter de choisir un nouveau kernel seulement depuis Mesa : comparer ancien
  scalaire et nouveau kernel sur les vrais poids, au chargement, une fois par
  forme/bits/codebook. Même validation/upload/readback pour les deux ; warmup,
  ordre alterné, quatre paires. Parité bit à bit avant sélection ; choisir le
  nouveau seulement avec marge et échantillons stables, sinon ancien kernel.
  Chronométrage host synchrone, pas timestamp GPU isolé ni débit modèle. Ne pas
  inclure cette qualification dans le débit decode ; coût de chargement séparé.
- Vérification prévue : CPU reconstruction indépendante, tous bits/codebooks,
  formes non carrées et réutilisation/erreurs ; baseline scalaire conservée
  comme oracle, parité non vide/finie. Fixture Engine avec oracle128logits,
  retour aux compteurs22/forward. Tests déterministes de la décision de tuning
  (bruit, absence de gain, parité invalide), Kani bornesgroupes ciblé ; fmt,
  Clippy, SPIR-V, CI APK/CPU/Vulkan/Kani/UI. Pas de benchmark physique local.
- Recherche : guides AMD mémoire/occupation et kernels amontEXL3/llama.cpp ;
  idées uniquement, aucun chiffre de leur matériel extrapolé au S24. Statut
  **préenregistré**, prototype/mesures non démarrés, APK non publié.

#### Résultats locaux de l'essai 08

- Chemin séquentiel `38f2d17` restauré, regroupement07 supprimé du chemin et
  buffers/API devenus inutiles retirés. Nouveau QMV quatre sorties/groupe avec
  bits spécialisés, sélection scalaire/tuilé sur poids réels au chargement.
  Médianes quatre paires alternées après warmup, parité bit à bit obligatoire,
  marge5% et max/min<=1,5 sinon ancien kernel. Coût de chargement supplémentaire
  assumé et non mesuré sur le téléphone ; aucun débit decode prédit.
- **16 tests CPU**, six contrôles fonctionnels Vulkan explicites réussis sur
  Mesa logiciel ; 72 comparaisons tuilé/scalaire bit à bit (8bits×3codebooks×
  3formes128×128/256×384/2048×512), oracle reconstructionCPU indépendant existant,
  196608codes, headdense et fixtureEngine128logits. Compteurs22/forward et
  champs de diagnostic de sélection vérifiés. Benchmark historique05 non lancé.
- Mutation stride erroné détectée (256×384/K1/cb0), shader restauré et contrôles
  affectés relancés avec succès. InitialwarningClippy range-loop corrigé ;
  fmt/Clippystrict/actionlint/diffcheck/SPIR-V réussis. Kanihelper réel :
  105 obligations/0échec/2covers, usize arbitraire, couvertureexacte des sorties
  multiples16 de16à32768 ; pas de preuve de shader/driver ni des tempsflottants.
- Commandes, décisions de repli et sources dans
  [le rapport](docs/android-kernels-2026-11-06.md). Logs temporaires
  `build/perf08-*`, tests/oracle permanents conservés. État : code local
  fonctionnellement validé, **gain S24 non mesuré/non concluant**, CI/APK à
  vérifier après publication autorisée. Aucun benchmark matériel local.
- Raffinement avant mesure physique : exiger aussi un gain>5% dans au moins
  trois des quatre paires, pour ne pas retenir une simple médiane favorable
  avec deux paires perdantes. Test déterministe de ce cas ajouté ; contrôles
  affectés relancés, protocole de sélection documenté avant déploiement.

#### Publication et réception de l'essai 08

- Source publique `85e9db1559b8d6e3d50e68539dc548fe21810300` ;
  [run 42246844081](https://github.com/0xZKnw/mlxl3-android/actions/runs/42246844081)
  terminé **cinq jobs réussis**. 16 tests CPU, six contrôles Vulkan (dont les
  72 cas tuilés bit à bit) et dix tests UI réussis ; benchmark historique
  facultatif sauté. Huit harnais Kani réussis : 686 obligations, 15 covers,
  deux obligations internesformat inatteignables ; le ciblé105 est aussi
  exécuté séparément et n'est pas compté comme une neuvième preuve.
- JobAPK **3min14**, caches/réseau propres au run, pas de gain CI extrapolé.
  [Artefact13518731858](https://github.com/0xZKnw/mlxl3-android/actions/runs/42246844081/artifacts/13518731858),
  fichier `debug/app-debug.apk`, **34 014 540 octets**, SHA-256
  `8151ba6655891263fde026db1d80677f972fc3654e83ba23806bc68874548768`
  conforme au manifeste CI ; digestZIP également vérifié. Signaturedebug,
  zipalign16K et packaging ARM64 validés après téléchargement. Moteur embarqué
  SHA-256 `8af305dc263d2c9fbaf5b7177bc7d31301d52c61996ab7b97d6df7b3d53bc81a`.
- État : **code public/APK CI**, aucune installation ni release. Validation
  fonctionnelle dans les domaines précités ; **débit et surcoût de chargement
  S24 non mesurés**, sélection sur téléphone à observer. Les timings internes
  du test logiciel de sélection n'ont pas été conservés comme benchmark ;
  aucun gain n'en est déduit. Aucun essai lancé encore en cours.
  Logs temporaires `build/perf08-ci-{apk,proofs,vulkan}.log`, APK local
  `build/apk-85e9db1/debug/app-debug.apk`, tests/oracle permanents conservés.

#### Révision utilisateur de l'essai 08 — nouvelle régression

- APK85e9db1 : **1,7tok/s**, même prompt, après2,7 puis1,9tok/s. Le sélecteur
  de microkernels au chargement n'a donc pas validé le débit de l'application.
  Résultat utilisateur négatif conservé ; attribution thermique, sélection,
  synchronisations et mémoire à établir, aucune cause exclusive affirmée.
  Décision : **rejeté comme accélération livrée**, pas de gain revendiqué.

### ANDROID-PERF-2026-11-06-09 — diagnostic de bout en bout — recherche terminée, attribution S24 non vérifiée

- Reprise : utilisateur rapporte1,7tok/s surS24/EXL3 4,49bpw, compare à34tok/s
  sous llama.cpp/Termux surS25Ultra. Modèle/GGUF/backend de la comparaison à
  demander, pas de cible20tok/s présumée acquise. Baseline85e9db1/18bd2b2.
- Lire08/07/06/05 et rapports, puis vrais kernels/graphes llama.cpp et EXLlama,
  documentation mémoire/Vulkan/Android. Distinguer constat de code, hypothèse,
  test logiciel et attribution physique. Pas de nouveau kernel choisi à l'aveugle.
- Pistes à auditer : dispersion/réutilisation des poids EXL3 et LUT, opération
  GPU trop petite + fence par projection, transformations/attention CPU,
  choix de mémoire/cohérence/map/unmap, headBF16, duplication/mmaps et pression
  mémoire, préfill token-par-token, biais/échauffement de la calibration.
- Si instrumentation nécessaire : mesurer le parcours de production, garder
  compteurs/latenceshost CPU/préparation/projection/head/readback distincts ;
  timestamps GPU seulement avec capacité physique vérifiée. Diagnostics
  copiables depuis l'app (pas seulement JSON inaccessible) avec build, GPU,
  paramètres, sélection et état mémoire/thermique disponibles. Aucun gain
  annoncé avant mesures S24. Pas de benchmark matériel Mac concurrent.
- Vérifications ciblées du diagnostic et invariants/transitions touchés,
  entréesréelles/fixtures nonvides, sortie sans informationspersonnelles ou
  prompt ; source-level Rust/Kani pertinent, checks CI/APK exact. Recherche
  et instrumentation ne constituent pas une validation de20tok/s.

### ANDROID-PERF-2026-11-06-10 — lot coûts CPU/driver et diagnostic — validé fonctionnellement en CI, gain S24 non mesuré

- Demande explicite : plusieurs défauts dans un lot, vérification rapide puis
  pushAPK. Baseline18bd2b2/85e9db1, retour1,7tok/s. Essai09 recherche en cours,
  pas de profilage physique possible (aucun appareilADB connecté).
- Changements prévus : mapping persistant des seuls buffers de travail pour
  retirer map/unmap par projection ; préférence DEVICE_LOCAL parmi les types
  HOST_VISIBLE|HOST_COHERENT compatibles ; SU/SV arrondies FP16 au chargement
  plutôt qu'à chaque token ; anglesRoPE calculés une fois par dimension et
  position au lieu de chaque tête. Retirer la calibration automatique et son
  choix expérimental du chemin livré, conserver le kernel scalaire de référence.
- Diagnostic : tempshost upload/commandes/attente/readback, tempshead et modèle,
  prefill/decode distincts, build/GPU/typesmémoire, étatmémoire/thermiqueAndroid,
  copiable depuis le menu sans texte de conversation ni chemin de modèle.
- Protocole : parité fixture/logits et projectionsGPU existantes, réutilisation
  des buffers persistants et erreurs de bornes ; tests CPU typesmémoire/RoPE/
  échelles, Kani sur sélection/bornes praticables ; fmt/Clippy/SPIR-V/CIAPK/UI.
  Compteurs d'appels supprimés ne seront pas présentés comme un gain de débit.
  Aucun nouveau kernel ni benchmark matériel local ; gainS24 non mesuré.
- Extension préenregistrée avant modification : les codebooks sont déjà des
  valeursFP16 exactes mais la LUT les stocke enFP32. Les empaqueter deux par
  uint32 pour réduire l'empreinte786432→393216octets, conversionexacte à la
  lecture (pas de nouvelle quantification). Vérifier les196608valeurs bit à
  bit contre le shaderprocédural indépendant, puis projections/fixture.
  Réduction d'empreinte seule, aucun gain de débit ni bande passante présumé.

#### Résultats locaux et décision avant publication du lot09/10

- Recherche09 **terminée pour les constats de code** : sourceVulkan llama.cpp
  6c01da6278ab6c69ed67d9124806b50d2b299ba7 lue (graphes/fusion/soumissions),
  GEMVEXLlama et guidesKhronos consultés. Hypothèses physiques non attribuées :
  S24 non connecté. Comparaison34tok/sLFM3/Q4_K_M probable, backend inconnu ;
  Termux n'implique pasGPU. Pas de benchmark relancé ni de chiffre inventé.
- Lot10 implémenté : buffers persistants, mémoirelocale compatible privilégiée,
  LUT2×pluscompacte sans changer lescodes, SU/SV précalculées, anglesRoPEpartagés,
  calibration automatique retirée ; diagnosticcopiable sans prompt/réponse/paths.
  Compteurs host parphase ethead imbriqué, mémoire/PSS/thermiqueAndroid séparés.
- **19 testsCPU**, sept contrôles fonctionnelsVulkan réussis explicitement sur
  Mesalogiciel :196608codesbitexact,72casQMV, buffers/réutilisation/erreurs,
  headdense, fixtureEngine/oracle128logits et diagnostics. Format/Clippystrict,
  actionlint/diffcheck, compilationKotlin/testKotlin et lintAndroid réussis.
- Kani mémoire :190obligations/2covers (4flagsu32symboliques, masqueu32arbitraire,
  unwind7,1obligationformatinatteignable) ; indicesLUT :27obligations/3covers,
  u32/u16complets. Premier comparateurtuple échoue enKani mais ses entrées
  concrètes passentnativement ; remplacé parboucleexplicite puis mêmeharnais
  inchangé réussi. Échecs/contre-exemples conservés `build/perf10-kani*`.
  Mutationretourpremiercompatible détectée par test, source restaurée et retestée.
- Détails/commandes/limites : [rapport](docs/android-perf-batch-2026-11-06.md).
  Logs temporaires `build/perf10-*` présents. État : **validé localement dans
  ces domaines**, codeprêtàpublier sousautorisationexistante, CI/APK à inspecter.
  **Aucun gain S24 mesuré**, causephysique de1,7tok/s nonconfirmée ; pas de
  preuve complète GPU/FFI/Android. Pas de benchmark matériel concurrent.

#### Publication et réception du lot10

- Source `4c07189b1676b6407a0db06f6038166b1b8f7985`,
  [run42251895779](https://github.com/0xZKnw/mlxl3-android/actions/runs/42251895779) :
  **cinq jobs réussis** inspectés. 19testsCPU, sept contrôlesVulkan explicites,
  douze testsUI (sixparécran, dont copieclipboard) ; neuf harnaisKani réussis,
  876obligations/17covers, trois obligationsformatinternes inatteignables.
  Le ciblé190 est aussi exécuté seul, pas une dixième preuve. Benchmark
  historique facultatif sauté ; aucune nouvelle vitesse de modèle mesurée.
- [Artefact13520937307](https://github.com/0xZKnw/mlxl3-android/actions/runs/42251895779/artifacts/13520937307),
  `debug/app-debug.apk`, **33 970 884 octets**, SHA-256
  `7ea37b96552d97cd221339af182944a40ecf76c5e8ed41d02c6e8ea258eb07d9`.
  ZIP et APK conformes auxdigests/manifeste, signaturedebug, zipalign16K et
  bibliothèquesARM64 vérifiés. SHA source du diagnostic retrouvé dans le moteur
  embarqué ; natif SHA-256
  `f39633fa909eae1e94373e61a97d0550441df3a86f988d024a56fa1d33cccda8`.
- État final : **code public/APK CI**, pas de release ni installation téléphone.
  Lot validé fonctionnellement dans ces domaines ; débit réel, mémoire GPU
  physique, impact thermique et causalitéS24 restent nonmesurés/nonconcluants.
  Recherche09 terminée sursources, diagnostic disponible ; aucun essai lancé
  encore en cours. Logs temporaires `build/perf10-ci-{apk,proofs,vulkan}.log`,
  APK `build/apk-4c07189/debug/app-debug.apk`, tests et rapports conservés.

#### Retour utilisateur sur le lot10

- APK4c07189 : **2,1tok/s** rapportés après1,7 ; pas de protocole thermique,
  répétitions ou diagnostic joint. Amélioration utilisateur partielle, mais
  débit toujours insuffisant ; aucun gain causal ou objectif20tok/s validé.

### ANDROID-PERF-2026-11-06-11 — recherche EXL3 portable et refonte du calcul — préenregistré

- Demande : chercher des essais réels EXL3 surVulkan/Android et reprendre le
  calcul. Baseline75e87bd/code4c07189. Historique05..10 et rapports relus ; ne
  pas refaire tuilage4/sélection ou LUT déjàdocumentés comme nouvelle découverte.
- Recherche prévue : dépôts/PR/issues amontEXLlama, moteurs portables EXL3,
  Vulkan/Metal/WebGPU/Android, papiers/rapports avec hardware et protocole.
  Identifier benchmarks véritables, prototypes et simples discussions ; ne pas
  attribuer des débitsGGUF àEXL3. Demander le diagnostic S24 en parallèle.
- Hypothèse à confronter : accès non coopératifs aux poids, deux lectures par
  poids logique, LUT dispersée et réduction128 ; calculHadamard/échellesCPU et
  154attentesGPU/token empêchent une implémentation orientéeGPU. Rechercher un
  algorithme qui décode des tuiles coopérativement et conserve les activations.
- Avant chaque prototype retenu : préciser configuration, ordre numérique et
  oracles, allocations/bornes, contexte/caches, provenance sources ; vérifier
  correction avant mesure, aucune campagneGPU Mac concurrente. Comparer le
  chemin production et référence indépendante, pas seulement deux clones.
  Tests ciblés/Kani applicables et CI exacte si code publié ; gain physique
  S24 inconnu tant que non mesuré. Statut : recherche en cours, pas de prototype.

#### Diagnostic physique reçu et recherche11

- Données utilisateur conservées dans `docs/measurements/s24-4c07189-user.json` :
  13étapesdecode/2002soumissions, 2,197566tok/s ; attente5,027s surmodèle5,899s
  (85,21%), CPUprojections0,424s, enregistrement/soumission0,210s, head0,257s
  imbriqué (4,35%). Mémoirelocale flags7, vraiXclipse940, pasd'économieénergie,
  statutthermique1 et low_memoryfalse. Pas de timestampGPUisolé : attente ne
  distingue pas exécution/pilote/réveilCPU. 154attentes/token confirmé.
- Recherche réelle : Strix-llama.cpp PR204/headbd64c24736eab68a006c1f44f99da3b994ba098c,
  portEXL3 Vulkan et benchmarksStrixHalo publiés (pas téléphone), shaders lus,
  licenceMIT ; PR201formatparent. vllm.cpp backendVulkanEXL3 est une référence
  validée surlavapipe sanschiffreGPUmatériel. infr issue6 fermée mais commentaires
  restent des plans, pas un benchmarkEXL3achevé. trellis-serve CUDA/XPU est
  horsVulkan/Android. Aucun benchmark EXL3surS24/Android public vérifiable trouvé
  danscesrecherches ; ce n'est pas une preuve d'absence.

### ANDROID-PERF-2026-11-06-12 — projection EXL3 GPU complète, fenêtres trellis et commandes préparées — préenregistré

- Refonte motivéeparle diagnostic, baseline4c07189/75e87bd. Le tuilage4 de08
  conservait2lecturesglobales/poids ; cettefois un thread décode8codes adjacents
  depuis une petitefenêtre deregistres, 16sorties/groupe128 et réductionquad/
  quatre partitionsK. Aucun requiswave32/FP16/coopmat, repliportable parmémoire
  partagée. Pas de matrice dense globale. Ordre FP32différent, pas promisbitexact.
- SU/Hadamard et Hadamard/SV passentdansdes kernelsGPU, mêmesfrontièresFP16 et
  Hadamard128, paddinglogiqueexplicite. Troisdispatchesavecbarrières dansune
  seulesoumission ; descripteurs et commandespréparés au chargement parmatrice,
  buffersrésidents. Gardercodec/scalar/HadamardCPUcommeoraclesindépendants.
- Ajouter timestampsGPU seulementsi file/périodeannoncéescompatibles, masque
  desbitsvalides et erreursnonfatales de lecture ; conserverlatenceshost et
  différencier tempsGPUmesuré des attentes. Aucuneautocalibration ni microbench
  destinéàchoisirle nouveaukernel depuisMesa.
- Protocole avantcode : forme/quantification/offsets validés, tests1..8bits et
  3codebooks contre reconstructionCPU/f64 (tolérance QMV1e-4+2e-5relative déjà
  utilisée), grandesformesrectangulaires/petitesfrontières et valeursfinies ;
  projectionsGPUcomplètes vsanciencheminCPU/scalaire avec toléranceFP16 explicite
  initiale2e-4+2e-3relative, oraclemodele128logits inchangé, caches/annulation.
  Tester barrières/réutilisation/erreurs et limitesmatrices/dispatches ; Kani
  contratsRust, pas prétendreprouverGLSL avecune réécritureRust. Vérifier CIexacte
  etAPK avant livraison. Pas de mesureGPUmatérielleMac, gainS24àmesurer ensuite.

#### Vérification12 en cours (avant publication)

- Nouveauchemin chargé et exercé : registre16, HadamardGPU, commandespréparées,
  timestampsoptionnels. Première parité96casQMV vsCPU/f64 réussie surMesa,
  projectioncomplète puis casétendus72matrices×réutilisation/zeros/overflow
  réussis ; fixture128logits/caches/annulation réussie sansmodifiertolérance.
- TestsCPU19/19 etClippystrict réussis ; Kanishapes83obligations/3covers réussi,
  suitecomplète lancée. SPIR-Vvalide ; Hadamard conserve OpFDiv/f32sqrt128 et
  NoContraction. Aucun résultat de débitS24 : toutceci est fonctionnel/logiciel.
- Avantcontrôlede sensibilité : mutationtemporaire du signe de la sortieQMV,
  lancer uniquement l'oracle stream_qmv (doitéchouer), restaurer même surerreur,
  reconstruire et relancerle contrôleaffecté. Pas de mesure de vitesse.

#### Résultats12 et décision locale

- **Validé fonctionnellement localement** :19testsCPU et8testsVulkan exécutés
  explicitement surMesa logiciel. QMV96cas, projectioncomplète72matrices et
  fixture128logits/caches/annulation ; tolérancesprévues inchangées. Mutation
  inversiondesigne détectée dès16×32/K1/codebookDefault (exit101), shader
  restauré enfinally, reconstruction et8contrôlesVulkan finaux tousréussis.
- Kani0.68/CBMC6.11 :11harnais,1005obligations,23covers réussis ; trois
  obligationsformatstd inatteignables. Domaines/bornes etlimites dansle
  [rapport détaillé](docs/android-exl3-rework-2026-11-06.md), dontlesdeux
  nouveauxcontrats timestamp/shape (46+83obligations/6covers). Pas de preuve
  GLSL/Vulkan/FFI ; harnaisflottantancien incompatibletoujoursnonvérifié.
- Clippystrict host, fmt, actionlint, diffcheck, compilation/validationSPIR-V
  réussis. Logs temporaires `build/perf12-{cpu,gpu-final,clippy-final,kani-all,
  kani-shapes,mutation,restored-build}.log` conservés ; pas de benchmarkmatériel.
- Recherche11 **terminée**, rapport etsources archivés. Essai12 : code local
  retenu, publication/CI/APK en attente sousautorisationexistante ; **gainS24
  nonmesuré/nonconcluant**, vrai checkpoint4,49bpw/Xclipse nonvalidés ici.
 154soumissions/tokenencoreprésentes ; pas de fusionglobale ni prefillmatriciel.
- Contrôle finalstaged a détecté une ligneblancheEOF dansle nouveauheaderGLSL
  (non suivi lorsdu premierdiffcheck), sansimpactnumérique ; supprimée dans
  lecommit suivant. ClippyARM64 réussit aussi ; compilation/validationSPIR-V
  relancées aprèscettecorrectiondeformat. Premiercommitcec814c supersédé.

#### Publication et clôture11/12

- Source `bd9671479d01f3e251c10d106a08856c4191ea6e`,
  [run42259603754](https://github.com/0xZKnw/mlxl3-android/actions/runs/42259603754) :
  **cinqjobs réussis** inspectés,19CPU/8Vulkan/12UI,11Kani/1005obligations/
  23covers (troisformatstdinatteignables). TestsJVM inchangésdepuiscache,
  Clippyhost/ARM64, fmt, SPIR-V, lint, builds et signatureAPKvalidés. Ancien
  runcec814c42259538294 annulé comme supersédé, pascompté commevalidé.
- [Artefact13525355179](https://github.com/0xZKnw/mlxl3-android/actions/runs/42259603754/artifacts/13525355179) :
  `debug/app-debug.apk`,33 965 868octets, SHA-256
  `eb42fe3caf056f73cd6aa4ffe168e12196b09bbe1fe65cbdfe9fdcadf211b56e`.
  DigestZIP/manifesteAPK, signaturev2, zipalign16K, deuxlibsARM64 etLOAD
  moteur0x4000 vérifiés. MoteurSHA-256
  `23cdb597886becdabae4fcb3030cb1d807effbc6629cb18f9422f3ae76d42fec`,
  SHA source/marqueursnouveauchemin retrouvés dansle binaire. Assertionlocale
  initialelibunique tropstricte corrigée pourAndroidX, pasunbugAPK.
- [Rapport final](docs/android-exl3-rework-2026-11-06.md), donnéesutilisateur et
  sourcesconservées ; logsCI `build/perf12-ci-*`, APK `build/apk-bd96714/`.
  État **public/APKCI**, pas de release/installation. Recherche11 terminée ;
  essai12 **validé fonctionnellement** danslesdomainesdécrits, **nonconcluant
  pour la vitesse** faute de mesureS24. Aucun essai lancé encore encours.
  Fusiongraphe/prefillmatriciel, vrai checkpoint etnumériqueXclipse restent
  nonvérifiés/nonimplémentés ; aucun chiffre20tok/s ni gainpromis.
