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
