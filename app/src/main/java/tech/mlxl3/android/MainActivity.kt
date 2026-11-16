package tech.mlxl3.android

import android.os.Bundle
import android.os.Build
import android.content.ClipData
import android.content.ClipboardManager
import android.widget.Toast
import android.view.View
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.input.TextFieldLineLimits
import androidx.compose.foundation.text.input.rememberTextFieldState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding
import kotlin.random.Random

private val hostFitsBottomInsets get() = Build.VERSION.SDK_INT >= 30

@Suppress("DEPRECATION") // Match the manifest resize mode, including Android 9/10.
internal fun ComponentActivity.configureStudioWindow() {
    window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
    enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
        navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
    if (hostFitsBottomInsets) {
        val content = findViewById<View>(android.R.id.content)
        ViewCompat.setOnApplyWindowInsetsListener(content) { view, insets ->
            // Give Compose and Android focus/popups the same visible viewport.
            view.updatePadding(bottom = insets.getInsets(WindowInsetsCompat.Type.ime() or WindowInsetsCompat.Type.navigationBars()).bottom)
            insets
        }
        ViewCompat.requestApplyInsets(content)
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        configureStudioWindow()
        setContent {
            val model: StudioViewModel = viewModel()
            val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
                if (uri != null) model.importModel(uri)
            }
            StudioScreen(model.messages, model.busy, model.ready, model.status, model.device, model.error,
                model.savedModel, { picker.launch(null) }, model::loadSaved, model::newChat,
                model::send, model::stop, model::dismissError, model.diagnostics, model.splitK, model::chooseSplitK,
                model.fastHead, model::chooseFastHead, model.gpuConvLayer, model::chooseGpuConvLayer,
                model.cooperativeK5, model::chooseK5, model.specializedMcg, model::chooseMcg,
                model.gpuGreedy, model::chooseGpuGreedy, model.gpuAttentionLayer, model::chooseGpuAttentionLayer,
                model.ffnK4Split, model.ffnK5Split, model::cycleFfnSplit, model.layerBatch, model::cycleLayerBatch,
                model.unroll2, model::chooseUnroll2, model::benchmark, model.subgroupHead, model::chooseSubgroupHead,
                model.headRows, model::cycleHeadRows, model.parallelNorm, model::chooseParallelNorm, model.gpuTimestamps, model::chooseGpuTimestamps, model.waveHead, model::chooseWaveHead, model.subgroupNorm, model::chooseSubgroupNorm, model.exl3Multiwave, model::chooseExl3Multiwave, model.fusedNorm, model::chooseFusedNorm, model.vectorHead, model::chooseVectorHead, model.shuffleHadamard, model::chooseShuffleHadamard,
                model.cooperativeK6, model::chooseCooperativeK6, model.exl3Packed4, model::chooseExl3Packed4)
        }
    }
}

// Desktop v2.0.5 design tokens (see StudioDesign.kt).
private val Ink = Studio.Ink
private val Quiet = Studio.Quiet

@Composable
fun StudioScreen(messages: List<ChatMessage>, busy: Boolean, ready: Boolean, status: String,
                 device: String, error: String?, savedModel: Boolean, onImport: () -> Unit,
                 onLoad: () -> Unit, onNew: () -> Unit, onSend: (String) -> Unit,
                 onStop: () -> Unit, onDismissError: () -> Unit, diagnostics: String? = null,
                 splitK: Int = 4, onSplitK: (Int) -> Unit = {},
                 fastHead: Boolean = false, onFastHead: (Boolean) -> Unit = {},
                 gpuConvLayer: Boolean = false, onGpuConvLayer: (Boolean) -> Unit = {},
                 cooperativeK5: Boolean = false, onK5: (Boolean) -> Unit = {},
                 specializedMcg: Boolean = false, onMcg: (Boolean) -> Unit = {},
                 gpuGreedy: Boolean = false, onGpuGreedy: (Boolean) -> Unit = {},
                 gpuAttentionLayer: Boolean = false, onGpuAttentionLayer: (Boolean) -> Unit = {},
                 ffnK4Split: Int = 0, ffnK5Split: Int = 0, onCycleFfnSplit: (Boolean) -> Unit = {},
                 layerBatch: Int = 1, onCycleLayerBatch: () -> Unit = {},
                 unroll2: Boolean = false, onUnroll2: (Boolean) -> Unit = {}, onBenchmark: () -> Unit = {},
                 subgroupHead: Boolean = false, onSubgroupHead: (Boolean) -> Unit = {},
                 headRows: Int = 4, onCycleHeadRows: () -> Unit = {},
                 parallelNorm: Boolean = false, onParallelNorm: (Boolean) -> Unit = {},
                 gpuTimestamps: Boolean = true, onGpuTimestamps: (Boolean) -> Unit = {},
                 waveHead: Boolean = false, onWaveHead: (Boolean) -> Unit = {},
                 subgroupNorm: Boolean = false, onSubgroupNorm: (Boolean) -> Unit = {},
                 exl3Multiwave: Boolean = false, onExl3Multiwave: (Boolean) -> Unit = {},
                 fusedNorm: Boolean = false, onFusedNorm: (Boolean) -> Unit = {},
                 vectorHead: Boolean = false, onVectorHead: (Boolean) -> Unit = {},
                 shuffleHadamard: Boolean = false, onShuffleHadamard: (Boolean) -> Unit = {},
                 cooperativeK6: Boolean = false, onCooperativeK6: (Boolean) -> Unit = {},
                 exl3Packed4: Boolean = false, onExl3Packed4: (Boolean) -> Unit = {}) {
    MaterialTheme(colorScheme = darkColorScheme(primary = Studio.Ink, onPrimary = Studio.Canvas,
        background = Studio.Canvas, surface = Studio.Menu, surfaceContainer = Studio.Menu,
        onSurface = Studio.Ink, onSurfaceVariant = Studio.Secondary, onBackground = Studio.Ink,
        outline = Studio.Edge)) {
        val draft = rememberTextFieldState()
        var menu by remember { mutableStateOf(false) }
        var advancedMenu by remember { mutableStateOf(false) }
        var focused by remember { mutableStateOf(false) }
        val context = LocalContext.current
        val list = rememberLazyListState()
        val imeBottom = WindowInsets.ime.getBottom(LocalDensity.current)
        LaunchedEffect(messages.size, messages.lastOrNull()?.text, imeBottom) {
            if (messages.isNotEmpty()) list.scrollToItem(messages.lastIndex)
        }
        val engine = when {
            error != null -> EngineState.Failed
            busy && ready -> EngineState.Generating
            busy -> EngineState.Loading
            ready -> EngineState.Ready
            else -> EngineState.Idle
        }
        // A new conversation draws a new greeting; it stays stable while it is open.
        val greetingSeed = remember(messages.isEmpty()) { Random.nextLong() }
        val mark = painterResource(R.drawable.mlxl3_mark)
        Box(Modifier.fillMaxSize().background(Studio.Canvas)) {
            StudioAurora()
            Column(Modifier.fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
                .then(if (hostFitsBottomInsets) Modifier.consumeWindowInsets(WindowInsets.ime.union(WindowInsets.navigationBars).only(WindowInsetsSides.Bottom))
                    else Modifier.imePadding())) {
                Row(Modifier.fillMaxWidth().height(64.dp).padding(start = 18.dp, end = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Image(mark, null, Modifier.size(24.dp), colorFilter = ColorFilter.tint(Studio.Ink))
                    Text("MLXL3", color = Studio.Ink, fontSize = 17.sp, fontWeight = FontWeight.SemiBold,
                        letterSpacing = 0.6.sp, modifier = Modifier.weight(1f).padding(start = 10.dp))
                    Box {
                        Surface(onClick = { advancedMenu = false; menu = true }, enabled = !busy, shape = CircleShape,
                            color = Studio.Glass, contentColor = Studio.Ink,
                            border = BorderStroke(1.dp, Studio.Contrast.copy(alpha = 0.12f)), modifier = Modifier.height(34.dp)) {
                            Row(Modifier.padding(horizontal = 13.dp), verticalAlignment = Alignment.CenterVertically) {
                                StatusDot(engine)
                                Text("LFM3 1.7B", fontSize = 13.sp, fontWeight = FontWeight.Medium,
                                    modifier = Modifier.padding(start = 8.dp))
                                Icon(painterResource(R.drawable.ic_chevron_down), null,
                                    Modifier.padding(start = 6.dp).size(13.dp), tint = Studio.Quiet)
                            }
                        }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false },
                            shape = RoundedCornerShape(14.dp), containerColor = Studio.Menu,
                            border = BorderStroke(1.dp, Studio.Edge), shadowElevation = 16.dp) {
                            DropdownMenuItem(text = { Text("Importer un dossier EXL3") }, onClick = { menu = false; onImport() })
                            if (savedModel) DropdownMenuItem(text = { Text("Charger le modèle enregistré") }, onClick = { menu = false; onLoad() })
                            if (ready) DropdownMenuItem(text = { Text("Benchmark fixe · 128 tokens") }, onClick = { menu = false; onBenchmark() })
                            if (diagnostics != null) DropdownMenuItem(text = { Text("Copier le diagnostic de performances") }, onClick = {
                                menu = false
                                context.getSystemService(ClipboardManager::class.java)
                                    .setPrimaryClip(ClipData.newPlainText("MLXL3 performance", diagnostics))
                                Toast.makeText(context, "Diagnostic copié", Toast.LENGTH_SHORT).show()
                            })
                            if (!advancedMenu) {
                                Text("Expériences à comparer", modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp), color = Quiet)
                                DropdownMenuItem(text = { Text("Hadamard shuffle (exp.) : ${if (shuffleHadamard) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onShuffleHadamard(!shuffleHadamard) })
                                DropdownMenuItem(text = { Text("Tête vectorisée (exp.) : ${if (vectorHead) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onVectorHead(!vectorHead) })
                                DropdownMenuItem(text = { Text("EXL3 K6 coopératif (exp.) : ${if (cooperativeK6) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onCooperativeK6(!cooperativeK6) })
                                DropdownMenuItem(text = { Text("EXL3 4 waves/groupe (exp.) : ${if (exl3Packed4) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onExl3Packed4(!exl3Packed4) })
                                DropdownMenuItem(text = { Text("Réglages avancés et retour arrière") },
                                    onClick = { advancedMenu = true })
                            } else {
                                DropdownMenuItem(text = { Text("Retour aux expériences") },
                                    onClick = { advancedMenu = false })
                                Text("Choix conservés · performances ≠ parité validée", modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp), color = Quiet)
                                DropdownMenuItem(text = { Text("Fusion RMSNorm (sans subgroup) : ${if (fusedNorm) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onFusedNorm(!fusedNorm) })
                                DropdownMenuItem(text = { Text("RMSNorm subgroup (exp.) : ${if (subgroupNorm) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onSubgroupNorm(!subgroupNorm) })
                                DropdownMenuItem(text = { Text("EXL3 multiwave (exp.) : ${if (exl3Multiwave) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onExl3Multiwave(!exl3Multiwave) })
                                DropdownMenuItem(text = { Text("Lot GPU (exp.) : $layerBatch couches · recharger") },
                                    onClick = { menu = false; onCycleLayerBatch() })
                                DropdownMenuItem(text = { Text("Tête une wave/ligne (exp.) : ${if (waveHead) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onWaveHead(!waveHead) })
                                listOf(1, 2, 4, 8).forEach { value ->
                                    DropdownMenuItem(text = { Text("Split-K $value${if (value == splitK) " ✓" else ""} · au prochain chargement") },
                                        onClick = { menu = false; onSplitK(value) })
                                }
                                DropdownMenuItem(text = { Text("FFN K4 : ${if (ffnK4Split == 0) "global" else ffnK4Split} · changer / recharger") },
                                    onClick = { menu = false; onCycleFfnSplit(true) })
                                DropdownMenuItem(text = { Text("FFN K5 : ${if (ffnK5Split == 0) "global" else ffnK5Split} · changer / recharger") },
                                    onClick = { menu = false; onCycleFfnSplit(false) })
                                DropdownMenuItem(text = { Text("Couches par lot : $layerBatch · changer / recharger") },
                                    onClick = { menu = false; onCycleLayerBatch() })
                                DropdownMenuItem(text = { Text("Boucle K4/K5 ×2 : ${if (unroll2) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onUnroll2(!unroll2) })
                                DropdownMenuItem(text = { Text("Tête expérimentale : ${if (fastHead) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onFastHead(!fastHead) })
                                DropdownMenuItem(text = { Text("Tête subgroup : ${if (subgroupHead) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onSubgroupHead(!subgroupHead) })
                                DropdownMenuItem(text = { Text("Lignes tête : $headRows · changer / recharger") },
                                    onClick = { menu = false; onCycleHeadRows() })
                                DropdownMenuItem(text = { Text("Couches convolution GPU : ${if (gpuConvLayer) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onGpuConvLayer(!gpuConvLayer) })
                                DropdownMenuItem(text = { Text("Kernel K5 expérimental : ${if (cooperativeK5) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onK5(!cooperativeK5) })
                                DropdownMenuItem(text = { Text("MCG spécialisé : ${if (specializedMcg) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onMcg(!specializedMcg) })
                                DropdownMenuItem(text = { Text("Sélection GPU : ${if (gpuGreedy) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onGpuGreedy(!gpuGreedy) })
                                DropdownMenuItem(text = { Text("Couches attention GPU : ${if (gpuAttentionLayer) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onGpuAttentionLayer(!gpuAttentionLayer) })
                                DropdownMenuItem(text = { Text("Profilage GPU : ${if (gpuTimestamps) "complet" else "désactivé"} · recharger") },
                                    onClick = { menu = false; onGpuTimestamps(!gpuTimestamps) })
                                DropdownMenuItem(text = { Text("Normalisations parallèles : ${if (parallelNorm) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onParallelNorm(!parallelNorm) })
                            }
                        }
                    }
                    IconButton(onClick = onNew, enabled = !busy, modifier = Modifier.padding(start = 8.dp).size(38.dp),
                        colors = IconButtonDefaults.iconButtonColors(containerColor = Studio.Glass, contentColor = Studio.Ink,
                            disabledContainerColor = Color.Transparent, disabledContentColor = Studio.Quiet.copy(alpha = 0.32f))) {
                        Icon(painterResource(R.drawable.ic_new_chat), "Nouvelle conversation", Modifier.size(19.dp))
                    }
                }
                if (messages.isEmpty()) {
                    BoxWithConstraints(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 28.dp), contentAlignment = Alignment.Center) {
                        val compact = maxHeight < 300.dp
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            if (!compact) {
                                Image(mark, null, Modifier.size(64.dp), colorFilter = ColorFilter.tint(Studio.Ink))
                                Spacer(Modifier.height(14.dp))
                            }
                            Text(WelcomeGreeting.now(greetingSeed), color = Studio.Ink, fontFamily = FontFamily.Serif,
                                fontSize = if (compact) 30.sp else 40.sp, lineHeight = if (compact) 36.sp else 46.sp,
                                letterSpacing = (-1.2).sp, textAlign = TextAlign.Center)
                            Text(if (ready) "LFM3 1.7B · EXL3 · 100 % local" else "Votre IA, sur votre téléphone · 100 % local",
                                color = Studio.Secondary, fontFamily = FontFamily.Monospace, fontSize = 11.sp,
                                textAlign = TextAlign.Center, modifier = Modifier.padding(top = 14.dp))
                            if (!ready && !busy) {
                                Spacer(Modifier.height(30.dp))
                                Button(onClick = if (savedModel) onLoad else onImport, shape = RoundedCornerShape(10.dp),
                                    colors = ButtonDefaults.buttonColors(containerColor = Studio.Ink, contentColor = Studio.Canvas),
                                    contentPadding = PaddingValues(horizontal = 24.dp, vertical = 13.dp),
                                    modifier = Modifier.shadow(6.dp, RoundedCornerShape(10.dp), clip = false)) {
                                    Text(if (savedModel) "Charger LFM3" else "Importer mon modèle", fontSize = 15.sp, fontWeight = FontWeight.Medium)
                                }
                                Text("Dossier EXL3 4bpw ou 8bpw", color = Studio.Quiet, fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp, modifier = Modifier.padding(top = 14.dp))
                            }
                        }
                    }
                } else {
                    LazyColumn(state = list, modifier = Modifier.weight(1f).fillMaxWidth().edgeFade(top = 18.dp, bottom = 22.dp),
                        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 22.dp),
                        verticalArrangement = Arrangement.spacedBy(26.dp)) {
                        items(messages) { message ->
                            Column(Modifier.fillMaxWidth(), horizontalAlignment = if (message.role == "user") Alignment.End else Alignment.Start) {
                                if (message.role == "assistant") Row(Modifier.padding(bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                                    Image(mark, null, Modifier.size(15.dp), colorFilter = ColorFilter.tint(Studio.Ink))
                                    Text("MLXL3", color = Studio.Quiet, fontSize = 12.sp, fontWeight = FontWeight.Medium,
                                        letterSpacing = 0.3.sp, modifier = Modifier.padding(start = 8.dp))
                                }
                                SelectionContainer {
                                    Text(message.text.ifEmpty { "…" }, color = Studio.Ink, fontSize = 16.sp, lineHeight = 25.sp,
                                        modifier = if (message.role == "user") Modifier.widthIn(max = 320.dp)
                                            .background(Studio.Contrast.copy(alpha = 0.055f), RoundedCornerShape(14.dp))
                                            .border(0.5.dp, Studio.Edge, RoundedCornerShape(14.dp))
                                            .padding(horizontal = 15.dp, vertical = 11.dp)
                                        else Modifier.fillMaxWidth())
                                }
                                if (message.role == "assistant" && message.speed != null) {
                                    Text(message.speed, color = Studio.Quiet, fontFamily = FontFamily.Monospace, fontSize = 11.sp,
                                        modifier = Modifier.padding(top = 8.dp))
                                }
                            }
                        }
                    }
                }
                if (error != null) {
                    Surface(color = Studio.Failed.copy(alpha = 0.10f), contentColor = Color(0xFFFFDAD9), shape = RoundedCornerShape(14.dp),
                        border = BorderStroke(1.dp, Studio.Failed.copy(alpha = 0.28f)),
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                        Row(Modifier.padding(start = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(error, fontSize = 13.sp, modifier = Modifier.weight(1f).padding(vertical = 12.dp))
                            TextButton(onClick = onDismissError) { Text("Fermer", color = Studio.Ink) }
                        }
                    }
                }
                val composerShape = RoundedCornerShape(18.dp)
                val composerEdge by animateColorAsState(
                    if (focused) Studio.Accent.copy(alpha = 0.45f) else Studio.Contrast.copy(alpha = 0.12f), label = "composerEdge")
                Column(Modifier.fillMaxWidth().then(if (hostFitsBottomInsets) Modifier else Modifier.navigationBarsPadding())
                    .padding(horizontal = 12.dp, vertical = 8.dp)) {
                    Column(Modifier.fillMaxWidth().testTag("composer")
                        // A tight contact shadow and a wide ambient lift, as on Desktop.
                        .shadow(if (focused) 18.dp else 12.dp, composerShape, clip = false)
                        .background(Studio.Composer, composerShape).border(1.dp, composerEdge, composerShape)
                        .padding(start = 18.dp, top = 15.dp, end = 10.dp, bottom = 10.dp)) {
                        BasicTextField(state = draft, enabled = !busy && ready,
                            textStyle = TextStyle(color = Studio.Ink, fontSize = 16.sp, lineHeight = 24.sp), cursorBrush = SolidColor(Studio.Accent),
                            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences), lineLimits = TextFieldLineLimits.MultiLine(maxHeightInLines = 5),
                            modifier = Modifier.fillMaxWidth().heightIn(min = 40.dp, max = 136.dp)
                                .onFocusChanged { focused = it.isFocused }
                                .semantics { contentDescription = "Message LFM3" },
                            decorator = { inner -> Box { if (draft.text.isEmpty()) Text("Écrivez un message…", color = Studio.Quiet, fontSize = 16.sp); inner() } })
                        Row(Modifier.fillMaxWidth().padding(top = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                            StatusDot(engine)
                            Text(if (busy) status else if (ready) "LFM3 1.7B · Local" else "Modèle non chargé",
                                color = Studio.Quiet, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f).padding(horizontal = 8.dp))
                            val canSend = ready && draft.text.isNotBlank()
                            val active = busy || canSend
                            FilledIconButton(onClick = { if (busy) onStop() else { onSend(draft.text.toString()); draft.edit { replace(0, length, "") } } },
                                enabled = active,
                                colors = IconButtonDefaults.filledIconButtonColors(containerColor = Studio.Ink, contentColor = Studio.Canvas,
                                    disabledContainerColor = Studio.Ink.copy(alpha = 0.16f), disabledContentColor = Studio.Ink.copy(alpha = 0.38f)),
                                modifier = Modifier.size(40.dp).shadow(if (active) 4.dp else 0.dp, CircleShape, clip = false)) {
                                Icon(painterResource(if (busy) R.drawable.ic_stop else R.drawable.ic_arrow_up),
                                    if (busy) "Arrêter la génération" else "Envoyer", Modifier.size(if (busy) 14.dp else 18.dp))
                            }
                        }
                    }
                    if (!ready && busy) Text(status, color = Studio.Quiet, fontSize = 11.sp,
                        modifier = Modifier.padding(start = 8.dp, top = 8.dp))
                }
            }
        }
    }
}
