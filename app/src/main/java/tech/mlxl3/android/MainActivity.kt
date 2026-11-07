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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
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
                model.headRows, model::cycleHeadRows, model.parallelNorm, model::chooseParallelNorm, model.gpuTimestamps, model::chooseGpuTimestamps, model.waveHead, model::chooseWaveHead)
        }
    }
}

private val Canvas = Color(0xFF121212)
private val Panel = Color(0xFF1C1C1E)
private val Ink = Color(0xFFF0F0EF)
private val Quiet = Color(0xFF969698)
private val Edge = Color(0xFF303032)

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
                 waveHead: Boolean = false, onWaveHead: (Boolean) -> Unit = {}) {
    MaterialTheme(colorScheme = darkColorScheme(primary = Ink, onPrimary = Canvas, background = Canvas,
        surface = Panel, onSurface = Ink, onBackground = Ink, outline = Edge)) {
        val draft = rememberTextFieldState()
        var menu by remember { mutableStateOf(false) }
        var advancedMenu by remember { mutableStateOf(false) }
        val context = LocalContext.current
        val list = rememberLazyListState()
        val imeBottom = WindowInsets.ime.getBottom(LocalDensity.current)
        LaunchedEffect(messages.size, messages.lastOrNull()?.text, imeBottom) {
            if (messages.isNotEmpty()) list.scrollToItem(messages.lastIndex)
        }
        Surface(modifier = Modifier.fillMaxSize(), color = Canvas, contentColor = Ink) {
            Column(Modifier.fillMaxSize()
                .windowInsetsPadding(WindowInsets.safeDrawing.only(WindowInsetsSides.Top + WindowInsetsSides.Horizontal))
                .then(if (hostFitsBottomInsets) Modifier.consumeWindowInsets(WindowInsets.ime.union(WindowInsets.navigationBars).only(WindowInsetsSides.Bottom))
                    else Modifier.imePadding())) {
                Row(Modifier.fillMaxWidth().height(68.dp).padding(horizontal = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                    Image(painterResource(R.drawable.mlxl3_brand), null, Modifier.size(34.dp))
                    Text("MLXL3", fontSize = 19.sp, fontWeight = FontWeight.SemiBold,
                        letterSpacing = 0.4.sp, modifier = Modifier.weight(1f).padding(start = 10.dp))
                    Box {
                        OutlinedButton(onClick = { advancedMenu = false; menu = true }, enabled = !busy,
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Edge), shape = CircleShape) {
                            Text("LFM3 1.7B", fontSize = 12.sp)
                            Icon(painterResource(R.drawable.ic_chevron_down), null, Modifier.padding(start = 6.dp).size(14.dp), tint = Quiet)
                        }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
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
                                DropdownMenuItem(text = { Text("Tête une wave/ligne (exp.) : ${if (waveHead) "oui" else "non"} · recharger") },
                                    onClick = { menu = false; onWaveHead(!waveHead) })
                                DropdownMenuItem(text = { Text("Réglages avancés et retour arrière") },
                                    onClick = { advancedMenu = true })
                            } else {
                                DropdownMenuItem(text = { Text("Retour aux expériences") },
                                    onClick = { advancedMenu = false })
                                Text("Choix conservés · performances ≠ parité validée", modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp), color = Quiet)
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
                    IconButton(onClick = onNew, enabled = !busy, modifier = Modifier.padding(start = 4.dp)) {
                        Icon(painterResource(R.drawable.ic_new_chat), "Nouvelle conversation", Modifier.size(21.dp))
                    }
                }
                HorizontalDivider(color = Color(0xFF232325))
                if (messages.isEmpty()) {
                    BoxWithConstraints(Modifier.weight(1f).fillMaxWidth().padding(horizontal = 28.dp), contentAlignment = Alignment.Center) {
                        val compact = maxHeight < 280.dp
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            if (!compact) {
                                Image(painterResource(R.drawable.mlxl3_brand), null, Modifier.size(72.dp))
                                Spacer(Modifier.height(28.dp))
                            }
                            Text(if (ready) "Qu’avez-vous en tête ?" else "Votre IA,\nsur votre téléphone.",
                                fontSize = if (compact) 22.sp else 28.sp, lineHeight = 35.sp,
                                fontWeight = FontWeight.Medium, textAlign = TextAlign.Center)
                            Text(if (ready) "Une conversation, entièrement en local." else "LFM3 1.7B · EXL3 · 100 % local",
                                color = Quiet, fontSize = 13.sp, textAlign = TextAlign.Center, modifier = Modifier.padding(top = 12.dp))
                            if (!ready && !busy) {
                                Spacer(Modifier.height(28.dp))
                                Button(onClick = if (savedModel) onLoad else onImport,
                                    shape = RoundedCornerShape(14.dp), contentPadding = PaddingValues(horizontal = 24.dp, vertical = 14.dp)) {
                                    Text(if (savedModel) "Charger LFM3" else "Importer mon modèle", fontWeight = FontWeight.Medium)
                                }
                                Text("Dossier EXL3 4bpw ou 8bpw", color = Quiet, fontSize = 12.sp, modifier = Modifier.padding(top = 12.dp))
                            }
                        }
                    }
                } else {
                    LazyColumn(state = list, modifier = Modifier.weight(1f).fillMaxWidth(),
                        contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp),
                        verticalArrangement = Arrangement.spacedBy(28.dp)) {
                        items(messages) { message ->
                            Column(Modifier.fillMaxWidth(), horizontalAlignment = if (message.role == "user") Alignment.End else Alignment.Start) {
                                if (message.role == "assistant") Row(Modifier.padding(bottom = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                                    Image(painterResource(R.drawable.mlxl3_brand), null, Modifier.size(20.dp))
                                    Text("MLXL3", color = Quiet, fontSize = 12.sp, fontWeight = FontWeight.Medium, modifier = Modifier.padding(start = 8.dp))
                                }
                                SelectionContainer {
                                    Text(message.text.ifEmpty { "…" }, color = Ink, fontSize = 16.sp, lineHeight = 25.sp,
                                        modifier = if (message.role == "user") Modifier.widthIn(max = 340.dp)
                                            .background(Panel, RoundedCornerShape(22.dp, 22.dp, 6.dp, 22.dp)).padding(horizontal = 16.dp, vertical = 12.dp)
                                        else Modifier.fillMaxWidth())
                                }
                                if (message.role == "assistant" && message.speed != null) {
                                    Text(message.speed, color = Quiet, fontSize = 11.sp, modifier = Modifier.padding(top = 8.dp))
                                }
                            }
                        }
                    }
                }
                if (error != null) {
                    Surface(color = Color(0xFF302124), contentColor = Color(0xFFFFDAD9), shape = RoundedCornerShape(14.dp),
                        modifier = Modifier.padding(horizontal = 16.dp, vertical = 8.dp)) {
                        Row(Modifier.padding(start = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                            Text(error, fontSize = 13.sp, modifier = Modifier.weight(1f).padding(vertical = 12.dp))
                            TextButton(onClick = onDismissError) { Text("Fermer") }
                        }
                    }
                }
                Column(Modifier.fillMaxWidth().then(if (hostFitsBottomInsets) Modifier else Modifier.navigationBarsPadding())
                    .padding(horizontal = 12.dp, vertical = 8.dp)) {
                    Column(Modifier.fillMaxWidth().testTag("composer")
                        .border(1.dp, Edge, RoundedCornerShape(24.dp)).background(Panel, RoundedCornerShape(24.dp))
                        .padding(start = 18.dp, top = 16.dp, end = 10.dp, bottom = 8.dp)) {
                        BasicTextField(state = draft, enabled = !busy && ready,
                            textStyle = TextStyle(color = Ink, fontSize = 16.sp, lineHeight = 24.sp), cursorBrush = SolidColor(Ink),
                            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.Sentences), lineLimits = TextFieldLineLimits.MultiLine(maxHeightInLines = 5),
                            modifier = Modifier.fillMaxWidth().heightIn(min = 40.dp, max = 136.dp)
                                .semantics { contentDescription = "Message LFM3" },
                            decorator = { inner -> Box { if (draft.text.isEmpty()) Text("Écrivez à LFM3…", color = Quiet, fontSize = 16.sp); inner() } })
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            Box(Modifier.size(5.dp).background(if (ready) Color(0xFF99B7A5) else Quiet, CircleShape))
                            Text(if (busy) status else if (ready) "LFM3 1.7B · Local" else "Modèle non chargé",
                                color = Quiet, fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
                                modifier = Modifier.weight(1f).padding(horizontal = 8.dp))
                            val canSend = ready && draft.text.isNotBlank()
                            FilledIconButton(onClick = { if (busy) onStop() else { onSend(draft.text.toString()); draft.edit { replace(0, length, "") } } },
                                enabled = busy || canSend,
                                colors = IconButtonDefaults.filledIconButtonColors(containerColor = Ink, contentColor = Canvas,
                                    disabledContainerColor = Color(0xFF303033), disabledContentColor = Color(0xFF727275)),
                                modifier = Modifier.size(48.dp)) {
                                Icon(painterResource(if (busy) R.drawable.ic_stop else R.drawable.ic_arrow_up),
                                    if (busy) "Arrêter la génération" else "Envoyer", Modifier.size(21.dp))
                            }
                        }
                    }
                    if (!ready && busy) Text(status, color = Quiet, fontSize = 11.sp,
                        modifier = Modifier.padding(start = 8.dp, top = 8.dp))
                }
            }
        }
    }
}
