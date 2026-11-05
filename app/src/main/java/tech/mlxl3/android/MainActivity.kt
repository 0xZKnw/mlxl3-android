package tech.mlxl3.android

import android.os.Bundle
import android.os.Build
import android.view.View
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

internal fun ComponentActivity.configureStudioWindow() {
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
                model::send, model::stop, model::dismissError)
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
                 onStop: () -> Unit, onDismissError: () -> Unit) {
    MaterialTheme(colorScheme = darkColorScheme(primary = Ink, onPrimary = Canvas, background = Canvas,
        surface = Panel, onSurface = Ink, onBackground = Ink, outline = Edge)) {
        val draft = rememberTextFieldState()
        var menu by remember { mutableStateOf(false) }
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
                        OutlinedButton(onClick = { menu = true }, enabled = !busy,
                            contentPadding = PaddingValues(horizontal = 12.dp, vertical = 0.dp),
                            border = androidx.compose.foundation.BorderStroke(1.dp, Edge), shape = CircleShape) {
                            Text("LFM3 1.7B", fontSize = 12.sp)
                            Icon(painterResource(R.drawable.ic_chevron_down), null, Modifier.padding(start = 6.dp).size(14.dp), tint = Quiet)
                        }
                        DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                            DropdownMenuItem(text = { Text("Importer un dossier EXL3") }, onClick = { menu = false; onImport() })
                            if (savedModel) DropdownMenuItem(text = { Text("Charger le modèle enregistré") }, onClick = { menu = false; onLoad() })
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
