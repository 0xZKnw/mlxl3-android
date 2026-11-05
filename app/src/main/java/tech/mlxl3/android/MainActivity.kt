package tech.mlxl3.android

import android.os.Bundle
import androidx.activity.ComponentActivity
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
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
private val Sidebar = Color(0xFF0C0C0C)
private val Composer = Color(0xFF1B1B1B)
private val Ink = Color(0xFFE8E8E8)
private val Quiet = Color(0xFF949494)
private val Edge = Color(0xFF2C2C2C)

@Composable
fun StudioScreen(messages: List<ChatMessage>, busy: Boolean, ready: Boolean, status: String,
                 device: String, error: String?, savedModel: Boolean, onImport: () -> Unit,
                 onLoad: () -> Unit, onNew: () -> Unit, onSend: (String) -> Unit,
                 onStop: () -> Unit, onDismissError: () -> Unit) {
    MaterialTheme(colorScheme = darkColorScheme(primary = Ink, onPrimary = Canvas, background = Canvas,
        surface = Composer, onSurface = Ink, onBackground = Ink, outline = Edge)) {
        var draft by rememberSaveable { mutableStateOf("") }
        var menu by remember { mutableStateOf(false) }
        val list = rememberLazyListState()
        LaunchedEffect(messages.size, messages.lastOrNull()?.text) {
            if (messages.isNotEmpty()) list.scrollToItem(messages.lastIndex)
        }
        Column(Modifier.fillMaxSize().background(Canvas).safeDrawingPadding().imePadding()) {
            Row(Modifier.fillMaxWidth().background(Sidebar).padding(horizontal = 16.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Image(painterResource(R.drawable.mlxl3_brand), null, Modifier.size(36.dp))
                Column(Modifier.weight(1f).padding(start = 10.dp)) {
                    Text("MLXL3", fontSize = 18.sp, letterSpacing = 1.sp)
                    Text("LFM3 1.7B", color = Quiet, fontSize = 12.sp)
                }
                Box {
                    TextButton(onClick = { menu = true }, enabled = !busy) { Text("Model") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(text = { Text("Import EXL3 folder") }, onClick = { menu = false; onImport() })
                        if (savedModel) DropdownMenuItem(text = { Text("Load saved model") }, onClick = { menu = false; onLoad() })
                        DropdownMenuItem(text = { Text("New chat") }, onClick = { menu = false; onNew() })
                    }
                }
            }
            HorizontalDivider(color = Edge)
            if (messages.isEmpty()) {
                Column(Modifier.weight(1f).fillMaxWidth().padding(32.dp), horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center) {
                    Image(painterResource(R.drawable.mlxl3_brand), null, Modifier.size(88.dp))
                    Spacer(Modifier.height(20.dp))
                    Text("What’s on your mind?", fontSize = 25.sp)
                    Spacer(Modifier.height(10.dp))
                    Text(if (ready) "LFM3 runs entirely on your phone." else "Your model. On your phone.", color = Quiet, fontSize = 14.sp)
                    if (!ready && !busy) {
                        Spacer(Modifier.height(24.dp))
                        OutlinedButton(onClick = if (savedModel) onLoad else onImport) {
                            Text(if (savedModel) "Load LFM3" else "Import LFM3 EXL3")
                        }
                        Text("Choose the 4bpw or 8bpw variant folder", color = Quiet, fontSize = 12.sp,
                            modifier = Modifier.padding(top = 8.dp))
                    }
                }
            } else {
                LazyColumn(state = list, modifier = Modifier.weight(1f).fillMaxWidth(),
                    contentPadding = PaddingValues(horizontal = 20.dp, vertical = 24.dp), verticalArrangement = Arrangement.spacedBy(24.dp)) {
                    items(messages) { message ->
                        Column(Modifier.fillMaxWidth(), horizontalAlignment = if (message.role == "user") Alignment.End else Alignment.Start) {
                            if (message.role == "assistant") Text("MLXL3", color = Quiet, fontSize = 12.sp, modifier = Modifier.padding(bottom = 8.dp))
                            SelectionContainer {
                                Text(message.text.ifEmpty { "…" }, fontSize = 16.sp, lineHeight = 25.sp,
                                    modifier = if (message.role == "user") Modifier.background(Composer, RoundedCornerShape(18.dp)).padding(14.dp) else Modifier)
                            }
                        }
                    }
                }
            }
            if (error != null) {
                Surface(color = Color(0xFF332222), shape = RoundedCornerShape(12.dp), modifier = Modifier.padding(horizontal = 16.dp)) {
                    Row(Modifier.padding(start = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(error, fontSize = 13.sp, modifier = Modifier.weight(1f).padding(vertical = 10.dp))
                        TextButton(onClick = onDismissError) { Text("Dismiss") }
                    }
                }
            }
            Column(Modifier.padding(12.dp).fillMaxWidth()
                .border(1.dp, Edge, RoundedCornerShape(22.dp)).background(Composer, RoundedCornerShape(22.dp)).padding(8.dp)) {
                TextField(value = draft, onValueChange = { draft = it }, enabled = !busy && ready,
                    placeholder = { Text("Message LFM3…", color = Quiet) }, maxLines = 5,
                    colors = TextFieldDefaults.colors(focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                        disabledContainerColor = Color.Transparent, focusedIndicatorColor = Color.Transparent,
                        unfocusedIndicatorColor = Color.Transparent, disabledIndicatorColor = Color.Transparent),
                    modifier = Modifier.fillMaxWidth().semantics { contentDescription = "Message LFM3" })
                Row(Modifier.fillMaxWidth().padding(start = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("LFM3 1.7B · EXL3", color = Quiet, fontSize = 12.sp, modifier = Modifier.weight(1f))
                    if (busy) FilledTonalButton(onClick = onStop) { Text("Stop") }
                    else Button(onClick = { onSend(draft); draft = "" }, enabled = ready && draft.isNotBlank()) { Text("Send") }
                }
            }
            Column(Modifier.fillMaxWidth().padding(start = 20.dp, end = 20.dp, bottom = 12.dp)) {
                Text(status, color = Quiet, fontSize = 11.sp)
                if (device.isNotEmpty()) Text(device, color = Quiet, fontSize = 10.sp)
            }
        }
    }
}
