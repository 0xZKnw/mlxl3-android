package tech.mlxl3.android

import android.app.Application
import android.annotation.SuppressLint
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.documentfile.provider.DocumentFile
import androidx.lifecycle.AndroidViewModel
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

data class ChatMessage(val role: String, val text: String, val speed: String? = null)

class StudioViewModel(application: Application) : AndroidViewModel(application) {
    var messages by mutableStateOf<List<ChatMessage>>(emptyList()); private set
    var busy by mutableStateOf(false); private set
    var ready by mutableStateOf(false); private set
    var status by mutableStateOf("Import LFM3-1.7B EXL3 to get started"); private set
    var device by mutableStateOf(""); private set
    var error by mutableStateOf<String?>(null); private set
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val stopped = AtomicBoolean(false)
    private val prefs = application.getSharedPreferences("studio", 0)
    val savedModel: Boolean get() = prefs.getString("model", null)?.let { File(it).isDirectory } == true

    fun dismissError() { error = null }
    fun newChat() { if (!busy) messages = emptyList() }
    fun stop() { stopped.set(true); NativeEngine.cancel(); status = "Stopping…" }

    private fun task(work: () -> Unit) {
        if (busy) return
        busy = true
        error = null
        stopped.set(false)
        NativeEngine.resetCancel()
        worker.execute {
            try { work() }
            catch (e: Exception) { main.post { error = if (stopped.get()) null else (e.message ?: "Operation failed"); status = if (stopped.get()) "Stopped" else "Operation failed" } }
            finally { main.post { busy = false } }
        }
    }

    private val callback = NativeEngine.Callback { json ->
        val event = JSONObject(json)
        main.post {
            when (event.getString("type")) {
                "status" -> status = event.getString("text")
                "ready" -> { ready = true; device = event.getString("device"); status = "LFM3-1.7B · EXL3 · Vulkan" }
                "text" -> {
                    messages = messages.dropLast(1) + ChatMessage("assistant", event.getString("text"))
                    status = "Generating on device…"
                }
                "done" -> {
                    val rate = event.optDouble("decode_tokens_per_second", Double.NaN)
                    val speed = if (rate.isFinite() && rate >= 0) "%.1f tok/s".format(rate) else "— tok/s"
                    messages.lastOrNull()?.takeIf { it.role == "assistant" }?.let {
                        messages = messages.dropLast(1) + it.copy(speed = speed)
                    }
                    status = "${event.getInt("tokens")} tokens · %.1f s".format(event.getDouble("decode_seconds") + event.getDouble("prefill_seconds"))
                }
            }
        }
    }

    fun loadSaved() = task {
        val path = prefs.getString("model", null) ?: error("Import a model first")
        main.post { ready = false; status = "Loading LFM3…" }
        val result = NativeEngine.load(path, callback)
        check(result.isEmpty()) { result }
    }

    // Persist synchronously on the worker before deleting the previous model files.
    @SuppressLint("ApplySharedPref", "UseKtx")
    fun importModel(uri: Uri) = task {
        val app = getApplication<Application>()
        val source = DocumentFile.fromTreeUri(app, uri) ?: error("Cannot open this folder")
        val files = source.listFiles().filter { it.isFile && ModelImport.accepts(it.name.orEmpty()) }
        check(files.size in 3..64) { "Select the 4bpw or 8bpw EXL3 folder with its tokenizer and weights" }
        check(files.mapNotNull { it.name }.distinct().size == files.size) { "Duplicate model filenames" }
        val stage = File(app.filesDir, "importing-${UUID.randomUUID()}")
        check(stage.mkdir()) { "Cannot create model folder" }
        var keep = false
        try {
            var total = 0L
            for (file in files) {
                val name = requireNotNull(file.name)
                val fileLimit = when (name) { "config.json" -> 1024L * 1024; "tokenizer.json" -> 64L * 1024 * 1024; else -> ModelImport.MAX_BYTES }
                val remaining = minOf(ModelImport.MAX_BYTES - total, fileLimit, StatFs(app.filesDir.path).availableBytes - 128L * 1024 * 1024)
                val destination = File(stage, name)
                app.contentResolver.openInputStream(file.uri).use { input ->
                    checkNotNull(input) { "Cannot read $name" }
                    destination.outputStream().use { output ->
                        total += ModelImport.copyBounded(input, output, remaining, stopped::get) { bytes ->
                            main.post { status = "Importing $name · ${bytes / (1024 * 1024)} MiB" }
                        }
                    }
                }
            }
            ModelImport.validateFolder(stage)
            main.post { ready = false; status = "Loading LFM3…" }
            val result = NativeEngine.load(stage.path, callback)
            check(result.isEmpty()) { result }
            val previous = prefs.getString("model", null)
            prefs.edit().putString("model", stage.path).commit().also { check(it) { "Cannot save model selection" } }
            keep = true
            if (previous != null && previous != stage.path && File(previous).parentFile == app.filesDir) File(previous).deleteRecursively()
        } finally {
            if (!keep) {
                NativeEngine.unload()
                main.post { ready = false }
                stage.deleteRecursively()
            }
        }
    }

    fun send(prompt: String) {
        if (busy || !ready || prompt.isBlank()) return
        val previous = messages
        val request = previous + ChatMessage("user", prompt.trim())
        messages = request + ChatMessage("assistant", "")
        task {
            val input = JSONArray()
            request.forEach { input.put(JSONObject().put("role", it.role).put("content", it.text)) }
            val result = NativeEngine.generate(input.toString(), 256, callback)
            if (result.isNotEmpty()) {
                main.post {
                    if (stopped.get()) {
                        if (messages.lastOrNull()?.text.isNullOrEmpty()) messages = previous
                        status = "Stopped"
                    } else messages = previous
                }
                check(stopped.get()) { result }
            }
        }
    }

    override fun onCleared() {
        stopped.set(true)
        NativeEngine.cancel()
        worker.execute { NativeEngine.unload() }
        worker.shutdown()
    }
}
