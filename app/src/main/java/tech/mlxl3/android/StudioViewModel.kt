package tech.mlxl3.android

import android.app.Application
import android.app.ActivityManager
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.annotation.SuppressLint
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.os.StatFs
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.documentfile.provider.DocumentFile
import androidx.core.content.edit
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
    var diagnostics by mutableStateOf<String?>(null); private set
    private var environmentBefore = JSONObject()
    var error by mutableStateOf<String?>(null); private set
    private val worker = Executors.newSingleThreadExecutor()
    private val main = Handler(Looper.getMainLooper())
    private val stopped = AtomicBoolean(false)
    private val prefs = application.getSharedPreferences("studio", 0)
    private val initialPerformance = PerformanceDefaults.forDevice(Build.MANUFACTURER, Build.MODEL)
        .withPreferences(prefs.all)
    var splitK by mutableStateOf(initialPerformance.splitK); private set
    var fastHead by mutableStateOf(prefs.getBoolean("fast_head", false)); private set
    var gpuConvLayer by mutableStateOf(initialPerformance.gpuConvLayer); private set
    var cooperativeK5 by mutableStateOf(initialPerformance.cooperativeK5); private set
    var specializedMcg by mutableStateOf(prefs.getBoolean("specialized_mcg", false)); private set
    var gpuGreedy by mutableStateOf(initialPerformance.gpuGreedy); private set
    var gpuAttentionLayer by mutableStateOf(initialPerformance.gpuAttentionLayer); private set
    var ffnK4Split by mutableStateOf(initialPerformance.ffnK4Split); private set
    var ffnK5Split by mutableStateOf(prefs.getInt("ffn_k5_split", 0).takeIf { it in listOf(0, 1, 2, 4, 8) } ?: 0); private set
    var layerBatch by mutableStateOf(initialPerformance.layerBatch); private set
    var unroll2 by mutableStateOf(prefs.getBoolean("unroll2", false)); private set
    var subgroupHead by mutableStateOf(prefs.getBoolean("subgroup_head", false)); private set
    var headRows by mutableStateOf(prefs.getInt("head_rows", 4).takeIf { it in listOf(1, 2, 4, 8) } ?: 4); private set
    var gpuTimestamps by mutableStateOf(prefs.getBoolean("gpu_timestamps", true)); private set
    var parallelNorm by mutableStateOf(prefs.getBoolean("parallel_norm", false)); private set
    val savedModel: Boolean get() = prefs.getString("model", null)?.let { File(it).isDirectory } == true

    fun dismissError() { error = null }
    fun chooseSplitK(value: Int) {
        if (busy || value !in listOf(1, 2, 4, 8)) return
        splitK = value
        prefs.edit { putInt("split_k", value) }
        status = "Split-K $value sélectionné · rechargez le modèle"
    }
    fun chooseFastHead(enabled: Boolean) {
        if (busy) return
        fastHead = enabled
        prefs.edit { putBoolean("fast_head", enabled) }
        status = "Tête expérimentale ${if (enabled) "activée" else "désactivée"} · rechargez le modèle"
    }
    fun chooseGpuConvLayer(enabled: Boolean) {
        if (busy) return
        gpuConvLayer = enabled
        prefs.edit { putBoolean("gpu_conv_layer", enabled) }
        status = "Couches convolution GPU ${if (enabled) "activées" else "désactivées"} · rechargez le modèle"
    }
    fun chooseK5(enabled: Boolean) {
        if (busy) return
        cooperativeK5 = enabled
        prefs.edit { putBoolean("cooperative_k5", enabled) }
        status = "Kernel K5 expérimental ${if (enabled) "activé" else "désactivé"} · rechargez le modèle"
    }
    fun chooseMcg(enabled: Boolean) {
        if (busy) return
        specializedMcg = enabled
        prefs.edit { putBoolean("specialized_mcg", enabled) }
        status = "MCG spécialisé ${if (enabled) "activé" else "désactivé"} · rechargez le modèle"
    }
    fun chooseGpuGreedy(enabled: Boolean) {
        if (busy) return
        gpuGreedy = enabled
        prefs.edit { putBoolean("gpu_greedy", enabled) }
        status = "Sélection GPU ${if (enabled) "activée" else "désactivée"} · rechargez le modèle"
    }
    fun chooseGpuAttentionLayer(enabled: Boolean) {
        if (busy) return
        gpuAttentionLayer = enabled
        prefs.edit { putBoolean("gpu_attention_layer", enabled) }
        status = "Couches attention GPU ${if (enabled) "activées" else "désactivées"} · rechargez le modèle"
    }
    fun cycleFfnSplit(k4: Boolean) {
        if (busy) return
        val choices = listOf(0, 1, 2, 4, 8)
        val previous = if (k4) ffnK4Split else ffnK5Split
        val next = choices[(choices.indexOf(previous) + 1) % choices.size]
        if (k4) ffnK4Split = next else ffnK5Split = next
        prefs.edit { putInt(if (k4) "ffn_k4_split" else "ffn_k5_split", next) }
        status = "Réglage FFN mis à jour · rechargez le modèle"
    }
    fun cycleLayerBatch() {
        if (busy) return
        val choices = listOf(1, 2, 4)
        layerBatch = choices[(choices.indexOf(layerBatch) + 1) % choices.size]
        prefs.edit { putInt("layer_batch", layerBatch) }
        status = "Regroupement des couches mis à jour · rechargez le modèle"
    }
    fun chooseUnroll2(enabled: Boolean) {
        if (busy) return
        unroll2 = enabled
        prefs.edit { putBoolean("unroll2", enabled) }
        status = "Boucle K4/K5 expérimentale ${if (enabled) "activée" else "désactivée"} · rechargez le modèle"
    }
    fun chooseSubgroupHead(enabled: Boolean) {
        if (busy) return
        subgroupHead = enabled
        prefs.edit { putBoolean("subgroup_head", enabled) }
        status = "Réduction subgroup ${if (enabled) "activée" else "désactivée"} · rechargez le modèle"
    }
    fun cycleHeadRows() {
        if (busy) return
        val choices = listOf(1, 2, 4, 8)
        headRows = choices[(choices.indexOf(headRows) + 1) % choices.size]
        prefs.edit { putInt("head_rows", headRows) }
        status = "Lignes de la tête mises à jour · rechargez le modèle"
    }
    fun chooseGpuTimestamps(enabled: Boolean) {
        if (busy) return
        gpuTimestamps = enabled
        prefs.edit { putBoolean("gpu_timestamps", enabled) }
        status = "Profilage GPU ${if (enabled) "activé" else "désactivé"} · rechargez le modèle"
    }
    fun chooseParallelNorm(enabled: Boolean) {
        if (busy) return
        parallelNorm = enabled
        prefs.edit { putBoolean("parallel_norm", enabled) }
        status = "Normalisations parallèles ${if (enabled) "activées" else "désactivées"} · rechargez le modèle"
    }
    fun newChat() { if (!busy) { messages = emptyList(); diagnostics = null } }

    private fun environment(): JSONObject = runCatching {
        val app = getApplication<Application>()
        val memory = ActivityManager.MemoryInfo()
        app.getSystemService(ActivityManager::class.java).getMemoryInfo(memory)
        val power = app.getSystemService(PowerManager::class.java)
        JSONObject().put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("default_performance_profile", initialPerformance.profile)
            .put("android_sdk", Build.VERSION.SDK_INT).put("available_ram_bytes", memory.availMem)
            .put("total_ram_bytes", memory.totalMem).put("low_memory", memory.lowMemory)
            .put("app_pss_kib", Debug.getPss()).put("power_save", power.isPowerSaveMode)
            .put("thermal_status", if (Build.VERSION.SDK_INT >= 29) power.currentThermalStatus else JSONObject.NULL)
    }.getOrElse { JSONObject().put("status", "unavailable").put("reason", it.javaClass.simpleName) }
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
        val report = if (event.optString("type") in listOf("done", "benchmark_done")) JSONObject()
            .put("engine", event).put("environment_before", environmentBefore)
            .put("environment_after", environment()).toString(2) else null
        main.post {
            when (event.getString("type")) {
                "status" -> status = event.getString("text")
                "ready" -> { ready = true; diagnostics = null; device = event.getString("device"); status = "LFM3-1.7B · EXL3 · Vulkan" }
                "text" -> {
                    messages = messages.dropLast(1) + ChatMessage("assistant", event.getString("text"))
                    status = "Generating on device…"
                }
                "done" -> {
                    diagnostics = report
                    val rate = event.optDouble("decode_tokens_per_second", Double.NaN)
                    val speed = if (rate.isFinite() && rate >= 0) "%.1f tok/s".format(rate) else "— tok/s"
                    messages.lastOrNull()?.takeIf { it.role == "assistant" }?.let {
                        messages = messages.dropLast(1) + it.copy(speed = speed)
                    }
                    status = "${event.getInt("tokens")} tokens · %.1f s".format(event.getDouble("decode_seconds") + event.getDouble("prefill_seconds"))
                }
                "benchmark_done" -> {
                    diagnostics = report
                    val rate = event.optDouble("decode_tokens_per_second", Double.NaN)
                    status = if (rate.isFinite() && rate >= 0) "Benchmark : %.1f tok/s".format(rate) else "Benchmark terminé"
                }
            }
        }
    }

    fun loadSaved() = task {
        val path = prefs.getString("model", null) ?: error("Import a model first")
        main.post { ready = false; status = "Loading LFM3…" }
        check(NativeEngine.setSplitK(splitK)) { "Invalid split-K" }
        NativeEngine.setFastHead(fastHead)
        NativeEngine.setGpuConvLayer(gpuConvLayer)
        NativeEngine.setExperimentalKernels(cooperativeK5, specializedMcg)
        NativeEngine.setGpuGreedy(gpuGreedy)
        NativeEngine.setGpuAttentionLayer(gpuAttentionLayer)
        check(NativeEngine.setFfnSplits(ffnK4Split, ffnK5Split)) { "Invalid FFN split-K" }
        check(NativeEngine.setLayerBatch(layerBatch)) { "Invalid layer batch" }
        NativeEngine.setUnroll2(unroll2)
        NativeEngine.setSubgroupHead(subgroupHead)
        check(NativeEngine.setHeadRows(headRows)) { "Invalid head rows" }
        NativeEngine.setGpuTimestamps(gpuTimestamps)
        NativeEngine.setParallelNorm(parallelNorm)
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
            check(NativeEngine.setSplitK(splitK)) { "Invalid split-K" }
            NativeEngine.setFastHead(fastHead)
            NativeEngine.setGpuConvLayer(gpuConvLayer)
            NativeEngine.setExperimentalKernels(cooperativeK5, specializedMcg)
            NativeEngine.setGpuGreedy(gpuGreedy)
            NativeEngine.setGpuAttentionLayer(gpuAttentionLayer)
            check(NativeEngine.setFfnSplits(ffnK4Split, ffnK5Split)) { "Invalid FFN split-K" }
            check(NativeEngine.setLayerBatch(layerBatch)) { "Invalid layer batch" }
            NativeEngine.setUnroll2(unroll2)
            NativeEngine.setSubgroupHead(subgroupHead)
            check(NativeEngine.setHeadRows(headRows)) { "Invalid head rows" }
            NativeEngine.setGpuTimestamps(gpuTimestamps)
            NativeEngine.setParallelNorm(parallelNorm)
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
            environmentBefore = environment()
            main.post { diagnostics = null }
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

    fun benchmark() {
        if (busy || !ready) return
        task {
            environmentBefore = environment()
            main.post { diagnostics = null; status = "Benchmark…" }
            val result = NativeEngine.benchmark(callback)
            if (result.isNotEmpty()) check(stopped.get()) { result }
        }
    }

    override fun onCleared() {
        stopped.set(true)
        NativeEngine.cancel()
        worker.execute { NativeEngine.unload() }
        worker.shutdown()
    }
}
