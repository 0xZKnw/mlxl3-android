package tech.mlxl3.android

import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.util.concurrent.CancellationException

object ModelImport {
    const val MAX_BYTES = 4L * 1024 * 1024 * 1024
    fun accepts(name: String): Boolean =
        name.matches(Regex("[a-zA-Z0-9_.-]+")) &&
            (name == "config.json" || name == "tokenizer.json" || name.endsWith(".safetensors"))

    fun copyBounded(input: InputStream, output: OutputStream, limit: Long, cancelled: () -> Boolean,
                    progress: (Long) -> Unit = {}): Long {
        require(limit >= 0) { "Insufficient model storage budget" }
        val buffer = ByteArray(1024 * 1024)
        var total = 0L
        while (true) {
            if (cancelled()) throw CancellationException("Import cancelled")
            val count = input.read(buffer)
            if (count < 0) return total
            check(count > 0) { "Model provider stopped returning data" }
            check(count.toLong() <= limit - total) { "Model file exceeds its storage budget" }
            output.write(buffer, 0, count)
            total += count
            progress(total)
        }
    }

    fun validateFolder(folder: File) {
        check(File(folder, "config.json").isFile && File(folder, "tokenizer.json").isFile) {
            "Select the EXL3 variant folder containing config.json and tokenizer.json"
        }
        check(folder.listFiles().orEmpty().any { it.name.endsWith(".safetensors") && it.length() > 8 }) {
            "No Safetensors weights found in this folder"
        }
    }
}
