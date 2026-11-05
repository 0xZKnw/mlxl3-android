package tech.mlxl3.android

import org.junit.Assert.*
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.util.concurrent.CancellationException

class ModelImportTest {
    @Test fun acceptsOnlyModelFilesAndSafeNames() {
        listOf("config.json", "tokenizer.json", "model-00001-of-00002.safetensors").forEach { assertTrue(ModelImport.accepts(it)) }
        listOf("../config.json", "/model.safetensors", "model.py", "tokenizer_config.json", "").forEach { assertFalse(ModelImport.accepts(it)) }
    }
    @Test fun copyRespectsExactBoundaryAndRejectsOverflow() {
        val bytes = byteArrayOf(1, 2, 3, 4)
        val output = ByteArrayOutputStream()
        assertEquals(4L, ModelImport.copyBounded(ByteArrayInputStream(bytes), output, 4, { false }))
        assertArrayEquals(bytes, output.toByteArray())
        assertThrows(IllegalStateException::class.java) { ModelImport.copyBounded(ByteArrayInputStream(bytes), ByteArrayOutputStream(), 3, { false }) }
    }
    @Test fun cancellationAndSilentProviderFail() {
        assertThrows(CancellationException::class.java) { ModelImport.copyBounded(ByteArrayInputStream(byteArrayOf(1)), ByteArrayOutputStream(), 1, { true }) }
        val silent = object : InputStream() { override fun read() = 0; override fun read(b: ByteArray) = 0 }
        assertThrows(IllegalStateException::class.java) { ModelImport.copyBounded(silent, ByteArrayOutputStream(), 4, { false }) }
    }
}
