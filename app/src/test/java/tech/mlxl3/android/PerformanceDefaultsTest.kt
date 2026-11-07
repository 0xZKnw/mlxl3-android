package tech.mlxl3.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PerformanceDefaultsTest {
    @Test fun measuredS24GetsRequestedProfile() {
        val settings = PerformanceDefaults.forDevice("Samsung", "SM-S921B").withPreferences(emptyMap<String, Any>())
        assertEquals(8, settings.splitK)
        assertEquals(4, settings.ffnK4Split)
        assertEquals(4, settings.layerBatch)
        assertTrue(settings.gpuConvLayer)
        assertTrue(settings.gpuAttentionLayer)
        assertTrue(settings.cooperativeK5)
        assertTrue(settings.gpuGreedy)
    }

    @Test fun unmeasuredDevicesKeepPortableDefaults() {
        for ((maker, model) in listOf("samsung" to "SM-S921U", "samsung" to "SM-S926B", "other" to "SM-S921B", "" to "")) {
            assertEquals(PerformanceDefaults(), PerformanceDefaults.forDevice(maker, model))
        }
    }

    @Test fun existingChoicesAreNotOverwritten() {
        val settings = PerformanceDefaults.forDevice("samsung", "SM-S921B").withPreferences(mapOf(
            "split_k" to 2, "ffn_k4_split" to 0, "layer_batch" to 1,
            "gpu_conv_layer" to false, "gpu_attention_layer" to false,
            "cooperative_k5" to false, "gpu_greedy" to false,
        ))
        assertEquals(2, settings.splitK)
        assertEquals(0, settings.ffnK4Split)
        assertEquals(1, settings.layerBatch)
        assertFalse(settings.gpuConvLayer)
        assertFalse(settings.gpuAttentionLayer)
        assertFalse(settings.cooperativeK5)
        assertFalse(settings.gpuGreedy)
    }

    @Test fun malformedValuesUseDeviceDefaultsWithoutChangingOtherChoices() {
        val settings = PerformanceDefaults.forDevice("samsung", "SM-S921B").withPreferences(mapOf(
            "split_k" to 3, "ffn_k4_split" to "8", "layer_batch" to 8,
            "gpu_greedy" to false, "gpu_conv_layer" to "true",
        ))
        assertEquals(8, settings.splitK)
        assertEquals(4, settings.ffnK4Split)
        assertEquals(4, settings.layerBatch)
        assertFalse(settings.gpuGreedy)
        assertTrue(settings.gpuConvLayer)
    }
}
