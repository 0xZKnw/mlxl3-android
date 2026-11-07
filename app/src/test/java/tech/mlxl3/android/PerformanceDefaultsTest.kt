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
        assertTrue(settings.parallelNorm)
        assertFalse(settings.gpuTimestamps)
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

    @Test fun s24MigrationReplacesOldChoicesOnlyOnce() {
        val profile = PerformanceDefaults.forDevice("Samsung", "SM-S921B")
        val old = mapOf<String, Any>("parallel_norm" to false, "gpu_timestamps" to true,
            "wave_head" to true, "split_k" to 2)
        val changes = profile.migrationPreferences(old)
        assertEquals(setOf("parallel_norm", "gpu_timestamps",
            PerformanceDefaults.S24_NORM_PROFILE_MIGRATION), changes.keys)
        val upgraded = old + changes
        val first = profile.withPreferences(upgraded)
        assertTrue(first.parallelNorm)
        assertFalse(first.gpuTimestamps)
        assertEquals(2, first.splitK)
        assertEquals(true, upgraded["wave_head"])
        val later = upgraded + mapOf("parallel_norm" to false, "gpu_timestamps" to true)
        assertTrue(profile.migrationPreferences(later).isEmpty())
        assertFalse(profile.withPreferences(later).parallelNorm)
        assertTrue(profile.withPreferences(later).gpuTimestamps)
    }

    @Test fun migrationDoesNotTouchOtherDevices() {
        for ((maker, model) in listOf("samsung" to "SM-S921U", "samsung" to "SM-S926B",
            "other" to "SM-S921B", "" to "")) {
            val profile = PerformanceDefaults.forDevice(maker, model)
            assertTrue(profile.migrationPreferences(emptyMap<String, Any>()).isEmpty())
            assertTrue(profile.migrationPreferences(mapOf("parallel_norm" to false,
                "gpu_timestamps" to true)).isEmpty())
            assertFalse(profile.parallelNorm)
            assertTrue(profile.gpuTimestamps)
        }
    }

    @Test fun newS24InstallGetsMigrationMarkerAndMeasuredDefaults() {
        val profile = PerformanceDefaults.forDevice("samsung", "SM-S921B")
        val changes = profile.migrationPreferences(emptyMap<String, Any>())
        assertEquals(true, changes[PerformanceDefaults.S24_NORM_PROFILE_MIGRATION])
        assertTrue(profile.withPreferences(changes).parallelNorm)
        assertFalse(profile.withPreferences(changes).gpuTimestamps)
    }

    @Test fun malformedValuesUseDeviceDefaultsWithoutChangingOtherChoices() {
        val settings = PerformanceDefaults.forDevice("samsung", "SM-S921B").withPreferences(mapOf(
            "split_k" to 3, "ffn_k4_split" to "8", "layer_batch" to 3,
            "gpu_greedy" to false, "gpu_conv_layer" to "true",
        ))
        assertEquals(8, settings.splitK)
        assertEquals(4, settings.ffnK4Split)
        assertEquals(4, settings.layerBatch)
        assertFalse(settings.gpuGreedy)
        assertTrue(settings.gpuConvLayer)
    }
    @Test fun experimentalLayerBatchesPersistWithoutChangingMeasuredDefault() {
        val profile = PerformanceDefaults.forDevice("samsung", "SM-S921B")
        assertEquals(4, profile.layerBatch)
        for (batch in listOf(1, 2, 4, 8, 12, 24)) {
            assertEquals(batch, profile.withPreferences(mapOf("layer_batch" to batch)).layerBatch)
        }
        for (batch in listOf(0, 3, 16, 25, Int.MAX_VALUE)) {
            assertEquals(4, profile.withPreferences(mapOf("layer_batch" to batch)).layerBatch)
        }
    }

}
