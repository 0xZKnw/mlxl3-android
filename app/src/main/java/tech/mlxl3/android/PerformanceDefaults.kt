package tech.mlxl3.android

/** Requested settings only: native capability checks still decide the effective kernels. */
internal data class PerformanceDefaults(
    val profile: String = "portable",
    val splitK: Int = 4,
    val ffnK4Split: Int = 0,
    val layerBatch: Int = 1,
    val gpuConvLayer: Boolean = false,
    val gpuAttentionLayer: Boolean = false,
    val cooperativeK5: Boolean = false,
    val gpuGreedy: Boolean = false,
    val parallelNorm: Boolean = false,
    val gpuTimestamps: Boolean = true,
) {
    /** Explicit choices, including false and global (0), always override defaults. */
    fun withPreferences(values: Map<String, *>): PerformanceDefaults = copy(
        splitK = (values["split_k"] as? Int)?.takeIf { it in listOf(1, 2, 4, 8) } ?: splitK,
        ffnK4Split = (values["ffn_k4_split"] as? Int)?.takeIf { it in listOf(0, 1, 2, 4, 8) } ?: ffnK4Split,
        layerBatch = (values["layer_batch"] as? Int)?.takeIf { it in listOf(1, 2, 4, 8, 12, 24) } ?: layerBatch,
        gpuConvLayer = values["gpu_conv_layer"] as? Boolean ?: gpuConvLayer,
        gpuAttentionLayer = values["gpu_attention_layer"] as? Boolean ?: gpuAttentionLayer,
        cooperativeK5 = values["cooperative_k5"] as? Boolean ?: cooperativeK5,
        gpuGreedy = values["gpu_greedy"] as? Boolean ?: gpuGreedy,
        parallelNorm = values["parallel_norm"] as? Boolean ?: parallelNorm,
        gpuTimestamps = values["gpu_timestamps"] as? Boolean ?: gpuTimestamps,
    )

    /**
     * One-time, explicitly requested S24 profile upgrade. Apply before withPreferences.
     * Only these two previously persisted options are replaced. Later choices win.
     */
    fun migrationPreferences(values: Map<String, *>): Map<String, Boolean> =
        if (profile == "s24-exynos-measured" && values[S24_NORM_PROFILE_MIGRATION] != true) {
            mapOf(
                "parallel_norm" to true,
                "gpu_timestamps" to false,
                S24_NORM_PROFILE_MIGRATION to true,
            )
        } else {
            emptyMap()
        }

    companion object {
        const val S24_NORM_PROFILE_MIGRATION = "s24_parallel_norm_profile_v1"
        fun forDevice(manufacturer: String, model: String): PerformanceDefaults =
            if (manufacturer.equals("samsung", ignoreCase = true) && model == "SM-S921B") {
                // Only this Exynos S24 variant has supplied on-device measurements.
                // Do not infer qualification of Snapdragon, Mali or other Xclipse devices.
                PerformanceDefaults("s24-exynos-measured", 8, 4, 4, true, true, true, true,
                    parallelNorm = true, gpuTimestamps = false)
            } else {
                PerformanceDefaults()
            }
    }
}
