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
) {
    /** Explicit choices, including false and global (0), always override defaults. */
    fun withPreferences(values: Map<String, *>): PerformanceDefaults = copy(
        splitK = (values["split_k"] as? Int)?.takeIf { it in listOf(1, 2, 4, 8) } ?: splitK,
        ffnK4Split = (values["ffn_k4_split"] as? Int)?.takeIf { it in listOf(0, 1, 2, 4, 8) } ?: ffnK4Split,
        layerBatch = (values["layer_batch"] as? Int)?.takeIf { it in listOf(1, 2, 4) } ?: layerBatch,
        gpuConvLayer = values["gpu_conv_layer"] as? Boolean ?: gpuConvLayer,
        gpuAttentionLayer = values["gpu_attention_layer"] as? Boolean ?: gpuAttentionLayer,
        cooperativeK5 = values["cooperative_k5"] as? Boolean ?: cooperativeK5,
        gpuGreedy = values["gpu_greedy"] as? Boolean ?: gpuGreedy,
    )

    companion object {
        fun forDevice(manufacturer: String, model: String): PerformanceDefaults =
            if (manufacturer.equals("samsung", ignoreCase = true) && model == "SM-S921B") {
                // Only this Exynos S24 variant has supplied on-device measurements.
                // Do not infer qualification of Snapdragon, Mali or other Xclipse devices.
                PerformanceDefaults("s24-exynos-measured", 8, 4, 4, true, true, true, true)
            } else {
                PerformanceDefaults()
            }
    }
}
