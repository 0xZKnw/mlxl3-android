package tech.mlxl3.android

object NativeEngine {
    init { System.loadLibrary("mlxl3_android") }
    fun interface Callback { fun onEvent(json: String) }
    external fun setSplitK(value: Int): Boolean
    external fun setFastHead(enabled: Boolean)
    external fun setGpuConvLayer(enabled: Boolean)
    external fun setExperimentalKernels(k5: Boolean, mcg: Boolean)
    external fun setGpuGreedy(enabled: Boolean)
    external fun setGpuAttentionLayer(enabled: Boolean)
    external fun setFfnSplits(k4: Int, k5: Int): Boolean
    external fun setLayerBatch(value: Int): Boolean
    external fun setExl3Multiwave(enabled: Boolean)
    external fun setUnroll2(enabled: Boolean)
    external fun setSubgroupHead(enabled: Boolean)
    external fun setShuffleHadamard(enabled: Boolean)
    external fun setVectorHead(enabled: Boolean)
    external fun setWaveHead(enabled: Boolean)
    external fun setHeadRows(value: Int): Boolean
    external fun setGpuTimestamps(enabled: Boolean)
    external fun setFusedNorm(enabled: Boolean)
    external fun setSubgroupNorm(enabled: Boolean)
    external fun setParallelNorm(enabled: Boolean)
    external fun load(path: String, callback: Callback): String
    external fun benchmark(callback: Callback): String
    external fun generate(messages: String, maxTokens: Int, callback: Callback): String
    external fun cancel()
    external fun resetCancel()
    external fun unload()
}
