package tech.mlxl3.android

object NativeEngine {
    init { System.loadLibrary("mlxl3_android") }
    fun interface Callback { fun onEvent(json: String) }
    external fun load(path: String, callback: Callback): String
    external fun generate(messages: String, maxTokens: Int, callback: Callback): String
    external fun cancel()
    external fun resetCancel()
    external fun unload()
}
