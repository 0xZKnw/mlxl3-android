# First S24 validation

Status: not executed. The first target is the Galaxy S24 Exynos 2400/Xclipse
940; record Android version, driver/device name, physical RAM and APK SHA-256.
Tests on Mesa or MoltenVK do not validate the Xclipse driver.

Use a pinned complete LFM3-1.7B-EXL3 checkpoint with recorded file hashes.
Prefer 4bpw for the first load. The current exported weights retain BF16
embeddings/head and FP16 EXL3 projection boundaries. CPU reductions and
Vulkan reductions may differ from ExLlamaV3; matching text alone is not
numeric parity.

Before a speed campaign:

1. Install the APK from an inspected successful CI commit; verify its checksum.
2. Exercise missing files, wrong model, truncated weights, insufficient storage,
   import cancellation, load cancellation and repeated load. A failure must
   leave the previous model files intact.
3. Record model load outcome and memory. Run short prompts, multi-turn chat,
   a full 256-token output, stop during prefill/decode, then a fresh chat.
4. Compare token IDs, nonempty finite logits and per-layer recurrent/KV states
   with an independent reference on the **same EXL3 checkpoint**. No full-model
   numeric tolerance has been qualified yet; record max/mean errors and
   near-tied argmax differences before adopting a tolerance. The initial app
   does not expose a state/logit export interface; add that targeted diagnostic
   before claiming model parity. It is a remaining qualification step.
5. Only after numeric qualification, pre-register a speed protocol in opti.md:
   prompts, context, token budget, warmup, repetitions, power/temperature,
   time to first token, decode tokens/s and process memory. Preserve failures.

The UI runs a single foreground task and preserves its ViewModel during
rotation. Android process termination unloads the model. This initial build
has no background inference service or conversation database.
