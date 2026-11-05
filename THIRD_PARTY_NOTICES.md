# Third-party notices

`engine/src/codec.rs` is reused from MLXL3, source
`1ac910d0a40efcad049c6e5a8cf081d770723465`. The Desktop artwork and visual
palette also come from MLXL3. See LICENSE (MIT).

The EXL3 format, trellis, codebooks and Hadamard reconstruction derive from
[ExLlamaV3](https://github.com/turboderp-org/exllamav3), MIT licensed,
Copyright (c) 2025 Turboderp. The shaders independently express those algorithms
for Vulkan; no CUDA or Metal runtime is bundled.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The LFM3 operator order follows LiquidAI/LFM3-1.7B,
revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`, `modeling_lfm3.py`,
Copyright 2026 The HuggingFace Team (Apache-2.0), and ExLlamaV3 1.6.6,
revision `bd1f519172406a9560bd5c90e61000691bb69a07`, `architecture/lfm3.py`.
The Apache-2.0 license text is supplied in `engine/APACHE-2.0.txt`.
No model Python file or model weights are included in the APK.
