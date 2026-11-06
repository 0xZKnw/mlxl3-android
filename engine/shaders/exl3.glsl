#ifdef EXL3_NATIVE
#include "exl3_native.glsl"
#elif defined(EXL3_REFERENCE)
#include "exl3_procedural.glsl"
#else
layout(set = 0, binding = 3, std430) readonly buffer Codebooks { uint codebooks[]; };
float codeword(uint state, uint cb) { return unpackHalf2x16(codebooks[cb * 32768u + state / 2u])[state % 2u]; }
#endif
