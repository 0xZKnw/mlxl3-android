#ifdef EXL3_REFERENCE
#include "exl3_procedural.glsl"
#else
layout(set = 0, binding = 3, std430) readonly buffer Codebooks { float codebooks[]; };
float codeword(uint state, uint cb) { return codebooks[cb * 65536u + state]; }
#endif
