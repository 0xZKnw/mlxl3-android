// Native FP16 pair addition for Default/MCG; Mul1 preserves the exact LUT path.
layout(set = 0, binding = 3, std430) readonly buffer Codebooks { uint codebooks[]; };
float mul1Lookup(uint state) {
    return unpackHalf2x16(codebooks[65536u + state / 2u])[state % 2u];
}

vec2 codewordPair(uint s0, uint s1, uint cb) {
    if (cb == 2u) return vec2(mul1Lookup(s0), mul1Lookup(s1));
    uvec2 state = uvec2(s0, s1);
    uvec2 z = cb == 0u ? state * 89226354u + 64248484u : state * 0xcbac1fedu;
    f16vec2 a = unpackFloat2x16(0x3b603b60u ^ (z.x & 0x8fff8fffu));
    f16vec2 b = unpackFloat2x16(0x3b603b60u ^ (z.y & 0x8fff8fffu));
    precise f16vec2 value = f16vec2(a.x, b.x) + f16vec2(a.y, b.y);
    return vec2(value);
}
