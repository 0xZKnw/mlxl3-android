// Explicit round-to-nearest, ties-to-even: do not rely on driver packHalf rounding.
float halfRound(float value) {
    uint bits = floatBitsToUint(value);
    uint sign = bits & 0x80000000u;
    uint mag = bits & 0x7fffffffu;
    if (mag >= 0x7f800000u) return value;
    if (mag >= 0x477ff000u) return uintBitsToFloat(sign | 0x7f800000u);
    if (mag < 0x38800000u) {
        if (mag <= 0x33000000u) return uintBitsToFloat(sign);
        uint shift = 126u - (mag >> 23u);
        uint mantissa = (mag & 0x7fffffu) | 0x800000u;
        uint base = mantissa >> shift;
        uint tail = mantissa & ((1u << shift) - 1u);
        uint midpoint = 1u << (shift - 1u);
        uint rounded = base + uint(tail > midpoint || (tail == midpoint && (base & 1u) != 0u));
        return unpackHalf2x16((sign >> 16u) | rounded).x;
    }
    uint rounded = (mag + 0xfffu + ((mag >> 13u) & 1u)) & 0xffffe000u;
    return uintBitsToFloat(sign | rounded);
}

float codeword(uint s, uint cb) {
    uint z;
    if (cb == 2u) {
        z = s * 0x83dcd12du;
        uint total = 0x6400u + (z & 255u) + ((z >> 8u) & 255u)
            + ((z >> 16u) & 255u) + (z >> 24u);
        float h = unpackHalf2x16(total).x;
        precise float value = fma(h, unpackHalf2x16(0x1eeeu).x, unpackHalf2x16(0xc931u).x);
        return halfRound(value);
    }
    z = cb == 0u ? s * 89226354u + 64248484u : s * 0xcbac1fedu;
    vec2 pair = unpackHalf2x16(0x3b603b60u ^ (z & 0x8fff8fffu));
    return halfRound(pair.x + pair.y);
}
