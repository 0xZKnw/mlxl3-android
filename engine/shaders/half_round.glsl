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

