#include "half_round.glsl"

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
