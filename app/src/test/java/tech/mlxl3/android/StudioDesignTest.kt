package tech.mlxl3.android

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StudioDesignTest {
    @Test fun greetingFollowsTheDesktopPoolsAndBoundaries() {
        assertEquals(WelcomeGreeting.pool(5), WelcomeGreeting.pool(11))
        assertEquals("Bonjour", WelcomeGreeting.pick(5, 0))
        assertEquals("Bon après-midi", WelcomeGreeting.pick(12, 0))
        assertEquals("Bonsoir", WelcomeGreeting.pick(18, 0))
        assertEquals("Encore debout ?", WelcomeGreeting.pick(23, 0))
        assertEquals("Encore debout ?", WelcomeGreeting.pick(4, 0))
        // Out-of-range hours fall back to the night pool, as on Desktop.
        assertEquals("Encore debout ?", WelcomeGreeting.pick(-1, 0))
        assertEquals("Encore debout ?", WelcomeGreeting.pick(99, 0))
        // One draw in four uses the any-hour pool.
        assertEquals("De quoi parle-t-on ?", WelcomeGreeting.pick(9, 3))
        assertEquals("Par où commencer ?", WelcomeGreeting.pick(9, 7))
        assertEquals("Bien réveillé ?", WelcomeGreeting.pick(9, 4))
    }

    @Test fun greetingIsTotalForEverySeedAndHour() {
        // Negative seeds are read as unsigned, like Swift's UInt64 hash seed.
        val seeds = listOf(Long.MIN_VALUE, -1L, 0L, 1L, 2L, 3L, Long.MAX_VALUE) + (0 until 200).map { it * 7919L - 500 }
        for (hour in -2..25) for (seed in seeds) {
            val phrase = WelcomeGreeting.pick(hour, seed)
            assertTrue(phrase, phrase.isNotBlank())
        }
    }

    @Test fun auroraLightIsBoundedAndShapedLikeTheDesktopPrint() {
        var brightest = 0.0
        var lit = 0
        for (row in 0..200) for (col in -100..100) {
            val (r, g, b, a) = AuroraArtwork.light(col / 100.0, row / 200.0)
            for (v in doubleArrayOf(r, g, b, a)) assertTrue("$col,$row: $v", v.isFinite() && v in 0.0..1.0)
            if (a > 0.13) lit++
            brightest = maxOf(brightest, a)
        }
        assertTrue("the print must contain glyph cells", lit > 2000)
        assertTrue(brightest > 0.9)
        // Fully transparent at the frame edges and at the very top.
        for (t in listOf(-1.0, 1.0)) assertEquals(0.0, AuroraArtwork.light(t, 0.5)[3], 1e-12)
        assertEquals(0.0, AuroraArtwork.light(0.0, 0.0)[3], 1e-12)
        assertEquals(0.0, AuroraArtwork.light(0.0, 1.0)[3], 1e-12)
        // The white-hot foot sits low and centred; the column above it is violet.
        val foot = AuroraArtwork.light(0.0, 0.66)
        val column = AuroraArtwork.light(0.0, 0.35)
        assertTrue(foot[1] > column[1] && foot[3] > 0.8)
        assertTrue(column[2] > column[1] && column[0] > column[1])
    }

    @Test fun hueRotationKeepsGreysAndIsIdentityAtZero() {
        val identity = hueRotation(0f).values
        for (i in 0 until 20) assertEquals(if (i % 6 == 0) 1f else 0f, identity[i], 1e-6f)
        for (deg in listOf(-38f, -26f, 9f, 180f)) {
            val m = hueRotation(deg).values
            for (row in 0 until 3) assertEquals("row $row at $deg°", 1f, m[row * 5] + m[row * 5 + 1] + m[row * 5 + 2], 1e-5f)
        }
    }
}
