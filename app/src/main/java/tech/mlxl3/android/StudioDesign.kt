package tech.mlxl3.android

import android.graphics.Bitmap
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.Typeface
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.background
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ColorFilter
import androidx.compose.ui.graphics.ColorMatrix
import androidx.compose.ui.graphics.CompositingStrategy
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.core.graphics.createBitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Calendar
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin

/** Design tokens of MLXL3 Desktop v2.0.5 (`StudioTheme`, dark appearance). */
internal object Studio {
    val Canvas = Color(0xFF121212)
    val Composer = Color(0xFF1B1B1B)
    val Menu = Color(0xFF1E1E1F)
    val Ink = Color(0xFFE8E8E8)
    val Contrast = Color.White
    val Panel = Contrast.copy(alpha = 0.035f)
    /** Toolbar controls over the aurora: dark glass keeps labels legible on the glow. */
    val Glass = Color(0xFF121212).copy(alpha = 0.55f)
    val Edge = Contrast.copy(alpha = 0.085f)
    val Quiet = Contrast.copy(alpha = 0.58f)
    val Secondary = Contrast.copy(alpha = 0.72f)
    /** "Iris", the default Desktop accent in dark mode. */
    val Accent = Color(0xFFBBA4F3)
    val Ready = Color(0xFF6BFFB0)
    val Loading = Color(0xFFFFC961)
    val Failed = Color(0xFFFF5C63)
}

internal enum class EngineState { Idle, Loading, Ready, Generating, Failed }

@Composable
internal fun StatusDot(state: EngineState, modifier: Modifier = Modifier) {
    val color = when (state) {
        EngineState.Ready -> Studio.Ready
        EngineState.Generating -> Studio.Accent
        EngineState.Loading -> Studio.Loading
        EngineState.Failed -> Studio.Failed
        EngineState.Idle -> Studio.Quiet
    }
    Box(modifier.size(7.dp).background(color, CircleShape))
}

/** Scrolling text fades out near the chrome instead of stopping on a hard edge. */
internal fun Modifier.edgeFade(top: Dp, bottom: Dp): Modifier =
    graphicsLayer { compositingStrategy = CompositingStrategy.Offscreen }
        .drawWithContent {
            drawContent()
            val h = size.height
            if (h <= 0f) return@drawWithContent
            val t = (top.toPx() / h).coerceIn(0f, 0.5f)
            val b = (bottom.toPx() / h).coerceIn(0f, 0.5f)
            drawRect(
                Brush.verticalGradient(
                    0f to Color.Transparent, t to Color.Black,
                    1f - b to Color.Black, 1f to Color.Transparent,
                ),
                blendMode = BlendMode.DstIn,
            )
        }

/** Welcome titles from Desktop v2.0.5: tied to the time of day, one draw in four any-hour. */
internal object WelcomeGreeting {
    private val morning = listOf("Bonjour", "Bien réveillé ?", "Un café, une idée ?", "On commence par quoi ?")
    private val afternoon = listOf("Bon après-midi", "On s’y remet ?", "Qu’est-ce qu’on construit ?")
    private val evening = listOf("Bonsoir", "Encore une idée ?", "Une dernière pour ce soir ?")
    private val night = listOf("Encore debout ?", "Les idées viennent la nuit", "Nuit blanche ?")
    private val anytime = listOf("De quoi parle-t-on ?", "Par où commencer ?", "Je vous écoute")

    fun pool(hour: Int): List<String> = when (hour) {
        in 5..11 -> morning
        in 12..17 -> afternoon
        in 18..22 -> evening
        else -> night
    }

    fun pick(hour: Int, seed: Long): String {
        val s = seed.toULong()
        if (s % 4u == 3uL) return anytime[((s / 4u) % anytime.size.toULong()).toInt()]
        val timed = pool(hour)
        return timed[((s / 4u) % timed.size.toULong()).toInt()]
    }

    fun now(seed: Long): String = pick(Calendar.getInstance().get(Calendar.HOUR_OF_DAY), seed)
}

/**
 * The still "print" of the Desktop aurora: a violet light column with a white-hot
 * curved foot, rendered as monospace glyphs whose density follows the light.
 * It is drawn once and never animated, so it costs no GPU time during inference.
 */
internal object AuroraArtwork {
    const val WIDTH = 740
    const val HEIGHT = 620

    /** Same light field as `StudioAurora.light` (x in -1…1, y in 0…1, top down). */
    fun light(x: Double, y: Double): DoubleArray {
        val foot = 0.70 + 0.20 * x * x + 0.027 * sin(4.5 * x)
        val d = foot - y
        val span = 0.25 + 0.16 * max(0.0, d)
        val column = exp(-(x / span).pow(2) - ((d - 0.22) / 0.43).pow(2))
        val rim = exp(-(x / 0.64).pow(2) - (d / 0.064).pow(2))
        val haze = exp(-(x / 0.66).pow(2) - ((d - 0.17) / 0.46).pow(2))
        val hot = exp(-(x / 0.32).pow(2) - ((d - 0.055) / 0.10).pow(2))
        val pink = exp(-((d - 0.14) / 0.19).pow(2))
        val r = (0.20 + 0.78 * pink + 0.64 * rim + 0.30 * hot).coerceIn(0.0, 1.0)
        val g = (0.022 + 0.10 * max(0.0, d) + 0.61 * hot).coerceIn(0.0, 1.0)
        val b = (0.98 - 0.55 * rim - 0.61 * hot).coerceIn(0.0, 1.0)
        val fade = min(1.0, max(0.0, y * 8)) * min(1.0, max(0.0, (1 - abs(x)) * 8)) *
            min(1.0, max(0.0, (1 - y) * 9))
        val a = min(1.0, 0.91 * column + 0.94 * rim + 0.24 * haze) * fade *
            exp(-(max(0.0, -d) / 0.09).pow(2))
        return doubleArrayOf(r, g, b, a)
    }

    /** Rendered once per process and shared across recreations (rotation, theme). */
    val still: ImageBitmap by lazy { render().asImageBitmap() }

    private fun channel(v: Double) = (v * 255).roundToInt().coerceIn(0, 255)

    /** Renders the print at [scale]× the 740×620 artwork (2× matches Desktop). */
    fun render(scale: Int = 2): Bitmap {
        val gw = WIDTH / 2
        val gh = HEIGHT / 2
        val pixels = IntArray(gw * gh)
        for (row in 0 until gh) for (col in 0 until gw) {
            val (r, g, b, a) = light(2 * (col + 0.5) / gw - 1, (row + 0.5) / gh)
            pixels[row * gw + col] = android.graphics.Color.argb(channel(a), channel(r), channel(g), channel(b))
        }
        val glow = Bitmap.createBitmap(pixels, gw, gh, Bitmap.Config.ARGB_8888)
        val plate = createBitmap(WIDTH * scale, HEIGHT * scale)
        val canvas = android.graphics.Canvas(plate)
        canvas.scale(scale.toFloat(), scale.toFloat())
        canvas.drawBitmap(glow, Rect(0, 0, gw, gh), Rect(0, 0, WIDTH, HEIGHT),
            Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
        glow.recycle()
        val alphabet = ".:-=+xX"
        val type = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            typeface = Typeface.MONOSPACE
            textSize = 7f
        }
        for (row in 0 until HEIGHT / 8) for (col in 0 until WIDTH / 5) {
            val x = (col * 5 + 2).toDouble()
            val y = (row * 8 + 5).toDouble()
            val (r, g, b, a) = light(2 * x / WIDTH - 1, y / HEIGHT)
            if (a <= 0.13) continue
            val grain = sin((col * 17 + row * 31).toDouble()) * 0.06
            val density = min(1.0, max(0.0, a + 0.13 * sin(y * 0.09) + grain))
            val glyph = alphabet[(density * (alphabet.length - 1)).toInt()]
            type.color = android.graphics.Color.argb(channel(a * 0.47), channel(r * 0.60 + 0.40),
                channel(g * 0.58 + 0.42), channel(b * 0.62 + 0.38))
            canvas.drawText(glyph.toString(), x.toFloat(), (y + 2).toFloat(), type)
        }
        return plate
    }
}

/** `(x, scale, hue°, strength)` of each copy of the print, as on Desktop. */
private val curtains = listOf(
    floatArrayOf(0.045f, 0.91f, -38f, 0.61f),
    floatArrayOf(0.77f, 1.08f, 0f, 1f),
    floatArrayOf(1.05f, 0.93f, 9f, 0.61f),
    floatArrayOf(0.34f, 0.67f, -26f, 0.52f),
)

/** Luminance-preserving hue rotation (same matrix as SVG `feColorMatrix hueRotate`). */
internal fun hueRotation(degrees: Float): ColorMatrix {
    val r = Math.toRadians(degrees.toDouble())
    val c = cos(r).toFloat()
    val s = sin(r).toFloat()
    return ColorMatrix(floatArrayOf(
        0.213f + c * 0.787f - s * 0.213f, 0.715f - c * 0.715f - s * 0.715f, 0.072f - c * 0.072f + s * 0.928f, 0f, 0f,
        0.213f - c * 0.213f + s * 0.143f, 0.715f + c * 0.285f + s * 0.140f, 0.072f - c * 0.072f - s * 0.283f, 0f, 0f,
        0.213f - c * 0.213f - s * 0.787f, 0.715f - c * 0.715f + s * 0.715f, 0.072f + c * 0.928f + s * 0.072f, 0f, 0f,
        0f, 0f, 0f, 1f, 0f,
    ))
}

private val curtainFilters = curtains.map { ColorFilter.colorMatrix(hueRotation(it[2])) }

/**
 * Background aurora of Desktop v2.0.5 (`StudioAurora` still mode). The print is
 * generated off the main thread; the canvas stays plain until it is ready.
 */
@Composable
internal fun StudioAurora(modifier: Modifier = Modifier, opacity: Float = 0.76f) {
    var art by remember { mutableStateOf<ImageBitmap?>(null) }
    LaunchedEffect(Unit) {
        art = withContext(Dispatchers.Default) { AuroraArtwork.still }
    }
    val image = art ?: return
    Canvas(modifier.fillMaxSize().graphicsLayer { alpha = opacity }) {
        val w = size.width
        val h = size.height
        // Phones are narrow: a deeper band than Desktop's 0.42 keeps the foot below the toolbar.
        val band = min(h * 0.48f, w * 0.56f)
        curtains.forEachIndexed { index, curtain ->
            val width = min(w * 0.68f, band * 2.1f) * curtain[1]
            val depth = band * curtain[1]
            val cx = w * (curtain[0] + 0.026f * sin(index.toFloat()))
            val cy = depth * (0.31f + 0.035f * sin(index.toFloat()))
            drawImage(
                image,
                srcOffset = IntOffset.Zero,
                srcSize = IntSize(image.width, image.height),
                dstOffset = IntOffset((cx - width / 2).roundToInt(), (cy - depth / 2).roundToInt()),
                dstSize = IntSize(width.roundToInt(), depth.roundToInt()),
                alpha = curtain[3],
                colorFilter = curtainFilters[index],
                filterQuality = FilterQuality.High,
            )
        }
    }
}
