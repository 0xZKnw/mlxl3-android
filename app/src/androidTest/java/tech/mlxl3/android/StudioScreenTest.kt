package tech.mlxl3.android

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.provider.MediaStore
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.SystemBarStyle
import androidx.activity.enableEdgeToEdge
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class StudioScreenTest {
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()
    private val answer = "Oui. LFM3 tourne directement sur ton téléphone, avec le moteur EXL3 de MLXL3."
    private val conversation get() = listOf(ChatMessage("user", "Salut, tu fonctionnes en local ?"), ChatMessage("assistant", answer))

    @Suppress("DEPRECATION") // Match the production resize mode, including Android 9/10.
    @Before fun prepareWindow() {
        compose.activityRule.scenario.onActivity {
            it.enableEdgeToEdge(statusBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT),
                navigationBarStyle = SystemBarStyle.dark(android.graphics.Color.TRANSPARENT))
            it.window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        }
        InstrumentationRegistry.getInstrumentation().uiAutomation
            .executeShellCommand("settings put secure show_ime_with_hard_keyboard 1").close()
    }

    @Test fun sendsTheDraftAndClearsTheInput() {
        var sent = ""
        compose.setContent { StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false, {}, {}, {}, { sent = it }, {}, {}) }
        compose.onNodeWithContentDescription("Message LFM3").performTextInput("Hello")
        compose.onNodeWithContentDescription("Envoyer").performClick()
        assertEquals("Hello", sent)
        compose.onNodeWithText("Hello").assertDoesNotExist()
    }

    @Test fun busyStateDisablesInputAndExposesCancellation() {
        compose.setContent { StudioScreen(emptyList(), true, false, "Chargement…", "", null, false, {}, {}, {}, {}, {}, {}) }
        compose.onNodeWithContentDescription("Arrêter la génération").assertIsEnabled()
        compose.onNodeWithContentDescription("Message LFM3").assertIsNotEnabled()
        compose.onAllNodesWithText("Chargement…").onFirst().assertIsDisplayed()
    }

    @Test fun conversationTextHasVisibleContrast() {
        compose.setContent { StudioScreen(conversation, false, true, "Ready", "Test GPU", null, false, {}, {}, {}, {}, {}, {}) }
        compose.onNodeWithText(answer).assertIsDisplayed()
        val pixels = compose.onNodeWithText(answer).captureToImage().toPixelMap()
        var lightPixels = 0
        for (y in 0 until pixels.height) for (x in 0 until pixels.width) {
            val p = pixels[x, y]
            if (p.red > 0.6f && p.green > 0.6f && p.blue > 0.6f) lightPixels++
        }
        assertTrue("Assistant text must contrast with the dark canvas", lightPixels > 100)
        screenshot("conversation.png")
    }

    @Test fun composerStaysAboveTheRealKeyboardWhileTyping() {
        compose.setContent { StudioScreen(conversation, false, true, "Ready", "Test GPU", null, false, {}, {}, {}, {}, {}, {}) }
        compose.onNodeWithContentDescription("Message LFM3").performClick()
        compose.waitUntil(15_000) {
            ViewCompat.getRootWindowInsets(compose.activity.window.decorView)?.isVisible(WindowInsetsCompat.Type.ime()) == true
        }
        compose.onNodeWithContentDescription("Message LFM3").performTextInput("Première ligne\nDeuxième ligne\nTroisième ligne")
        compose.waitForIdle()
        val view = compose.activity.window.decorView
        val keyboardHeight = requireNotNull(ViewCompat.getRootWindowInsets(view)).getInsets(WindowInsetsCompat.Type.ime()).bottom
        assertTrue("A real software keyboard must be visible", keyboardHeight > 0)
        val composer = compose.onNodeWithTag("composer").fetchSemanticsNode().boundsInWindow
        assertTrue("Composer overlaps the keyboard: $composer", composer.bottom <= view.height - keyboardHeight + 2f)
        assertTrue("The input and send control must remain fully visible", composer.height > 80f)
        compose.onNodeWithContentDescription("Envoyer").assertIsDisplayed().assertIsEnabled()
        screenshot("keyboard.png")
    }

    @Test fun initialModelImportScreen() {
        compose.setContent { StudioScreen(emptyList(), false, false, "Import a model", "", null, false, {}, {}, {}, {}, {}, {}) }
        compose.onNodeWithText("Importer mon modèle").assertIsDisplayed()
        compose.onNodeWithContentDescription("Envoyer").assertIsNotEnabled()
        screenshot("welcome.png")
    }

    private fun screenshot(name: String) {
        if (Build.VERSION.SDK_INT >= 29) {
            val context = InstrumentationRegistry.getInstrumentation().targetContext
            val metadata = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, name)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MLXL3-tests")
            }
            val uri = requireNotNull(context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, metadata))
            requireNotNull(context.contentResolver.openOutputStream(uri)).use {
                requireNotNull(InstrumentationRegistry.getInstrumentation().uiAutomation.takeScreenshot())
                    .compress(Bitmap.CompressFormat.PNG, 100, it)
            }
        }
    }
}
