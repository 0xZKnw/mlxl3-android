package tech.mlxl3.android

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.test.platform.app.InstrumentationRegistry
import android.graphics.Bitmap
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Rule
import org.junit.Test

class StudioScreenTest {
    @get:Rule val compose = createComposeRule()
    @Test fun sendsTheDraftAndExposesCancellation() {
        var sent = ""
        compose.setContent { StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false, {}, {}, {}, { sent = it }, {}, {}) }
        compose.onNodeWithContentDescription("Message LFM3").performTextInput("Hello")
        compose.onNodeWithText("Send").performClick()
        assertEquals("Hello", sent)
        compose.onNodeWithText("Hello").assertDoesNotExist()
    }
    @Test fun missingModelAndBusyStatesDisableSending() {
        compose.setContent { StudioScreen(emptyList(), true, false, "Loading…", "", null, false, {}, {}, {}, {}, {}, {}) }
        compose.onNodeWithText("Stop").assertIsEnabled()
        compose.onNodeWithContentDescription("Message LFM3").assertIsNotEnabled()
        compose.onNodeWithText("Loading…").assertIsDisplayed()
    }
    @Test fun conversationMatchesDesktopPaletteAndLayout() {
        compose.setContent {
            StudioScreen(listOf(ChatMessage("user", "Salut, tu fonctionnes en local ?"),
                ChatMessage("assistant", "Oui. LFM3 tourne directement sur ton téléphone, avec le moteur EXL3 de MLXL3.")),
                false, true, "LFM3-1.7B · EXL3 · Vulkan", "Xclipse · preview", null, false,
                {}, {}, {}, {}, {}, {})
        }
        compose.onNodeWithText("Salut, tu fonctionnes en local ?").assertIsDisplayed()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val folder = File(context.getExternalFilesDir(null), "screenshots").also { it.mkdirs() }
        File(folder, "conversation.png").outputStream().use {
            compose.onRoot().captureToImage().asAndroidBitmap().compress(Bitmap.CompressFormat.PNG, 100, it)
        }
    }
}
