package tech.mlxl3.android

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.graphics.asAndroidBitmap
import androidx.test.platform.app.InstrumentationRegistry
import android.graphics.Bitmap
import android.content.ContentValues
import android.provider.MediaStore
import android.os.Build
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
        if (Build.VERSION.SDK_INT >= 29) {
            // Public test output survives Gradle's automatic uninstall on the disposable emulator.
            val metadata = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, "conversation.png")
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MLXL3-tests")
            }
            val uri = requireNotNull(context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, metadata))
            requireNotNull(context.contentResolver.openOutputStream(uri)).use {
                compose.onRoot().captureToImage().asAndroidBitmap().compress(Bitmap.CompressFormat.PNG, 100, it)
            }
        }
    }
}
