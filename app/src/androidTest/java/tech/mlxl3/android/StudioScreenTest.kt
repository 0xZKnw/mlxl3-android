package tech.mlxl3.android

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
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
}
