package tech.mlxl3.android

import android.content.ContentValues
import android.graphics.Bitmap
import android.os.Build
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.compose.ui.graphics.toPixelMap
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.layout.positionOnScreen
import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.test.platform.app.InstrumentationRegistry
import android.view.KeyEvent
import org.junit.Assert.*
import org.junit.Before
import org.junit.Rule
import org.junit.Test

class StudioScreenTest {
    @get:Rule val compose = createAndroidComposeRule<ComponentActivity>()
    private val answer = "Oui. LFM3 tourne directement sur ton téléphone, avec le moteur EXL3 de MLXL3."
    private val conversation get() = listOf(ChatMessage("user", "Salut, tu fonctionnes en local ?"), ChatMessage("assistant", answer, "12,5 tok/s"))

    @Before fun prepareWindow() {
        compose.activityRule.scenario.onActivity { it.configureStudioWindow() }
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

    @Test fun copiesPerformanceReportFromTheModelMenu() {
        val report = "{\"source_sha\":\"test\",\"wait_ns\":42}"
        compose.setContent { StudioScreen(conversation, false, true, "Ready", "Test GPU", null, false, {}, {}, {}, {}, {}, {}, report) }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Copier le diagnostic de performances").performClick()
        compose.runOnIdle {
            val clipboard = compose.activity.getSystemService(android.content.ClipboardManager::class.java)
            assertEquals(report, clipboard.primaryClip?.getItemAt(0)?.text.toString())
            clipboard.clearPrimaryClip()
        }
    }

    @Test fun gpuProfilingToggleCanBeChangedAndRestored() {
        val enabled = androidx.compose.runtime.mutableStateOf(true)
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, gpuTimestamps = enabled.value,
                onGpuTimestamps = { enabled.value = it })
        }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Profilage GPU : complet · recharger").performScrollTo().performClick()
        compose.runOnIdle { assertFalse(enabled.value) }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Profilage GPU : désactivé · recharger").performScrollTo().performClick()
        compose.runOnIdle { assertTrue(enabled.value) }
    }

    @Test fun measuredOptionsAreHiddenButRecoverableWithoutChangingSettings() {
        var changed = false
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, gpuConvLayer = true,
                onGpuConvLayer = { changed = true })
        }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Couches convolution GPU : oui · recharger").assertDoesNotExist()
        compose.onNodeWithText("Tête une wave/ligne (exp.) : non · recharger").assertDoesNotExist()
        compose.onNodeWithText("Tête vectorisée (exp.) : non · recharger").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Couches convolution GPU : oui · recharger").performScrollTo().assertIsDisplayed()
        compose.onNodeWithText("Retour aux expériences").performScrollTo().performClick()
        compose.onNodeWithText("Couches convolution GPU : oui · recharger").assertDoesNotExist()
        compose.runOnIdle { assertFalse(changed) }
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Couches convolution GPU : oui · recharger").performScrollTo().performClick()
        compose.runOnIdle { assertTrue(changed) }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Retour aux expériences").assertDoesNotExist()
        compose.onNodeWithText("Couches convolution GPU : oui · recharger").assertDoesNotExist()
    }

    @Test fun shuffleHadamardExperimentCanBeToggledAndRestored() {
        val enabled = androidx.compose.runtime.mutableStateOf(false)
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, shuffleHadamard = enabled.value,
                onShuffleHadamard = { enabled.value = it })
        }
        listOf("non", "oui").forEach { state ->
            compose.onNodeWithText("LFM3 1.7B").performClick()
            compose.onNodeWithText("Hadamard shuffle (exp.) : $state · recharger").performScrollTo().performClick()
            compose.runOnIdle { assertEquals(state == "non", enabled.value) }
        }
    }

    @Test fun vectorHeadExperimentCanBeToggledAndRestored() {
        val enabled = androidx.compose.runtime.mutableStateOf(false)
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, vectorHead = enabled.value,
                onVectorHead = { enabled.value = it })
        }
        listOf("non", "oui").forEach { state ->
            compose.onNodeWithText("LFM3 1.7B").performClick()
            compose.onNodeWithText("Tête vectorisée (exp.) : $state · recharger").performScrollTo().performClick()
            compose.runOnIdle { assertEquals(state == "non", enabled.value) }
        }
    }

    @Test fun newWaveHeadExperimentCanBeToggledAndRestored() {
        val enabled = androidx.compose.runtime.mutableStateOf(false)
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, waveHead = enabled.value,
                onWaveHead = { enabled.value = it })
        }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Tête une wave/ligne (exp.) : non · recharger").performScrollTo().performClick()
        compose.runOnIdle { assertTrue(enabled.value) }
        compose.onNodeWithText("LFM3 1.7B").performClick()
        compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
        compose.onNodeWithText("Tête une wave/ligne (exp.) : oui · recharger").performScrollTo().performClick()
        compose.runOnIdle { assertFalse(enabled.value) }
    }

    @Test fun newKernelExperimentsCanBeToggledAndRestored() {
        val fused = androidx.compose.runtime.mutableStateOf(false)
        val rms = androidx.compose.runtime.mutableStateOf(false)
        val exl3 = androidx.compose.runtime.mutableStateOf(false)
        compose.setContent {
            StudioScreen(emptyList(), false, true, "Ready", "Test GPU", null, false,
                {}, {}, {}, {}, {}, {}, subgroupNorm = rms.value,
                onSubgroupNorm = { rms.value = it }, exl3Multiwave = exl3.value,
                onExl3Multiwave = { exl3.value = it }, fusedNorm = fused.value,
                onFusedNorm = { fused.value = it })
        }
        listOf("non", "oui").forEach { state ->
            compose.onNodeWithText("LFM3 1.7B").performClick()
            compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
            compose.onNodeWithText("RMSNorm subgroup (exp.) : $state · recharger").performScrollTo().performClick()
            compose.runOnIdle { assertEquals(state == "non", rms.value) }
            compose.onNodeWithText("LFM3 1.7B").performClick()
            compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
            compose.onNodeWithText("EXL3 multiwave (exp.) : $state · recharger").performScrollTo().performClick()
            compose.runOnIdle { assertEquals(state == "non", exl3.value) }
            compose.onNodeWithText("LFM3 1.7B").performClick()
            compose.onNodeWithText("Réglages avancés et retour arrière").performScrollTo().performClick()
            compose.onNodeWithText("Fusion RMSNorm (sans subgroup) : $state · recharger").performScrollTo().performClick()
            compose.runOnIdle { assertEquals(state == "non", fused.value) }
        }
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
        val speed = compose.onNodeWithText("12,5 tok/s").assertIsDisplayed().fetchSemanticsNode().boundsInWindow
        assertTrue("Speed must appear below its answer", speed.top >= compose.onNodeWithText(answer).fetchSemanticsNode().boundsInWindow.bottom)
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
        assertComposerAboveKeyboard()
        InstrumentationRegistry.getInstrumentation().sendKeyDownUpSync(KeyEvent.KEYCODE_BACK)
        compose.waitUntil(15_000) {
            ViewCompat.getRootWindowInsets(compose.activity.window.decorView)?.isVisible(WindowInsetsCompat.Type.ime()) == false
        }
        compose.onNodeWithContentDescription("Message LFM3").performClick()
        compose.waitUntil(15_000) {
            ViewCompat.getRootWindowInsets(compose.activity.window.decorView)?.isVisible(WindowInsetsCompat.Type.ime()) == true
        }
        assertComposerAboveKeyboard()
        compose.onNodeWithContentDescription("Message LFM3").assertTextEquals("Première ligne\nDeuxième ligne\nTroisième ligne")
        val composerNode = compose.onNodeWithTag("composer").fetchSemanticsNode()
        val popupNodes = compose.onAllNodes(isPopup()).fetchSemanticsNodes()
        assertTrue("The cursor handle must be visible after tapping the field", popupNodes.isNotEmpty())
        compose.runOnIdle {
            val coordinates = composerNode.layoutInfo.coordinates
            val origin = coordinates.positionOnScreen()
            val bounds = Rect(origin.x, origin.y, origin.x + coordinates.size.width, origin.y + coordinates.size.height)
            popupNodes.forEach {
                val popup = it.layoutInfo.coordinates
                val center = popup.localToScreen(Offset(popup.size.width / 2f, popup.size.height / 2f))
                assertTrue("Cursor popup is outside the composer: popup=$center composer=$bounds", bounds.contains(center))
            }
        }
        screenshot("keyboard.png")
    }

    private fun assertComposerAboveKeyboard() {
        // IME/window updates continue outside Compose; measure after they settle.
        InstrumentationRegistry.getInstrumentation().uiAutomation.waitForIdle(300, 5_000)
        compose.waitForIdle()
        val view = compose.activity.window.decorView
        val keyboardHeight = requireNotNull(ViewCompat.getRootWindowInsets(view)).getInsets(WindowInsetsCompat.Type.ime()).bottom
        assertTrue("A real software keyboard must be visible", keyboardHeight > 0)
        val composer = compose.onNodeWithTag("composer").fetchSemanticsNode().boundsInWindow
        val focusedRect = compose.runOnIdle { android.graphics.Rect().also { view.findFocus()?.getFocusedRect(it) } }
        val keyboardTop = view.height - keyboardHeight
        val gap = keyboardTop - composer.bottom
        assertTrue("Composer must sit just above the keyboard: gap=$gap, composer=$composer, window=${view.height}, ime=$keyboardHeight, mode=${compose.activity.window.attributes.softInputMode}, focus=$focusedRect",
            gap >= -2f && gap <= 12f * view.resources.displayMetrics.density)
        compose.onAllNodesWithText("MLXL3").onFirst().assertIsDisplayed()
        assertTrue("The input and send control must remain fully visible", composer.height > 80f)
        compose.onNodeWithContentDescription("Envoyer").assertIsDisplayed().assertIsEnabled()
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
