import { html, css, LitElement } from '../../assets/lit-core-2.7.4.min.js';
import { AppHeader } from './AppHeader.js';
import { MainView } from '../views/MainView.js';
import { CustomizeView } from '../views/CustomizeView.js';
import { HelpView } from '../views/HelpView.js';
import { AssistantView } from '../views/AssistantView.js';
import { OnboardingView } from '../views/OnboardingView.js';
import { AdvancedView } from '../views/AdvancedView.js';

export class CheatingDaddyApp extends LitElement {
    static styles = css`
        * {
            box-sizing: border-box;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            margin: 0px;
            padding: 0px;
            cursor: default;
            user-select: none;
        }

        :host {
            display: block;
            width: 100%;
            height: 100vh;
            background-color: var(--background-transparent);
            color: var(--text-color);
        }

        .window-container {
            height: 100vh;
            border-radius: 7px;
            overflow: hidden;
        }

        .container {
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .main-content {
            flex: 1;
            padding: var(--main-content-padding);
            overflow-y: auto;
            margin-top: var(--main-content-margin-top);
            border-radius: var(--content-border-radius);
            transition: all 0.15s ease-out;
            background: var(--main-content-background);
        }

        .main-content.with-border {
            border: 1px solid var(--border-color);
        }

        .main-content.assistant-view {
            padding: 10px;
            border: none;
        }

        .main-content.onboarding-view {
            padding: 0;
            border: none;
            background: transparent;
        }

        .view-container {
            opacity: 1;
            transform: translateY(0);
            transition: opacity 0.15s ease-out, transform 0.15s ease-out;
            height: 100%;
        }

        .view-container.entering {
            opacity: 0;
            transform: translateY(10px);
        }

        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: var(--scrollbar-background);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--scrollbar-thumb);
            border-radius: 3px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: var(--scrollbar-thumb-hover);
        }
    `;

    static properties = {
        currentView: { type: String },
        statusText: { type: String },
        startTime: { type: Number },
        isRecording: { type: Boolean },
        sessionActive: { type: Boolean },
        selectedProfile: { type: String },
        selectedLanguage: { type: String },
        responses: { type: Array },
        currentResponseIndex: { type: Number },
        selectedScreenshotInterval: { type: String },
        selectedImageQuality: { type: String },
        layoutMode: { type: String },
        advancedMode: { type: Boolean },
        currentMode: { type: String },
        currentModel: { type: String },
        _viewInstances: { type: Object, state: true },
        _isClickThrough: { state: true },
        _awaitingNewResponse: { state: true },
        shouldAnimateResponse: { type: Boolean },
        updateAvailable: { type: Boolean },
    };

    constructor() {
        super();
        this.migrateLegacyModels();
        this.currentView = localStorage.getItem('onboardingCompleted') ? 'main' : 'onboarding';
        this.statusText = '';
        this.startTime = null;
        this.isRecording = false;
        this.sessionActive = false;
        this.selectedProfile = localStorage.getItem('selectedProfile') || 'interview';
        this.selectedLanguage = localStorage.getItem('selectedLanguage') || 'en-US';
        this.selectedScreenshotInterval = 'manual'; // Always manual to prevent rate limits
        this.selectedImageQuality = localStorage.getItem('selectedImageQuality') || 'high';
        this.layoutMode = localStorage.getItem('layoutMode') || 'compact';
        this.advancedMode = localStorage.getItem('advancedMode') === 'true';
        this.currentMode = localStorage.getItem('selectedMode') || 'interview';
        this.currentModel = '';
        this.responses = [];
        this.currentResponseIndex = -1;
        this._viewInstances = new Map();
        this._isClickThrough = false;
        this._awaitingNewResponse = false;
        this._currentResponseIsComplete = true;
        this.shouldAnimateResponse = false;
        this.updateAvailable = false;

        // Apply layout mode to document root
        this.updateLayoutMode();
    }

    // One-time migration for users upgrading from older versions:
    // Llama 4 Maverick/Scout were shut down by Groq (replaced by Qwen 3.6 27B)
    migrateLegacyModels() {
        const legacyModelMap = {
            'llama-4-maverick': 'qwen-3.6-27b',
            'llama-4-scout': 'qwen-3.6-27b',
        };
        const storedModel = localStorage.getItem('selectedModel');
        if (storedModel && legacyModelMap[storedModel]) {
            console.log(`[MIGRATION] Replacing deprecated model ${storedModel} with ${legacyModelMap[storedModel]}`);
            localStorage.setItem('selectedModel', legacyModelMap[storedModel]);
        }
        // Also migrate per-mode remembered models so mode switches don't restore dead models
        for (const key of ['lastModel_interview', 'lastModel_coding']) {
            const lastModel = localStorage.getItem(key);
            if (lastModel && legacyModelMap[lastModel]) {
                localStorage.setItem(key, legacyModelMap[lastModel]);
            }
        }
        // Google Search default flipped to OFF — grounding is paid-only on Gemini 3.x / 2.5 Pro,
        // and the old always-on 'true' caused constant 429s for free-tier keys in exam mode.
        // (Gemini 2.5 Flash models get search forced ON in gemini.js regardless of this setting.)
        if (!localStorage.getItem('googleSearchDefaultMigrated')) {
            localStorage.setItem('googleSearchEnabled', 'false');
            localStorage.setItem('googleSearchDefaultMigrated', 'true');
        }
    }

    connectedCallback() {
        super.connectedCallback();

        // Set up IPC listeners if needed
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.on('update-response', (_, response) => {
                this.setResponse(response);
            });
            ipcRenderer.on('update-status', (_, status) => {
                this.setStatus(status);
            });
            ipcRenderer.on('click-through-toggled', (_, isEnabled) => {
                this._isClickThrough = isEnabled;
            });
            ipcRenderer.on('update-available', (_, available) => {
                this.updateAvailable = available;
            });
        }

        // Add global keyboard event listener for Ctrl+G
        this.boundKeydownHandler = this.handleGlobalKeydown.bind(this);
        document.addEventListener('keydown', this.boundKeydownHandler);
    }

    disconnectedCallback() {
        super.disconnectedCallback();
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.removeAllListeners('update-response');
            ipcRenderer.removeAllListeners('update-status');
            ipcRenderer.removeAllListeners('click-through-toggled');
            ipcRenderer.removeAllListeners('update-available');
        }
        
        // Remove global keyboard event listener
        if (this.boundKeydownHandler) {
            document.removeEventListener('keydown', this.boundKeydownHandler);
        }
    }

    setStatus(text) {
        this.statusText = text;

        // Mark response as complete when we get certain status messages
        if (text.includes('Ready') || text.includes('Listening') || text.includes('Error') ||
            text.includes('Quota') || text.includes('Session closed')) {
            this._currentResponseIsComplete = true;
            // Response marked complete
        }
    }

    setResponse(response) {
        // Check if this looks like a filler response (very short responses to hmm, ok, etc)
        const isFillerResponse =
            response.length < 30 &&
            (response.toLowerCase().includes('hmm') ||
                response.toLowerCase().includes('okay') ||
                response.toLowerCase().includes('next') ||
                response.toLowerCase().includes('go on') ||
                response.toLowerCase().includes('continue'));

        if (this._awaitingNewResponse || this.responses.length === 0) {
            // Always add as new response when explicitly waiting for one
            this.responses = [...this.responses, response];
            this.currentResponseIndex = this.responses.length - 1;
            this._awaitingNewResponse = false;
            this._currentResponseIsComplete = false;
        } else if (!this._currentResponseIsComplete && !isFillerResponse && this.responses.length > 0) {
            // For substantial responses, update the last one (streaming behavior)
            // Only update if the current response is not marked as complete
            this.responses = [...this.responses.slice(0, this.responses.length - 1), response];
        } else {
            // For filler responses or when current response is complete, add as new
            this.responses = [...this.responses, response];
            this.currentResponseIndex = this.responses.length - 1;
            this._currentResponseIsComplete = false;
        }
        this.shouldAnimateResponse = true;
        this.requestUpdate();
    }

    // Header event handlers
    handleCustomizeClick() {
        this.currentView = 'customize';
        this.requestUpdate();
    }

    handleHelpClick() {
        this.currentView = 'help';
        this.requestUpdate();
    }


    handleAdvancedClick() {
        this.currentView = 'advanced';
        this.requestUpdate();
    }

    async handleUpdateCheckClick() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            const result = await ipcRenderer.invoke('check-for-updates');
            if (result && result.updateAvailable) {
                this.updateAvailable = true;
            } else {
                this.updateAvailable = false;
            }
        }
    }

    async handleClose() {
        if (this.currentView === 'customize' || this.currentView === 'help' || this.currentView === 'advanced') {
            this.currentView = 'main';
        } else if (this.currentView === 'assistant') {
            cheddar.stopCapture();

            // Close the session
            if (window.require) {
                const { ipcRenderer } = window.require('electron');
                await ipcRenderer.invoke('close-session');
            }
            this.sessionActive = false;
            this.currentView = 'main';
            console.log('Session closed');
        } else {
            // Quit the entire application
            if (window.require) {
                const { ipcRenderer } = window.require('electron');
                await ipcRenderer.invoke('quit-application');
            }
        }
    }

    async handleHideToggle() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            await ipcRenderer.invoke('toggle-window-visibility');
        }
    }

    // Main view event handlers
    async handleStart() {
        const selectedModel = localStorage.getItem('selectedModel') || 'qwen-3.6-27b';
        const isGroqModel = selectedModel && (selectedModel.includes('qwen') || selectedModel.includes('groq'));
        const needsBothKeys = ['gemini-2.5-flash-lite', 'gemini-3.1-flash-lite'].includes(selectedModel) && this.selectedProfile !== 'exam';
        const mainView = this.shadowRoot.querySelector('main-view');

        // Validate the right API key(s) based on model
        if (needsBothKeys) {
            const geminiKey = localStorage.getItem('apiKey')?.trim();
            const groqKey = localStorage.getItem('groqApiKey')?.trim();
            let hasError = false;
            if (!geminiKey) {
                if (mainView?.triggerApiKeyError) mainView.triggerApiKeyError();
                hasError = true;
            }
            if (!groqKey) {
                if (mainView?.triggerGroqApiKeyError) mainView.triggerGroqApiKeyError();
                hasError = true;
            }
            if (hasError) return;
        } else if (isGroqModel) {
            const groqKey = localStorage.getItem('groqApiKey')?.trim();
            if (!groqKey) {
                if (mainView?.triggerApiKeyError) mainView.triggerApiKeyError();
                return;
            }
        } else {
            const apiKey = localStorage.getItem('apiKey')?.trim();
            if (!apiKey) {
                if (mainView?.triggerApiKeyError) mainView.triggerApiKeyError();
                return;
            }
        }

        // Auto-set mode based on profile
        let selectedMode;
        if (this.selectedProfile === 'exam') {
            // Exam Assistant -> Coding/OA mode (forced)
            selectedMode = 'coding';
            localStorage.setItem('selectedMode', 'coding');
        } else {
            // All other profiles -> Interview mode (forced)
            selectedMode = 'interview';
            localStorage.setItem('selectedMode', 'interview');
        }

        // selectedModel already retrieved above for key validation

        await cheddar.initializeGemini(this.selectedProfile, this.selectedLanguage, selectedMode, selectedModel);

        // Set current mode and model for header display
        this.currentMode = selectedMode;
        // Use the actual selected model (could be Gemini or Llama/Groq)
        this.currentModel = selectedModel;

        // Always use manual mode to prevent rate limits and token exhaustion
        const screenshotMode = 'manual';

        console.log('📸 Manual capture mode: Press Ctrl+Enter to capture and analyze screenshot');

        cheddar.startCapture(screenshotMode, this.selectedImageQuality);
        this.responses = [];
        this.currentResponseIndex = -1;
        this.startTime = Date.now();
        this.currentView = 'assistant';
    }

    async handleAPIKeyHelp() {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            const selectedModel = localStorage.getItem('selectedModel') || 'qwen-3.6-27b';
            const isGroqModel = selectedModel && (selectedModel.includes('qwen') || selectedModel.includes('groq'));

            if (['gemini-2.5-flash-lite', 'gemini-3.1-flash-lite'].includes(selectedModel) && this.selectedProfile !== 'exam') {
                // Dual key mode (interview): open both API key pages
                await ipcRenderer.invoke('open-external', 'https://aistudio.google.com/');
                await ipcRenderer.invoke('open-external', 'https://groq.com/');
            } else {
                const url = isGroqModel ? 'https://groq.com/' : 'https://aistudio.google.com/';
                await ipcRenderer.invoke('open-external', url);
            }
        }
    }

        handleClearAndRestart() {
        // Clear the current session and responses
        this.responses = [];
        this.currentResponseIndex = -1;
        this.startTime = null;

        // Stop any ongoing capture if in assistant view
        if (this.currentView === 'assistant' && window.cheddar) {
            window.cheddar.stopCapture();
        }

        // Return to main view
        this.currentView = 'main';
        this.setStatus('Session cleared. Starting new session...');

        // Request update to refresh the UI
        this.requestUpdate();

        // Automatically start a new session after a brief delay
        setTimeout(() => {
            this.handleStart();
        }, 100);
    }

    handleGlobalKeydown(e) {
        // Handle Ctrl+Alt+R (or Cmd+Option+R on Mac) for clearing and restarting session
        const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
        const isClearShortcut = isMac
            ? e.metaKey && e.altKey && e.key === 'r'
            : e.ctrlKey && e.altKey && e.key === 'r';

        if (isClearShortcut) {
            e.preventDefault();
            e.stopPropagation();
            this.handleClearAndRestart();
        }
    }

    // Customize view event handlers
    handleProfileChange(profile) {
        this.selectedProfile = profile;
    }

    handleLanguageChange(language) {
        this.selectedLanguage = language;
    }

    handleScreenshotIntervalChange(interval) {
        this.selectedScreenshotInterval = interval;
    }

    handleImageQualityChange(quality) {
        this.selectedImageQuality = quality;
        localStorage.setItem('selectedImageQuality', quality);
    }

    handleAdvancedModeChange(advancedMode) {
        this.advancedMode = advancedMode;
        localStorage.setItem('advancedMode', advancedMode.toString());
    }

    handleBackClick() {
        this.currentView = 'main';
        this.requestUpdate();
    }

    // Help view event handlers
    async handleExternalLinkClick(url) {
        if (window.require) {
            const { ipcRenderer } = window.require('electron');
            await ipcRenderer.invoke('open-external', url);
        }
    }

    // Assistant view event handlers
    async handleSendText(message) {
        const result = await window.cheddar.sendTextMessage(message);

        if (!result.success) {
            console.error('Failed to send message:', result.error);
            this.setStatus('Error sending message: ' + result.error);
        } else {
            // Don't set "Message sent..." here - status is already managed by sendRealtimeInput
            // which sends: "Analyzing..." → "Ready" (success) or "Server overloaded" (error)
            this._awaitingNewResponse = true;
        }
    }

    handleResponseIndexChanged(e) {
        this.currentResponseIndex = e.detail.index;
        this.shouldAnimateResponse = false;
        this.requestUpdate();
    }

    // Onboarding event handlers
    handleOnboardingComplete() {
        this.currentView = 'main';
    }

    updated(changedProperties) {
        super.updated(changedProperties);

        // Only notify main process of view change if the view actually changed
        if (changedProperties.has('currentView') && window.require) {
            const { ipcRenderer } = window.require('electron');
            ipcRenderer.send('view-changed', this.currentView);

            // Add a small delay to smooth out the transition
            const viewContainer = this.shadowRoot?.querySelector('.view-container');
            if (viewContainer) {
                viewContainer.classList.add('entering');
                requestAnimationFrame(() => {
                    viewContainer.classList.remove('entering');
                });
            }
        }

        // Only update localStorage when these specific properties change
        if (changedProperties.has('selectedProfile')) {
            localStorage.setItem('selectedProfile', this.selectedProfile);
        }
        if (changedProperties.has('selectedLanguage')) {
            localStorage.setItem('selectedLanguage', this.selectedLanguage);
        }
        if (changedProperties.has('selectedScreenshotInterval')) {
            localStorage.setItem('selectedScreenshotInterval', this.selectedScreenshotInterval);
        }
        if (changedProperties.has('selectedImageQuality')) {
            localStorage.setItem('selectedImageQuality', this.selectedImageQuality);
        }
        if (changedProperties.has('layoutMode')) {
            this.updateLayoutMode();
        }
        if (changedProperties.has('advancedMode')) {
            localStorage.setItem('advancedMode', this.advancedMode.toString());
        }
    }

    renderCurrentView() {
        // Only re-render the view if it hasn't been cached or if critical properties changed
        const viewKey = `${this.currentView}-${this.selectedProfile}-${this.selectedLanguage}`;

        switch (this.currentView) {
            case 'onboarding':
                return html`
                    <onboarding-view .onComplete=${() => this.handleOnboardingComplete()} .onClose=${() => this.handleClose()}></onboarding-view>
                `;

            case 'main':
                return html`
                    <main-view
                        .onStart=${() => this.handleStart()}
                        .onAPIKeyHelp=${() => this.handleAPIKeyHelp()}
                        .onLayoutModeChange=${layoutMode => this.handleLayoutModeChange(layoutMode)}
                        .onClearAndRestart=${() => this.handleClearAndRestart()}
                    ></main-view>
                `;

            case 'customize':
                return html`
                    <customize-view
                        .selectedProfile=${this.selectedProfile}
                        .selectedLanguage=${this.selectedLanguage}
                        .selectedScreenshotInterval=${this.selectedScreenshotInterval}
                        .selectedImageQuality=${this.selectedImageQuality}
                        .layoutMode=${this.layoutMode}
                        .advancedMode=${this.advancedMode}
                        .onProfileChange=${profile => this.handleProfileChange(profile)}
                        .onLanguageChange=${language => this.handleLanguageChange(language)}
                        .onScreenshotIntervalChange=${interval => this.handleScreenshotIntervalChange(interval)}
                        .onImageQualityChange=${quality => this.handleImageQualityChange(quality)}
                        .onLayoutModeChange=${layoutMode => this.handleLayoutModeChange(layoutMode)}
                        .onAdvancedModeChange=${advancedMode => this.handleAdvancedModeChange(advancedMode)}
                    ></customize-view>
                `;

            case 'help':
                return html` <help-view .onExternalLinkClick=${url => this.handleExternalLinkClick(url)}></help-view> `;

            case 'advanced':
                return html` <advanced-view></advanced-view> `;

            case 'assistant':
                return html`
                    <assistant-view
                        .responses=${this.responses}
                        .currentResponseIndex=${this.currentResponseIndex}
                        .selectedProfile=${this.selectedProfile}
                        .selectedLanguage=${this.selectedLanguage}
                        .onSendText=${message => this.handleSendText(message)}
                        .shouldAnimateResponse=${this.shouldAnimateResponse}
                        @response-index-changed=${this.handleResponseIndexChanged}
                        @response-animation-complete=${() => {
                            this.shouldAnimateResponse = false;
                            this._currentResponseIsComplete = true;
                            // Response animation complete
                            this.requestUpdate();
                        }}
                    ></assistant-view>
                `;

            default:
                return html`<div>Unknown view: ${this.currentView}</div>`;
        }
    }

    render() {
        const mainContentClass = `main-content ${
            this.currentView === 'assistant' ? 'assistant-view' : this.currentView === 'onboarding' ? 'onboarding-view' : 'with-border'
        }`;

        return html`
            <div class="window-container">
                <div class="container">
                    <app-header
                        .currentView=${this.currentView}
                        .statusText=${this.statusText}
                        .startTime=${this.startTime}
                        .currentMode=${this.currentMode}
                        .currentModel=${this.currentModel}
                        .advancedMode=${this.advancedMode}
                        .updateAvailable=${this.updateAvailable}
                        .onCustomizeClick=${() => this.handleCustomizeClick()}
                        .onHelpClick=${() => this.handleHelpClick()}
                        .onAdvancedClick=${() => this.handleAdvancedClick()}
                        .onUpdateCheckClick=${() => this.handleUpdateCheckClick()}
                        .onCloseClick=${() => this.handleClose()}
                        .onBackClick=${() => this.handleBackClick()}
                        .onHideToggleClick=${() => this.handleHideToggle()}
                        .onRestartClick=${() => this.handleClearAndRestart()}
                        ?isClickThrough=${this._isClickThrough}
                    ></app-header>
                    <div class="${mainContentClass}">
                        <div class="view-container">${this.renderCurrentView()}</div>
                    </div>
                </div>
            </div>
        `;
    }

    updateLayoutMode() {
        // Apply or remove compact layout class to document root
        if (this.layoutMode === 'compact') {
            document.documentElement.classList.add('compact-layout');
        } else {
            document.documentElement.classList.remove('compact-layout');
        }
    }

    async handleLayoutModeChange(layoutMode) {
        this.layoutMode = layoutMode;
        localStorage.setItem('layoutMode', layoutMode);
        this.updateLayoutMode();

        // Notify main process about layout change for window resizing
        if (window.require) {
            try {
                const { ipcRenderer } = window.require('electron');
                await ipcRenderer.invoke('update-sizes');
            } catch (error) {
                console.error('Failed to update sizes in main process:', error);
            }
        }

        this.requestUpdate();
    }
}

customElements.define('cheating-daddy-app', CheatingDaddyApp);
