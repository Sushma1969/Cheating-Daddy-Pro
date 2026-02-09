const { GoogleGenAI } = require('@google/genai');
const { BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const { saveDebugAudio } = require('../audioUtils');
const { getSystemPrompt, getGeminiMessageHint } = require('./prompts');
const { VADProcessor } = require('./vad');

// Session tracking
let isInitializingSession = false;
let storedLanguageName = 'English';

// Audio capture variables
let systemAudioProc = null;

// macOS audio VAD variables (used by macOS audio capture)
let macVADProcessor = null;
let macVADEnabled = false;
let macVADMode = 'automatic';
let macMicrophoneEnabled = true;

// Track current session mode and profile
let currentMode = 'interview';
let currentProfile = 'interview';

// Rate limit countdown (similar to groq.js)
let rateLimitCountdownInterval = null;

/**
 * Start a live countdown in the header after a 429 rate limit error from Gemini.
 * Updates the status every second: "Rate Limit (Gemini): 30s" → "29s" → ... → "Ready"
 */
function scheduleGeminiRateLimitRecovery(statusMessage, recoveryMs = 60 * 1000) {
    if (rateLimitCountdownInterval) {
        clearInterval(rateLimitCountdownInterval);
        rateLimitCountdownInterval = null;
    }

    let remainingSec = Math.ceil(recoveryMs / 1000);
    console.log(`[GEMINI] Rate limit hit - countdown ${remainingSec}s`);

    sendToRenderer('update-status', `${statusMessage} (${remainingSec}s)`);

    rateLimitCountdownInterval = setInterval(() => {
        remainingSec--;
        if (remainingSec <= 0) {
            clearInterval(rateLimitCountdownInterval);
            rateLimitCountdownInterval = null;
            console.log('[GEMINI] Rate limit countdown done');
            sendToRenderer('update-status', 'Ready');
        } else {
            sendToRenderer('update-status', `${statusMessage} (${remainingSec}s)`);
        }
    }, 1000);
}

/**
 * Parse the Gemini 429 error to determine rate limit type and extract retry wait time.
 * Gemini errors include retryDelay in details: {"retryDelay":"30s"}
 * or "try again in XX.XXs" in the message text.
 */
function parseGeminiRateLimitError(errorMessage) {
    let statusMessage = 'Rate Limit (Gemini)';
    let recoveryMs = 60 * 1000; // Fallback 60s for Gemini

    try {
        // Try to extract retryDelay from the JSON in the error message
        // SDK wraps the error with escaped quotes: \"retryDelay\": \"26s\"
        const retryDelayMatch = errorMessage.match(/retryDelay[\\"\s:]+(\d+\.?\d*)s/i);
        if (retryDelayMatch) {
            const retrySec = parseFloat(retryDelayMatch[1]);
            recoveryMs = Math.ceil((retrySec + 2) * 1000);
        }

        // Also try "Please retry in 26.326453387s." pattern (Gemini's wording)
        if (!retryDelayMatch) {
            const retryInMatch = errorMessage.match(/retry in (\d+\.?\d*)s/i);
            if (retryInMatch) {
                const retrySec = parseFloat(retryInMatch[1]);
                recoveryMs = Math.ceil((retrySec + 2) * 1000);
            }
        }

        // Determine the rate limit type
        const msgLower = errorMessage.toLowerCase();
        if (msgLower.includes('resource_exhausted') || msgLower.includes('resource has been exhausted')) {
            statusMessage = 'Rate Limit (Gemini): Quota exhausted';
        } else if (msgLower.includes('tokens') || msgLower.includes('tpm')) {
            statusMessage = 'Rate Limit (Gemini): Tokens exceeded';
        } else if (msgLower.includes('requests') || msgLower.includes('rpm')) {
            statusMessage = 'Rate Limit (Gemini): Requests exceeded';
        }

        console.log(`[GEMINI] Rate limit details: ${errorMessage.substring(0, 200)}`);
    } catch (e) {
        console.warn('[GEMINI] Could not parse 429 error:', e.message);
    }

    return { statusMessage, recoveryMs };
}

// Model generation settings (can be updated via IPC from renderer)
let generationSettings = {
    temperature: 0.7,
    topP: 0.95,
    maxOutputTokens: 8192,
};

// Model-specific max output token limits
const MODEL_MAX_OUTPUT_TOKENS = {
    // Gemini models
    'gemini-2.5-flash': 65536,
    'gemini-3-flash-preview': 65536,
    'gemini-3-pro-preview': 65536,
    // Groq Llama models
    'llama-4-maverick': 8192,
    'llama-4-scout': 8192,
};

// Get max output tokens for a specific model
function getMaxOutputTokensForModel(model) {
    return MODEL_MAX_OUTPUT_TOKENS[model] || 8192;
}

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

async function getEnabledTools() {
    const tools = [];

    // Check if Google Search is enabled (default: true)
    const googleSearchEnabled = await getStoredSetting('googleSearchEnabled', 'true');
    console.log('Google Search enabled:', googleSearchEnabled);

    if (googleSearchEnabled === 'true') {
        tools.push({ googleSearch: {} });
        console.log('Added Google Search tool');
    } else {
        console.log('Google Search tool disabled');
    }

    return tools;
}

async function getStoredSetting(key, defaultValue) {
    try {
        const windows = BrowserWindow.getAllWindows();
        if (windows.length > 0) {
            // Wait a bit for the renderer to be ready
            await new Promise(resolve => setTimeout(resolve, 100));

            // Try to get setting from renderer process localStorage
            const value = await windows[0].webContents.executeJavaScript(`
                (function() {
                    try {
                        if (typeof localStorage === 'undefined') {
                            console.log('localStorage not available yet for ${key}');
                            return '${defaultValue}';
                        }
                        const stored = localStorage.getItem('${key}');
                        console.log('Retrieved setting ${key}:', stored);
                        return stored || '${defaultValue}';
                    } catch (e) {
                        console.error('Error accessing localStorage for ${key}:', e);
                        return '${defaultValue}';
                    }
                })()
            `);
            return value;
        }
    } catch (error) {
        console.error('Error getting stored setting for', key, ':', error.message);
    }
    console.log('Using default value for', key, ':', defaultValue);
    return defaultValue;
}

async function initializeGeminiSession(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', _isReconnection = false, mode = 'interview', model = 'gemini-2.5-flash') {
    if (isInitializingSession) {
        console.log('Session initialization already in progress');
        return false;
    }

    isInitializingSession = true;
    sendToRenderer('session-initializing', true);

    // Clear any active rate limit countdown from previous session
    if (rateLimitCountdownInterval) {
        clearInterval(rateLimitCountdownInterval);
        rateLimitCountdownInterval = null;
    }

    const client = new GoogleGenAI({
        vertexai: false,
        apiKey: apiKey,
    });

    // Get enabled tools first to determine Google Search status
    const enabledTools = await getEnabledTools();
    const googleSearchEnabled = enabledTools.some(tool => tool.googleSearch);

    let systemPrompt = getSystemPrompt(profile, customPrompt, googleSearchEnabled);

    // Add explicit language instruction based on user's selected language
    const languageMap = {
        'en-US': 'English',
        'en-GB': 'English',
        'en-AU': 'English',
        'en-IN': 'English',
        'es-ES': 'Spanish',
        'es-US': 'Spanish',
        'fr-FR': 'French',
        'fr-CA': 'French',
        'de-DE': 'German',
        'it-IT': 'Italian',
        'pt-BR': 'Portuguese',
        'pt-PT': 'Portuguese',
        'ru-RU': 'Russian',
        'ja-JP': 'Japanese',
        'ko-KR': 'Korean',
        'zh-CN': 'Chinese (Simplified)',
        'cmn-CN': 'Chinese (Simplified)',
        'zh-TW': 'Chinese (Traditional)',
        'ar-SA': 'Arabic',
        'ar-XA': 'Arabic',
        'hi-IN': 'Hindi',
        'nl-NL': 'Dutch',
        'pl-PL': 'Polish',
        'tr-TR': 'Turkish',
        'sv-SE': 'Swedish',
        'da-DK': 'Danish',
        'fi-FI': 'Finnish',
        'no-NO': 'Norwegian',
        'th-TH': 'Thai',
        'te-IN': 'Telugu',
        'ta-IN': 'Tamil',
        'mr-IN': 'Marathi',
        'ml-IN': 'Malayalam',
        'kn-IN': 'Kannada',
        'gu-IN': 'Gujarati',
        'bn-IN': 'Bengali',
        'vi-VN': 'Vietnamese',
        'id-ID': 'Indonesian',
    };

    const selectedLanguageName = languageMap[language] || 'English';
    storedLanguageName = selectedLanguageName; // Store for use in text/screenshot prompts

    // Add critical language instruction to system prompt
    systemPrompt += `\n\n=== CRITICAL LANGUAGE INSTRUCTION ===
The user has selected ${selectedLanguageName} as their preferred language.
YOU MUST respond ONLY in ${selectedLanguageName}, regardless of what language the interviewer or other person uses.
Even if they speak in mixed languages (e.g., English + Hindi, Russian + English, etc.), you MUST respond entirely in ${selectedLanguageName}.
This is mandatory and cannot be overridden by any other instruction.`;

    try {
        let session;
        const regularModel = model || 'gemini-2.5-flash';
        currentMode = mode;
        currentProfile = profile;
        console.log(`Initializing Gemini session: ${regularModel} (mode: ${mode}, profile: ${profile})`);

            // Enhanced prompt for coding/interview mode - for coding mode add aggressive direct answer instructions
            const isProModel = regularModel.includes('pro');
            const codingPrompt = systemPrompt + `

============ CRITICAL CODING MODE INSTRUCTIONS ============

YOU ARE A CODING ASSISTANT IN A TIMED ASSESSMENT. FOLLOW THESE RULES EXACTLY:

${isProModel ? `
WARNING GEMINI PRO: YOU MUST BE EXTREMELY CONCISE. NO VERBOSE RESPONSES.
MAXIMUM 10 LINES OF EXPLANATION TOTAL. FOCUS ON CODE ONLY.
` : ''}

1. WHEN YOU SEE A SCREENSHOT:
   - DO NOT DESCRIBE the screenshot or UI elements
   - DO NOT explain what you see on screen
   - IMMEDIATELY read the coding problem text
   - IMMEDIATELY solve that problem
   - Your ONLY job is to provide the CODE SOLUTION

2. RESPONSE FORMAT (STRICTLY FOLLOW):
   - Line 1: One sentence approach (MAX 15 words)
   - Line 2+: COMPLETE working code ONLY (language shown on screen)
   - Last line: Time/Space complexity
   - DO NOT include: screenshot descriptions, UI analysis, or anything not related to the code solution

3. CODE REQUIREMENTS:
   - ZERO comments in code
   - ZERO explanations before or after code
   - ZERO tutorial text
   - ONLY the language shown on screen
   - CLEAN, optimized code that passes all test cases

4. ABSOLUTELY FORBIDDEN:
   - NO long explanations or theory
   - NO comments in code
   - NO multiple language versions
   - NO step-by-step walkthroughs
   - NO example inputs/outputs
   - NO alternative approaches discussion

5. RESPONSE LENGTH LIMIT:
   - Approach: 1 line (max 15 words)
   - Code: As needed
   - Complexity: 1 line
   - TOTAL NON-CODE TEXT: MAX 2 LINES

EXAMPLE OF PERFECT RESPONSE:
"HashMap to count frequencies, find max group size, return chars with that frequency.

class Solution {
    public String majorityFrequencyGroup(String s) {
        Map<Character, Integer> freq = new HashMap<>();
        for (char c : s.toCharArray()) freq.put(c, freq.getOrDefault(c, 0) + 1);
        Map<Integer, List<Character>> groups = new HashMap<>();
        for (Map.Entry<Character, Integer> e : freq.entrySet()) {
            groups.computeIfAbsent(e.getValue(), k -> new ArrayList<>()).add(e.getKey());
        }
        int maxSize = 0, maxFreq = 0;
        for (Map.Entry<Integer, List<Character>> e : groups.entrySet()) {
            int size = e.getValue().size();
            if (size > maxSize || (size == maxSize && e.getKey() > maxFreq)) {
                maxSize = size;
                maxFreq = e.getKey();
            }
        }
        StringBuilder sb = new StringBuilder();
        for (char c : groups.get(maxFreq)) sb.append(c);
        return sb.toString();
    }
}

Time: O(n), Space: O(n)"

CRITICAL FINAL REMINDER:
- DO NOT describe what you see in the screenshot
- DO NOT analyze UI elements, browser tabs, taskbar, or any visual elements
- DO NOT write "This is a screenshot of..." or "I see a problem on LeetCode..."
- IMMEDIATELY jump to solving the coding problem
- Your response MUST START with the approach sentence, NOT a description

NOW SOLVE THE CODING PROBLEM SHOWN IN THE SCREENSHOT.
RESPONSE FORMAT: [approach sentence] + [code] + [complexity]`;

            // Add conciseness override for interview mode (Gemini tends to be very verbose)
            const interviewPrompt = systemPrompt + `

============ CRITICAL INTERVIEW BREVITY RULES ============

You are helping someone in a LIVE SPOKEN interview. They will READ your answer aloud.
Long responses are HARMFUL — the interviewer will notice they are reading, not speaking.

RULES:
1. NON-CODING questions: MAX 2-4 sentences. Give the direct answer, one brief reason, done.
2. Formula questions: State the formula, define variables in ONE line each. NO derivations, NO proofs, NO examples, NO step-by-step walkthroughs.
3. Concept questions: Define it in 1-2 sentences. Give 1 example ONLY if asked. NO listing every variant or type.
4. NEVER use tables, numbered lists longer than 3 items, or multi-section responses for spoken answers.
5. CODING questions (screenshots OR audio requests like "write code for X"): Use the FULL 5-section format: Approach → Intuition (2-4 paragraphs) → Implementation (code) → Complexity → Algorithm. Do NOT shorten coding answers.

BAD (too long): "Linear Regression uses the formula y = β0 + β1*x where... [followed by OLS derivation, error terms, 5 examples]"
GOOD (interview-ready): "The formula is **ŷ = β₀ + β₁x**, where β₀ is the intercept and β₁ is the slope representing the change in y per unit x."

REMEMBER: If someone asked you this face-to-face, you would NOT recite a textbook chapter. Keep it natural and brief.`;

            // Create a "session" object that uses generateContentStream internally
            // For coding/exam mode: use codingPrompt with aggressive direct answer instructions
            // For interview mode: use interviewPrompt with brevity override
            const sessionPrompt = (mode === 'coding') ? codingPrompt : interviewPrompt;

            session = {
                model: regularModel,
                client: client,
                systemPrompt: sessionPrompt,
                tools: enabledTools,
                isClosed: false,
                conversationHistory: [], // Track conversation history for context

                async sendRealtimeInput(input) {
                    if (this.isClosed) {
                        console.log('Session is closed, ignoring input');
                        return;
                    }

                    try {
                        // Only process image and text inputs for coding mode
                        if (input.media || input.text) {
                            const requestStartTime = Date.now();
                            console.log(`📸 Sending to ${this.model}`);
                            // In interview mode, groq.js already sets "Generating..." — don't override
                            // In coding/exam mode, show "Analyzing..." for screenshot processing
                            if (currentMode !== 'interview') {
                                sendToRenderer('update-status', 'Analyzing...');
                            }

                            // Build the parts array for current message
                            const parts = [];

                            if (input.text) {
                                // Add language reminder for non-English languages
                                let finalText = input.text;
                                if (storedLanguageName !== 'English') {
                                    finalText = `${input.text} (Remember: Respond in ${storedLanguageName})`;
                                }
                                parts.push({ text: finalText });
                            }

                            if (input.media) {
                                parts.push({
                                    inlineData: {
                                        mimeType: input.media.mimeType,
                                        data: input.media.data
                                    }
                                });
                                // Add language reminder for non-English when only screenshot (no text provided)
                                if (!input.text && storedLanguageName !== 'English') {
                                    parts.push({ text: `(Remember: Respond in ${storedLanguageName})` });
                                }
                            }

                            const hasImage = !!input.media;

                            // Build conversation with history
                            // For text-only requests: strip ALL images from history to reduce payload
                            // (images are 200KB+ base64 each — re-sending them adds seconds of latency)
                            let historyForRequest;
                            if (!hasImage && this.conversationHistory.some(e => e.role === 'user' && e.parts?.some(p => p.inlineData))) {
                                historyForRequest = this.conversationHistory.map(entry => {
                                    if (entry.role === 'user' && entry.parts && entry.parts.some(p => p.inlineData)) {
                                        return {
                                            role: entry.role,
                                            parts: entry.parts.map(p => p.inlineData ? { text: '[screenshot]' } : p)
                                        };
                                    }
                                    return entry;
                                });
                            } else {
                                historyForRequest = this.conversationHistory;
                            }

                            const contents = [
                                ...historyForRequest,
                                { role: 'user', parts: parts }
                            ];

                            if (this.conversationHistory.length > 0) {
                                console.log(`Context: ${this.conversationHistory.length / 2} turns (request #${this.conversationHistory.length / 2 + 1})`);
                            }

                            const modelMaxTokens = getMaxOutputTokensForModel(this.model);
                            let effectiveMaxTokens = Math.min(generationSettings.maxOutputTokens, modelMaxTokens);

                            // Interview mode token limits:
                            // - Screenshot/image requests (coding problems): force 4096 for full structured answer
                            //   (overrides UI setting since the default 1024 is too low for Intuition + code)
                            // - Text-only spoken answers: cap at 1024 for concise interview responses
                            // Exam/coding mode needs full output for detailed code solutions
                            if (currentMode === 'interview') {
                                if (hasImage) {
                                    effectiveMaxTokens = Math.min(4096, modelMaxTokens);
                                } else {
                                    effectiveMaxTokens = Math.min(effectiveMaxTokens, 1024);
                                }
                            }

                            // Thinking levels:
                            // Gemini 3 Flash interview mode → 'minimal' (fastest responses)
                            // Gemini 3 Flash exam mode → 'low'
                            // Gemini 3 Pro → 'high' (best accuracy)
                            const thinkingConfig = this.model === 'gemini-3-flash-preview'
                                ? { thinkingLevel: currentMode === 'interview' ? 'minimal' : 'low' }
                                : this.model === 'gemini-3-pro-preview'
                                    ? { thinkingLevel: 'high' }
                                    : undefined;

                            // Interview mode: skip Google Search tool entirely (saves 2-5s latency per request)
                            // - Text Q&A ("what is polymorphism?") doesn't need web search
                            // - Coding screenshots (LeetCode) don't need web search
                            // Exam/coding mode: keep Search enabled (may need current info for complex problems)
                            const requestTools = (currentMode === 'interview')
                                ? undefined
                                : (this.tools.length > 0 ? this.tools : undefined);

                            const streamResult = await this.client.models.generateContentStream({
                                model: this.model,
                                contents: contents,
                                systemInstruction: { parts: [{ text: this.systemPrompt }] },
                                generationConfig: {
                                    temperature: generationSettings.temperature,
                                    topP: generationSettings.topP,
                                    maxOutputTokens: effectiveMaxTokens,
                                    ...(thinkingConfig ? { thinkingConfig } : {}),
                                },
                                tools: requestTools,
                            });

                            // Stream the response as it arrives
                            let responseText = '';

                            // Check if it's iterable stream or has stream property
                            const streamToIterate = streamResult.stream || streamResult;

                            // Streaming optimization: Batch UI updates for smoother rendering
                            let lastUpdateTime = Date.now();
                            const UPDATE_INTERVAL = 50; // Update UI every 50ms for smooth rendering
                            let firstChunkLogged = false;

                            try {
                                for await (const chunk of streamToIterate) {
                                    if (chunk && chunk.candidates && chunk.candidates.length > 0) {
                                        const candidate = chunk.candidates[0];
                                        if (candidate.content && candidate.content.parts) {
                                            for (const part of candidate.content.parts) {
                                                if (part.text) {
                                                    if (!firstChunkLogged) {
                                                        console.log(`⏱️ First token: ${Date.now() - requestStartTime}ms`);
                                                        firstChunkLogged = true;
                                                    }
                                                    responseText += part.text;

                                                    // Batch updates: Only send to UI every 50ms for smoother rendering
                                                    const now = Date.now();
                                                    if (now - lastUpdateTime >= UPDATE_INTERVAL) {
                                                        sendToRenderer('update-response', responseText);
                                                        lastUpdateTime = now;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                // Send final update to ensure last chunk is displayed
                                sendToRenderer('update-response', responseText);

                                if (responseText && responseText.trim()) {
                                    console.log(`✅ Got response: ${responseText.length} chars in ${Date.now() - requestStartTime}ms`);

                                    // Save to conversation history with full data
                                    this.conversationHistory.push(
                                        { role: 'user', parts: parts },
                                        { role: 'model', parts: [{ text: responseText }] }
                                    );

                                    // Cap at 8 turns (16 entries)
                                    if (this.conversationHistory.length > 16) {
                                        this.conversationHistory = this.conversationHistory.slice(-16);
                                    }

                                    // Strip images from older turns, keep only last 3 screenshots
                                    // Images are 200KB+ base64 — re-sending them all balloons latency from 2s to 10s+
                                    let imageCount = 0;
                                    for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
                                        const entry = this.conversationHistory[i];
                                        if (entry.role === 'user' && entry.parts) {
                                            const hasImage = entry.parts.some(p => p.inlineData);
                                            if (hasImage) {
                                                imageCount++;
                                                if (imageCount > 3) {
                                                    // Replace old images with lightweight placeholder
                                                    entry.parts = entry.parts.map(p =>
                                                        p.inlineData ? { text: '[screenshot]' } : p
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    console.log(`💬 Conversation history: ${this.conversationHistory.length / 2} turns`);
                                    // Interview mode: "Listening..." (matches Groq Llama flow)
                                    // Coding/exam mode: "Ready" (waiting for next screenshot)
                                    sendToRenderer('update-status', currentMode === 'interview' ? 'Listening...' : 'Ready');
                                    return responseText;
                                } else {
                                    console.error('❌ No response text received');
                                    sendToRenderer('update-status', currentMode === 'interview' ? 'Listening...' : 'No response generated');
                                    return null;
                                }
                            } catch (streamError) {
                                console.error('❌ Streaming error:', streamError);
                                // Fallback: try to get the complete result
                                const finalResult = await streamResult;
                                if (finalResult && finalResult.candidates && finalResult.candidates.length > 0) {
                                    const candidate = finalResult.candidates[0];
                                    if (candidate.content && candidate.content.parts) {
                                        for (const part of candidate.content.parts) {
                                            if (part.text) {
                                                responseText += part.text;
                                            }
                                        }
                                    }
                                }
                                if (responseText && responseText.trim()) {
                                    console.log(`✅ Got response (fallback): ${responseText.length} chars`);

                                    // Save to conversation history (same image-stripping as main path)
                                    this.conversationHistory.push(
                                        { role: 'user', parts: parts },
                                        { role: 'model', parts: [{ text: responseText }] }
                                    );
                                    if (this.conversationHistory.length > 16) {
                                        this.conversationHistory = this.conversationHistory.slice(-16);
                                    }
                                    let imgCount = 0;
                                    for (let i = this.conversationHistory.length - 1; i >= 0; i--) {
                                        const entry = this.conversationHistory[i];
                                        if (entry.role === 'user' && entry.parts) {
                                            if (entry.parts.some(p => p.inlineData)) {
                                                imgCount++;
                                                if (imgCount > 3) {
                                                    entry.parts = entry.parts.map(p =>
                                                        p.inlineData ? { text: '[screenshot]' } : p
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    sendToRenderer('update-response', responseText);
                                    sendToRenderer('update-status', currentMode === 'interview' ? 'Listening...' : 'Ready');
                                    return responseText;
                                } else {
                                    throw streamError;
                                }
                            }
                        }
                        return null;
                    } catch (error) {
                        console.error('❌ Error in Gemini session:', error);

                        // Show user-friendly short error message
                        let shortMsg = 'Error';

                        const errMsg = (error.message || '').toLowerCase();
                        if (errMsg.includes('429')) {
                            const rateLimit = parseGeminiRateLimitError(error.message || '');
                            scheduleGeminiRateLimitRecovery(rateLimit.statusMessage, rateLimit.recoveryMs);
                            return null;
                        } else if (errMsg.includes('503') || errMsg.includes('overloaded') || errMsg.includes('unavailable')) {
                            // Server overloaded — common on free tier, retry after 15s
                            scheduleGeminiRateLimitRecovery('Server Overloaded (Gemini)', 15 * 1000);
                            return null;
                        } else if (errMsg.includes('401') || errMsg.includes('api_key_invalid') || errMsg.includes('api key not valid')) {
                            shortMsg = 'Invalid API Key (Gemini)';
                        } else if (errMsg.includes('404') || errMsg.includes('not found') || errMsg.includes('not_found')) {
                            // Model removed or deprecated by Google
                            shortMsg = 'Model Unavailable (Gemini)';
                            console.error('[GEMINI] Model not found — it may have been removed or deprecated by Google.');
                        } else if (errMsg.includes('deprecated') || errMsg.includes('decommission')) {
                            shortMsg = 'Model Deprecated (Gemini)';
                            console.error('[GEMINI] Model deprecated by Google.');
                        }

                        sendToRenderer('update-status', shortMsg);
                        return null;
                    }
                },

                async close() {
                    this.isClosed = true;
                    console.log('Coding mode session closed');
                    sendToRenderer('update-status', 'Session closed');
                }
            };

            sendToRenderer('update-status', `${regularModel} ready (screenshot mode)`);

        isInitializingSession = false;
        sendToRenderer('session-initializing', false);
        return session;
    } catch (error) {
        console.error('Failed to initialize Gemini session:', error);
        isInitializingSession = false;
        sendToRenderer('session-initializing', false);
        return null;
    }
}

function killExistingSystemAudioDump() {
    return new Promise(resolve => {
        console.log('Checking for existing SystemAudioDump processes...');

        // Kill any existing SystemAudioDump processes
        const killProc = spawn('pkill', ['-f', 'SystemAudioDump'], {
            stdio: 'ignore',
        });

        killProc.on('close', code => {
            if (code === 0) {
                console.log('Killed existing SystemAudioDump processes');
            } else {
                console.log('No existing SystemAudioDump processes found');
            }
            resolve();
        });

        killProc.on('error', err => {
            console.log('Error checking for existing processes (this is normal):', err.message);
            resolve();
        });

        // Timeout after 2 seconds
        setTimeout(() => {
            killProc.kill();
            resolve();
        }, 2000);
    });
}

async function startMacOSAudioCapture(geminiSessionRef, vadEnabled = false, vadMode = 'automatic') {
    if (process.platform !== 'darwin') return false;

    // Kill any existing SystemAudioDump processes first
    await killExistingSystemAudioDump();

    console.log('Starting macOS audio capture with SystemAudioDump...');

    const { app } = require('electron');
    const path = require('path');
    const fs = require('fs');

    let systemAudioPath;
    if (app.isPackaged) {
        systemAudioPath = path.join(process.resourcesPath, 'SystemAudioDump');
    } else {
        systemAudioPath = path.join(__dirname, '../assets', 'SystemAudioDump');
    }

    console.log('SystemAudioDump path:', systemAudioPath);

    // Check if SystemAudioDump binary exists before attempting to spawn
    if (!fs.existsSync(systemAudioPath)) {
        console.warn('⚠ SystemAudioDump binary not found at:', systemAudioPath);
        console.warn('ℹ macOS system audio capture will not be available.');
        console.warn('ℹ The app will continue but audio from other apps will not be captured.');
        console.warn('ℹ To enable audio capture, ensure SystemAudioDump is in src/assets/ (development) or Resources/ (packaged).');
        return false;
    }

    // Spawn SystemAudioDump with stealth options
    const spawnOptions = {
        stdio: ['ignore', 'pipe', 'pipe'],
        env: {
            ...process.env,
            // Set environment variables that might help with stealth
            PROCESS_NAME: 'AudioService',
            APP_NAME: 'System Audio Service',
        },
    };

    // On macOS, apply additional stealth measures
    if (process.platform === 'darwin') {
        spawnOptions.detached = false;
        spawnOptions.windowsHide = false;
    }

    try {
        systemAudioProc = spawn(systemAudioPath, [], spawnOptions);
    } catch (error) {
        console.error(' Failed to spawn SystemAudioDump:', error.message);
        console.error('Path:', systemAudioPath);
        console.error('Hint: Make sure SystemAudioDump binary has execute permissions (chmod +x)');
        return false;
    }

    if (!systemAudioProc.pid) {
        console.error(' Failed to start SystemAudioDump - no PID');
        console.error('Path:', systemAudioPath);
        console.error('Hint: Binary may not have execute permissions or wrong architecture');
        return false;
    }

    console.log(' SystemAudioDump started with PID:', systemAudioProc.pid);

    const CHUNK_DURATION = 0.1;
    const SAMPLE_RATE = 24000;
    const BYTES_PER_SAMPLE = 2;
    const CHANNELS = 2;
    const CHUNK_SIZE = SAMPLE_RATE * BYTES_PER_SAMPLE * CHANNELS * CHUNK_DURATION;

    let audioBuffer = Buffer.alloc(0);

    // Initialize VAD for macOS using settings passed from renderer
    macVADEnabled = vadEnabled;
    macVADMode = vadMode;

    console.log(`🔧 [macOS] VAD Settings: enabled=${macVADEnabled}, mode=${macVADMode}`);

    if (macVADEnabled) {
        console.log(`🔧 [macOS] Initializing VAD in ${macVADMode.toUpperCase()} mode`);

        // Initialize microphone state based on mode
        if (macVADMode === 'automatic') {
            macMicrophoneEnabled = true;
            console.log('🎤 [macOS AUTOMATIC] Microphone enabled by default');
        } else {
            macMicrophoneEnabled = false;
            console.log('🔴 [macOS MANUAL] Microphone OFF - click button to enable');
        }

        // Create VAD processor for macOS
        macVADProcessor = new VADProcessor(
            async (audioSegment, metadata) => {
                try {
                    // Convert Float32Array to PCM Buffer
                    const pcmBuffer = convertFloat32ToPCMBuffer(audioSegment);
                    const base64Data = pcmBuffer.toString('base64');
                    await sendAudioToGemini(base64Data, geminiSessionRef);
                    console.log('🎤 [macOS VAD] Audio segment sent:', metadata);
                } catch (error) {
                    console.error('❌ [macOS VAD] Failed to send audio segment:', error);
                }
            },
            null, // onStateChange callback
            macVADMode // VAD mode
        );

        console.log('✅ [macOS] VAD processor initialized');
    }

    systemAudioProc.stdout.on('data', data => {
        audioBuffer = Buffer.concat([audioBuffer, data]);

        while (audioBuffer.length >= CHUNK_SIZE) {
            const chunk = audioBuffer.slice(0, CHUNK_SIZE);
            audioBuffer = audioBuffer.slice(CHUNK_SIZE);

            const monoChunk = CHANNELS === 2 ? convertStereoToMono(chunk) : chunk;

            if (macVADEnabled && macVADProcessor) {
                // VAD mode: process through VAD
                if (!macMicrophoneEnabled) {
                    // Skip audio processing if mic is OFF in manual mode
                    continue;
                }

                // Convert PCM Buffer to Float32Array for VAD
                const float32Audio = convertPCMBufferToFloat32(monoChunk);
                macVADProcessor.processAudio(float32Audio);
            } else {
                // No VAD: send directly to Gemini (legacy behavior)
                const base64Data = monoChunk.toString('base64');
                sendAudioToGemini(base64Data, geminiSessionRef);
            }

            if (process.env.DEBUG_AUDIO) {
                console.log(`Processed audio chunk: ${chunk.length} bytes`);
                saveDebugAudio(monoChunk, 'system_audio');
            }
        }

        const maxBufferSize = SAMPLE_RATE * BYTES_PER_SAMPLE * 1;
        if (audioBuffer.length > maxBufferSize) {
            audioBuffer = audioBuffer.slice(-maxBufferSize);
        }
    });

    systemAudioProc.stderr.on('data', data => {
        console.error('SystemAudioDump stderr:', data.toString());
    });

    systemAudioProc.on('close', code => {
        console.log('SystemAudioDump process closed with code:', code);
        systemAudioProc = null;
    });

    systemAudioProc.on('error', err => {
        console.error(' SystemAudioDump process error:', err);

        // Provide helpful error message for architecture issues
        if (err.code === 'ENOENT') {
            console.error('\n TROUBLESHOOTING:');
            console.error('1. SystemAudioDump may not have execute permissions');
            console.error('   Run: chmod +x /path/to/SystemAudioDump');
            console.error('\n2. Binary may be wrong architecture for your Mac');
            console.error('   - ARM64 binary requires Apple Silicon (M1/M2/M3)');
            console.error('   - x64 binary requires Intel Mac or Rosetta 2');
            console.error('\n3. For Intel Macs: Install Rosetta 2');
            console.error('   Run: softwareupdate --install-rosetta');
        }

        systemAudioProc = null;
    });

    return true;
}

function convertStereoToMono(stereoBuffer) {
    const samples = stereoBuffer.length / 4;
    const monoBuffer = Buffer.alloc(samples * 2);

    for (let i = 0; i < samples; i++) {
        const leftSample = stereoBuffer.readInt16LE(i * 4);
        monoBuffer.writeInt16LE(leftSample, i * 2);
    }

    return monoBuffer;
}

// Convert PCM Buffer (Int16) to Float32Array for VAD processing
function convertPCMBufferToFloat32(pcmBuffer) {
    const samples = pcmBuffer.length / 2; // 2 bytes per sample (Int16)
    const float32Array = new Float32Array(samples);

    for (let i = 0; i < samples; i++) {
        const int16Sample = pcmBuffer.readInt16LE(i * 2);
        // Convert from Int16 range [-32768, 32767] to Float32 range [-1, 1]
        float32Array[i] = int16Sample / (int16Sample < 0 ? 32768 : 32767);
    }

    return float32Array;
}

// Convert Float32Array back to PCM Buffer (Int16) for sending to Gemini
function convertFloat32ToPCMBuffer(float32Array) {
    const pcmBuffer = Buffer.alloc(float32Array.length * 2); // 2 bytes per sample

    for (let i = 0; i < float32Array.length; i++) {
        // Clamp to [-1, 1] range
        const sample = Math.max(-1, Math.min(1, float32Array[i]));
        // Convert from Float32 range [-1, 1] to Int16 range [-32768, 32767]
        const int16Sample = sample < 0 ? sample * 32768 : sample * 32767;
        pcmBuffer.writeInt16LE(Math.round(int16Sample), i * 2);
    }

    return pcmBuffer;
}

function stopMacOSAudioCapture() {
    // Clean up VAD processor
    if (macVADProcessor) {
        macVADProcessor.destroy();
        macVADProcessor = null;
        console.log('[macOS] VAD processor destroyed');
    }

    if (systemAudioProc) {
        console.log('Stopping SystemAudioDump...');
        systemAudioProc.kill('SIGTERM');
        systemAudioProc = null;
    }
}

async function sendAudioToGemini(base64Data, geminiSessionRef) {
    if (!geminiSessionRef.current) return;

    try {
        process.stdout.write('.');
        await geminiSessionRef.current.sendRealtimeInput({
            audio: {
                data: base64Data,
                mimeType: 'audio/pcm;rate=24000',
            },
        });
    } catch (error) {
        console.error('Error sending audio to Gemini:', error);
    }
}

function setupGeminiIpcHandlers(geminiSessionRef) {
    // Store the geminiSessionRef globally for reconnection access
    global.geminiSessionRef = geminiSessionRef;

    ipcMain.handle('initialize-gemini', async (event, apiKey, customPrompt, profile = 'interview', language = 'en-US', mode = 'interview', model = 'gemini-2.5-flash') => {
        const session = await initializeGeminiSession(apiKey, customPrompt, profile, language, false, mode, model);
        if (session) {
            geminiSessionRef.current = session;
            return true;
        }
        return false;
    });

    // Update model generation settings from renderer
    ipcMain.handle('update-generation-settings', async (event, settings) => {
        if (settings.temperature !== undefined) {
            generationSettings.temperature = settings.temperature;
        }
        if (settings.topP !== undefined) {
            generationSettings.topP = settings.topP;
        }
        if (settings.maxOutputTokens !== undefined) {
            generationSettings.maxOutputTokens = settings.maxOutputTokens;
        }
        console.log('[GEMINI] Generation settings updated:', generationSettings);
        return { success: true };
    });

    // Get model-specific max output tokens
    ipcMain.handle('get-model-max-tokens', async (event, model) => {
        return getMaxOutputTokensForModel(model);
    });

    ipcMain.handle('send-audio-content', async (event, { data, mimeType }) => {
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };
        try {
            process.stdout.write('.');
            await geminiSessionRef.current.sendRealtimeInput({
                audio: { data: data, mimeType: mimeType },
            });
            return { success: true };
        } catch (error) {
            console.error('Error sending system audio:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-image-content', async (event, { data }) => {
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };

        try {
            if (!data || typeof data !== 'string') {
                console.error('Invalid image data received');
                return { success: false, error: 'Invalid image data' };
            }

            const buffer = Buffer.from(data, 'base64');

            if (buffer.length < 1000) {
                console.error(`Image buffer too small: ${buffer.length} bytes`);
                return { success: false, error: 'Image buffer too small' };
            }

            process.stdout.write('!');

            // Send screenshot to Gemini session (exam/coding mode)
            // Interview mode screenshots route through groq.js instead
            await geminiSessionRef.current.sendRealtimeInput({
                media: { data: data, mimeType: 'image/jpeg' },
            });

            return { success: true };
        } catch (error) {
            console.error('Error sending image:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('send-text-message', async (event, text) => {
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };

        try {
            if (!text || typeof text !== 'string' || text.trim().length === 0) {
                return { success: false, error: 'Invalid text message' };
            }

            // Add language reminder for non-English languages
            let finalText = text.trim();
            if (storedLanguageName !== 'English') {
                finalText += ` (Remember: Respond in ${storedLanguageName})`;
            }

            console.log('Sending text message:', finalText);
            await geminiSessionRef.current.sendRealtimeInput({ text: finalText });
            return { success: true };
        } catch (error) {
            console.error('Error sending text:', error);
            return { success: false, error: error.message };
        }
    });

    // Combined handler: Send screenshot + text
    ipcMain.handle('send-screenshot-with-text', async (event, { imageData, text }) => {
        if (!geminiSessionRef.current) return { success: false, error: 'No active Gemini session' };

        try {
            if (!imageData || typeof imageData !== 'string') {
                return { success: false, error: 'Invalid image data' };
            }

            if (!text || typeof text !== 'string' || text.trim().length === 0) {
                return { success: false, error: 'Invalid text message' };
            }

            // Add language reminder for non-English languages
            let finalText = text.trim();
            if (storedLanguageName !== 'English') {
                finalText += ` (Remember: Respond in ${storedLanguageName})`;
            }

            console.log('Sending screenshot + text:', finalText);
            process.stdout.write('!');

            await geminiSessionRef.current.sendRealtimeInput({
                media: { data: imageData, mimeType: 'image/jpeg' },
                text: finalText
            });

            return { success: true };
        } catch (error) {
            console.error('Error sending screenshot with text:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('start-macos-audio', async (event, vadEnabled = false, vadMode = 'automatic') => {
        if (process.platform !== 'darwin') {
            return {
                success: false,
                error: 'macOS audio capture only available on macOS',
            };
        }

        try {
            const success = await startMacOSAudioCapture(geminiSessionRef, vadEnabled, vadMode);
            return { success };
        } catch (error) {
            console.error('Error starting macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('stop-macos-audio', async event => {
        try {
            stopMacOSAudioCapture();
            return { success: true };
        } catch (error) {
            console.error('Error stopping macOS audio capture:', error);
            return { success: false, error: error.message };
        }
    });

    // macOS microphone toggle handler (for manual VAD mode)
    ipcMain.handle('toggle-macos-microphone', async (event, enabled) => {
        try {
            if (process.platform !== 'darwin') {
                return { success: false, error: 'macOS only' };
            }

            macMicrophoneEnabled = enabled;
            console.log(`🎤 [macOS] Microphone ${enabled ? 'enabled' : 'disabled'}`);

            if (macVADProcessor && macVADMode === 'manual') {
                if (enabled) {
                    // Manual mode: enable mic and start recording
                    macVADProcessor.resume();
                    console.log('[macOS MANUAL] Mic ON - now recording');
                } else {
                    // Manual mode: disable mic and commit audio
                    if (macVADProcessor.audioBuffer && macVADProcessor.audioBuffer.length > 0) {
                        console.log('[macOS MANUAL] Mic OFF - committing audio');
                        macVADProcessor.commit();
                    } else {
                        macVADProcessor.pause();
                        console.log('[macOS MANUAL] Mic OFF - no audio to commit');
                    }
                }
            }

            return { success: true, enabled: macMicrophoneEnabled };
        } catch (error) {
            console.error('Error toggling macOS microphone:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('close-session', async event => {
        try {
            stopMacOSAudioCapture();

            // Cleanup any pending resources and stop audio/video capture
            if (geminiSessionRef.current) {
                await geminiSessionRef.current.close();
                geminiSessionRef.current = null;
            }

            return { success: true };
        } catch (error) {
            console.error('Error closing session:', error);
            return { success: false, error: error.message };
        }
    });

    // VAD mode update handler
    ipcMain.handle('update-vad-mode', async (event, vadMode) => {
        try {
            console.log(`VAD mode updated to: ${vadMode}`);
            // The renderer process will handle the VAD mode change
            // This handler is mainly for logging and potential future use
            return { success: true, mode: vadMode };
        } catch (error) {
            console.error('Error updating VAD mode:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('update-google-search-setting', async (event, enabled) => {
        try {
            console.log('Google Search setting updated to:', enabled);
            // The setting is already saved in localStorage by the renderer
            // This is just for logging/confirmation
            return { success: true };
        } catch (error) {
            console.error('Error updating Google Search setting:', error);
            return { success: false, error: error.message };
        }
    });
}

/**
 * Chat with Gemini using text (and optional image).
 * Used by groq.js when interview model is gemini-3-flash-preview:
 *   Groq Whisper (STT) → transcription → chatWithGeminiText() → Gemini response
 *
 * @param {string} text - The transcription or prompt text
 * @param {string|null} imageData - Optional base64 image data for screenshot analysis
 * @returns {Promise<string|null>} The response text, or null on error
 */
async function chatWithGeminiText(text, imageData = null) {
    const session = global.geminiSessionRef?.current;
    if (!session) {
        console.error('[GEMINI] No active session for text chat');
        sendToRenderer('update-status', 'No Gemini session');
        return null;
    }

    const input = {};
    const hasImage = !!imageData;
    if (text) {
        // Append full formatting instructions from prompts.js (profile-aware)
        // Gemini follows per-message instructions better than system prompt alone
        input.text = text + getGeminiMessageHint(hasImage, currentProfile);
    }
    if (imageData) {
        input.media = { data: imageData, mimeType: 'image/jpeg' };
    }

    return await session.sendRealtimeInput(input);
}

module.exports = {
    initializeGeminiSession,
    chatWithGeminiText,
    getEnabledTools,
    getStoredSetting,
    sendToRenderer,
    killExistingSystemAudioDump,
    startMacOSAudioCapture,
    convertStereoToMono,
    convertPCMBufferToFloat32,
    convertFloat32ToPCMBuffer,
    stopMacOSAudioCapture,
    sendAudioToGemini,
    setupGeminiIpcHandlers,
};
