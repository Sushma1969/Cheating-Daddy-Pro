// groq.js - Groq API integration for Speech-to-Text (Whisper) and Chat Completion (Qwen models)
const { BrowserWindow, ipcMain } = require('electron');
const https = require('https');
const { URL } = require('url');
const { getCondensedSystemPrompt, getSystemPrompt, getExamMessageHint } = require('./prompts');
const { chatWithGeminiText, clearGeminiRateLimitCountdown } = require('./gemini');

// Groq API configuration
const GROQ_API_BASE = 'https://api.groq.com/openai/v1';
const WHISPER_MODEL = 'whisper-large-v3-turbo';

// Available Groq models for chat completion
// Qwen 3.6 27B replaced Llama 4 Maverick/Scout (both deprecated by Groq in 2026)
// It's the only vision-capable Groq model, needed for screenshot analysis (max 3 images/request)
const GROQ_CHAT_MODELS = {
    'qwen-3.6-27b': 'qwen/qwen3.6-27b'
};

// Audio buffer for accumulating audio chunks before sending to Groq
let speechBuffer = []; // Only contains speech segments
let contextBuffer = []; // Rolling buffer for pre-speech context
let isProcessing = false;
let groqApiKey = null;

// Conversation history for context
let conversationHistory = [];
let currentSystemPrompt = '';

// Minimum audio duration in seconds before sending to Groq (to avoid sending tiny clips)
const MIN_AUDIO_DURATION_SECONDS = 1.5; // Reduced since we now only buffer speech
const SAMPLE_RATE = 24000; // 24kHz as used in the app
const BYTES_PER_SAMPLE = 2; // 16-bit PCM

// Context buffer settings - keep some silence before speech starts
const MAX_CONTEXT_CHUNKS = 10; // Keep ~1 second of pre-speech context

// Speech detection thresholds
const SILENCE_RMS_THRESHOLD = 300; // RMS below this is considered silence (lowered for sensitivity)
const SPEECH_RMS_THRESHOLD = 500; // RMS above this is considered speech

// Speech state tracking for smarter flush
let lastSpeechTime = 0; // Timestamp of last detected speech
let isSpeaking = false; // Currently in speech segment
const SILENCE_AFTER_SPEECH_MS = 1500; // Wait 1.5 seconds of silence after speech before flushing
const POST_SPEECH_CONTEXT_MS = 500; // Include 0.5s of silence after speech ends

// Periodic check timer
let checkTimer = null;
const CHECK_INTERVAL_MS = 500; // Check every 500ms for faster response

// Store selected model for chat completion
let selectedGroqModel = 'qwen-3.6-27b';

// Track mode: 'interview' (audio + concise answers) or 'exam' (screenshot-based, thinking ON)
let currentGroqMode = 'interview';

// Rate limit countdown - auto-reset status after 429 errors with live countdown in header
let rateLimitCountdownInterval = null;

// Store selected language name for use in prompts
let storedLanguageName = 'English';

// Generation settings (can be updated from AdvancedView like Gemini)
let generationSettings = {
    temperature: 0.7,
    topP: 0.95,
    maxOutputTokens: 4096, // Default for interview mode, allows detailed technical answers
};

function sendToRenderer(channel, data) {
    const windows = BrowserWindow.getAllWindows();
    if (windows.length > 0) {
        windows[0].webContents.send(channel, data);
    }
}

/**
 * Start a live countdown in the header after a 429 rate limit error.
 * Updates the status every second: "Rate Limit: Tokens/min exceeded (15s)" → (14s) → ... → "Listening..."
 * Edge case: if user restarts app and hits 429 again, Groq returns fresh retry time and countdown restarts.
 */
function scheduleRateLimitRecovery(statusMessage, recoveryMs = 30 * 1000) {
    // Clear any existing countdown to avoid duplicates
    if (rateLimitCountdownInterval) {
        clearInterval(rateLimitCountdownInterval);
        rateLimitCountdownInterval = null;
    }

    let remainingSec = Math.ceil(recoveryMs / 1000);
    console.log(`[GROQ] Rate limit hit - countdown ${remainingSec}s`);

    // Show initial status with countdown
    sendToRenderer('update-status', `${statusMessage} (${remainingSec}s)`);

    rateLimitCountdownInterval = setInterval(() => {
        remainingSec--;

        if (remainingSec <= 0) {
            clearInterval(rateLimitCountdownInterval);
            rateLimitCountdownInterval = null;
            console.log('[GROQ] Rate limit countdown done - resetting to Listening...');
            sendToRenderer('update-status', 'Listening...');
        } else {
            sendToRenderer('update-status', `${statusMessage} (${remainingSec}s)`);
        }
    }, 1000);
}

/**
 * Clear any active Groq rate limit countdown.
 * Also called from gemini.js on session init — quota is per model, so a stale Groq
 * countdown must not keep overwriting the header after switching to a Gemini model.
 */
function clearGroqRateLimitCountdown() {
    if (rateLimitCountdownInterval) {
        clearInterval(rateLimitCountdownInterval);
        rateLimitCountdownInterval = null;
    }
}

/**
 * Parse the Groq 429 error response to determine rate limit type and extract the actual retry wait time.
 * Groq error messages contain "Please try again in XX.XXs" with the exact wait time.
 */
function parseRateLimitError(errorBody) {
    let statusMessage = 'API Quota Exceeded';
    let recoveryMs = 30 * 1000; // Fallback 30s if we can't parse

    try {
        const parsed = JSON.parse(errorBody);
        const msg = parsed.error?.message || '';
        const msgLower = msg.toLowerCase();

        // Extract the exact retry time from "Please try again in XX.XXs"
        const retryMatch = msg.match(/try again in (\d+\.?\d*)s/i);
        if (retryMatch) {
            const retrySec = parseFloat(retryMatch[1]);
            // Add 2s buffer to ensure the limit has fully reset
            recoveryMs = Math.ceil((retrySec + 2) * 1000);
        }

        // Determine the rate limit type for the status message
        if (msgLower.includes('tokens per minute') || msgLower.includes('tpm')) {
            statusMessage = 'Rate Limit: Tokens/min exceeded';
        } else if (msgLower.includes('tokens per day') || msgLower.includes('tpd')) {
            statusMessage = 'Rate Limit: Daily token limit reached';
        } else if (msgLower.includes('requests per minute') || msgLower.includes('rpm')) {
            statusMessage = 'Rate Limit: Requests/min exceeded';
        } else if (msgLower.includes('requests per day') || msgLower.includes('rpd')) {
            statusMessage = 'Rate Limit: Daily request limit reached';
        } else if (msgLower.includes('tokens per hour') || msgLower.includes('tph')) {
            statusMessage = 'Rate Limit: Tokens/hour exceeded';
        } else if (msgLower.includes('requests per hour') || msgLower.includes('rph')) {
            statusMessage = 'Rate Limit: Requests/hour exceeded';
        }

        console.log(`[GROQ] Rate limit details: ${msg}`);
    } catch (e) {
        // Failed to parse error body, use default message
        console.warn('[GROQ] Could not parse 429 error body:', errorBody);
    }

    return { statusMessage, recoveryMs };
}

/**
 * Calculate RMS (Root Mean Square) energy of PCM audio buffer
 */
function calculateRMS(pcmBuffer) {
    const samples = pcmBuffer.length / 2; // 2 bytes per sample (16-bit)
    let sumSquares = 0;

    for (let i = 0; i < samples; i++) {
        const sample = pcmBuffer.readInt16LE(i * 2);
        sumSquares += sample * sample;
    }

    return Math.sqrt(sumSquares / samples);
}

/**
 * Initialize Groq API with the provided API key
 */
function initializeGroq(apiKey, customPrompt = '', profile = 'interview', language = 'en-US', model = 'qwen-3.6-27b') {
    groqApiKey = apiKey;
    conversationHistory = [];
    selectedGroqModel = model;
    console.log(`[GROQ] Chat model set to: ${selectedGroqModel}`);

    // Clear any active rate limit countdown from previous session — BOTH providers,
    // so a stale Gemini countdown doesn't survive a switch to Qwen (quota is per model)
    clearGroqRateLimitCountdown();
    clearGeminiRateLimitCountdown();

    // Exam mode: use the SAME full exam prompt the Gemini models follow (compact enough for Groq)
    // Interview mode: use CONDENSED prompt (high request rate, full interview prompt ~27KB is too heavy)
    currentGroqMode = profile === 'exam' ? 'exam' : 'interview';
    if (currentGroqMode === 'exam') {
        currentSystemPrompt = getSystemPrompt(profile, customPrompt, false); // false = no Google Search on Groq
    } else {
        currentSystemPrompt = getCondensedSystemPrompt(profile, customPrompt);
    }

    // Add language instruction - matches Gemini's full language support
    const languageMap = {
        'en-US': 'English', 'en-GB': 'English', 'en-AU': 'English', 'en-IN': 'English',
        'es-ES': 'Spanish', 'es-US': 'Spanish', 'fr-FR': 'French', 'fr-CA': 'French',
        'de-DE': 'German', 'it-IT': 'Italian', 'pt-BR': 'Portuguese', 'pt-PT': 'Portuguese',
        'ru-RU': 'Russian', 'ja-JP': 'Japanese', 'ko-KR': 'Korean',
        'zh-CN': 'Chinese (Simplified)', 'cmn-CN': 'Chinese (Simplified)', 'zh-TW': 'Chinese (Traditional)',
        'ar-SA': 'Arabic', 'ar-XA': 'Arabic', 'hi-IN': 'Hindi',
        'nl-NL': 'Dutch', 'pl-PL': 'Polish', 'tr-TR': 'Turkish',
        'sv-SE': 'Swedish', 'da-DK': 'Danish', 'fi-FI': 'Finnish', 'no-NO': 'Norwegian',
        'th-TH': 'Thai', 'te-IN': 'Telugu', 'ta-IN': 'Tamil', 'mr-IN': 'Marathi',
        'ml-IN': 'Malayalam', 'kn-IN': 'Kannada', 'gu-IN': 'Gujarati', 'bn-IN': 'Bengali',
        'vi-VN': 'Vietnamese', 'id-ID': 'Indonesian',
    };

    const selectedLanguageName = languageMap[language] || 'English';
    storedLanguageName = selectedLanguageName; // Store for use in text/screenshot prompts
    currentSystemPrompt += `\n\n=== CRITICAL LANGUAGE INSTRUCTION ===
The user has selected ${selectedLanguageName} as their preferred language.
YOU MUST respond ONLY in ${selectedLanguageName}, regardless of what language the interviewer uses.`;

    console.log('[GROQ] Initialized with API key for profile:', profile);
    return true;
}

/**
 * Convert PCM audio buffer to WAV format
 */
function pcmToWav(pcmBuffer) {
    const numChannels = 1;
    const sampleRate = SAMPLE_RATE;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
    const blockAlign = numChannels * (bitsPerSample / 8);
    const dataSize = pcmBuffer.length;
    const fileSize = 36 + dataSize;

    const wavBuffer = Buffer.alloc(44 + dataSize);
    let offset = 0;

    // RIFF header
    wavBuffer.write('RIFF', offset); offset += 4;
    wavBuffer.writeUInt32LE(fileSize, offset); offset += 4;
    wavBuffer.write('WAVE', offset); offset += 4;

    // fmt chunk
    wavBuffer.write('fmt ', offset); offset += 4;
    wavBuffer.writeUInt32LE(16, offset); offset += 4;
    wavBuffer.writeUInt16LE(1, offset); offset += 2;
    wavBuffer.writeUInt16LE(numChannels, offset); offset += 2;
    wavBuffer.writeUInt32LE(sampleRate, offset); offset += 4;
    wavBuffer.writeUInt32LE(byteRate, offset); offset += 4;
    wavBuffer.writeUInt16LE(blockAlign, offset); offset += 2;
    wavBuffer.writeUInt16LE(bitsPerSample, offset); offset += 2;

    // data chunk
    wavBuffer.write('data', offset); offset += 4;
    wavBuffer.writeUInt32LE(dataSize, offset); offset += 4;

    // Copy PCM data
    pcmBuffer.copy(wavBuffer, offset);

    return wavBuffer;
}

/**
 * Send audio to Groq Whisper API for transcription
 */
async function transcribeWithGroq(wavBuffer) {
    return new Promise((resolve, reject) => {
        if (!groqApiKey) {
            reject(new Error('Groq API key not initialized'));
            return;
        }

        const boundary = '----FormBoundary' + Math.random().toString(36).substring(2);

        // Build multipart form data
        const formDataParts = [];

        formDataParts.push(
            `--${boundary}\r\n`,
            `Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n`,
            `Content-Type: audio/wav\r\n\r\n`
        );

        const filePartHeader = Buffer.from(formDataParts.join(''));

        const modelPart = Buffer.from(
            `\r\n--${boundary}\r\n` +
            `Content-Disposition: form-data; name="model"\r\n\r\n` +
            `${WHISPER_MODEL}\r\n` +
            `--${boundary}--\r\n`
        );

        const requestBody = Buffer.concat([filePartHeader, wavBuffer, modelPart]);

        const url = new URL(`${GROQ_API_BASE}/audio/transcriptions`);

        const options = {
            hostname: url.hostname,
            port: 443,
            path: url.pathname,
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${groqApiKey}`,
                'Content-Type': `multipart/form-data; boundary=${boundary}`,
                'Content-Length': requestBody.length
            }
        };

        const req = https.request(options, (res) => {
            let data = '';

            res.on('data', (chunk) => {
                data += chunk;
            });

            res.on('end', () => {
                if (res.statusCode === 200) {
                    try {
                        const response = JSON.parse(data);
                        const transcription = response.text || '';
                        console.log(`[GROQ WHISPER] Transcription: ${transcription.length} chars`);
                        resolve(transcription);
                    } catch (e) {
                        console.error('[GROQ] Failed to parse response:', e);
                        reject(e);
                    }
                } else {
                    console.error('[GROQ] Whisper API Error:', res.statusCode, data);
                    // Handle specific error codes like Gemini does - user-friendly messages
                    if (res.statusCode === 401) {
                        sendToRenderer('update-status', 'Invalid API Key (Groq)');
                        reject(new Error('Invalid API Key (Groq)'));
                    } else if (res.statusCode === 429) {
                        const rateLimit = parseRateLimitError(data);
                        scheduleRateLimitRecovery(rateLimit.statusMessage, rateLimit.recoveryMs);
                        reject(new Error(rateLimit.statusMessage));
                    } else if (res.statusCode === 413) {
                        sendToRenderer('update-status', 'Audio too long');
                        reject(new Error('Audio too long'));
                    } else if (res.statusCode === 403) {
                        const errLower = data.toLowerCase();
                        if (errLower.includes('location') || errLower.includes('region') || errLower.includes('country')) {
                            sendToRenderer('update-status', 'Region Not Supported (Groq)');
                            reject(new Error('Region Not Supported (Groq)'));
                        } else {
                            sendToRenderer('update-status', 'Access Denied (Groq)');
                            reject(new Error('Access Denied (Groq)'));
                        }
                    } else if (res.statusCode === 400) {
                        sendToRenderer('update-status', 'Invalid request');
                        reject(new Error('Invalid request'));
                    } else if (res.statusCode >= 500) {
                        sendToRenderer('update-status', 'Server error');
                        reject(new Error('Server error'));
                    } else {
                        sendToRenderer('update-status', 'Connection error');
                        reject(new Error('Connection error'));
                    }
                }
            });
        });

        req.on('error', (e) => {
            console.error('[GROQ] Request error:', e);
            reject(e);
        });

        req.write(requestBody);
        req.end();
    });
}

/**
 * Send chat completion request to Groq chat model (Qwen)
 * maxTokensOverride: set by the 413 auto-retry to shrink max_tokens into the TPM window
 * forceNoThink: set by the empty-answer retry when thinking consumed the whole token budget
 */
async function chatWithGroq(userMessage, model = 'qwen-3.6-27b', imageData = null, maxTokensOverride = null, forceNoThink = false) {
    return new Promise((resolve, reject) => {
        if (!groqApiKey) {
            reject(new Error('Groq API key not initialized'));
            return;
        }

        const modelId = GROQ_CHAT_MODELS[model] || GROQ_CHAT_MODELS['qwen-3.6-27b'];
        const isQwen = modelId.startsWith('qwen/');

        // Build messages array with conversation history
        const messages = [
            { role: 'system', content: currentSystemPrompt }
        ];

        // Add conversation history for context
        for (const turn of conversationHistory) {
            messages.push({ role: 'user', content: turn.userMessage });
            messages.push({ role: 'assistant', content: turn.assistantResponse });
        }

        // Add current user message
        if (imageData) {
            // Multimodal message with image
            messages.push({
                role: 'user',
                content: [
                    { type: 'text', text: userMessage },
                    { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${imageData}` } }
                ]
            });
        } else {
            messages.push({ role: 'user', content: userMessage });
        }

        // Qwen 3.6 dual-mode reasoning (per Groq/Qwen docs):
        // Interview → thinking OFF ('none') for low-latency replies
        // Exam → thinking ON ('default'); 'parsed' streams reasoning in a separate delta.reasoning
        // field so we can show "Thinking..." progress while keeping it out of the response
        // (Mode-specific sampling temps are set via AdvancedView defaults)
        const qwenParams = isQwen
            ? (currentGroqMode === 'exam' && !forceNoThink
                ? { reasoning_effort: 'default', reasoning_format: 'parsed' }
                : { reasoning_effort: 'none' })
            : {};

        // Groq counts input tokens + max_tokens against the TPM limit (free tier: 8000 for Qwen),
        // so max_tokens must stay modest — the 413 auto-retry below shrinks it further if needed
        const effectiveMaxTokens = maxTokensOverride || generationSettings.maxOutputTokens;

        let requestBody = JSON.stringify({
            model: modelId,
            messages: messages,
            temperature: generationSettings.temperature,
            top_p: generationSettings.topP,
            max_tokens: effectiveMaxTokens,
            stream: true,
            ...qwenParams
        });

        // Log request size for debugging
        let requestSizeKB = (Buffer.byteLength(requestBody) / 1024).toFixed(1);
        console.log(`[GROQ] Request body size: ${requestSizeKB}KB (${messages.length} messages, ${conversationHistory.length} history turns)`);

        // Groq has ~4MB request limit, but we should stay well under for performance
        // If request is too large, clear history and retry with just current message
        const MAX_REQUEST_SIZE = 500 * 1024; // 500KB safety limit (system prompt is already huge)
        if (Buffer.byteLength(requestBody) > MAX_REQUEST_SIZE) {
            console.warn(`[GROQ] Request too large (${requestSizeKB}KB > 500KB limit), clearing history and retrying...`);
            conversationHistory = []; // Clear history
            
            // Rebuild messages with just system prompt and current message
            const trimmedMessages = [
                { role: 'system', content: currentSystemPrompt }
            ];
            if (imageData) {
                trimmedMessages.push({
                    role: 'user',
                    content: [
                        { type: 'image_url', image_url: { url: `data:image/png;base64,${imageData}` } },
                        { type: 'text', text: userMessage }
                    ]
                });
            } else {
                trimmedMessages.push({ role: 'user', content: userMessage });
            }
            
            // Recreate request body with trimmed messages
            requestBody = JSON.stringify({
                model: modelId,
                messages: trimmedMessages,
                temperature: generationSettings.temperature,
                top_p: generationSettings.topP,
                max_tokens: effectiveMaxTokens,
                stream: true,
                ...qwenParams
            });
            requestSizeKB = (Buffer.byteLength(requestBody) / 1024).toFixed(1);
            console.log(`[GROQ] Trimmed request size: ${requestSizeKB}KB`);
        }

        const url = new URL(`${GROQ_API_BASE}/chat/completions`);

        const options = {
            hostname: url.hostname,
            port: 443,
            path: url.pathname,
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${groqApiKey}`,
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(requestBody)
            }
        };

        let responseText = '';
        let rawErrorBody = '';
        let sseBuffer = ''; // SSE lines can split across TCP chunks — buffer incomplete lines
        let reasoningChars = 0;
        let thinkingStatusSent = false;
        let firstTokenLogged = false;
        const requestStartTime = Date.now();
        let lastActivityTime = Date.now();

        const req = https.request(options, (res) => {
            res.on('data', (chunk) => {
                lastActivityTime = Date.now();
                if (res.statusCode !== 200) {
                    // Accumulate error response body for detailed error messages
                    rawErrorBody += chunk.toString();
                    return;
                }
                sseBuffer += chunk.toString();
                const lines = sseBuffer.split('\n');
                sseBuffer = lines.pop(); // keep the (possibly incomplete) last line for the next chunk
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') continue;
                        try {
                            const parsed = JSON.parse(data);
                            const delta = parsed.choices?.[0]?.delta;
                            // Qwen reasoning streams separately (parsed format) — show progress, never render it
                            if (delta?.reasoning) {
                                reasoningChars += delta.reasoning.length;
                                if (!thinkingStatusSent) {
                                    thinkingStatusSent = true;
                                    console.log(`[GROQ] Model started thinking (${Date.now() - requestStartTime}ms after request)`);
                                    sendToRenderer('update-status', 'Thinking...');
                                }
                            }
                            const content = delta?.content;
                            if (content) {
                                if (!firstTokenLogged) {
                                    firstTokenLogged = true;
                                    console.log(`[GROQ] First answer token: ${Date.now() - requestStartTime}ms (hidden reasoning so far: ${reasoningChars} chars)`);
                                }
                                responseText += content;
                                // Stream to renderer
                                sendToRenderer('update-response', responseText);
                            }
                            // Groq attaches usage stats to the final chunk — queue_time here reveals
                            // whether a slow request was Groq queuing vs actual generation/thinking
                            const usage = parsed.x_groq?.usage || parsed.usage;
                            if (usage && usage.total_time !== undefined) {
                                console.log(`[GROQ] Usage: queue ${Number(usage.queue_time || 0).toFixed(2)}s | prompt ${usage.prompt_tokens} tok | completion ${usage.completion_tokens} tok in ${Number(usage.completion_time || 0).toFixed(2)}s`);
                            }
                        } catch (e) {
                            // Skip invalid JSON lines
                        }
                    }
                }
            });

            res.on('end', () => {
                clearInterval(idleWatchdog);
                if (res.statusCode === 200 && responseText) {
                    console.log(`[GROQ CHAT] Response: ${responseText.length} chars in ${((Date.now() - requestStartTime) / 1000).toFixed(1)}s${reasoningChars ? ` (+${reasoningChars} hidden reasoning chars)` : ''}`);

                    // Save to conversation history
                    conversationHistory.push({
                        userMessage: userMessage,
                        assistantResponse: responseText
                    });

                    // Limit history to last 10 turns for better context
                    // Size budget: Prompt ~2KB + 10 turns × ~3KB avg = ~32KB (well under 500KB limit)
                    // Even with screenshots, we have room since images aren't stored in history
                    if (conversationHistory.length > 10) {
                        conversationHistory = conversationHistory.slice(-10);
                    }

                    // Interview mode: back to "Listening..." | Exam mode: "Ready" (waiting for next screenshot)
                    sendToRenderer('update-status', currentGroqMode === 'exam' ? 'Ready' : 'Listening...');
                    resolve(responseText);
                } else if (res.statusCode !== 200) {
                    console.error('[GROQ] Chat API Error:', res.statusCode, rawErrorBody);
                    // Handle specific error codes like Gemini does
                    if (res.statusCode === 401) {
                        sendToRenderer('update-status', 'Invalid API Key (Groq)');
                        reject(new Error('Invalid API Key (Groq)'));
                    } else if (res.statusCode === 429) {
                        const rateLimit = parseRateLimitError(rawErrorBody);
                        scheduleRateLimitRecovery(rateLimit.statusMessage, rateLimit.recoveryMs);
                        reject(new Error(rateLimit.statusMessage));
                    } else if (res.statusCode === 403) {
                        const errLower = rawErrorBody.toLowerCase();
                        if (errLower.includes('location') || errLower.includes('region') || errLower.includes('country')) {
                            sendToRenderer('update-status', 'Region Not Supported (Groq)');
                            reject(new Error('Region Not Supported (Groq)'));
                        } else {
                            sendToRenderer('update-status', 'Access Denied (Groq)');
                            reject(new Error('Access Denied (Groq)'));
                        }
                    } else if (res.statusCode === 413) {
                        // Groq's TPM 413 includes exact numbers: "Limit 8000, Requested 12271"
                        // Retry ONCE with max_tokens shrunk to fit the remaining TPM window
                        const tpmMatch = rawErrorBody.match(/Limit (\d+), Requested (\d+)/i);
                        if (tpmMatch && maxTokensOverride === null) {
                            const tpmLimit = parseInt(tpmMatch[1], 10);
                            const requested = parseInt(tpmMatch[2], 10);
                            const inputTokens = requested - effectiveMaxTokens;
                            const reducedMax = tpmLimit - inputTokens - 256; // 256 token safety buffer
                            if (reducedMax >= 512) {
                                console.log(`[GROQ] TPM limit ${tpmLimit}, input ~${inputTokens} tokens — retrying with max_tokens ${reducedMax}`);
                                resolve(chatWithGroq(userMessage, model, imageData, reducedMax));
                                return;
                            }
                        }
                        sendToRenderer('update-status', 'Request too large');
                        reject(new Error('Request too large'));
                    } else if (res.statusCode === 400) {
                        sendToRenderer('update-status', 'Invalid request');
                        reject(new Error('Invalid request'));
                    } else if (res.statusCode >= 500) {
                        sendToRenderer('update-status', 'Server error');
                        reject(new Error('Server error'));
                    } else {
                        sendToRenderer('update-status', 'Connection error');
                        reject(new Error('Connection error'));
                    }
                } else {
                    // HTTP 200 but empty answer: thinking consumed the entire max_tokens budget
                    // (happens on free tier where the 413 retry shrinks the budget) — retry once without thinking
                    if (reasoningChars > 0 && !forceNoThink && currentGroqMode === 'exam') {
                        console.log(`[GROQ] Thinking ate the whole ${effectiveMaxTokens}-token budget with no answer — retrying without thinking`);
                        resolve(chatWithGroq(userMessage, model, imageData, maxTokensOverride, true));
                        return;
                    }
                    sendToRenderer('update-status', currentGroqMode === 'exam' ? 'Ready' : 'Listening...');
                    resolve(responseText);
                }
            });
        });

        // Idle watchdog: with 'parsed' reasoning even thinking keeps the stream active,
        // so 120s of total silence means a hung request/queue — fail fast instead of hanging forever
        const idleWatchdog = setInterval(() => {
            if (Date.now() - lastActivityTime > 120 * 1000) {
                clearInterval(idleWatchdog);
                console.error('[GROQ] Stream idle for 120s — aborting request');
                req.destroy(new Error('Request Timeout, Please Try Again'));
            }
        }, 10 * 1000);

        req.on('error', (e) => {
            clearInterval(idleWatchdog);
            console.error('[GROQ] Chat request error:', e);
            if (e.message === 'Request Timeout, Please Try Again') {
                sendToRenderer('update-status', 'Request Timeout, Please Try Again');
            }
            reject(e);
        });

        req.write(requestBody);
        req.end();
    });
}

/**
 * Add audio chunk to buffer with speech detection
 */
function addAudioChunk(pcmBuffer) {
    const chunkRMS = calculateRMS(pcmBuffer);
    const now = Date.now();
    const isSpeechChunk = chunkRMS >= SPEECH_RMS_THRESHOLD;

    if (isSpeechChunk) {
        // Speech detected!
        if (!isSpeaking) {
            // Speech just started - add context buffer first
            if (contextBuffer.length > 0) {
                speechBuffer.push(...contextBuffer);
                contextBuffer = [];
            }
            isSpeaking = true;
            console.log(`[GROQ] Speech started (RMS: ${chunkRMS.toFixed(0)})`);
        }
        // Add speech chunk to buffer
        speechBuffer.push(pcmBuffer);
        lastSpeechTime = now;
    } else {
        // Silence or low audio
        if (isSpeaking) {
            // Was speaking, now silence - add some post-speech context
            const timeSinceSpeech = now - lastSpeechTime;
            if (timeSinceSpeech < POST_SPEECH_CONTEXT_MS) {
                // Still in post-speech window, keep adding
                speechBuffer.push(pcmBuffer);
            } else {
                // Post-speech context complete
                isSpeaking = false;
            }
        } else {
            // Pure silence - add to rolling context buffer (trim to max size)
            contextBuffer.push(pcmBuffer);
            while (contextBuffer.length > MAX_CONTEXT_CHUNKS) {
                contextBuffer.shift();
            }
        }
    }

    // Start periodic check timer if we have speech and timer not running
    if (speechBuffer.length > 0 && !checkTimer) {
        checkTimer = setInterval(() => {
            checkAndFlush();
        }, CHECK_INTERVAL_MS);
    }
}

/**
 * Check if we should flush the speech buffer
 */
async function checkAndFlush() {
    if (isProcessing || speechBuffer.length === 0) {
        return;
    }

    const now = Date.now();
    const timeSinceSpeech = now - lastSpeechTime;
    const speechBytes = speechBuffer.reduce((sum, buf) => sum + buf.length, 0);
    const speechDuration = speechBytes / (SAMPLE_RATE * BYTES_PER_SAMPLE);

    // Time-based speech end detection (handles push-to-talk mode where mic turns off)
    // If enough time has passed since last speech, mark speaking as ended
    if (isSpeaking && timeSinceSpeech >= SILENCE_AFTER_SPEECH_MS) {
        console.log(`[GROQ] No audio for ${(timeSinceSpeech/1000).toFixed(1)}s - marking speech ended`);
        isSpeaking = false;
    }

    // Log status occasionally
    if (speechDuration >= 1) {
        console.log(`[GROQ] Speech buffer: ${speechDuration.toFixed(1)}s | Silence: ${(timeSinceSpeech/1000).toFixed(1)}s | Speaking: ${isSpeaking}`);
    }

    // Flush conditions:
    // 1. Have enough audio AND sustained silence after speech
    // 2. Buffer too large (>20s) - force flush
    const shouldFlush =
        (speechDuration >= MIN_AUDIO_DURATION_SECONDS && timeSinceSpeech >= SILENCE_AFTER_SPEECH_MS && !isSpeaking) ||
        (speechDuration > 20);

    if (shouldFlush) {
        if (speechDuration > 20) {
            console.log(`[GROQ] Buffer large (${speechDuration.toFixed(1)}s) - flushing...`);
        } else {
            console.log(`[GROQ] Speech ended ${(timeSinceSpeech/1000).toFixed(1)}s ago - flushing...`);
        }
        await processAudioBuffer();
    }

    // Stop timer if no more speech
    if (speechBuffer.length === 0 && checkTimer) {
        clearInterval(checkTimer);
        checkTimer = null;
    }
}

/**
 * Process accumulated audio buffer: transcribe with Whisper, then send to chat model
 */
async function processAudioBuffer(model = null) {
    if (isProcessing || speechBuffer.length === 0) {
        return null;
    }

    // Use provided model or stored model
    const chatModel = model || selectedGroqModel;

    // Calculate total duration
    const totalBytes = speechBuffer.reduce((sum, buf) => sum + buf.length, 0);
    const totalDuration = totalBytes / (SAMPLE_RATE * BYTES_PER_SAMPLE);

    if (totalDuration < MIN_AUDIO_DURATION_SECONDS) {
        console.log(`[GROQ] Audio buffer too short (${totalDuration.toFixed(2)}s), waiting for more audio...`);
        return null;
    }

    isProcessing = true;
    sendToRenderer('update-status', 'Transcribing...');

    try {
        // Combine all audio chunks
        const combinedPcm = Buffer.concat(speechBuffer);
        speechBuffer = [];

        // Check audio energy
        const rms = calculateRMS(combinedPcm);
        console.log(`[GROQ] Audio RMS: ${rms.toFixed(0)}, threshold: ${SILENCE_RMS_THRESHOLD}`);

        if (rms < SILENCE_RMS_THRESHOLD) {
            console.log(`[GROQ] Audio too quiet (RMS: ${rms.toFixed(0)}), likely silence - skipping`);
            isProcessing = false;
            sendToRenderer('update-status', 'Listening...');
            return null;
        }

        console.log(`\n[GROQ] Processing ${totalDuration.toFixed(2)}s of audio (RMS: ${rms.toFixed(0)})...`);

        // Step 1: Transcribe with Whisper
        const wavBuffer = pcmToWav(combinedPcm);
        const transcription = await transcribeWithGroq(wavBuffer);

        if (!transcription || !transcription.trim()) {
            console.log('[GROQ] Empty transcription, skipping chat');
            isProcessing = false;
            sendToRenderer('update-status', 'Listening...');
            return null;
        }

        // Send transcription to renderer
        sendToRenderer('groq-transcription', transcription);
        sendToRenderer('update-status', 'Generating...');

        // Step 2: Send transcription to chat model for response
        let response;
        if (chatModel.startsWith('gemini-')) {
            // Gemini Flash Lite models (2.5 / 3.1) route to Gemini for text generation
            response = await chatWithGeminiText(transcription);
        } else {
            response = await chatWithGroq(transcription, chatModel);
        }

        // Reset speech tracking state
        isSpeaking = false;
        lastSpeechTime = 0;

        return { transcription, response };
    } catch (error) {
        console.error('[GROQ] Error processing audio:', error);
        // Only update status if it's not already showing a user-friendly error
        if (!['Invalid API Key (Groq)', 'Invalid API Key (Gemini)', 'API Quota Exceeded', 'Audio too long', 'Request too large', 'Server error', 'Connection error', 'Invalid request', 'Request Timeout, Please Try Again'].includes(error.message) && !error.message.startsWith('Rate Limit:')) {
            sendToRenderer('update-status', 'Processing failed');
        }
        isSpeaking = false;
        lastSpeechTime = 0;
        return null;
    } finally {
        isProcessing = false;
    }
}

/**
 * Force process the current audio buffer (called when VAD detects end of speech)
 */
async function flushAudioBuffer(model = null) {
    if (speechBuffer.length === 0) {
        return null;
    }

    // Cancel any pending check timer
    if (checkTimer) {
        clearInterval(checkTimer);
        checkTimer = null;
    }

    const totalBytes = speechBuffer.reduce((sum, buf) => sum + buf.length, 0);
    const totalDuration = totalBytes / (SAMPLE_RATE * BYTES_PER_SAMPLE);

    // Need at least 0.5 seconds for meaningful transcription
    // VAD triggered this flush, so we trust the end-of-speech detection
    if (totalDuration < 0.5) {
        // Too short - silently discard
        speechBuffer = [];
        contextBuffer = [];
        return null;
    }

    // Use provided model or stored model
    const chatModel = model || selectedGroqModel;

    isProcessing = true;
    sendToRenderer('update-status', 'Transcribing...');
    console.log(`\n[GROQ] Flush processing ${totalDuration.toFixed(2)}s of audio...`);

    try {
        const combinedPcm = Buffer.concat(speechBuffer);
        speechBuffer = [];
        contextBuffer = [];

        const wavBuffer = pcmToWav(combinedPcm);
        const transcription = await transcribeWithGroq(wavBuffer);

        if (!transcription || !transcription.trim()) {
            console.log('[GROQ] Empty transcription from flush');
            isProcessing = false;
            sendToRenderer('update-status', 'Listening...');
            return null;
        }

        sendToRenderer('groq-transcription', transcription);
        sendToRenderer('update-status', 'Generating...');

        // Send transcription to chat model for response
        let response;
        if (chatModel.startsWith('gemini-')) {
            // Gemini Flash Lite models (2.5 / 3.1) route to Gemini for text generation
            response = await chatWithGeminiText(transcription);
        } else {
            response = await chatWithGroq(transcription, chatModel);
        }

        // Reset state
        isSpeaking = false;
        lastSpeechTime = 0;

        return { transcription, response };
    } catch (error) {
        console.error('[GROQ] Error flushing audio:', error);
        // Only update status if it's not already showing a user-friendly error
        if (!['Invalid API Key (Groq)', 'Invalid API Key (Gemini)', 'API Quota Exceeded', 'Audio too long', 'Request too large', 'Server error', 'Connection error', 'Invalid request', 'Request Timeout, Please Try Again'].includes(error.message) && !error.message.startsWith('Rate Limit:')) {
            sendToRenderer('update-status', 'Processing failed');
        }
        return null;
    } finally {
        isProcessing = false;
    }
}

/**
 * Send screenshot + text to chat model for analysis
 */
async function analyzeWithGroq(text, imageData, model = 'qwen-3.6-27b') {
    if (!groqApiKey) {
        console.error('[GROQ] No API key initialized');
        sendToRenderer('update-status', 'No API Key Found');
        return null;
    }

    sendToRenderer('update-status', 'Analyzing...');
    console.log('[GROQ] Analyzing screenshot with text:', text.substring(0, 100) + '...');

    try {
        let finalText = text;
        // Exam mode: append per-message exam hints (code only / MCQ answer) — same as the Gemini path
        if (currentGroqMode === 'exam') {
            finalText += getExamMessageHint();
        }
        // Add language reminder for non-English languages
        if (storedLanguageName !== 'English') {
            finalText = `${finalText} (Remember: Respond in ${storedLanguageName})`;
        }

        let response;
        if (model.startsWith('gemini-')) {
            // Route to Gemini for screenshot analysis (Flash Lite 2.5 / 3.1)
            response = await chatWithGeminiText(finalText, imageData);
        } else {
            response = await chatWithGroq(finalText, model, imageData);
        }
        // Status will be set to 'Listening...' / 'Ready' by the respective handler
        return response;
    } catch (error) {
        console.error('[GROQ] Error analyzing:', error);
        // Only update status if it's not already showing a user-friendly error
        if (!['Invalid API Key (Groq)', 'Invalid API Key (Gemini)', 'API Quota Exceeded', 'Audio too long', 'Request too large', 'Server error', 'Connection error', 'Invalid request', 'Request Timeout, Please Try Again'].includes(error.message) && !error.message.startsWith('Rate Limit:')) {
            sendToRenderer('update-status', 'Analysis failed');
        }
        return null;
    }
}

/**
 * Clear the audio buffer without processing
 */
function clearAudioBuffer() {
    speechBuffer = [];
    contextBuffer = [];
    isSpeaking = false;
    lastSpeechTime = 0;
    if (checkTimer) {
        clearInterval(checkTimer);
        checkTimer = null;
    }
    console.log('[GROQ] Audio buffer cleared');
}

/**
 * Clear conversation history
 */
function clearConversationHistory() {
    conversationHistory = [];
    console.log('[GROQ] Conversation history cleared');
}

/**
 * Get current buffer duration in seconds
 */
function getBufferDuration() {
    const totalBytes = speechBuffer.reduce((sum, buf) => sum + buf.length, 0);
    return totalBytes / (SAMPLE_RATE * BYTES_PER_SAMPLE);
}

/**
 * Check if Groq is initialized
 */
function isGroqInitialized() {
    return groqApiKey !== null;
}

/**
 * Get conversation history
 */
function getConversationHistory() {
    return conversationHistory;
}

/**
 * Update generation settings (temperature, topP, maxOutputTokens)
 */
function updateGenerationSettings(settings) {
    if (settings.temperature !== undefined) {
        generationSettings.temperature = settings.temperature;
    }
    if (settings.topP !== undefined) {
        generationSettings.topP = settings.topP;
    }
    if (settings.maxOutputTokens !== undefined) {
        generationSettings.maxOutputTokens = settings.maxOutputTokens;
    }
    console.log('[GROQ] Generation settings updated:', generationSettings);
}
/**
 * Setup IPC handlers for Groq
 */
function setupGroqIpcHandlers() {
    ipcMain.handle('initialize-groq', async (event, apiKey, customPrompt = '', profile = 'interview', language = 'en-US', model = 'qwen-3.6-27b') => {
        try {
            initializeGroq(apiKey, customPrompt, profile, language, model);
            return { success: true };
        } catch (error) {
            console.error('[GROQ] Initialization error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-add-audio', async (event, { data }) => {
        try {
            const pcmBuffer = Buffer.from(data, 'base64');
            addAudioChunk(pcmBuffer);
            return { success: true };
        } catch (error) {
            console.error('[GROQ] Add audio error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-process-audio', async (event, model = 'qwen-3.6-27b') => {
        try {
            selectedGroqModel = model;
            const result = await processAudioBuffer(model);
            return { success: true, result };
        } catch (error) {
            console.error('[GROQ] Process audio error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-flush-audio', async (event, model = 'qwen-3.6-27b') => {
        try {
            selectedGroqModel = model;
            const result = await flushAudioBuffer(model);
            return { success: true, result };
        } catch (error) {
            console.error('[GROQ] Flush audio error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-clear-audio', async (event) => {
        clearAudioBuffer();
        return { success: true };
    });

    ipcMain.handle('groq-chat', async (event, { message, model, imageData }) => {
        try {
            // Add language reminder for non-English languages
            let finalMessage = message;
            if (storedLanguageName !== 'English') {
                finalMessage = `${message} (Remember: Respond in ${storedLanguageName})`;
            }
            const response = await chatWithGroq(finalMessage, model, imageData);
            return { success: true, response };
        } catch (error) {
            console.error('[GROQ] Chat error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-analyze-image', async (event, { text, imageData, model }) => {
        try {
            const response = await analyzeWithGroq(text, imageData, model);
            return { success: true, response };
        } catch (error) {
            console.error('[GROQ] Analyze image error:', error);
            return { success: false, error: error.message };
        }
    });

    ipcMain.handle('groq-clear-history', async (event) => {
        clearConversationHistory();
        return { success: true };
    });

    ipcMain.handle('groq-get-history', async (event) => {
        return { success: true, history: getConversationHistory() };
    });

    ipcMain.handle('groq-update-generation-settings', async (event, settings) => {
        try {
            updateGenerationSettings(settings);
            return { success: true };
        } catch (error) {
            console.error('[GROQ] Update settings error:', error);
            return { success: false, error: error.message };
        }
    });

    console.log('[GROQ] IPC handlers registered');
}

module.exports = {
    initializeGroq,
    pcmToWav,
    transcribeWithGroq,
    chatWithGroq,
    analyzeWithGroq,
    addAudioChunk,
    processAudioBuffer,
    flushAudioBuffer,
    clearAudioBuffer,
    clearConversationHistory,
    getBufferDuration,
    isGroqInitialized,
    updateGenerationSettings,
    getConversationHistory,
    setupGroqIpcHandlers,
    sendToRenderer,
    clearGroqRateLimitCountdown,
    GROQ_CHAT_MODELS
};
