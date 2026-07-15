/**
 * Language Source Sync Tests
 *
 * The existing language tests (groqLanguages.test.js / languages.test.js) verify DUPLICATED
 * copies of the language maps — they would still pass if the real maps in groq.js/gemini.js
 * drifted or lost entries. This suite closes that gap by checking the REAL source files:
 * 1. Every UI-supported language mapping exists in groq.js AND gemini.js languageMap
 * 2. The CRITICAL LANGUAGE INSTRUCTION template exists in both provider files
 * 3. The Qwen model mapping + exam-mode prompt routing + reasoning params exist in groq.js
 * 4. CustomizeView offers exactly these languages in its dropdown
 * 5. prompts.js is imported DIRECTLY (pure module) and its real output is verified
 *
 * Honest limitation: none of this proves a model actually replies in Hindi/Tamil/etc. —
 * that needs a live API call. These tests prove the app SENDS the correct instructions.
 */

const fs = require('fs');
const path = require('path');
const { getSystemPrompt, getCondensedSystemPrompt, getExamMessageHint } = require('../utils/prompts');

// The languages offered in the CustomizeView UI (code -> expected mapped name)
const SUPPORTED_LANGUAGES = [
    ['en-US', 'English'], ['en-GB', 'English'], ['en-AU', 'English'], ['en-IN', 'English'],
    ['de-DE', 'German'], ['es-US', 'Spanish'], ['es-ES', 'Spanish'],
    ['fr-FR', 'French'], ['fr-CA', 'French'], ['hi-IN', 'Hindi'],
    ['pt-BR', 'Portuguese'], ['ar-XA', 'Arabic'], ['id-ID', 'Indonesian'],
    ['it-IT', 'Italian'], ['ja-JP', 'Japanese'], ['tr-TR', 'Turkish'],
    ['vi-VN', 'Vietnamese'], ['bn-IN', 'Bengali'], ['gu-IN', 'Gujarati'],
    ['kn-IN', 'Kannada'], ['ml-IN', 'Malayalam'], ['mr-IN', 'Marathi'],
    ['ta-IN', 'Tamil'], ['te-IN', 'Telugu'], ['nl-NL', 'Dutch'],
    ['ko-KR', 'Korean'], ['cmn-CN', 'Chinese (Simplified)'], ['pl-PL', 'Polish'],
    ['ru-RU', 'Russian'], ['th-TH', 'Thai'],
];

const groqSource = fs.readFileSync(path.join(__dirname, '../utils/groq.js'), 'utf8');
const geminiSource = fs.readFileSync(path.join(__dirname, '../utils/gemini.js'), 'utf8');
const customizeSource = fs.readFileSync(path.join(__dirname, '../components/views/CustomizeView.js'), 'utf8');

describe('Language maps in the REAL source files (no duplicated-constant drift)', () => {
    it('groq.js languageMap contains every UI-supported language', () => {
        const missing = SUPPORTED_LANGUAGES.filter(([code, name]) => !groqSource.includes(`'${code}': '${name}'`)).map(([code]) => code);
        expect(missing).toEqual([]);
    });

    it('gemini.js languageMap contains every UI-supported language', () => {
        const missing = SUPPORTED_LANGUAGES.filter(([code, name]) => !geminiSource.includes(`'${code}': '${name}'`)).map(([code]) => code);
        expect(missing).toEqual([]);
    });

    it('CustomizeView offers every supported language in its dropdown', () => {
        const missing = SUPPORTED_LANGUAGES.filter(([code]) => !customizeSource.includes(`value: '${code}'`)).map(([code]) => code);
        expect(missing).toEqual([]);
    });

    it('both providers inject the critical language instruction into the system prompt', () => {
        for (const source of [groqSource, geminiSource]) {
            expect(source).toContain('=== CRITICAL LANGUAGE INSTRUCTION ===');
            expect(source).toContain('YOU MUST respond ONLY in ${selectedLanguageName}');
        }
    });

    it('both providers fall back to English for unknown language codes', () => {
        expect(groqSource).toContain("languageMap[language] || 'English'");
        expect(geminiSource).toContain("languageMap[language] || 'English'");
    });
});

describe('Qwen wiring in the REAL groq.js source', () => {
    it('maps the app model id to the Groq API model id', () => {
        expect(groqSource).toContain("'qwen-3.6-27b': 'qwen/qwen3.6-27b'");
    });

    it('routes exam profile to the full Gemini exam prompt (not the condensed interview one)', () => {
        expect(groqSource).toContain("currentGroqMode = profile === 'exam' ? 'exam' : 'interview'");
        expect(groqSource).toContain('getSystemPrompt(profile, customPrompt, false)');
    });

    it('uses mode-aware Qwen reasoning parameters', () => {
        expect(groqSource).toContain("reasoning_effort: 'default'");
        expect(groqSource).toContain("reasoning_effort: 'none'");
    });

    it('appends the exam message hint in exam mode', () => {
        expect(groqSource).toContain('getExamMessageHint()');
    });
});

describe('REAL prompts.js output (module imported directly)', () => {
    it('exam system prompt has MCQ + comment-free code rules that Qwen and Gemini share', () => {
        const examPrompt = getSystemPrompt('exam', '', false);

        expect(examPrompt).toContain('MCQ');
        expect(examPrompt).toContain('COMMENT-FREE');
        expect(examPrompt).toContain('Aptitude');
    });

    it('exam prompt drops the Google Search section when search is disabled (Qwen path)', () => {
        const withSearch = getSystemPrompt('exam', '', true);
        const withoutSearch = getSystemPrompt('exam', '', false);

        expect(withSearch).toContain('SEARCH TOOL USAGE');
        expect(withoutSearch).not.toContain('SEARCH TOOL USAGE');
    });

    it('condensed interview prompt embeds the user custom context', () => {
        const prompt = getCondensedSystemPrompt('interview', 'I am a React developer with 5 years experience');

        expect(prompt).toContain('interview assistant');
        expect(prompt).toContain('I am a React developer with 5 years experience');
    });

    it('exam message hint covers coding, MCQ and theory question types', () => {
        const hint = getExamMessageHint();

        expect(hint).toContain('IF CODING QUESTION');
        expect(hint).toContain('IF MCQ');
        expect(hint).toContain('THEORETICAL');
        expect(hint).toContain('DO NOT describe the screenshot');
    });
});
