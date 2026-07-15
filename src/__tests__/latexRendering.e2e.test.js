/**
 * LaTeX Rendering Tests (higher mathematics / ML formulas)
 *
 * Verifies the response render pipeline (marked -> KaTeX) handles math from BOTH providers:
 * 1. Gemini-style delimiters: $...$ (inline) and $$...$$ (display)
 * 2. Qwen/OpenAI-style delimiters: \(...\) (inline) and \[...\] (display)
 *    - marked strips those backslashes (\[ becomes [), so they must be normalized
 *      to $-style BEFORE parsing (AssistantView.normalizeLaTeXDelimiters)
 * 3. LaTeX-looking text inside code blocks must stay raw
 *
 * Uses the REAL marked + KaTeX assets loaded from index.html via JSDOM.
 */

const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

// Mirrors AssistantView.normalizeLaTeXDelimiters — keep in sync
function normalizeLaTeXDelimiters(content) {
    const collapse = latex => latex.trim().replace(/\s*\n\s*/g, ' ');
    return content
        .split(/(```[\s\S]*?```|`[^`\n]+`)/g)
        .map(segment => {
            if (segment.startsWith('`')) return segment;
            return segment
                .replace(/\\\[([\s\S]+?)\\\]/g, (match, latex) => `$$${collapse(latex)}$$`)
                .replace(/\\\(([\s\S]+?)\\\)/g, (match, latex) => `$${collapse(latex)}$`)
                .replace(/\$\$([\s\S]+?)\$\$/g, (match, latex) => `$$${collapse(latex)}$$`);
        })
        .join('');
}

// Mirrors AssistantView.renderLaTeX — keep in sync
function renderLaTeX(win, html) {
    if (!win.katex) return html;

    const doc = new win.DOMParser().parseFromString(html, 'text/html');

    function isInsideCode(node) {
        let parent = node.parentElement;
        while (parent) {
            if (parent.tagName === 'CODE' || parent.tagName === 'PRE') return true;
            parent = parent.parentElement;
        }
        return false;
    }

    function processTextNode(textNode) {
        if (isInsideCode(textNode)) return;

        let text = textNode.textContent;
        if (!/\$\$[\s\S]+?\$\$|\$[^\$\n]+?\$/.test(text)) return;

        text = text.replace(/\$\$([^\$]+?)\$\$/g, (match, latex) => {
            try {
                const rendered = win.katex.renderToString(latex.trim(), { displayMode: true, throwOnError: false, output: 'html' });
                return `<span class="katex-display">${rendered}</span>`;
            } catch (err) {
                return match;
            }
        });

        text = text.replace(/\$(?!\s)([^\$\n]+?)(?<!\s)\$/g, (match, latex) => {
            try {
                return win.katex.renderToString(latex.trim(), { displayMode: false, throwOnError: false, output: 'html' });
            } catch (err) {
                return match;
            }
        });

        if (text !== textNode.textContent) {
            const span = doc.createElement('span');
            span.innerHTML = text;
            textNode.parentNode.replaceChild(span, textNode);
        }
    }

    const walker = doc.createTreeWalker(doc.body, win.NodeFilter.SHOW_TEXT, null, false);
    const textNodes = [];
    let node;
    while ((node = walker.nextNode())) {
        if (node.textContent.trim()) textNodes.push(node);
    }
    textNodes.forEach(processTextNode);

    return doc.body.innerHTML;
}

// Full pipeline as AssistantView.renderMarkdown runs it: normalize -> marked -> KaTeX
function renderPipeline(win, content) {
    win.marked.setOptions({ breaks: true, gfm: true });
    const parsed = win.marked.parse(normalizeLaTeXDelimiters(content));
    return renderLaTeX(win, parsed);
}

describe('LaTeX rendering for mathematical responses', () => {
    let dom;

    beforeAll(async () => {
        const htmlPath = path.join(__dirname, '../index.html');
        const html = fs.readFileSync(htmlPath, 'utf8');
        dom = new JSDOM(html, {
            runScripts: 'dangerously',
            resources: 'usable',
            url: 'file://' + path.resolve(__dirname, '..') + '/',
        });
        await new Promise(resolve => {
            dom.window.addEventListener('load', () => resolve());
        });
    }, 30000);

    it('loads marked and KaTeX from index.html', () => {
        expect(dom.window.marked).toBeDefined();
        expect(dom.window.katex).toBeDefined();
    });

    describe('Gemini-style delimiters ($ and $$)', () => {
        it('renders display math (MSE loss function)', () => {
            const response = 'The cost function is:\n\n$$J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2$$';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect(html).toContain('class="katex"');
            expect(html).not.toContain('$$');
        });

        it('renders inline math (learning rate)', () => {
            const response = 'Here $\\alpha$ is the learning rate and $\\theta_j$ the parameter.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex"');
            expect(html).not.toContain('$\\alpha$');
        });
    });

    describe('Qwen/OpenAI-style delimiters (\\(...\\) and \\[...\\])', () => {
        it('renders display math (gradient descent update rule)', () => {
            const response = 'The update rule is:\n\n\\[ \\theta_{t+1} = \\theta_t - \\eta \\nabla_\\theta J(\\theta_t) \\]';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            // Raw LaTeX must not leak into the visible output
            expect(html).not.toContain('\\nabla_\\theta');
            expect(html).not.toContain('\\[');
        });

        it('renders display math (softmax function)', () => {
            const response = '\\[ \\sigma(z_i) = \\frac{e^{z_i}}{\\sum_{j=1}^{K} e^{z_j}} \\]';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect(html).not.toContain('\\frac');
        });

        it('renders display math (scaled dot-product attention)', () => {
            const response = '\\[ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V \\]';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect(html).not.toContain('\\sqrt');
        });

        it('renders inline math inside sentences', () => {
            const response = 'where \\( \\eta \\) is the learning rate and \\( \\nabla_\\theta J \\) the gradient.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex"');
            expect(html).not.toContain('\\( \\eta \\)');
        });

        it('renders a full mixed ML answer (text + display + inline)', () => {
            const response = [
                'Linear regression minimizes the mean squared error:',
                '',
                '\\[ J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m} (h_\\theta(x^{(i)}) - y^{(i)})^2 \\]',
                '',
                'Gradient descent updates \\( \\theta \\) iteratively with learning rate \\( \\alpha \\).',
            ].join('\n');
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect((html.match(/class="katex"/g) || []).length).toBeGreaterThanOrEqual(3);
            expect(html).not.toContain('\\[');
            expect(html).not.toContain('\\(');
        });
    });

    describe('Per-model response fixtures (all models used in the app)', () => {
        // Realistic math responses in the delimiter style each model family actually emits.
        // Gemini models typically use $-style; Gemini 3.x and Qwen also emit \(...\)/\[...\]
        // (OpenAI convention), so every model is tested with BOTH styles.
        const MODEL_FIXTURES = {
            'gemini-3.5-flash': 'The gradient is $$\\nabla J(\\theta) = \\frac{1}{m} X^T (X\\theta - y)$$ where $m$ is the sample count.',
            'gemini-3-flash-preview': 'Bayes theorem: \\[ P(A|B) = \\frac{P(B|A)P(A)}{P(B)} \\] with prior \\( P(A) \\).',
            'gemini-3.1-pro-preview': 'The eigenvalue equation is $$A v = \\lambda v$$ where $\\lambda$ is the eigenvalue.',
            'gemini-2.5-pro': 'Cross-entropy loss: \\[ L = -\\sum_{i=1}^{N} y_i \\log(\\hat{y}_i) \\] for \\( N \\) classes.',
            'gemini-2.5-flash': 'Sigmoid activation: $$\\sigma(x) = \\frac{1}{1 + e^{-x}}$$ maps to $(0, 1)$.',
            'gemini-2.5-flash-lite': 'Variance is $\\sigma^2 = \\frac{1}{N}\\sum_{i=1}^{N}(x_i - \\mu)^2$ for the population.',
            'gemini-3.1-flash-lite': 'Euler identity: \\( e^{i\\pi} + 1 = 0 \\) combines five constants.',
            'qwen-3.6-27b': 'Backpropagation chain rule: \\[ \\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial a} \\cdot \\frac{\\partial a}{\\partial z} \\cdot \\frac{\\partial z}{\\partial w} \\]',
        };

        for (const [model, response] of Object.entries(MODEL_FIXTURES)) {
            it(`renders math from ${model} responses`, () => {
                const html = renderPipeline(dom.window, response);

                expect(html).toContain('class="katex"');
                // No raw delimiters or LaTeX commands may leak into the visible output
                expect(html).not.toContain('$$');
                expect(html).not.toContain('\\[');
                expect(html).not.toContain('\\frac');
                expect(html).not.toContain('\\sigma');
            });
        }
    });

    describe('Code block safety', () => {
        it('does not touch LaTeX-like content inside fenced code blocks', () => {
            const response = 'Compute cost in Python:\n\n```python\ncost = "$total$"\ngrad = r"\\[ \\nabla J \\]"\n```';
            const html = renderPipeline(dom.window, response);

            // Code content must stay raw — no KaTeX inside <code>
            expect(html).toContain('<code');
            expect(html).toContain('$total$');
            expect(html).not.toContain('class="katex-display"');
        });
    });

    describe('Robustness', () => {
        it('does not crash on malformed LaTeX', () => {
            const response = 'Broken math: $\\frac{1}{$ and \\[ \\invalidcommand{ \\]';
            expect(() => renderPipeline(dom.window, response)).not.toThrow();
        });
    });

    describe('Edge cases (both Gemini and Qwen output styles)', () => {
        it('does not render currency mentions as math (sales/negotiation answers)', () => {
            const response = 'The subscription costs $20 per seat and the addon $5 per month.';
            const html = renderPipeline(dom.window, response);

            expect(html).not.toContain('class="katex"');
            expect(html).toContain('$20');
            expect(html).toContain('$5');
        });

        it('renders Qwen multi-line display math (delimiters on their own lines)', () => {
            const response = 'The update rule:\n\\[\n\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta J(\\theta)\n\\]\nrepeated until convergence.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect(html).not.toContain('\\leftarrow');
        });

        it('renders Gemini multi-line display math ($$ on their own lines)', () => {
            const response = 'The loss is:\n$$\nL(y, \\hat{y}) = -y \\log(\\hat{y}) - (1-y) \\log(1-\\hat{y})\n$$\nknown as binary cross-entropy.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex-display"');
            expect(html).not.toContain('\\log');
        });

        it('renders inline math right after bold markdown labels', () => {
            const response = '**Gemini style:** $R^2 = 1 - \\frac{SS_{res}}{SS_{tot}}$ and **Qwen style:** \\( F_1 = 2 \\cdot \\frac{P \\cdot R}{P + R} \\)';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('<strong>');
            expect((html.match(/class="katex"/g) || []).length).toBeGreaterThanOrEqual(2);
            expect(html).not.toContain('\\frac');
        });

        it('renders math inside markdown list items (mixed styles)', () => {
            const response = ['Key formulas:', '- Precision: $P = \\frac{TP}{TP + FP}$', '- Recall: \\( R = \\frac{TP}{TP + FN} \\)'].join('\n');
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('<li>');
            expect((html.match(/class="katex"/g) || []).length).toBeGreaterThanOrEqual(2);
            expect(html).not.toContain('\\frac');
        });

        it('renders subscripts/superscripts without markdown emphasis corruption', () => {
            const response = 'The squared terms $x_i^2 + y_j^2$ stay intact.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('class="katex"');
            // Underscores inside math must not become <em> tags
            expect(html).not.toContain('<em>');
        });

        it('leaves LaTeX delimiters inside inline code spans untouched', () => {
            const response = 'In regex, escape parens like `\\(x\\)` to match literally.';
            const html = renderPipeline(dom.window, response);

            expect(html).toContain('<code>');
            expect(html).not.toContain('class="katex"');
            // The code span must keep its original \(x\) content
            expect(html).toContain('\\(x\\)');
        });

        it('does not treat dollar amounts across line breaks as math', () => {
            const response = 'Plan A costs $10\nPlan B costs $25';
            const html = renderPipeline(dom.window, response);

            expect(html).not.toContain('class="katex"');
            expect(html).toContain('$10');
            expect(html).toContain('$25');
        });
    });
});
