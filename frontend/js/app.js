/**
 * Goodreads RAG — Frontend Application Logic
 * Vanilla JS — calls the FastAPI /api/v1/query endpoint
 */

(function () {
    "use strict";

    // ── DOM References ──────────────────────────────────────────
    const questionInput    = document.getElementById("question-input");
    const charCount        = document.getElementById("char-count");
    const topkSlider       = document.getElementById("topk-slider");
    const topkValue        = document.getElementById("topk-value");
    const askBtn           = document.getElementById("ask-btn");
    const answerSection    = document.getElementById("answer-section");
    const answerBody       = document.getElementById("answer-body");
    const latencyBadge     = document.getElementById("latency-badge");
    const sourcesSection   = document.getElementById("sources-section");
    const sourcesGrid      = document.getElementById("sources-grid");
    const toggleSourcesBtn = document.getElementById("toggle-sources-btn");
    const statusBadge      = document.getElementById("status-badge");
    const errorToast       = document.getElementById("error-toast");
    const errorMsg         = document.getElementById("error-msg");
    const errorClose       = document.getElementById("error-close");

    const API_BASE = "";  // same origin

    // ── Readiness Check ─────────────────────────────────────────
    async function checkReady() {
        try {
            const resp = await fetch(`${API_BASE}/ready`);
            const data = await resp.json();
            if (resp.ok && data.status === "ready") {
                statusBadge.className = "header__badge header__badge--ready";
                statusBadge.querySelector(".header__badge-text").textContent = "System ready";
            } else {
                statusBadge.className = "header__badge header__badge--error";
                statusBadge.querySelector(".header__badge-text").textContent = "Loading index…";
                setTimeout(checkReady, 3000);
            }
        } catch {
            statusBadge.className = "header__badge header__badge--error";
            statusBadge.querySelector(".header__badge-text").textContent = "API offline";
            setTimeout(checkReady, 5000);
        }
    }

    checkReady();

    // ── Input Handlers ──────────────────────────────────────────
    questionInput.addEventListener("input", () => {
        const len = questionInput.value.length;
        charCount.textContent = len;
        askBtn.disabled = len < 3;
    });

    topkSlider.addEventListener("input", () => {
        topkValue.textContent = topkSlider.value;
    });

    // Submit on Ctrl+Enter
    questionInput.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !askBtn.disabled) {
            e.preventDefault();
            askBtn.click();
        }
    });

    // ── Ask Button ──────────────────────────────────────────────
    askBtn.addEventListener("click", async () => {
        const question = questionInput.value.trim();
        if (question.length < 3) return;

        setLoading(true);
        hideError();
        hideResults();

        try {
            const resp = await fetch(`${API_BASE}/api/v1/query`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    question: question,
                    top_k: parseInt(topkSlider.value, 10),
                }),
            });

            if (!resp.ok) {
                const errData = await resp.json().catch(() => ({}));
                throw new Error(errData.detail || `Server error (${resp.status})`);
            }

            const data = await resp.json();
            renderAnswer(data);
            renderSources(data.sources);
        } catch (err) {
            showError(err.message || "Failed to reach the API.");
        } finally {
            setLoading(false);
        }
    });

    // ── Toggle Sources ──────────────────────────────────────────
    toggleSourcesBtn.addEventListener("click", () => {
        const isHidden = sourcesGrid.style.display === "none";
        sourcesGrid.style.display = isHidden ? "flex" : "none";
        toggleSourcesBtn.textContent = isHidden ? "Hide sources" : "Show sources";
    });

    // ── Error Toast ─────────────────────────────────────────────
    errorClose.addEventListener("click", hideError);

    function showError(msg) {
        errorMsg.textContent = msg;
        errorToast.hidden = false;
    }

    function hideError() {
        errorToast.hidden = true;
    }

    // ── Loading State ───────────────────────────────────────────
    function setLoading(isLoading) {
        const btnText   = askBtn.querySelector(".btn__text");
        const btnLoader = askBtn.querySelector(".btn__loader");

        if (isLoading) {
            askBtn.disabled    = true;
            btnText.hidden     = true;
            btnLoader.hidden   = false;
            questionInput.disabled = true;
        } else {
            askBtn.disabled    = questionInput.value.trim().length < 3;
            btnText.hidden     = false;
            btnLoader.hidden   = true;
            questionInput.disabled = false;
        }
    }

    // ── Hide Results ────────────────────────────────────────────
    function hideResults() {
        answerSection.hidden  = true;
        sourcesSection.hidden = true;
    }

    // ── Render Answer ───────────────────────────────────────────
    function renderAnswer(data) {
        answerBody.innerHTML = formatAnswer(data.answer);
        latencyBadge.textContent = `${data.latency_ms}ms`;
        answerSection.hidden = false;
        answerSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    /**
     * Convert plain text answer to paragraphs.
     * Also handles simple markdown-like bullet lists.
     */
    function formatAnswer(text) {
        return text
            .split(/\n\n+/)
            .map((para) => {
                // Check if it's a bullet list
                const lines = para.split("\n");
                const isList = lines.every((l) => /^\s*[-*•]\s/.test(l) || l.trim() === "");
                if (isList && lines.length > 1) {
                    const items = lines
                        .filter((l) => l.trim())
                        .map((l) => `<li>${escapeHtml(l.replace(/^\s*[-*•]\s*/, ""))}</li>`)
                        .join("");
                    return `<ul>${items}</ul>`;
                }
                return `<p>${escapeHtml(para)}</p>`;
            })
            .join("");
    }

    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Render Sources ──────────────────────────────────────────
    function renderSources(sources) {
        if (!sources || sources.length === 0) return;

        sourcesGrid.innerHTML = "";
        sourcesGrid.style.display = "flex";
        toggleSourcesBtn.textContent = "Hide sources";

        sources.forEach((src, i) => {
            const card = createSourceCard(src, i);
            // Stagger animation
            card.style.animationDelay = `${i * 80}ms`;
            sourcesGrid.appendChild(card);
        });

        sourcesSection.hidden = false;
    }

    function createSourceCard(src, index) {
        const card = document.createElement("div");
        card.className = "source-card";

        const title  = src.title  || "(no title)";
        const author = src.author || "";
        const rating = src.rating || "";

        card.innerHTML = `
            <div class="source-card__header">
                <span class="source-card__rank">${index + 1}</span>
                <div class="source-card__info">
                    <span class="source-card__title">${escapeHtml(title)}</span>
                    <span class="source-card__meta">
                        ${author ? `<span>by ${escapeHtml(author)}</span>` : ""}
                        ${rating ? `<span>⭐ ${escapeHtml(rating)}</span>` : ""}
                    </span>
                </div>
                <span class="source-card__distance">${src.distance.toFixed(4)}</span>
                <span class="source-card__chevron">▼</span>
            </div>
            <div class="source-card__body">
                <div class="source-card__snippet">${escapeHtml(src.snippet)}</div>
            </div>
        `;

        // Toggle expand/collapse
        const header = card.querySelector(".source-card__header");
        header.addEventListener("click", () => {
            card.classList.toggle("source-card--expanded");
        });

        return card;
    }
})();
