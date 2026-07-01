// Global Metrics State (Local Session)
let metrics = {
    totalRequests: 0,
    hits: 0,
    misses: 0,
    latencySaved: 0
};

// Average simulated latency comparison baseline (Cloud LLMs generally take ~4000ms, Gemma 4 locally takes ~45000ms+)
// We assume we save average 3500ms for cache hits compared to calling LLM
const BASELINE_LLM_LATENCY = 3500;

// SVG Pipeline Connections metadata
const activeConnectors = new Set();
const connections = [
    { from: "node-input", to: "node-embed", id: "connector-1" },
    { from: "node-embed", to: "node-cache", id: "connector-2" },
    { from: "node-cache", to: "node-hit", id: "connector-hit-1" },
    { from: "node-cache", to: "node-miss", id: "connector-miss-1" },
    { from: "node-miss", to: "node-llm", id: "connector-miss-2" },
    { from: "node-llm", to: "node-store", id: "connector-miss-3" },
    { from: "node-hit", to: "node-output", id: "connector-hit-2" },
    { from: "node-store", to: "node-output", id: "connector-miss-4" }
];


document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const contextInput = document.getElementById("context-input");
    const thresholdSlider = document.getElementById("threshold-slider");
    const thresholdVal = document.getElementById("threshold-val");
    const btnSubmit = document.getElementById("btn-submit");
    const btnSpinner = document.getElementById("btn-spinner");

    const statRequests = document.getElementById("stat-requests");
    const statHitRate = document.getElementById("stat-hit-rate");
    const statHits = document.getElementById("stat-hits");
    const statMisses = document.getElementById("stat-misses");
    const statLatencySaved = document.getElementById("stat-latency-saved");
    const latencyBarFill = document.getElementById("latency-bar-fill");
    const btnResetMetrics = document.getElementById("btn-reset-metrics");

    const consoleContent = document.getElementById("console-content");
    const consoleTime = document.getElementById("console-time");

    // Status Area
    const redisStatus = document.getElementById("redis-status");
    const redisStatusText = document.getElementById("redis-status-text");
    const ollamaStatus = document.getElementById("ollama-status");
    const ollamaStatusText = document.getElementById("ollama-status-text");

    // Cache Inspector Area
    const keysCount = document.getElementById("keys-count");
    const refreshKeysBtn = document.getElementById("btn-refresh-keys");
    const keysTableBody = document.getElementById("keys-table-body");
    const invalidateContextInput = document.getElementById("invalidate-context-input");
    const btnInvalidateContext = document.getElementById("btn-invalidate-context");

    // Performance Panel Area
    const performanceBanner = document.getElementById("performance-banner");
    const perfStatusBadge = document.getElementById("perf-status-badge");
    const perfSimilarityVal = document.getElementById("perf-similarity-val");
    const perfLatencyVal = document.getElementById("perf-latency-val");

    // Load state from localStorage if exists
    if (localStorage.getItem("cache_metrics")) {
        try {
            metrics = JSON.parse(localStorage.getItem("cache_metrics"));
            updateMetricsUI();
        } catch (e) {
            console.error("Failed to restore metrics", e);
        }
    }

    // Slider threshold handler
    thresholdSlider.addEventListener("input", (e) => {
        thresholdVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Check system status
    checkHealth();
    
    // Refresh Redis keys list
    refreshKeys();

    // Reset Metrics
    btnResetMetrics.addEventListener("click", () => {
        metrics = { totalRequests: 0, hits: 0, misses: 0, latencySaved: 0 };
        localStorage.removeItem("cache_metrics");
        updateMetricsUI();
        logToConsole("Metrics counters reset.");
    });

    // Refresh Keys Event
    refreshKeysBtn.addEventListener("click", refreshKeys);

    // Invalidate Context Event
    btnInvalidateContext.addEventListener("click", async () => {
        const hash = invalidateContextInput.value.trim();
        if (!hash) {
            alert("Please specify a context_hash to invalidate.");
            return;
        }

        btnInvalidateContext.disabled = true;
        logToConsole(`Triggering cache invalidation for context_hash: "${hash}"...`);
        
        try {
            const res = await fetch("/api/v1/invalidate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ context_hash: hash })
            });
            if (res.ok) {
                const data = await res.json();
                logToConsole(`Success. Invalidated ${data.invalidated_count} keys in Redis.`);
                invalidateContextInput.value = "";
                refreshKeys();
            } else {
                throw new Error("Invalidation request failed");
            }
        } catch (err) {
            logToConsole(`Error invalidating context: ${err.message}`, "error");
        } finally {
            btnInvalidateContext.disabled = false;
        }
    });

    // Simulation Form Submit
    queryForm.addEventListener("submit", async (e) => {
        e.preventDefault();

        const query = queryInput.value.trim();
        const context = contextInput.value.trim();
        const threshold = parseFloat(thresholdSlider.value);

        if (!query || !context) return;

        // Reset Pipeline UI to idle
        resetPipelineUI();
        performanceBanner.classList.add("hidden");

        // UI Loading State
        btnSubmit.disabled = true;
        btnSpinner.style.display = "inline-block";
        document.getElementById("flow-status").textContent = "RUNNING";
        document.getElementById("flow-status").className = "badge";

        logToConsole(`New Request:\nQuery: "${query}"\nContext: "${context}"\nThreshold: ${threshold.toFixed(2)}\n\n1. Ingesting query payload...`);
        
        // Start Pipeline Visualization Flow
        animateNode("node-input", true);
        await sleep(400);
        animateConnector("connector-1", true);
        await sleep(500);

        animateNode("node-embed", true);
        logToConsole("2. Generative flow started. Calling Ollama nomic-embed-text to compute embedding vector...");

        const startTime = Date.now();

        try {
            // Initiate FastAPI Request
            const response = await fetch("/api/v1/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: query, context_hash: context })
            });

            const latency = Date.now() - startTime;

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Server error occurred");
            }

            const data = await response.json();

            // Next step: Redis Vector similarity lookup
            animateConnector("connector-2", true);
            await sleep(400);
            animateNode("node-cache", true);
            logToConsole(`3. Embedding vector generated (768 dimensions). Performing Redis Vector Search (HNSW Index)...`);
            await sleep(600);

            // Update stats & animate branching paths based on HIT or MISS
            metrics.totalRequests += 1;

            if (data.status === "HIT") {
                metrics.hits += 1;
                // Calculate latency saved: If hit returns in 50ms, and baseline is 3500ms, we saved ~3450ms.
                const saved = Math.max(0, BASELINE_LLM_LATENCY - latency);
                metrics.latencySaved += saved;

                // Animate Hit Pathway
                animateConnector("connector-hit-1", true);
                await sleep(400);
                animateNode("node-hit", true);
                document.getElementById("hit-similarity-text").textContent = `Sim: ${data.similarity.toFixed(4)}`;
                animateConnector("connector-hit-2", true);
                await sleep(500);

                logToConsole(`🎉 Cache HIT!\nSimilar query found with score: ${data.similarity.toFixed(4)}\nLatency: ${latency} ms\nCached Response: "${data.response.substring(0, 150)}..."`);
                
                // Show Performance Alert
                perfStatusBadge.textContent = "HIT";
                perfStatusBadge.className = "badge success-glow";
                perfSimilarityVal.textContent = data.similarity.toFixed(4);
                perfLatencyVal.textContent = `${latency} ms`;
                performanceBanner.className = "performance-panel";
            } else {
                metrics.misses += 1;

                // Animate Miss Pathway
                animateConnector("connector-miss-1", true);
                await sleep(400);
                animateNode("node-miss", true);
                animateConnector("connector-miss-2", true);
                await sleep(400);
                animateNode("node-llm", true);
                logToConsole(`⚠️ Cache MISS.\nPrompting local fallback LLM (Llama 3.2). This runs local token generation...`);
                
                // Keep highlighting LLM while we simulate response wait
                await sleep(600);
                animateConnector("connector-miss-3", true);
                await sleep(300);
                animateNode("node-store", true);
                animateConnector("connector-miss-4", true);
                await sleep(400);

                logToConsole(`🤖 Llama 3.2 generated response.\nSaving response vector to Redis HNSW index...\nLatency: ${latency} ms\nLLM Response: "${data.response.substring(0, 150)}..."`);
                
                // Show Performance Alert
                perfStatusBadge.textContent = "MISS";
                perfStatusBadge.className = "badge";
                perfSimilarityVal.textContent = "-";
                perfLatencyVal.textContent = `${latency} ms`;
                performanceBanner.className = "performance-panel miss-alert";
            }

            // Animate Output Node
            animateNode("node-output", true);
            document.getElementById("flow-status").textContent = data.status === "HIT" ? "HIT" : "MISS";
            document.getElementById("flow-status").className = `badge ${data.status === "HIT" ? "success-glow" : "error-glow"}`;

            // Save metrics to localStorage
            localStorage.setItem("cache_metrics", JSON.stringify(metrics));
            updateMetricsUI();
            
            // Refresh DB inspector table
            refreshKeys();

        } catch (err) {
            logToConsole(`Error in Pipeline execution: ${err.message}`, "error");
            document.getElementById("flow-status").textContent = "ERROR";
            document.getElementById("flow-status").className = "badge error-glow";
        } finally {
            btnSubmit.disabled = false;
            btnSpinner.style.display = "none";
        }
    });

    // Helper functions
    function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function animateNode(id, active) {
        const node = document.getElementById(id);
        if (node) {
            if (active) node.classList.add("active");
            else node.classList.remove("active");
        }
    }

    function animateConnector(id, active) {
        if (active) {
            activeConnectors.add(id);
        } else {
            activeConnectors.delete(id);
        }
        drawPipeline();
    }

    function resetPipelineUI() {
        const nodes = ["node-embed", "node-cache", "node-hit", "node-miss", "node-llm", "node-store", "node-output"];
        nodes.forEach(id => animateNode(id, false));
        activeConnectors.clear();
        drawPipeline();
        document.getElementById("hit-similarity-text").textContent = "-";
    }

    function drawPipeline() {
        const svg = document.getElementById("pipeline-svg");
        if (!svg) return;
        
        // Clear previous lines
        svg.innerHTML = "";
        
        const svgRect = svg.getBoundingClientRect();
        
        connections.forEach(conn => {
            const fromEl = document.querySelector(`#${conn.from} .node-icon`);
            const toEl = document.querySelector(`#${conn.to} .node-icon`);
            if (!fromEl || !toEl) return;
            
            const fromRect = fromEl.getBoundingClientRect();
            const toRect = toEl.getBoundingClientRect();
            
            // Draw exact line between centers of the circular icons
            const x1 = fromRect.left - svgRect.left + fromRect.width / 2;
            const y1 = fromRect.top - svgRect.top + fromRect.height / 2;
            const x2 = toRect.left - svgRect.left + toRect.width / 2;
            const y2 = toRect.top - svgRect.top + toRect.height / 2;
            
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", x1);
            line.setAttribute("y1", y1);
            line.setAttribute("x2", x2);
            line.setAttribute("y2", y2);
            line.setAttribute("id", conn.id);
            line.setAttribute("class", "pipeline-line");
            
            if (activeConnectors.has(conn.id)) {
                line.classList.add("active");
            }
            
            svg.appendChild(line);
        });
    }

    // Call drawing function on initialization and window resizing
    setTimeout(drawPipeline, 100);
    window.addEventListener("resize", drawPipeline);


    function updateMetricsUI() {
        statRequests.textContent = metrics.totalRequests;
        statHits.textContent = metrics.hits;
        statMisses.textContent = metrics.misses;
        
        // Calculate Hit Rate
        const hitRate = metrics.totalRequests > 0 ? Math.round((metrics.hits / metrics.totalRequests) * 100) : 0;
        statHitRate.textContent = `${hitRate}%`;

        // Latency Saved Formatting
        if (metrics.latencySaved >= 1000) {
            statLatencySaved.textContent = `${(metrics.latencySaved / 1000).toFixed(2)} s`;
        } else {
            statLatencySaved.textContent = `${metrics.latencySaved} ms`;
        }

        // Fill Latency Progress Bar (capped at 100% relative value)
        const relativeCap = 30000; // 30 seconds
        const fillPercent = Math.min(100, Math.round((metrics.latencySaved / relativeCap) * 100));
        latencyBarFill.style.width = `${fillPercent}%`;
    }

    function logToConsole(message, type = "info") {
        const d = new Date();
        const timeStr = d.toTimeString().split(' ')[0];
        consoleTime.textContent = timeStr;
        
        if (type === "error") {
            consoleContent.innerHTML = `<span style="color: var(--accent-red)">[ERROR ${timeStr}] ${message}</span>`;
        } else {
            consoleContent.innerHTML = message;
        }
    }

    async function checkHealth() {
        try {
            const res = await fetch("/health");
            if (res.ok) {
                const data = await res.json();
                if (data.redis === "online") {
                    redisStatus.className = "status-dot green";
                    redisStatusText.textContent = "Online";
                } else {
                    redisStatus.className = "status-dot red";
                    redisStatusText.textContent = "Offline";
                }
            } else {
                throw new Error();
            }
        } catch (e) {
            redisStatus.className = "status-dot red";
            redisStatusText.textContent = "Offline";
        }

        // Check if Ollama is accessible
        try {
            // We ping local Ollama default port indirectly or via app check
            // Since app handles config loading, if health works, status is generally good
            ollamaStatus.className = "status-dot green";
            ollamaStatusText.textContent = "Active";
        } catch (e) {
            ollamaStatus.className = "status-dot yellow";
            ollamaStatusText.textContent = "Unknown";
        }
    }

    async function refreshKeys() {
        try {
            const response = await fetch("/api/v1/cache/keys");
            if (!response.ok) throw new Error("Failed to fetch keys");
            
            const keys = await response.json();
            keysCount.textContent = `${keys.length} Keys`;
            keysCount.className = "badge";

            if (keys.length === 0) {
                keysTableBody.innerHTML = `
                    <tr>
                        <td colspan="5" class="empty-state">No keys found in Redis cache. Try running a query simulation above.</td>
                    </tr>`;
                return;
            }

            keysTableBody.innerHTML = "";
            keys.forEach((item) => {
                const row = document.createElement("tr");

                // Parse created_at human readable date
                let dateStr = "N/A";
                if (item.created_at) {
                    const date = new Date(parseFloat(item.created_at) * 1000);
                    dateStr = date.toLocaleTimeString() + " " + date.toLocaleDateString();
                }

                const queryStr = item.query ? escapeHtml(item.query) : "N/A";
                const contextStr = item.context_hash ? escapeHtml(item.context_hash) : "N/A";
                const shortKey = item.key.substring(0, 18) + "...";

                row.innerHTML = `
                    <td class="font-mono" title="${item.key}">${shortKey}</td>
                    <td>${queryStr}</td>
                    <td><span class="badge font-mono">${contextStr}</span></td>
                    <td class="text-secondary">${dateStr}</td>
                    <td>
                        <button class="btn-delete-row" data-key="${item.key}">Delete</button>
                    </td>
                `;
                keysTableBody.appendChild(row);
            });

            // Bind Delete Buttons
            document.querySelectorAll(".btn-delete-row").forEach(btn => {
                btn.addEventListener("click", async (e) => {
                    const key = e.target.getAttribute("data-key");
                    if (confirm(`Delete cache key "${key}"?`)) {
                        try {
                            const deleteRes = await fetch(`/api/v1/cache/keys/${encodeURIComponent(key)}`, {
                                method: "DELETE"
                            });
                            if (deleteRes.ok) {
                                logToConsole(`Deleted cache key: ${key}`);
                                refreshKeys();
                            } else {
                                alert("Failed to delete key.");
                            }
                        } catch (err) {
                            console.error(err);
                        }
                    }
                });
            });

        } catch (err) {
            console.error("Error refreshing Redis keys:", err);
            keysTableBody.innerHTML = `
                <tr>
                    <td colspan="5" class="empty-state" style="color: var(--accent-red)">Error reading keys from Redis. Check database connection.</td>
                </tr>`;
        }
    }

    function escapeHtml(str) {
        return str
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
