let gaugeChart = null;
let radarChart = null;
let rocChart = null;
let metadataCache = null;

document.addEventListener("DOMContentLoaded", async () => {
    initCharts();
    await loadMetadata();
    await loadBenchmarks();
    await loadKMeansClusters();
    setupLiveRecalculation();
    // Default preset
    loadPreset("moderate_standard_risk");
});

function initCharts() {
    // 1. Gauge / Doughnut Chart for Risk
    const gaugeCtx = document.getElementById("gaugeChart").getContext("2d");
    gaugeChart = new Chart(gaugeCtx, {
        type: "doughnut",
        data: {
            datasets: [{
                data: [0, 100],
                backgroundColor: ["#4f46e5", "#e2e8f0"],
                borderWidth: 0,
                circumference: 240,
                rotation: 240,
                cutout: "82%",
                borderRadius: 4
            }]
        },
        options: {
            responsive: false,
            maintainAspectRatio: true,
            plugins: {
                tooltip: { enabled: false },
                legend: { display: false }
            }
        }
    });

    // 2. Radar Chart for Risk Profiling
    const radarCtx = document.getElementById("radarChart").getContext("2d");
    radarChart = new Chart(radarCtx, {
        type: "radar",
        data: {
            labels: ["Duration", "Amount", "Installment", "Age", "Credits"],
            datasets: [
                {
                    label: "Applicant",
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: "rgba(79, 70, 229, 0.25)",
                    borderColor: "#4f46e5",
                    pointBackgroundColor: "#4f46e5",
                    borderWidth: 2
                },
                {
                    label: "Cluster Average",
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: "rgba(100, 116, 139, 0.15)",
                    borderColor: "#64748b",
                    pointBackgroundColor: "#64748b",
                    borderDash: [4, 4],
                    borderWidth: 1.5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    min: 0,
                    max: 100,
                    ticks: { display: false },
                    grid: { color: "#e2e8f0" },
                    pointLabels: {
                        font: { size: 10, family: "Inter" },
                        color: "#64748b"
                    }
                }
            },
            plugins: {
                legend: {
                    position: "top",
                    labels: { boxWidth: 12, font: { size: 11, family: "Inter" } }
                }
            }
        }
    });
}

async function loadMetadata() {
    try {
        const res = await fetch("/api/meta");
        const data = await res.json();
        metadataCache = data;

        // Populate categorical selects
        for (const [key, options] of Object.entries(data.choices)) {
            const selectEl = document.getElementById(key);
            if (selectEl) {
                selectEl.innerHTML = "";
                options.forEach(opt => {
                    const optEl = document.createElement("option");
                    optEl.value = opt;
                    optEl.textContent = opt;
                    selectEl.appendChild(optEl);
                });
            }
        }
    } catch (err) {
        console.error("Failed to load metadata:", err);
    }
}

function loadPreset(presetKey) {
    if (!metadataCache || !metadataCache.presets[presetKey]) return;
    const preset = metadataCache.presets[presetKey].data;

    for (const [key, val] of Object.entries(preset)) {
        const el = document.getElementById(key);
        if (el) {
            el.value = val;
        }
    }

    submitAssessment();
}

let liveDebounceTimer = null;

function setupLiveRecalculation() {
    const form = document.getElementById("assessmentForm");
    if (!form) return;

    form.addEventListener("input", (e) => {
        clearTimeout(liveDebounceTimer);
        const liveBadge = document.getElementById("liveStatusBadge");
        if (liveBadge) {
            liveBadge.innerHTML = "⚡ Calculating...";
            liveBadge.style.backgroundColor = "#fef3c7";
            liveBadge.style.color = "#b45309";
        }
        liveDebounceTimer = setTimeout(() => {
            submitAssessment(null, true);
        }, 300);
    });

    form.addEventListener("change", (e) => {
        clearTimeout(liveDebounceTimer);
        const liveBadge = document.getElementById("liveStatusBadge");
        if (liveBadge) {
            liveBadge.innerHTML = "⚡ Calculating...";
            liveBadge.style.backgroundColor = "#fef3c7";
            liveBadge.style.color = "#b45309";
        }
        liveDebounceTimer = setTimeout(() => {
            submitAssessment(null, true);
        }, 80);
    });
}

async function submitAssessment(e, isLive = false) {
    if (e) e.preventDefault();

    const alertEl = document.getElementById("assessmentAlert");
    if (alertEl && !isLive) {
        alertEl.classList.add("d-none");
        alertEl.innerHTML = "";
    }

    const fields = [
        "credit_amount", "duration", "installment_rate", "purpose", "existing_credits",
        "checking_status", "savings_status", "credit_history", "other_payment_plans",
        "age", "employment", "job", "housing", "property_magnitude", "personal_status",
        "residence_since", "other_parties", "num_dependents", "foreign_worker", "own_telephone",
        "model_name"
    ];

    const payload = {};
    for (const f of fields) {
        const el = document.getElementById(f);
        if (!el) continue;
        const valStr = String(el.value).replace(/,/g, "").trim();

        if (f === "credit_amount") {
            const num = parseFloat(valStr);
            if (isNaN(num) || num <= 0) {
                if (!isLive) showAssessmentError("Please enter a valid credit amount.");
                return;
            }
            payload[f] = num;
        } else if (["duration", "installment_rate", "existing_credits", "age", "residence_since", "num_dependents"].includes(f)) {
            const num = parseInt(valStr, 10);
            if (isNaN(num)) {
                if (!isLive) showAssessmentError(`Please enter a valid number for ${f.replace('_', ' ')}.`);
                return;
            }
            payload[f] = num;
        } else {
            payload[f] = el.value;
        }
    }

    const submitBtn = document.getElementById("submitBtn");
    if (!isLive) {
        submitBtn.disabled = true;
        submitBtn.innerHTML = "Assessing...";
    }

    try {
        const res = await fetch("/api/assess", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            let errorText = "Assessment failed. Please review your input.";
            try {
                const errData = await res.json();
                if (errData.detail) {
                    if (Array.isArray(errData.detail)) {
                        errorText = errData.detail.map(d => `${d.loc ? d.loc.slice(1).join(' ') : ''}: ${d.msg}`).join("<br>");
                    } else {
                        errorText = errData.detail;
                    }
                }
            } catch (_) {}
            throw new Error(errorText);
        }

        const data = await res.json();
        renderAssessmentResults(data);

        // Update live badge
        const liveBadge = document.getElementById("liveStatusBadge");
        if (liveBadge) {
            liveBadge.innerHTML = "✓ Live Updated";
            liveBadge.style.backgroundColor = "#dcfce7";
            liveBadge.style.color = "#15803d";
        }
    } catch (err) {
        console.error("Assessment error:", err);
        if (!isLive) {
            showAssessmentError(err.message);
        }
    } finally {
        if (!isLive) {
            submitBtn.disabled = false;
            submitBtn.innerHTML = `
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
                Assess Borrower
            `;
        }
    }
}

function showAssessmentError(message) {
    const alertEl = document.getElementById("assessmentAlert");
    if (alertEl) {
        alertEl.innerHTML = `<strong>Attention:</strong> ${message}`;
        alertEl.classList.remove("d-none");
        alertEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } else {
        alert(message);
    }
}

function renderAssessmentResults(data) {
    const prob = data.default_probability;
    const badge = data.recommendation_badge;

    // 1. Update Gauge & Risk Percent
    document.getElementById("probPercent").textContent = `${prob}%`;
    const recBadgeEl = document.getElementById("recommendationBadge");
    recBadgeEl.className = `badge-tag badge-${badge}`;
    recBadgeEl.textContent = prob < 35 ? "Low Risk" : (prob < 60 ? "Moderate Risk" : "High Risk Defaulter");

    document.getElementById("recommendationText").textContent = data.recommendation;
    document.getElementById("decisionDetails").textContent =
        prob < 35
            ? "Applicant satisfies risk thresholds. Recommended for fast-track loan authorization."
            : (prob < 60
                ? "Applicant falls in the marginal risk tier. Manual underwriting / guarantor review suggested."
                : "Applicant exceeds default tolerance threshold. High probability of credit distress.");

    let gaugeColor = "#10b981"; // Emerald
    if (badge === "warning") gaugeColor = "#f59e0b"; // Amber
    if (badge === "danger") gaugeColor = "#ef4444"; // Red

    gaugeChart.data.datasets[0].data = [prob, 100 - prob];
    gaugeChart.data.datasets[0].backgroundColor = [gaugeColor, "#e2e8f0"];
    gaugeChart.update();

    // 2. Cross-Model Comparison Table
    const tbody = document.getElementById("crossModelTableBody");
    tbody.innerHTML = "";
    const modelNames = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "adaboost": "AdaBoost"
    };

    for (const [key, val] of Object.entries(data.all_models_probabilities)) {
        const isCurrent = key === data.model_used;
        const badgeClass = val.default_probability < 35 ? "badge-success" : (val.default_probability < 60 ? "badge-warning" : "badge-danger");
        const statusText = val.default_probability < 50 ? "Approve" : "Default Risk";

        const tr = document.createElement("tr");
        if (isCurrent) tr.style.backgroundColor = "#e0e7ff33";
        tr.innerHTML = `
            <td>
                <strong>${modelNames[key] || key}</strong>
                ${isCurrent ? '<span class="badge-tag badge-success ms-2">Active</span>' : ''}
            </td>
            <td><strong>${val.default_probability}%</strong></td>
            <td><span class="badge-tag ${badgeClass}">${statusText}</span></td>
        `;
        tbody.appendChild(tr);
    }

    // 3. Top Drivers
    const driversContainer = document.getElementById("topDriversContainer");
    driversContainer.innerHTML = "";
    if (data.top_drivers && data.top_drivers.length > 0) {
        let driversHtml = '<ul class="list-unstyled mb-0">';
        data.top_drivers.forEach(d => {
            const isNegative = d.weight < 0;
            const badgeC = isNegative ? "badge-success" : "badge-warning";
            const cleanName = d.feature.replace(/^(num__|cat__)/, "").replace(/_/g, " ");
            driversHtml += `
                <li class="d-flex justify-content-between align-items-center py-1 border-bottom">
                    <span class="text-truncate" style="max-width: 220px;">${cleanName}</span>
                    <span class="badge-tag ${badgeC}">${d.impact} (${Math.abs(d.weight)})</span>
                </li>
            `;
        });
        driversHtml += '</ul>';
        driversContainer.innerHTML = driversHtml;
    }

    // 4. Update K-Means Profiler
    const prof = data.risk_profile;
    const kmBadge = document.getElementById("kmeansPersonaBadge");
    kmBadge.className = `badge-tag badge-${prof.badge_type}`;
    kmBadge.textContent = `Cluster #${prof.cluster_id}`;

    document.getElementById("personaTitle").textContent = prof.persona_title;
    document.getElementById("personaDescription").textContent = prof.description;
    document.getElementById("clusterDefRate").textContent = `${prof.historical_default_rate}%`;
    document.getElementById("clusterSizePct").textContent = `${prof.cluster_size_percentage}%`;

    // Update Radar Chart
    radarChart.data.labels = prof.radar_data.labels;
    radarChart.data.datasets[0].data = prof.radar_data.applicant_scores;
    radarChart.data.datasets[1].data = prof.radar_data.cluster_scores;
    radarChart.update();
}

async function loadKMeansClusters() {
    try {
        const res = await fetch("/api/kmeans/clusters");
        if (!res.ok) return;
        const data = await res.json();

        const container = document.getElementById("clustersDetailCards");
        container.innerHTML = "";

        for (const [id, c] of Object.entries(data.clusters)) {
            const col = document.createElement("div");
            col.className = "col-md-4";
            col.innerHTML = `
                <div class="mc-metric-card h-100">
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="badge-tag badge-${c.badge_type}">Cluster #${id}</span>
                        <span class="text-muted small">${c.percentage}% share</span>
                    </div>
                    <h4 class="h6 fw-bold mb-2">${c.persona_title}</h4>
                    <p class="text-muted small mb-3">${c.description}</p>
                    <div class="p-2 bg-light rounded small">
                        <div class="d-flex justify-content-between mb-1">
                            <span>Default Rate:</span>
                            <strong>${c.historical_default_rate}%</strong>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>Avg Loan Amount:</span>
                            <strong>${Math.round(c.feature_means.credit_amount)} DM</strong>
                        </div>
                        <div class="d-flex justify-content-between mb-1">
                            <span>Avg Duration:</span>
                            <strong>${Math.round(c.feature_means.duration)} months</strong>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>Avg Age:</span>
                            <strong>${Math.round(c.feature_means.age)} yrs</strong>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(col);
        }
    } catch (err) {
        console.error("Failed to load K-Means clusters:", err);
    }
}

async function loadBenchmarks() {
    try {
        const res = await fetch("/api/models/comparison");
        if (!res.ok) return;
        const data = await res.json();

        // Table
        const tbody = document.getElementById("benchmarkTableBody");
        tbody.innerHTML = "";

        const cmContainer = document.getElementById("confusionMatricesContainer");
        cmContainer.innerHTML = "";

        const datasets = [];
        const colors = {
            "random_forest": { border: "#4f46e5", bg: "rgba(79, 70, 229, 0.1)" },
            "logistic_regression": { border: "#10b981", bg: "rgba(16, 185, 129, 0.1)" },
            "adaboost": { border: "#f59e0b", bg: "rgba(245, 158, 11, 0.1)" }
        };

        for (const [key, m] of Object.entries(data)) {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${m.display_name}</strong></td>
                <td>${m.accuracy}%</td>
                <td>${m.precision}%</td>
                <td>${m.recall}%</td>
                <td>${m.f1_score}%</td>
                <td><strong class="text-primary">${m.roc_auc}%</strong></td>
            `;
            tbody.appendChild(tr);

            // Confusion Matrix Card
            const cm = m.confusion_matrix;
            const cmCard = document.createElement("div");
            cmCard.className = "col-md-4";
            cmCard.innerHTML = `
                <div class="p-3 bg-light rounded border">
                    <h5 class="small fw-bold mb-2">${m.display_name}</h5>
                    <div class="row g-2 text-center small">
                        <div class="col-6">
                            <div class="p-2 bg-white rounded border">
                                <div class="text-success fw-bold">${cm.tn}</div>
                                <div class="text-muted" style="font-size: 0.7rem;">True Neg (Good)</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="p-2 bg-white rounded border">
                                <div class="text-danger fw-bold">${cm.fp}</div>
                                <div class="text-muted" style="font-size: 0.7rem;">False Pos (Type I)</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="p-2 bg-white rounded border">
                                <div class="text-danger fw-bold">${cm.fn}</div>
                                <div class="text-muted" style="font-size: 0.7rem;">False Neg (Type II)</div>
                            </div>
                        </div>
                        <div class="col-6">
                            <div class="p-2 bg-white rounded border">
                                <div class="text-success fw-bold">${cm.tp}</div>
                                <div class="text-muted" style="font-size: 0.7rem;">True Pos (Default)</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            cmContainer.appendChild(cmCard);

            // ROC curve points
            const clr = colors[key] || { border: "#3b82f6", bg: "rgba(59, 130, 246, 0.1)" };
            datasets.push({
                label: `${m.display_name} (AUC: ${m.roc_auc}%)`,
                data: m.roc_curve,
                borderColor: clr.border,
                backgroundColor: clr.bg,
                borderWidth: 2,
                pointRadius: 0,
                fill: false,
                tension: 0.2
            });
        }

        // Add 45-degree diagonal baseline
        datasets.push({
            label: "Random Baseline",
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: "#cbd5e1",
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        });

        // Initialize ROC Chart
        const rocCtx = document.getElementById("rocChart").getContext("2d");
        if (rocChart) rocChart.destroy();
        rocChart = new Chart(rocCtx, {
            type: "line",
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: "linear",
                        min: 0,
                        max: 1,
                        title: { display: true, text: "False Positive Rate (1 - Specificity)", font: { family: "Inter", size: 11 } }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: { display: true, text: "True Positive Rate (Sensitivity / Recall)", font: { family: "Inter", size: 11 } }
                    }
                },
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: { boxWidth: 12, font: { family: "Inter", size: 10 } }
                    }
                }
            }
        });

        // Ensure charts resize accurately when switching Bootstrap tabs
        document.addEventListener("shown.bs.tab", (e) => {
            if (e.target.id === "benchmarks-tab" && rocChart) {
                rocChart.resize();
            }
            if (e.target.id === "assessment-tab") {
                if (gaugeChart) gaugeChart.resize();
                if (radarChart) radarChart.resize();
            }
        });

    } catch (err) {
        console.error("Failed to load benchmarks:", err);
    }
}

function filterGuideFields() {
    const input = document.getElementById("guideSearchInput");
    if (!input) return;
    const query = input.value.toLowerCase().trim();
    const cards = document.querySelectorAll("#guideCardsList .field-guide-card");
    const categories = document.querySelectorAll("#guideCardsList .guide-category");

    cards.forEach(card => {
        const text = ((card.getAttribute("data-field") || "") + " " + card.textContent).toLowerCase();
        if (text.includes(query)) {
            card.style.display = "";
        } else {
            card.style.display = "none";
        }
    });

    categories.forEach(cat => {
        const visibleCards = Array.from(cat.querySelectorAll(".field-guide-card")).filter(c => c.style.display !== "none");
        cat.style.display = visibleCards.length === 0 ? "none" : "";
    });
}
