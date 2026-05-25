/* ── Cross-Modal Cartographer — Frontend Logic ─────────────────────────── */

// ── State ───────────────────────────────────────────────────────────────────
let cities = [];           // [{name, lat, lon}, ...] from /api/cities
let selectedCity = null;   // {name, lat, lon} or null
let detectedCoords = null; // {lat, lon} from geolocation

// ── DOM refs ────────────────────────────────────────────────────────────────
const canvasEl        = document.getElementById('sketch-canvas');
const btnPen          = document.getElementById('btn-pen');
const btnEraser       = document.getElementById('btn-eraser');
const btnUndo         = document.getElementById('btn-undo');
const btnClear        = document.getElementById('btn-clear');
const btnUpload       = document.getElementById('btn-upload');
const fileUpload      = document.getElementById('file-upload');
const colorPicker     = document.getElementById('color-picker');
const brushSize       = document.getElementById('brush-size');
const textQuery       = document.getElementById('text-query');
const detectLocation  = document.getElementById('detect-location');
const detectedCityEl  = document.getElementById('detected-city');
const locationWrapper = document.getElementById('location-search-wrapper');
const locationSearch  = document.getElementById('location-search');
const suggestionsEl   = document.getElementById('location-suggestions');
const toggleFaiss     = document.getElementById('toggle-faiss');
const toggleKg        = document.getElementById('toggle-kg');
const alphaSlider     = document.getElementById('alpha-slider');
const alphaValue      = document.getElementById('alpha-value');
const textWeight      = document.getElementById('text-weight');
const btnSearch       = document.getElementById('btn-search');
const statsBar        = document.getElementById('stats-bar');
const resultsTabs     = document.getElementById('results-tabs');
const loadingEl       = document.getElementById('loading');
const resultsGrid     = document.getElementById('results-grid');
const emptyState      = document.getElementById('empty-state');

// Tab state
let lastSearchData = null;  // stores full response for tab switching
let activeTab = 'accepted';

// ── Fabric.js Canvas Setup ──────────────────────────────────────────────────
const CANVAS_SIZE = 400;
canvasEl.width = CANVAS_SIZE;
canvasEl.height = CANVAS_SIZE;

const canvas = new fabric.Canvas('sketch-canvas', {
    isDrawingMode: true,
    width: CANVAS_SIZE,
    height: CANVAS_SIZE,
    backgroundColor: '#ffffff',
});

canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
canvas.freeDrawingBrush.color = '#000000';
canvas.freeDrawingBrush.width = 4;

// Handle responsive canvas
function resizeCanvas() {
    const wrapper = document.querySelector('.canvas-wrapper');
    const w = wrapper.clientWidth;
    const scale = w / CANVAS_SIZE;
    const outer = canvasEl.parentElement; // .canvas-container created by Fabric
    if (outer) {
        outer.style.transform = `scale(${scale})`;
        outer.style.transformOrigin = 'top left';
        outer.style.width = CANVAS_SIZE + 'px';
        outer.style.height = CANVAS_SIZE + 'px';
        wrapper.style.height = (CANVAS_SIZE * scale) + 'px';
    }
}

window.addEventListener('resize', resizeCanvas);
setTimeout(resizeCanvas, 100);

// ── Canvas Toolbar ──────────────────────────────────────────────────────────
let currentTool = 'pen';

btnPen.addEventListener('click', () => {
    currentTool = 'pen';
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.color = colorPicker.value;
    canvas.freeDrawingBrush.width = parseInt(brushSize.value);
    btnPen.classList.add('active');
    btnEraser.classList.remove('active');
});

btnEraser.addEventListener('click', () => {
    currentTool = 'eraser';
    canvas.isDrawingMode = true;
    canvas.freeDrawingBrush.color = '#ffffff';
    canvas.freeDrawingBrush.width = 20;
    btnEraser.classList.add('active');
    btnPen.classList.remove('active');
});

btnUndo.addEventListener('click', () => {
    const objects = canvas.getObjects();
    if (objects.length > 0) {
        canvas.remove(objects[objects.length - 1]);
        canvas.renderAll();
    }
});

btnClear.addEventListener('click', () => {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    canvas.renderAll();
    uploadedFile = null;
});

// ── Upload Image ────────────────────────────────────────────────────────────
let uploadedFile = null; // stores the raw File if user uploaded an image

btnUpload.addEventListener('click', () => fileUpload.click());

fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadedFile = file;

    // Draw uploaded image onto canvas
    const reader = new FileReader();
    reader.onload = (ev) => {
        fabric.Image.fromURL(ev.target.result, (img) => {
            canvas.clear();
            canvas.backgroundColor = '#ffffff';

            // Scale image to fit canvas
            const scale = Math.min(CANVAS_SIZE / img.width, CANVAS_SIZE / img.height);
            img.scale(scale);
            img.set({
                left: (CANVAS_SIZE - img.width * scale) / 2,
                top: (CANVAS_SIZE - img.height * scale) / 2,
                selectable: false,
                evented: false,
            });

            canvas.add(img);
            canvas.renderAll();
        });
    };
    reader.readAsDataURL(file);

    // Reset file input so the same file can be re-selected
    fileUpload.value = '';
});

colorPicker.addEventListener('input', (e) => {
    if (currentTool === 'pen') {
        canvas.freeDrawingBrush.color = e.target.value;
    }
});

brushSize.addEventListener('input', (e) => {
    if (currentTool === 'pen') {
        canvas.freeDrawingBrush.width = parseInt(e.target.value);
    }
});

// ── Alpha Slider ────────────────────────────────────────────────────────────
alphaSlider.addEventListener('input', () => {
    const v = parseInt(alphaSlider.value);
    alphaValue.textContent = v + '%';
    textWeight.textContent = (100 - v) + '%';
});

// ── Location: Fetch cities ──────────────────────────────────────────────────
async function loadCities() {
    try {
        const resp = await fetch('/api/cities');
        cities = await resp.json();
    } catch (e) {
        console.error('Failed to load cities:', e);
    }
}
loadCities();

// ── Location: Detect toggle ─────────────────────────────────────────────────
detectLocation.addEventListener('change', () => {
    if (detectLocation.checked) {
        locationWrapper.classList.add('disabled');
        detectedCityEl.textContent = 'Detecting...';
        selectedCity = null;
        locationSearch.value = '';

        if (!navigator.geolocation) {
            detectedCityEl.textContent = 'Not supported';
            return;
        }

        navigator.geolocation.getCurrentPosition(
            (pos) => {
                detectedCoords = { lat: pos.coords.latitude, lon: pos.coords.longitude };
                // Find nearest Egyptian city client-side
                let bestCity = null;
                let bestDist = Infinity;
                for (const c of cities) {
                    const d = haversine(detectedCoords.lat, detectedCoords.lon, c.lat, c.lon);
                    if (d < bestDist) {
                        bestDist = d;
                        bestCity = c;
                    }
                }
                if (bestCity && bestDist < 200) {
                    detectedCityEl.textContent = bestCity.name;
                    selectedCity = bestCity;
                } else {
                    detectedCityEl.textContent = 'Outside Egypt';
                    detectedCoords = null;
                }
            },
            (err) => {
                detectedCityEl.textContent = 'Permission denied';
                console.error('Geolocation error:', err);
            },
            { timeout: 10000 }
        );
    } else {
        locationWrapper.classList.remove('disabled');
        detectedCityEl.textContent = '';
        detectedCoords = null;
    }
});

// ── Location: Search autocomplete ───────────────────────────────────────────
let highlightIdx = -1;

locationSearch.addEventListener('input', () => {
    const q = locationSearch.value.trim().toLowerCase();
    selectedCity = null;
    highlightIdx = -1;

    if (q.length < 1) {
        suggestionsEl.classList.remove('show');
        return;
    }

    const matches = cities.filter(c => c.name.toLowerCase().includes(q)).slice(0, 8);

    if (matches.length === 0) {
        suggestionsEl.classList.remove('show');
        return;
    }

    suggestionsEl.innerHTML = matches.map((c, i) =>
        `<li data-idx="${i}">${c.name}</li>`
    ).join('');
    suggestionsEl.classList.add('show');

    // Click handlers
    suggestionsEl.querySelectorAll('li').forEach((li, i) => {
        li.addEventListener('click', () => {
            selectCity(matches[i]);
        });
    });
});

locationSearch.addEventListener('keydown', (e) => {
    const items = suggestionsEl.querySelectorAll('li');
    if (!items.length) return;

    if (e.key === 'ArrowDown') {
        e.preventDefault();
        highlightIdx = Math.min(highlightIdx + 1, items.length - 1);
        updateHighlight(items);
    } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        highlightIdx = Math.max(highlightIdx - 1, 0);
        updateHighlight(items);
    } else if (e.key === 'Enter' && highlightIdx >= 0) {
        e.preventDefault();
        items[highlightIdx].click();
    } else if (e.key === 'Escape') {
        suggestionsEl.classList.remove('show');
    }
});

function updateHighlight(items) {
    items.forEach((li, i) => {
        li.classList.toggle('highlighted', i === highlightIdx);
    });
}

function selectCity(city) {
    selectedCity = city;
    locationSearch.value = city.name;
    suggestionsEl.classList.remove('show');
}

// Close suggestions on outside click
document.addEventListener('click', (e) => {
    if (!locationWrapper.contains(e.target)) {
        suggestionsEl.classList.remove('show');
    }
});

// ── Haversine (client-side, for geolocation nearest-city) ───────────────────
function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// ── Search ──────────────────────────────────────────────────────────────────
btnSearch.addEventListener('click', doSearch);

// Also search on Enter in text input
textQuery.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') doSearch();
});

async function doSearch() {
    // Use uploaded file directly if available, otherwise export canvas
    let blob;
    if (uploadedFile) {
        blob = uploadedFile;
    } else {
        const dataUrl = canvas.toDataURL({ format: 'png' });
        const resp = await fetch(dataUrl);
        blob = await resp.blob();
    }

    // Build location string
    let locationStr = '';
    if (detectLocation.checked && detectedCoords) {
        locationStr = `${detectedCoords.lat},${detectedCoords.lon}`;
    } else if (selectedCity) {
        locationStr = selectedCity.name;
    } else if (locationSearch.value.trim()) {
        locationStr = locationSearch.value.trim();
    }

    // Build form data
    const formData = new FormData();
    formData.append('sketch', blob, 'sketch.png');
    formData.append('text_query', textQuery.value.trim());
    formData.append('location', locationStr);
    formData.append('alpha', (parseInt(alphaSlider.value) / 100).toFixed(2));
    formData.append('k', '10');
    formData.append('use_faiss', toggleFaiss.checked ? 'true' : 'false');
    formData.append('use_kg', toggleKg.checked ? 'true' : 'false');
    formData.append('host', window.location.origin);

    // UI: loading state
    btnSearch.disabled = true;
    btnSearch.textContent = 'Searching...';
    loadingEl.style.display = 'flex';
    resultsGrid.innerHTML = '';
    statsBar.style.display = 'none';
    resultsTabs.style.display = 'none';
    emptyState.style.display = 'none';

    try {
        const response = await fetch('/search', { method: 'POST', body: formData });
        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.detail || 'Search failed');
        }

        const data = await response.json();
        lastSearchData = data;
        activeTab = 'accepted';
        renderResults(data);
    } catch (e) {
        resultsGrid.innerHTML = `<div class="empty-state"><p>Error: ${e.message}</p></div>`;
    } finally {
        btnSearch.disabled = false;
        btnSearch.textContent = 'Search';
        loadingEl.style.display = 'none';
    }
}

// ── Tab switching ───────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        if (!lastSearchData) return;
        activeTab = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderTabContent(lastSearchData);
    });
});

// ── Render Results ──────────────────────────────────────────────────────────
function renderResults(data) {
    // Stats bar
    statsBar.style.display = 'flex';
    let statsHtml = `<span class="query-summary">${escHtml(data.query_summary)}</span>`;

    if (data.predicted_type) {
        statsHtml += `<span class="stat"><span class="analysis-metric metric-type">${escHtml(data.predicted_type)}</span> predicted type</span>`;
    }

    if (!data.use_faiss) {
        statsHtml += `<span class="toggle-disabled-note">Vector search disabled</span>`;
    } else {
        statsHtml += `
            <span class="stat"><span class="stat-dot" style="background:var(--success)"></span>${data.n_accepted} accepted</span>
            <span class="stat"><span class="stat-dot" style="background:var(--warning)"></span>${data.n_rejected_tau} below threshold</span>
        `;
        if (data.use_kg) {
            statsHtml += `<span class="stat"><span class="stat-dot" style="background:var(--danger)"></span>${data.n_s_violations} schema violations</span>`;
        } else {
            statsHtml += `<span class="toggle-disabled-note">KG disabled &mdash; no schema verification</span>`;
        }
        statsHtml += `<span class="stat"><span class="analysis-metric metric-tau">tau=${(data.tau * 100).toFixed(0)}%</span> threshold</span>`;
    }
    statsBar.innerHTML = statsHtml;

    // Compute tab counts
    const acceptedOnly = data.results.filter(r => r.schema_pass);
    const violations = data.schema_violations || [];
    const rejected = data.rejected_tau || [];

    document.getElementById('tab-count-accepted').textContent = acceptedOnly.length;
    document.getElementById('tab-count-rejected').textContent = rejected.length;
    document.getElementById('tab-count-violations').textContent = violations.length;

    // Show tabs
    if (data.use_faiss) {
        resultsTabs.style.display = 'flex';
        // Reset to accepted tab
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelector('.tab-btn[data-tab="accepted"]').classList.add('active');
    }

    renderTabContent(data);
}

function renderTabContent(data) {
    let items;
    let emptyMsg;

    if (activeTab === 'accepted') {
        items = data.results.filter(r => r.schema_pass);
        emptyMsg = !data.use_faiss
            ? 'Vector search is disabled. Enable FAISS to see results.'
            : 'No accepted results. All results were below the similarity threshold or had schema violations.';
    } else if (activeTab === 'rejected') {
        items = data.rejected_tau || [];
        emptyMsg = 'No results below the threshold — all retrieved results passed the similarity check.';
    } else {
        items = data.schema_violations || [];
        emptyMsg = 'No schema violations — all accepted results match the predicted landmark type.';
    }

    if (items.length === 0) {
        resultsGrid.innerHTML = '';
        emptyState.style.display = 'block';
        emptyState.querySelector('p').textContent = emptyMsg;
        return;
    }

    emptyState.style.display = 'none';
    resultsGrid.innerHTML = items.map(r => renderCard(r, data)).join('');
}

function renderCard(r, data) {
    const scorePct = Math.round(r.score * 100);
    const scoreClass = r.score >= 0.75 ? 'high' : r.score >= 0.65 ? 'medium' : 'low';

    let tagsHtml = `<span class="tag tag-category">${escHtml(r.category)}</span>`;

    if (data.use_kg) {
        if (r.schema_pass) {
            tagsHtml += `<span class="badge badge-pass">PASS</span>`;
        } else {
            tagsHtml += `<span class="badge badge-fail">S-FAIL</span>`;
        }
        if (r.historical_era) tagsHtml += `<span class="tag tag-era">${escHtml(r.historical_era)}</span>`;
        if (r.architectural_style) tagsHtml += `<span class="tag tag-style">${escHtml(r.architectural_style)}</span>`;
        if (r.geographic_region) tagsHtml += `<span class="tag tag-region">${escHtml(r.geographic_region)}</span>`;
    }

    if (r.is_nearby) tagsHtml += `<span class="tag tag-nearby">Nearby</span>`;

    let locationHtml = escHtml(r.location);
    if (r.distance_km !== null && r.distance_km !== undefined) {
        locationHtml += ` <span class="distance">(${r.distance_km} km)</span>`;
    }

    let nearbyHtml = '';
    if (data.use_kg && r.nearby_landmarks && r.nearby_landmarks.length > 0) {
        nearbyHtml = `<div class="card-nearby"><strong>Nearby:</strong> ${r.nearby_landmarks.map(escHtml).join(', ')}</div>`;
    }

    // Analysis box
    let analysisHtml = '';
    if (r.analysis) {
        const analysisClass = r.verdict === 'S' ? 'analysis-s'
            : r.verdict === 'TAU' ? 'analysis-tau'
            : 'analysis-ok';

        // Build metrics line
        let metricsHtml = `<span class="analysis-metric metric-score">score: ${scorePct}%</span> `;
        if (r.tau_threshold !== null && r.tau_threshold !== undefined) {
            metricsHtml += `<span class="analysis-metric metric-tau">tau: ${Math.round(r.tau_threshold * 100)}%</span> `;
        }
        if (r.score_gap !== null && r.score_gap !== undefined) {
            metricsHtml += `<span class="analysis-metric metric-gap">gap: -${Math.round(r.score_gap * 100)}%</span> `;
        }
        if (r.expected_type) {
            metricsHtml += `<span class="analysis-metric metric-type">expected: ${escHtml(r.expected_type)}</span> `;
        }

        analysisHtml = `
            <div class="card-analysis ${analysisClass}">
                <div style="margin-bottom:0.3rem">${metricsHtml}</div>
                ${escHtml(r.analysis)}
            </div>
        `;
    }

    return `
        <div class="result-card">
            <img class="card-image" src="${escAttr(r.image_url)}" alt="${escAttr(r.name)}" loading="lazy"
                 onerror="this.style.background='#ddd'; this.alt='Image not found';">
            <div class="card-body">
                <div class="card-name">${escHtml(r.name)}</div>
                <div class="card-location">${locationHtml}</div>
                <div class="score-row">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill ${scoreClass}" style="width:${scorePct}%"></div>
                    </div>
                    <span class="score-label ${scoreClass}">${scorePct}%</span>
                </div>
                <div class="card-tags">${tagsHtml}</div>
                ${nearbyHtml}
                ${analysisHtml}
            </div>
        </div>
    `;
}

// ── Utils ───────────────────────────────────────────────────────────────────
function escHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function escAttr(str) {
    if (!str) return '';
    return str.replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
