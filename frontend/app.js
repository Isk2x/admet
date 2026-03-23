// PharmaKinetics MVP — клиентская логика

const API_URL = '/api/predict';
const MODELS_URL = '/api/models';

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    document.getElementById('smiles-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') predict();
    });
});

async function loadModels() {
    try {
        const resp = await fetch(MODELS_URL);
        if (!resp.ok) return;
        const models = await resp.json();
        const select = document.getElementById('model-select');
        select.innerHTML = '';
        for (const m of models) {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = `${m.name} (${m.params})`;
            opt.disabled = !m.available;
            if (!m.available) opt.textContent += ' [не обучена]';
            select.appendChild(opt);
        }
    } catch (e) {
        console.warn('Не удалось загрузить список моделей:', e);
    }
}

function setSmiles(smiles) {
    document.getElementById('smiles-input').value = smiles;
    predict();
}

async function predict() {
    const input = document.getElementById('smiles-input');
    const smiles = input.value.trim();
    if (!smiles) return;

    const modelSelect = document.getElementById('model-select');
    const modelId = modelSelect.value;

    const btn = document.getElementById('predict-btn');
    const btnText = btn.querySelector('.btn-text');
    const btnSpinner = btn.querySelector('.btn-spinner');
    const errorBox = document.getElementById('error-box');
    const results = document.getElementById('results');

    btn.disabled = true;
    btnText.textContent = 'Анализ...';
    btnSpinner.classList.add('visible');
    errorBox.hidden = true;
    results.hidden = true;

    try {
        const resp = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles, model: modelId }),
        });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || `Ошибка ${resp.status}`);
        }

        const data = await resp.json();
        renderResults(data);
    } catch (e) {
        errorBox.textContent = e.message;
        errorBox.hidden = false;
    } finally {
        btn.disabled = false;
        btnText.textContent = 'Анализировать';
        btnSpinner.classList.remove('visible');
    }
}

function renderResults(data) {
    const results = document.getElementById('results');

    document.getElementById('canonical-smiles').textContent = data.canonical_smiles;

    renderMolecule(data.molecule_svg_base64);
    renderGauge(data.overall_toxicity);
    renderModelInfo(data);
    renderTasks(data.tasks);
    renderAtomsChart(data.atom_importance);

    results.hidden = false;
}

function renderMolecule(svgBase64) {
    const container = document.getElementById('molecule-svg');
    if (!svgBase64) {
        container.innerHTML = '<p style="color:var(--text-muted)">Визуализация недоступна</p>';
        return;
    }
    container.innerHTML = atob(svgBase64);
}

function renderGauge(toxicity) {
    const container = document.getElementById('overall-gauge');
    const pct = Math.round(toxicity * 100);
    const circumference = 2 * Math.PI * 60;
    const offset = circumference * (1 - toxicity);

    let color, riskText;
    if (toxicity < 0.3) { color = '#22c55e'; riskText = 'Низкий риск'; }
    else if (toxicity < 0.6) { color = '#f59e0b'; riskText = 'Средний риск'; }
    else { color = '#ef4444'; riskText = 'Высокий риск'; }

    container.innerHTML = `
        <svg width="160" height="160" viewBox="0 0 160 160">
            <circle class="gauge-bg" cx="80" cy="80" r="60" />
            <circle class="gauge-ring gauge-fill" cx="80" cy="80" r="60"
                stroke="${color}"
                stroke-dasharray="${circumference}"
                stroke-dashoffset="${offset}" />
            <text x="80" y="76" text-anchor="middle" class="gauge-label" fill="${color}">${pct}%</text>
            <text x="80" y="100" text-anchor="middle" class="gauge-sublabel">${riskText}</text>
        </svg>
    `;
}

function renderModelInfo(data) {
    document.getElementById('model-badge').textContent = data.model_name;
    document.getElementById('params-badge').textContent = data.model_params;
    document.getElementById('atoms-badge').textContent = `${data.num_atoms} атомов`;

    const badge = document.getElementById('model-badge');
    if (data.model_type === 'transformer') {
        badge.className = 'badge badge-transformer';
    } else {
        badge.className = 'badge';
    }
}

function renderTasks(tasks) {
    const container = document.getElementById('tasks-list');
    const sorted = [...tasks].sort((a, b) => b.probability - a.probability);

    container.innerHTML = sorted.map(t => {
        const pct = Math.round(t.probability * 100);
        const barColor = t.toxic ? 'var(--danger)' : 'var(--success)';
        const badgeClass = t.toxic ? 'toxic' : 'safe';
        const badgeText = t.toxic ? 'TOX' : 'OK';

        return `
            <div class="task-row">
                <span class="task-name">${t.task}</span>
                <div class="task-bar-container">
                    <div class="task-bar" style="width:${pct}%;background:${barColor}"></div>
                </div>
                <span class="task-prob">${pct}%</span>
                <span class="task-badge ${badgeClass}">${badgeText}</span>
            </div>
        `;
    }).join('');
}

function renderAtomsChart(atoms) {
    const container = document.getElementById('atoms-chart');
    const maxImportance = Math.max(...atoms.map(a => a.importance), 0.01);
    const MAX_BAR_PX = 100;

    container.innerHTML = atoms.map(a => {
        const barHeight = Math.max(Math.round((a.importance / maxImportance) * MAX_BAR_PX), 3);
        const color = importanceToColor(a.importance);
        const pct = Math.round(a.importance * 100);
        return `
            <div class="atom-bar-group" title="${a.symbol}${a.index}: ${pct}%">
                <div class="atom-value">${pct}%</div>
                <div class="atom-bar" style="height:${barHeight}px;background:${color}"></div>
                <span class="atom-label">${a.symbol}<sub>${a.index}</sub></span>
            </div>
        `;
    }).join('');
}

function importanceToColor(val) {
    const r = Math.round(34 + val * (239 - 34));
    const g = Math.round(197 - val * (197 - 68));
    const b = Math.round(94 - val * (94 - 68));
    return `rgb(${r}, ${g}, ${b})`;
}
