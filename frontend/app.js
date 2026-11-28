'use strict';

const fields = [
  'id_deudor', 'monto_pendiente', 'dias_vencimiento', 'incumplimientos_anteriores',
  'pagos_parciales', 'intentos_contacto', 'respuesta_contacto', 'situacion_adeudo',
];
const wholeFields = new Set(['dias_vencimiento', 'incumplimientos_anteriores', 'intentos_contacto', 'respuesta_contacto', 'situacion_adeudo']);
const pageSize = 15;
const maxCsvBytes = 2 * 1024 * 1024;
const currency = new Intl.NumberFormat('es-MX', { style: 'currency', currency: 'MXN', currencyDisplay: 'code' });
const number = new Intl.NumberFormat('es-MX');
const percentage = new Intl.NumberFormat('es-MX', { style: 'percent', maximumFractionDigits: 1 });
const decimal = new Intl.NumberFormat('es-MX', { minimumFractionDigits: 3, maximumFractionDigits: 3 });
const get = (id) => document.getElementById(id);

const sampleButton = get('sample-button');
const fileButton = get('file-button');
const fileInput = get('file-input');
const operation = get('operation');
const errorPanel = get('error-panel');
const dialog = get('case-dialog');
const previousPage = get('previous-page');
const nextPage = get('next-page');
let modelReady = false;
let busy = false;
let cases = [];
let warnings = new Map();
let page = 0;
let detailTrigger;
let explanationRequest = 0;
let analysisModelVersion = null;

function element(tag, content, className) {
  const node = document.createElement(tag);
  if (content !== undefined) node.textContent = content;
  if (className) node.className = className;
  return node;
}

function renderList(container, items, describe) {
  const fragment = document.createDocumentFragment();
  for (const item of items) fragment.append(element('li', describe(item)));
  container.replaceChildren(fragment);
}

function updateControls() {
  sampleButton.disabled = !modelReady || busy;
  fileButton.disabled = !modelReady || busy;
  get('results').setAttribute('aria-busy', String(busy));
}

function setModelState(state, label, description) {
  const status = get('model-status');
  status.dataset.state = state;
  status.textContent = label;
  const message = get('model-message');
  message.textContent = description || '';
  message.hidden = !description;
  modelReady = state === 'ready';
  updateControls();
}

async function request(path, body) {
  const response = await fetch(path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : {});
  const payload = await response.json().catch(() => ({}));
  if (response.ok) return payload;

  const issues = Array.isArray(payload.detail) ? payload.detail.map((issue) => {
    const parts = Array.isArray(issue.loc) ? issue.loc : [];
    const row = typeof parts[2] === 'number' ? `Caso ${parts[2] + 1}` : '';
    const field = parts.slice(3).join('.');
    return [row, field, issue.msg].filter(Boolean).join(' · ');
  }) : [typeof payload.detail === 'string' ? payload.detail : `El servidor devolvió el código ${response.status}.`];
  const error = new Error(issues[0]);
  error.issues = issues;
  error.status = response.status;
  throw error;
}

function renderMetrics(metrics) {
  const container = get('metrics');
  const values = [['Random Forest', metrics?.random_forest], ['Naive Bayes', metrics?.naive_bayes]];
  container.replaceChildren();
  for (const [label, scores] of values) {
    if (!Number.isFinite(scores?.f1) || !Number.isFinite(scores?.roc_auc)) continue;
    container.append(element('p', `${label} · F1 ${decimal.format(scores.f1)} · ROC AUC ${decimal.format(scores.roc_auc)}`));
  }
  container.hidden = !container.childElementCount;
}

async function loadStatus() {
  try {
    const status = await request('/api/status');
    renderMetrics(status.metrics);
    if (status.ready) {
      setModelState('ready', 'Modelo disponible');
    } else {
      setModelState('unavailable', 'Modelo no disponible', 'Consulta las instrucciones del proyecto para preparar o verificar el modelo. Después, recarga esta página.');
    }
  } catch {
    setModelState('unavailable', 'Servicio no disponible', 'Comprueba que el servidor esté en ejecución y recarga esta página.');
  }
}

async function loadEvaluation() {
  const container = get('evaluation-content');
  try {
    const result = await request('/api/evaluation');
    const cards = element('div', undefined, 'evaluation-cards');
    for (const [key, label] of [['modelo', 'Modelo Random Forest'], ['monto', 'Mayor monto'], ['dias', 'Más días vencidos']]) {
      const count = result.recovered[key];
      if (!Number.isInteger(count) || count < 0 || count > result.top_n) throw new Error('Resultado inválido');
      const card = element('article', undefined, 'evaluation-card');
      const bar = element('progress', undefined, 'evaluation-bar');
      bar.max = result.top_n;
      bar.value = count;
      bar.setAttribute('aria-label', `${label}: ${count} de ${result.top_n} casos recuperados`);
      card.append(element('h3', label, 'evaluation-label'), element('p', `${count} / ${result.top_n}`, 'evaluation-value'), bar,
        element('p', 'recuperados observados', 'evaluation-caption'));
      cards.append(card);
    }
    const context = element('p', `Mismo conjunto de prueba: ${number.format(result.test_cases)} casos disponibles. Se muestran los primeros ${result.top_n} de cada orden.`, 'evaluation-caption');
    container.replaceChildren(context, cards);
  } catch {
    container.textContent = 'No fue posible cargar la comparación. Comprueba que el modelo esté preparado y recarga la página.';
  }
}

function detectDelimiter(source) {
  let commas = 0;
  let semicolons = 0;
  let quoted = false;
  for (const char of source) {
    if (char === '"') quoted = !quoted;
    if (quoted) continue;
    if (char === '\n' || char === '\r') break;
    if (char === ',') commas++;
    if (char === ';') semicolons++;
  }
  return semicolons > commas ? ';' : ',';
}

function readRows(source, delimiter) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  let quoteClosed = false;
  const finishField = () => {
    row.push(field.trim());
    field = '';
    quoteClosed = false;
  };
  const finishRow = () => {
    finishField();
    if (row.length > 1 || row[0]) rows.push(row);
    row = [];
  };

  for (let index = 0; index < source.length; index++) {
    const char = source[index];
    if (quoted) {
      if (char !== '"') {
        field += char;
      } else if (source[index + 1] === '"') {
        field += '"';
        index++;
      } else {
        quoted = false;
        quoteClosed = true;
      }
    } else if (char === delimiter) {
      finishField();
    } else if (char === '\r' || char === '\n') {
      if (char === '\r' && source[index + 1] === '\n') index++;
      finishRow();
    } else if (quoteClosed && /\s/.test(char)) {
      continue;
    } else if (char === '"' && !field.trim() && !quoteClosed) {
      field = '';
      quoted = true;
    } else if (char === '"' || quoteClosed) {
      throw new Error(`Revisa las comillas en el registro ${rows.length + 1} del CSV.`);
    } else {
      field += char;
    }
  }

  if (quoted) throw new Error('El CSV contiene un campo entre comillas sin cerrar.');
  finishRow();
  return rows;
}

function parseCsv(text) {
  const source = text.replace(/^\uFEFF/, '').trimStart();
  const [headers, ...rows] = readRows(source, detectDelimiter(source));
  if (!headers) throw new Error('El archivo CSV está vacío.');
  if (new Set(headers).size !== headers.length) throw new Error('El CSV contiene encabezados duplicados.');
  const missing = fields.filter((name) => !headers.includes(name));
  if (missing.length) throw new Error(`Faltan columnas obligatorias: ${missing.join(', ')}.`);
  if (!rows.length) throw new Error('Agrega al menos un caso debajo de los encabezados.');

  const columns = fields.map((name) => headers.indexOf(name));
  const numeric = /^\+?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;
  return rows.map((row, index) => {
    const context = `Caso ${index + 1}`;
    if (row.length !== headers.length) throw new Error(`${context}: la cantidad de campos difiere de la de los encabezados.`);
    return Object.fromEntries(fields.map((name, position) => {
      const value = row[columns[position]];
      if (!value) throw new Error(`${context}: completa el campo ${name}.`);
      if (name === 'id_deudor') return [name, value];
      if (!numeric.test(value) || !Number.isFinite(Number(value))) {
        throw new Error(`${context}: usa un número no negativo en ${name}, con punto decimal si aplica.`);
      }
      const parsed = Number(value);
      if (wholeFields.has(name) && !Number.isSafeInteger(parsed)) {
        throw new Error(`${context}: usa un número entero válido en ${name}.`);
      }
      if ((name === 'respuesta_contacto' && parsed > 2) || (name === 'situacion_adeudo' && parsed > 1)) {
        throw new Error(`${context}: usa uno de los códigos permitidos en ${name}.`);
      }
      return [name, parsed];
    }));
  });
}

function openDetail(item, trigger) {
  const details = [
    ['Orden original en el archivo', number.format(item.rank_original)],
    ['Posición original entre activos', number.format(item.rank_active)],
    ['Saldo pendiente', currency.format(item.monto_pendiente)],
    ['Días vencidos', number.format(item.dias_vencimiento)],
    ['Probabilidad de recuperación sintética', percentage.format(item.probability)],
    ['Posición por modelo', number.format(item.rank_model)],
    ['Posición por saldo', number.format(item.rank_amount)],
    ['Posición por días', number.format(item.rank_days)],
    ['Incumplimientos anteriores', number.format(item.incumplimientos_anteriores)],
    ['Pagos parciales', currency.format(item.pagos_parciales)],
    ['Intentos de contacto', number.format(item.intentos_contacto)],
    ['Respuesta de contacto (código)', number.format(item.respuesta_contacto)],
    ['Situación del adeudo', item.situacion_adeudo === 1 ? 'Disponible para cobro' : 'Requiere revisión'],
  ];
  get('detail-title').textContent = item.id_deudor;
  const warning = get('detail-warning');
  const flagged = warnings.get(item.id_deudor);
  warning.textContent = flagged ? `Fuera del rango de entrenamiento: ${flagged.join(', ')}.` : '';
  warning.hidden = !flagged;
  get('detail-fields').replaceChildren(...details.map(([term, value]) => {
    const group = element('div');
    group.append(element('dt', term), element('dd', value));
    return group;
  }));
  detailTrigger = trigger;
  dialog.showModal();
  showExplanation(item);
}

async function showExplanation(item) {
  const token = ++explanationRequest;
  const container = get('detail-explanation');
  container.replaceChildren(element('h3', '¿Qué influye en la estimación?'), element('p', 'Calculando sensibilidad con el modelo local…'));
  const names = {monto_pendiente: 'Saldo', pagos_parciales: 'Pagos parciales', dias_vencimiento: 'Días vencidos', incumplimientos_anteriores: 'Incumplimientos', intentos_contacto: 'Intentos', respuesta_contacto: 'Respuesta (código)'};
  const describe = (values) => Object.entries(values).map(([key, value]) => `${names[key] || key}: ${key === 'monto_pendiente' || key === 'pagos_parciales' ? currency.format(value) : number.format(value)}`).join(' · ');
  try {
    const result = await request('/api/explain', {case: Object.fromEntries(fields.map((key) => [key, item[key]]))});
    if (token !== explanationRequest || !dialog.open) return;
    if (!analysisModelVersion || result.model_version !== analysisModelVersion) {
      container.replaceChildren(element('h3', 'El modelo cambió'), element('p', 'Vuelve a analizar los datos para obtener una explicación consistente con los resultados.'));
      return;
    }
    const list = element('div', undefined, 'explanation-list');
    for (const entry of [...result.items].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta))) {
      const card = element('div', undefined, 'explanation-item');
      const magnitude = Math.abs(entry.delta);
      const change = magnitude < 0.05 ? 'Sin cambio apreciable' : `${entry.delta > 0 ? 'Baja' : 'Sube'} ${number.format(Number(magnitude.toFixed(1)))} puntos porcentuales`;
      card.append(element('strong', entry.label), element('p', change, 'explanation-delta'),
        element('p', `Caso: ${describe(entry.original)}`), element('p', `Referencia: ${describe(entry.reference)}`));
      list.append(card);
    }
    container.replaceChildren(element('h3', '¿Qué influye en la estimación?'),
      element('p', 'Para cada grupo cambiamos sus datos por una referencia del entrenamiento y volvemos a calcular la probabilidad. El cambio indicado es respecto a este caso.'), list,
      element('p', 'Son cambios aislados que dependen de la referencia; no son causas ni contribuciones que puedan sumarse. Algunas combinaciones podrían no haber aparecido en los datos sintéticos.', 'explanation-note'));
  } catch {
    if (token === explanationRequest && dialog.open) container.replaceChildren(element('h3', 'Explicación no disponible'), element('p', 'No se pudo calcular la sensibilidad del modelo. Comprueba que esté preparado y vuelve a intentarlo.'));
  }
}

function detailButton(item) {
  const button = element('button', 'Detalle', 'button secondary detail-button');
  button.type = 'button';
  button.setAttribute('aria-label', `Ver detalle del adeudo ${item.id_deudor}`);
  button.addEventListener('click', () => openDetail(item, button));
  return button;
}

const criteria = {
  rank_model: { title: 'Orden del modelo', description: 'Mayor probabilidad sintética primero', value: (item) => percentage.format(item.probability), valueHeading: 'Probabilidad sintética' },
  rank_amount: { title: 'Orden por saldo', description: 'Mayor saldo pendiente primero', value: (item) => currency.format(item.monto_pendiente), valueHeading: 'Saldo pendiente' },
  rank_days: { title: 'Orden por días', description: 'Más días vencidos primero', value: (item) => number.format(item.dias_vencimiento), valueHeading: 'Días vencidos' },
};

function renderComparisonRow(item, view) {
  const selected = get('comparison-criterion').value;
  const rank = item[selected];
  const row = element('tr');
  const id = element('th', item.id_deudor);
  id.scope = 'row';
  const actions = element('td');
  actions.append(detailButton(item));
  if (view === 'original') {
    row.append(element('td', number.format(item.rank_original), 'position-cell'), element('td', number.format(item.rank_active)), id,
      element('td', currency.format(item.monto_pendiente)), element('td', number.format(rank)), actions);
  } else {
    const difference = item.rank_active - rank;
    const description = difference > 0 ? `↑ Sube ${number.format(difference)} ${difference === 1 ? 'posición' : 'posiciones'}`
      : difference < 0 ? `↓ Baja ${number.format(-difference)} ${difference === -1 ? 'posición' : 'posiciones'}` : 'Igual';
    const change = element('td');
    const transition = element('span', undefined, 'position-transition');
    transition.append(element('span', number.format(item.rank_active)), element('span', ' → ', 'transition-arrow'),
      element('span', number.format(rank)));
    change.append(transition, element('span', description, `position-change ${difference > 0 ? 'position-up' : difference < 0 ? 'position-down' : 'position-same'}`));
    const criterionValue = element('td', criteria[selected].value(item));
    criterionValue.hidden = selected === 'rank_model';
    row.append(element('td', number.format(rank), 'position-cell'), id,
      element('td', percentage.format(item.probability)), criterionValue, change, actions);
  }
  return row;
}

function renderPage() {
  const query = get('case-search').value.trim();
  const selected = get('comparison-criterion').value;
  const criterion = criteria[selected];
  get('model-title').textContent = criterion.title;
  get('criterion-description').textContent = criterion.description;
  get('criterion-value-heading').textContent = criterion.valueHeading;
  get('criterion-value-heading').hidden = selected === 'rank_model';
  const matching = cases.filter((item) => !query || item.id_deudor === query);
  const original = [...matching].sort((a, b) => a.rank_original - b.rank_original);
  const model = [...matching].sort((a, b) => a[selected] - b[selected]);
  const start = page * pageSize;
  const end = Math.min(start + pageSize, matching.length);
  get('original-rows').replaceChildren(...original.slice(start, end).map((item) => renderComparisonRow(item, 'original')));
  get('model-rows').replaceChildren(...model.slice(start, end).map((item) => renderComparisonRow(item, 'model')));
  get('pagination').hidden = matching.length <= pageSize;
  get('page-status').textContent = `Tramo ${number.format(start + 1)}–${number.format(end)} de ${number.format(matching.length)} coincidencias en cada orden`;
  previousPage.disabled = page === 0;
  nextPage.disabled = end >= matching.length;
  get('search-status').textContent = matching.length ? `${number.format(matching.length)} ${matching.length === 1 ? 'caso encontrado' : 'casos encontrados'} en cada lista.` : 'No se encontró un caso activo con ese identificador exacto.';
}

function renderAnalysis(payload, origin, inputRecords) {
  if (!Array.isArray(payload.cases) || !Array.isArray(payload.excluded)) {
    throw new Error('El servidor devolvió un resultado de análisis incompleto.');
  }

  // Preserve the original CSV position, including gaps for excluded records.
  const positions = new Map(inputRecords.map((item, index) => [item.id_deudor.trim(), index + 1]));
  cases = payload.cases.map((item) => ({ ...item, rank_original: positions.get(item.id_deudor) }));
  [...cases].sort((a, b) => a.rank_original - b.rank_original).forEach((item, index) => { item.rank_active = index + 1; });
  get('case-search').value = '';
  get('comparison-criterion').value = 'rank_model';
  analysisModelVersion = payload.model_version || null;
  const excluded = payload.excluded;
  warnings = new Map((payload.warnings ?? []).map((item) => [item.id_deudor, item.fields]));
  const caseCount = number.format(cases.length);
  const excludedCount = number.format(excluded.length);
  const analyzedLabel = cases.length === 1 ? 'caso analizado' : 'casos analizados';
  const excludedLabel = excluded.length === 1 ? 'excluido' : 'excluidos';
  get('results-summary').textContent = `${origin} · ${caseCount} ${analyzedLabel} · ${excludedCount} ${excludedLabel}`;
  get('ranking-note').hidden = !cases.length;
  get('table-region').hidden = !cases.length;
  const empty = get('empty-results');
  empty.hidden = !!cases.length;
  empty.lastElementChild.textContent = excluded.length
    ? 'Todos los casos requieren revisión y quedan fuera de la comparativa.'
    : 'El servidor no devolvió casos para comparar.';

  const excludedPanel = get('excluded-panel');
  excludedPanel.open = false;
  excludedPanel.hidden = !excluded.length;
  get('excluded-summary').textContent = `Casos que requieren revisión (${excludedCount})`;
  renderList(get('excluded-list'), excluded, (item) => `${item.id_deudor} · ${currency.format(item.monto_pendiente)} · ${number.format(item.dias_vencimiento)} días vencidos`);
  const warningPanel = get('warning-panel');
  warningPanel.open = false;
  warningPanel.hidden = !warnings.size;
  get('warning-summary').textContent = `Casos fuera del rango de entrenamiento (${number.format(warnings.size)})`;
  renderList(get('warning-list'), warnings, ([id, flagged]) => `${id} · ${flagged.join(', ')}`);
  page = 0;
  renderPage();
}

async function analyze(origin, loadCases, trigger) {
  if (busy || !modelReady) return;
  busy = true;
  updateControls();
  errorPanel.hidden = true;
  operation.textContent = `Procesando ${origin}…`;
  operation.hidden = false;
  let finished = false;
  try {
    const records = await loadCases();
    const result = await request('/api/analyze', { cases: records });
    renderAnalysis(result, origin, records);
    finished = true;
  } catch (error) {
    if (error.status === 503) {
      setModelState('unavailable', 'Modelo no disponible', 'Comprueba el estado del modelo en el servidor y recarga esta página.');
    }
    const issues = error.issues || [error instanceof TypeError ? 'La conexión con el servidor se interrumpió. Intenta nuevamente.' : error.message];
    const messages = issues.slice(0, 12);
    if (issues.length > messages.length) messages.push(`Hay ${issues.length - messages.length} errores adicionales.`);
    get('error-list').replaceChildren(...messages.map((message) => element('li', message)));
    errorPanel.hidden = false;
    errorPanel.focus();
  } finally {
    busy = false;
    operation.hidden = true;
    updateControls();
    if (finished && (document.activeElement === document.body || document.activeElement === trigger)) {
      get('results-title').focus();
    }
  }
}

sampleButton.addEventListener('click', () => analyze('Muestra sintética', async () => {
  const sample = await request('/api/sample');
  if (!Array.isArray(sample.cases)) throw new Error('El servidor devolvió una muestra incompleta.');
  return sample.cases;
}, sampleButton));

fileButton.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  const [file] = fileInput.files;
  fileInput.value = '';
  if (!file) return;
  analyze(file.name, async () => {
    if (!/\.csv$/i.test(file.name)) throw new Error('Selecciona un archivo con extensión .csv.');
    if (file.size > maxCsvBytes) throw new Error('Selecciona un archivo CSV de hasta 2 MB.');
    return parseCsv(await file.text());
  }, fileButton);
});

get('comparison-criterion').addEventListener('change', () => { page = 0; renderPage(); });
get('case-search').addEventListener('input', () => { page = 0; renderPage(); });
get('replay-comparison').addEventListener('click', () => {
  const region = get('table-region');
  region.classList.remove('replaying');
  void region.offsetWidth;
  region.classList.add('replaying');
});
previousPage.addEventListener('click', () => {
  page--;
  renderPage();
  if (previousPage.disabled) nextPage.focus();
});
nextPage.addEventListener('click', () => {
  page++;
  renderPage();
  if (nextPage.disabled) previousPage.focus();
});
dialog.addEventListener('close', () => { explanationRequest++; detailTrigger?.focus(); });

loadStatus();
loadEvaluation();
