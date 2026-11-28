'use strict';

const fields = [
  'id_deudor', 'monto_pendiente', 'dias_vencimiento', 'incumplimientos_anteriores',
  'pagos_parciales', 'intentos_contacto', 'respuesta_contacto', 'situacion_adeudo',
];
const rankFields = ['rank_model', 'rank_amount', 'rank_days'];
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
const sort = get('sort');
const dialog = get('case-dialog');
const previousPage = get('previous-page');
const nextPage = get('next-page');
let modelReady = false;
let busy = false;
let cases = [];
let warnings = new Map();
let page = 0;
let detailTrigger;

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
}

function renderRow(item) {
  const row = element('tr');
  const id = element('th', item.id_deudor);
  id.scope = 'row';
  const score = element('td');
  const scoreContents = element('div', undefined, 'probability');
  const progress = element('progress');
  progress.value = item.probability;
  progress.max = 1;
  progress.setAttribute('aria-hidden', 'true');
  scoreContents.append(progress, element('span', percentage.format(item.probability)));
  score.append(scoreContents);
  row.append(id, element('td', number.format(item.rank_original), 'rank-cell'), element('td', currency.format(item.monto_pendiente), 'numeric'),
    element('td', number.format(item.dias_vencimiento), 'numeric'), score);

  for (const field of rankFields) {
    const cell = element('td', undefined, 'rank-cell');
    cell.append(element('span', number.format(item[field]), field === sort.value ? 'rank-active' : undefined));
    row.append(cell);
  }

  const actions = element('td');
  const detail = element('button', 'Ver detalle', 'button secondary');
  detail.type = 'button';
  detail.setAttribute('aria-label', `Ver detalle del adeudo ${item.id_deudor}`);
  detail.addEventListener('click', () => openDetail(item, detail));
  actions.append(detail);
  row.append(actions);
  return row;
}

function renderPage() {
  const ordered = [...cases].sort((first, second) => first[sort.value] - second[sort.value]);
  const start = page * pageSize;
  const end = Math.min(start + pageSize, ordered.length);
  get('ranking-rows').replaceChildren(...ordered.slice(start, end).map(renderRow));
  get('pagination').hidden = ordered.length <= pageSize;
  get('page-status').textContent = `Casos ${number.format(start + 1)}–${number.format(end)} de ${number.format(ordered.length)}`;
  previousPage.disabled = page === 0;
  nextPage.disabled = end >= ordered.length;
}

function renderAnalysis(payload, origin, inputRecords) {
  if (!Array.isArray(payload.cases) || !Array.isArray(payload.excluded)) {
    throw new Error('El servidor devolvió un resultado de análisis incompleto.');
  }

  // Preserve the original CSV position, including gaps for excluded records.
  const positions = new Map(inputRecords.map((item, index) => [item.id_deudor.trim(), index + 1]));
  cases = payload.cases.map((item) => ({ ...item, rank_original: positions.get(item.id_deudor) }));
  const excluded = payload.excluded;
  warnings = new Map((payload.warnings ?? []).map((item) => [item.id_deudor, item.fields]));
  const caseCount = number.format(cases.length);
  const excludedCount = number.format(excluded.length);
  const analyzedLabel = cases.length === 1 ? 'caso analizado' : 'casos analizados';
  const excludedLabel = excluded.length === 1 ? 'excluido' : 'excluidos';
  get('results-summary').textContent = `${origin} · ${caseCount} ${analyzedLabel} · ${excludedCount} ${excludedLabel}`;
  get('sort-control').hidden = !cases.length;
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
  sort.value = rankFields[0];
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

sort.addEventListener('change', () => { page = 0; renderPage(); });
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
dialog.addEventListener('close', () => detailTrigger?.focus());

loadStatus();
