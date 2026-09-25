/* Dashboard state and scene building.
 *
 * Every dataset arrives once as JSON; filtering, ranking and redrawing then
 * happen entirely in the page, so nothing waits on the Python process. */
'use strict';

const SOC_VIEWS = new Set(['SoC Average Power Draw', 'SoC Average Efficiency']);
const SOC_TOGGLE_VIEWS = new Set(['Energy efficiency', 'Average power draw']);
const REFERENCE_VIEWS = new Set(['Performance curve', 'Runtime vs capacity', 'Energy efficiency', 'Average power draw', 'SoC Average Power Draw']);
// GPU efficiency varies less across architectures, so its rays sit closer.
const EFFICIENCY_LEVELS = [1, 0.5, 0.25, 0.125];
const GPU_EFFICIENCY_LEVELS = [1, 0.875, 0.75, 0.625, 0.5];
const GPU_DATASETS = new Set(['GPU', 'Laptop GPU']);
const CORE_GROUP_LABELS = { super: 'Super', large: 'Large', medium: 'Medium', small: 'Small' };

const element = (id) => document.getElementById(id);
const ui = {
  tabs: element('tabs'),
  kicker: element('kicker'),
  title: element('title'),
  view: element('view'),
  search: element('search'),
  quick: element('quick'),
  coreFilter: element('core-filter'),
  coreGroups: element('core-groups'),
  coreNamesButton: element('core-names-button'),
  coreNamesMenu: element('core-names-menu'),
  profileList: element('profile-list'),
  selectionNote: element('selection-note'),
  sourceNote: element('source-note'),
  chartNote: element('chart-note'),
  chartGuide: element('chart-guide'),
  rankHint: element('rank-hint'),
  statProfiles: element('stat-profiles'),
  statPoints: element('stat-points'),
  statLeader: element('stat-leader'),
  listTabs: element('list-tabs'),
  searchLabel: element('search-label'),
  overlayToggle: element('overlay-toggle'),
  referenceToggle: element('reference-toggle'),
  referenceNote: element('reference-note'),
  overlayWrap: element('overlay-wrap'),
  measuredToggle: element('measured-toggle'),
  measuredWrap: element('measured-wrap'),
  reload: element('reload'),
  exportButton: element('export'),
  exportMenu: element('export-menu'),
  toast: element('toast'),
};

const state = {
  payload: null,
  datasets: new Map(),
  key: '',
  palette: [],
  views: {},
  // Devices and processors are two selections over the same tab, so their
  // search text, picks and latched mode are tracked per list.
  lists: {},
  searches: { devices: {}, socs: {} },
  selection: { devices: {}, socs: {} },
  modes: { devices: {}, socs: {} },
  cores: {},
  rankOffsets: {},
  colorMaps: new Map(),
  showMeasured: true,
  showReference: true,
  showSocAverages: false,
  visible: [],
  lastRank: null,
};

const surface = new Chart.Surface(element('chart-stage'), element('chart-base'), element('chart-overlay'));
let hoverPoints = [];

/* ---------- helpers ---------- */

function dataset() { return state.datasets.get(state.key); }

/** Which point fields each view puts on each axis. */
function curveAxes(entry, view) {
  if (view === 'Performance curve') {
    return { xIndex: 0, yIndex: 1, xLabel: 'Board power (W)', yLabel: entry.scoreLabel,
             rankIndex: 1, rankLabel: 'Peak score' };
  }
  if (view === 'Efficiency vs score') {
    return { xIndex: 1, yIndex: 2, xLabel: entry.scoreLabel, yLabel: 'Score per watt',
             rankIndex: 2, rankLabel: 'Peak score/W' };
  }
  return { xIndex: 0, yIndex: 2, xLabel: 'Board power (W)', yLabel: 'Score per watt',
           rankIndex: 2, rankLabel: 'Peak score/W' };
}

function batteryAxes(view) {
  if (view === 'Energy efficiency') {
    return { xKey: 'capacityWh', yKey: 'minutes', title: 'Energy efficiency by battery size',
             xLabel: 'Battery capacity (Wh)', yLabel: 'Runtime (minutes)',
             rankKey: 'minPerWh', rankLabel: 'Minutes per Wh', higher: true };
  }
  if (view === 'Average power draw') {
    return { xKey: 'capacityWh', yKey: 'avgPowerW', title: 'Power draw by battery size',
             xLabel: 'Battery capacity (Wh)', yLabel: 'Average power draw (W)',
             rankKey: 'avgPowerW', rankLabel: 'Average power (W)', higher: false };
  }
  return { xKey: 'capacityWh', yKey: 'hours', title: 'Endurance landscape',
           xLabel: 'Battery capacity (Wh)', yLabel: 'Runtime (hours)',
           rankKey: 'hours', rankLabel: 'Runtime (hours)', higher: true };
}

function socAxes(view) {
  return view === 'SoC Average Power Draw'
    ? { xKey: 'capacityWh', yKey: 'powerW', xLabel: 'Average battery capacity (Wh)',
        yLabel: 'Average power draw (W)', title: 'Processor-average power draw',
        rankLabel: 'Average power (W)', higher: false }
    : { xKey: 'powerW', yKey: 'minPerWh', xLabel: 'Average power draw (W)',
        yLabel: 'Average minutes per Wh', title: 'Processor-average efficiency',
        rankLabel: 'Average minutes per Wh', higher: true };
}

/* Where the better results sit on each view, and a one-line reading of it. */
const BETTER = {
  'Performance curve': ['top-left', 'more score for less power'],
  'Efficiency curve': ['top-left', 'more score per watt at lower power'],
  'Efficiency vs score': ['top-right', 'higher score and more score per watt'],
  'Runtime vs capacity': ['top-left', 'longer runtime from a smaller battery'],
  'Energy efficiency': ['top-left', 'longer runtime from a smaller battery'],
  'Average power draw': ['bottom-right', 'lower power draw with a bigger battery'],
  'SoC Average Power Draw': ['bottom-right', 'lower power draw with a bigger battery'],
  'SoC Average Efficiency': ['top-left', 'more minutes per Wh at lower power'],
};

/* ---------- colours ---------- */

function extentOf(items) {
  const xs = [];
  const ys = [];
  for (const item of items) {
    for (const point of item.points) {
      if (Number.isFinite(point[0]) && Number.isFinite(point[1])) { xs.push(point[0]); ys.push(point[1]); }
    }
  }
  if (!xs.length) return { x0: 0, x1: 1, y0: 0, y1: 1 };
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

/** Where every profile of a list sits on one figure, selection ignored. */
function figureGeometry(entry, view, list) {
  if (list === 'socs') {
    const averages = entry.socAverages || [];
    if (SOC_VIEWS.has(view)) {
      const axes = socAxes(view);
      const items = averages.map((average) => ({
        id: average.soc, points: [[average[axes.xKey], average[axes.yKey]]],
      }));
      return { items, extent: extentOf(items) };
    }
    // Over a device chart the averages are full-width rules, so only their y
    // separates them, against the device chart's own range.
    const axes = batteryAxes(view);
    const averageKeys = { avgPowerW: 'powerW', minutes: 'minPerWh', hours: 'runtimeHours' };
    const devices = figureGeometry(entry, view, 'devices');
    const items = averages.map((average) => {
      const y = average[averageKeys[axes.yKey]];
      return {
        id: average.soc,
        kind: axes.yKey === 'minutes' ? 'line' : 'hline',
        points: [devices.extent.x0, devices.extent.x1].map((x) =>
          [x, axes.yKey === 'minutes' ? x * y : y]),
      };
    });
    return { items, extent: extentOf([...devices.items, ...items]) };
  }

  if (entry.kind === 'curve') {
    const axes = curveAxes(entry, view);
    const items = entry.profiles.map((profile) => ({
      id: profile.label,
      points: profile.series.flatMap((series) =>
        series.points.map((point) => [point[axes.xIndex], point[axes.yIndex]])),
    }));
    return { items, extent: extentOf(items) };
  }

  const axes = batteryAxes(view);
  const items = entry.profiles.map((profile) => {
    const points = [[profile[axes.xKey], profile[axes.yKey]]];
    if (profile.measured) {
      const measuredKeys = { capacityWh: 'capacityWh', avgPowerW: 'avgPowerW', minPerWh: 'minPerWh' };
      points.push([
        measuredKeys[axes.xKey] ? profile.measured[measuredKeys[axes.xKey]] : profile[axes.xKey],
        measuredKeys[axes.yKey] ? profile.measured[measuredKeys[axes.yKey]] : profile[axes.yKey],
      ]);
    }
    return { id: profile.label, points };
  });
  return { items, extent: extentOf(items) };
}

/** Colours for one figure, computed once and reused for every selection. */
function colorMap(list) {
  const entry = dataset();
  const view = currentView();
  const cacheKey = `${state.key}|${view}|${list}`;
  let map = state.colorMaps.get(cacheKey);
  if (!map) {
    if (entry.kind === 'battery' && SOC_TOGGLE_VIEWS.has(view)) {
      // Devices and processor-average rules can share one chart. Assign them
      // together even while the overlay is hidden, so a later toggle neither
      // recolours devices nor introduces an avoidable same-colour collision.
      const groups = ['devices', 'socs'].map((name) => ({
        name,
        geometry: figureGeometry(entry, view, name),
      }));
      const tagged = groups.flatMap(({ name, geometry }) =>
        geometry.items.map((item, index) => ({
          id: `${name}:${index}`,
          kind: item.kind,
          points: item.points,
        })));
      const assigned = Colors.assign(tagged, state.palette, extentOf(tagged));
      groups.forEach(({ name, geometry }) => {
        const groupMap = new Map();
        geometry.items.forEach((item, index) => {
          groupMap.set(item.id, assigned.get(`${name}:${index}`));
        });
        state.colorMaps.set(`${state.key}|${view}|${name}`, groupMap);
      });
    } else {
      const { items, extent } = figureGeometry(entry, view, list);
      state.colorMaps.set(cacheKey, Colors.assign(items, state.palette, extent));
    }
    map = state.colorMaps.get(cacheKey);
  }
  return map;
}

function colorFor(id, list) {
  return colorMap(list || activeList()).get(id) || state.palette[0];
}

function fixed(value, decimals) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: decimals, maximumFractionDigits: decimals,
  });
}

function shorten(text, length) {
  return text.length <= length ? text : text.slice(0, length - 1).trimEnd() + '…';
}

function toast(message, tone) {
  ui.toast.textContent = message;
  ui.toast.dataset.tone = tone || 'info';
  ui.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { ui.toast.hidden = true; }, 3200);
}

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, '0');
}

/* ---------- per-dataset state ---------- */

function cores(key) {
  if (!state.cores[key]) state.cores[key] = { groups: new Set(), names: new Set() };
  return state.cores[key];
}

/** Processor averages get their own list only where a battery CSV carries SoCs. */
function hasSocList(entry) {
  return Boolean(entry) && entry.kind === 'battery' && (entry.socAverages || []).length > 0;
}

function activeList(key) {
  const entry = state.datasets.get(key || state.key);
  return hasSocList(entry) && state.lists[key || state.key] === 'socs' ? 'socs' : 'devices';
}

function selection(key, list) {
  const store = state.selection[list || activeList(key)];
  if (!store[key]) store[key] = new Set();
  return store[key];
}

function mode(key, list) {
  return state.modes[list || activeList(key)][key] || 'manual';
}

function setMode(key, value, list) {
  state.modes[list || activeList(key)][key] = value;
}

function search(key, list) {
  return state.searches[list || activeList(key)][key] || '';
}

/** Every processor in the dataset, as sidebar items. */
function socItems() {
  const entry = dataset();
  return (entry.socAverages || []).map((average) => ({
    id: average.soc,
    label: average.soc,
    meta: String(average.deviceCount),
    average,
  }));
}

function deviceItems() {
  return filteredProfiles().map((profile) => ({ id: profile.label, label: profile.label, profile }));
}

/** Whether any loaded device has a Geekerwan measured-capacity match. */
function hasMeasured(entry) {
  return Boolean(entry) && entry.kind === 'battery'
    && entry.profiles.some((profile) => profile.measured);
}

/** The processors whose averages the charts should draw. */
function selectedSocs(entry) {
  const chosen = selection(state.key, 'socs');
  return (entry.socAverages || []).filter((average) => chosen.has(average.soc));
}

function matchesCoreFilter(profile, filter) {
  if (filter.groups.size && !filter.groups.has(profile.coreGroup)) return false;
  if (filter.names.size && !filter.names.has(profile.core)) return false;
  return true;
}

function filteredProfiles() {
  const entry = dataset();
  if (!entry) return [];
  const filter = cores(state.key);
  return entry.profiles.filter((profile) => matchesCoreFilter(profile, filter));
}

function visibleFor(list) {
  const needle = search(state.key, list).trim().toLowerCase();
  const items = list === 'socs' ? socItems() : deviceItems();
  return items.filter((item) => item.label.toLowerCase().includes(needle));
}

function visibleItems() {
  return visibleFor(activeList());
}

function currentView() {
  const entry = dataset();
  if (!entry) return '';
  const stored = state.views[state.key];
  return entry.views.includes(stored) ? stored : entry.views[0];
}

/** How the current chart orders processor averages, for the Top 5 button. */
function socRankMetric(view) {
  if (view === 'SoC Average Power Draw' || view === 'Average power draw') {
    return { key: 'powerW', higher: false };
  }
  if (view === 'Runtime vs capacity') return { key: 'runtimeHours', higher: true };
  return { key: 'minPerWh', higher: true };
}

/** Rank a list by the metric the current chart uses, for the Top 5 button. */
function rankedIds(list) {
  const entry = dataset();
  if (!entry) return [];
  const view = currentView();
  if (list === 'socs') {
    const metric = socRankMetric(view);
    return (entry.socAverages || [])
      .slice()
      .sort((left, right) => (metric.higher ? 1 : -1) * ((right[metric.key] || 0) - (left[metric.key] || 0)))
      .map((average) => average.soc);
  }
  const profiles = filteredProfiles();
  if (entry.kind === 'curve') {
    const index = view === 'Performance curve' ? 1 : 2;
    return profiles
      .map((profile) => ({
        label: profile.label,
        value: Math.max(...profile.series.flatMap((item) => item.points.map((point) => point[index]))),
      }))
      .sort((left, right) => right.value - left.value)
      .map((item) => item.label);
  }
  const metric = batteryAxes(view);
  return profiles
    .slice()
    .sort((left, right) => (metric.higher ? 1 : -1) * (right[metric.rankKey] - left[metric.rankKey]))
    .map((profile) => profile.label);
}

/** Top 5 and All shown stay latched as filters, search and views change.
 *  Both lists are re-applied every refresh, so the list you are not looking
 *  at still follows the chart metric. */
function applySelectionMode() {
  for (const list of hasSocList(dataset()) ? ['devices', 'socs'] : ['devices']) {
    const current = mode(state.key, list);
    const chosen = selection(state.key, list);
    if (current === 'all' || current === 'top5') {
      const visible = new Set(visibleFor(list).map((item) => item.id));
      const ids = current === 'all'
        ? [...visible]
        : rankedIds(list).filter((id) => visible.has(id)).slice(0, 5);
      chosen.clear();
      ids.forEach((id) => chosen.add(id));
    } else if (current === 'clear') {
      chosen.clear();
    }
  }
  const active = mode(state.key);
  for (const button of ui.quick.querySelectorAll('button')) {
    button.dataset.active = String(button.dataset.mode === active);
  }
  return active;
}

function selectedProfiles() {
  const chosen = selection(state.key, 'devices');
  return filteredProfiles().filter((profile) => chosen.has(profile.label));
}

/* ---------- rendering the shell ---------- */

function renderTabs() {
  ui.tabs.replaceChildren();
  for (const entry of state.payload.datasets) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = entry.tabLabel;
    button.setAttribute('aria-selected', String(entry.key === state.key));
    button.dataset.empty = String(!entry.available);
    button.title = entry.available ? entry.title : `No snapshot yet — ${entry.hintFile}`;
    button.addEventListener('click', () => setDataset(entry.key));
    ui.tabs.append(button);
  }
}

function renderCoreFilter() {
  const entry = dataset();
  const hasCores = Boolean(entry && entry.coreGroups.length);
  ui.coreFilter.hidden = !hasCores;
  if (!hasCores) return;

  const filter = cores(state.key);
  ui.coreGroups.replaceChildren();
  const groups = ['', ...entry.coreGroups];
  for (const group of groups) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'chip';
    button.textContent = group ? CORE_GROUP_LABELS[group] || group : 'All';
    button.dataset.active = String(group ? filter.groups.has(group) : filter.groups.size === 0);
    button.addEventListener('click', () => {
      if (!group) { filter.groups.clear(); filter.names.clear(); }
      else if (filter.groups.has(group)) filter.groups.delete(group);
      else filter.groups.add(group);
      refresh();
    });
    ui.coreGroups.append(button);
  }

  ui.coreNamesButton.textContent = filter.names.size
    ? `${filter.names.size} core names selected`
    : 'All core names';
  ui.coreNamesMenu.replaceChildren();
  const all = document.createElement('label');
  all.innerHTML = '<input type="checkbox"><span>All core names</span>';
  all.querySelector('input').checked = filter.names.size === 0;
  all.querySelector('input').addEventListener('change', () => { filter.names.clear(); refresh(); });
  ui.coreNamesMenu.append(all, document.createElement('hr'));
  for (const name of entry.coreNames) {
    const row = document.createElement('label');
    row.innerHTML = '<input type="checkbox"><span></span>';
    row.querySelector('span').textContent = name;
    const box = row.querySelector('input');
    box.checked = filter.names.has(name);
    box.addEventListener('change', () => {
      if (box.checked) filter.names.add(name); else filter.names.delete(name);
      refresh();
    });
    ui.coreNamesMenu.append(row);
  }
}

function renderProfileList() {
  const list = activeList();
  const chosen = selection(state.key);
  ui.profileList.dataset.list = list;
  ui.profileList.replaceChildren();
  if (!state.visible.length) {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = !dataset().available
      ? 'This dataset has no snapshot yet.'
      : list === 'socs'
        ? 'No processor matches the current search.'
        : 'No profile matches the current search and core filters.';
    ui.profileList.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  state.visible.forEach((item, index) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.setAttribute('role', 'option');
    const picked = chosen.has(item.id);
    row.setAttribute('aria-selected', String(picked));
    row.dataset.index = String(index);
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    if (picked) swatch.style.background = colorFor(item.id, list);
    const text = document.createElement('span');
    text.textContent = item.label;
    text.title = item.label;
    row.append(swatch, text);
    if (item.meta) {
      const meta = document.createElement('span');
      meta.className = 'meta';
      meta.textContent = item.meta;
      meta.title = `${item.meta} device profile${item.meta === '1' ? '' : 's'} averaged`;
      row.append(meta);
    }
    fragment.append(row);
  });
  ui.profileList.append(fragment);
}

function renderListTabs() {
  const entry = dataset();
  const show = hasSocList(entry);
  ui.listTabs.hidden = !show;
  if (show) {
    for (const button of ui.listTabs.querySelectorAll('button')) {
      button.setAttribute('aria-selected', String(button.dataset.list === activeList()));
    }
  }
  const socs = activeList() === 'socs';
  ui.searchLabel.textContent = socs ? 'Search processors' : 'Search profiles';
  ui.search.placeholder = socs ? 'Processor name' : 'Chip or core name';
}

function renderViews() {
  const entry = dataset();
  ui.view.replaceChildren();
  if (!entry) return;
  for (const view of entry.views) {
    const option = document.createElement('option');
    option.value = view;
    option.textContent = view;
    ui.view.append(option);
  }
  ui.view.value = currentView();
}

function syncMeasuredToggle() {
  // Only the device charts carry the overlay, so it belongs to that list.
  const show = activeList() === 'devices' && hasMeasured(dataset());
  ui.measuredWrap.hidden = !show;
  if (!show) return;
  ui.measuredToggle.disabled = SOC_VIEWS.has(currentView());
  ui.measuredToggle.checked = state.showMeasured;
  ui.measuredWrap.title = ui.measuredToggle.disabled
    ? 'The processor-average views plot no devices'
    : "Hollow diamonds for Geekerwan's measured usable capacity, joined to the pulled point";
}

function syncReferenceToggle() {
  ui.referenceToggle.disabled = !REFERENCE_VIEWS.has(currentView());
  ui.referenceToggle.checked = state.showReference && !ui.referenceToggle.disabled;
  ui.referenceToggle.parentElement.title = ui.referenceToggle.disabled
    ? 'This view has no equal-efficiency line; the ranking shows the best value'
    : 'Equal-efficiency lines, or the low-power / large-battery Pareto frontier on power-draw views';
}

function syncOverlayToggle() {
  // The overlay belongs to the processor list; the dedicated SoC views already
  // plot the averages themselves, so it has nothing to add there.
  const show = activeList() === 'socs';
  ui.overlayWrap.hidden = !show;
  if (!show) return;
  ui.overlayToggle.disabled = !SOC_TOGGLE_VIEWS.has(currentView());
  ui.overlayToggle.checked = state.showSocAverages && !ui.overlayToggle.disabled;
  ui.overlayWrap.title = ui.overlayToggle.disabled
    ? 'Available on the Energy efficiency and Average power draw views'
    : 'Draw a labelled average line per selected processor';
}

/* ---------- scene building ---------- */

function unavailableScene(entry) {
  return {
    main: {
      title: `No ${entry.title.toLowerCase()} data yet`,
      subtitle: `Expected ${entry.hintFile}`,
      message: {
        title: 'Generate the dataset, then choose Reload data',
        lines: [{ text: entry.hintCommand, mono: true, size: 11, color: Chart.THEME.cyan, box: true }],
      },
      polylines: [], dots: [],
    },
    rank: { title: 'Ranking', subtitle: 'waiting for data', items: [], emptyText: 'Nothing to rank yet' },
  };
}

function emptyScene() {
  return {
    main: {
      title: 'Choose profiles to compare',
      subtitle: '',
      message: { lines: [{ text: 'Select one or more profiles from the left panel' }] },
      polylines: [], dots: [],
    },
    rank: { title: 'Ranking', subtitle: '', items: [], emptyText: 'Nothing to rank yet' },
  };
}

function rankItems(rows, higherIsBetter, decimals) {
  return rows
    .filter((row) => Number.isFinite(row.value))
    .sort((left, right) => (higherIsBetter ? right.value - left.value : left.value - right.value))
    .map((row) => ({
      label: row.label,
      value: row.value,
      color: row.color,
      text: decimals ? fixed(row.value, decimals) : Chart.formatValue(row.value),
    }));
}

function curveScene(entry, profiles) {
  const view = currentView();
  const decimals = entry.scoreDecimals;
  const { xIndex, yIndex, xLabel, yLabel, rankIndex, rankLabel } = curveAxes(entry, view);
  const colors = colorMap('devices');

  const ordered = profiles.slice().sort((left, right) =>
    left.label.localeCompare(right.label, undefined, { sensitivity: 'base' }));
  const polylines = [];
  const dots = [];
  const legend = [];
  const points = [];
  const ranking = [];
  let total = 0;

  ordered.forEach((profile) => {
    const color = colors.get(profile.label);
    legend.push({ text: profile.legend, color, shape: 'line' });
    let best = -Infinity;
    profile.series.forEach((series, seriesIndex) => {
      const sorted = series.points.slice().sort((left, right) => left[xIndex] - right[xIndex]);
      const dense = sorted.length >= 4;
      const dotted = entry.sparseCurves && sorted.length > 1 && sorted.length < 4;
      const alpha = seriesIndex === 0 ? 0.94 : 0.5;
      total += sorted.length;
      if (dense || dotted) {
        polylines.push({
          pts: sorted.map((point) => [point[xIndex], point[yIndex]]),
          color,
          width: dotted ? 1.8 : 2.4,
          dash: dotted ? [2, 3] : null,
          alpha,
        });
      }
      for (const point of sorted) {
        best = Math.max(best, point[rankIndex]);
        dots.push({
          x: point[xIndex],
          y: point[yIndex],
          r: dense ? 2.3 : 4.6,
          fill: color,
          alpha: dense ? 0.4 : 0.95,
          stroke: dense ? null : Chart.THEME.text,
          lineWidth: 0.9,
        });
        points.push({
          x: point[xIndex],
          y: point[yIndex],
          color,
          details: [
            profile.label,
            `${entry.scoreLabel}: ${fixed(point[1], decimals)}`,
            `Board power: ${point[0].toFixed(2)} W`,
            `Efficiency: ${fixed(point[2], decimals)} score/W`,
          ].join('\n'),
        });
      }
    });
    ranking.push({ label: profile.legend, value: best, color });
  });

  hoverPoints = points;
  const sparseNote = entry.sparseCurves ? ' · dotted lines join sparse samples' : '';
  ui.chartNote.textContent = `${total.toLocaleString()} curve points shown${sparseNote}`;

  return {
    main: {
      title: `${entry.kicker.replace(/-/g, ' ').toUpperCase()} CURVE`,
      subtitle: `${ordered.length} profiles`,
      xLabel,
      yLabel,
      polylines,
      dots,
      legend: { items: legend, cols: legend.length > 5 ? 2 : 1 },
    },
    rank: {
      title: 'Ranking',
      subtitle: `↑ better`,
      xLabel: rankLabel,
      items: rankItems(ranking, true, decimals),
      offset: state.rankOffsets[state.key] || 0,
    },
  };
}

function batteryScene(entry, profiles) {
  const view = currentView();
  const measuredColumns = {
    capacityWh: 'capacityWh', avgPowerW: 'avgPowerW', minPerWh: 'minPerWh', hours: null,
  };
  const { xKey, yKey, xLabel, yLabel, title, rankKey, rankLabel, higher } = batteryAxes(view);
  const colors = colorMap('devices');

  const ordered = profiles.slice().sort((left, right) =>
    left.label.localeCompare(right.label, undefined, { sensitivity: 'base' }));
  const dots = [];
  const connectors = [];
  const labels = [];
  const points = [];
  const ranking = [];
  const showLabels = ordered.length <= 18;
  const showMeasured = state.showMeasured;
  let matched = 0;
  let rows = 0;

  ordered.forEach((profile) => {
    const color = colors.get(profile.label);
    const x = profile[xKey];
    const y = profile[yKey];
    rows += profile.count;
    ranking.push({ label: profile.label, value: profile[rankKey], color });
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    dots.push({ x, y, r: 4.9, fill: color, stroke: Chart.THEME.plot, lineWidth: 1.5 });
    points.push({
      x, y, color,
      details: [
        profile.label,
        profile.soc ? `SoC: ${profile.soc}` : null,
        `Runtime: ${profile.hours.toFixed(2)} h (${profile.minutes.toFixed(0)} min)`,
        `Capacity: ${profile.capacityWh.toFixed(2)} Wh`,
        `Average power: ${profile.avgPowerW.toFixed(2)} W`,
        `Efficiency: ${profile.minPerWh.toFixed(2)} min/Wh`,
      ].filter(Boolean).join('\n'),
    });
    if (showLabels) labels.push({ x, y, text: shorten(profile.label, 24), color: Chart.THEME.text });

    if (profile.measured) {
      matched += 1;
      if (!showMeasured) return;
      const measuredX = measuredColumns[xKey] ? profile.measured[measuredColumns[xKey]] : x;
      const measuredY = measuredColumns[yKey] ? profile.measured[measuredColumns[yKey]] : y;
      connectors.push({ x1: x, y1: y, x2: measuredX, y2: measuredY, color });
      dots.push({
        x: measuredX, y: measuredY, r: 5.3, shape: 'diamond',
        fill: Chart.THEME.plot, stroke: color, lineWidth: 2,
      });
      const shortfall = profile.measured;
      points.push({
        x: measuredX, y: measuredY, color,
        details: [
          profile.label,
          'Geekerwan measured usable capacity',
          `Measured: ${fixed(shortfall.measuredMah, 0)} mAh of ${fixed(shortfall.advertisedMah, 0)} mAh`,
          `Locked/lost: ${fixed(shortfall.shortfallMah, 0)} mAh (${shortfall.shortfallPct.toFixed(2)}%)`,
          `Adjusted capacity: ${shortfall.capacityWh.toFixed(2)} Wh`,
          `Adjusted average power: ${shortfall.avgPowerW.toFixed(2)} W`,
          `Adjusted efficiency: ${shortfall.minPerWh.toFixed(2)} min/Wh`,
        ].join('\n'),
      });
    }
  });

  const hlines = [];
  const references = [];
  const showAverages = state.showSocAverages && SOC_TOGGLE_VIEWS.has(view);
  if (showAverages) {
    const averageKeys = { avgPowerW: 'powerW', minutes: 'minPerWh', hours: 'runtimeHours' };
    const socColors = colorMap('socs');
    const socs = selectedSocs(entry)
      .sort((left, right) => left.soc.localeCompare(right.soc, undefined, { sensitivity: 'base' }));
    socs.forEach((average) => {
      const value = average[averageKeys[yKey]];
      if (!Number.isFinite(value)) return;
      if (yKey === 'minutes') {
        references.push({ slope: value, intercept: 0, color: socColors.get(average.soc),
          source: average.soc, text: `${average.soc} · ${fixed(value, 2)} min/Wh` });
        return;
      }
      hlines.push({
        y: value,
        color: socColors.get(average.soc),
        text: `${shorten(average.soc, 22)}  ${value.toFixed(2)}`,
      });
    });
  }

  const legend = [];
  if (matched && showMeasured) {
    legend.push({ text: 'SoCPK / advertised capacity', color: Chart.THEME.muted, shape: 'ring' });
    legend.push({ text: 'Geekerwan measured capacity', color: Chart.THEME.muted, shape: 'diamond' });
  }
  if (hlines.length) {
    legend.push({ text: 'Processor average (y-axis)', color: '#ffb15c', shape: 'dash' });
  }
  references.forEach((line) => legend.push({ text: line.text, color: line.color, shape: 'dash' }));

  hoverPoints = points;
  const measuredNote = !matched
    ? 'no Geekerwan measured matches'
    : showMeasured
      ? `${matched} Geekerwan measured overlays`
      : `${matched} Geekerwan measured overlays hidden`;
  ui.chartNote.textContent = `${rows.toLocaleString()} battery tests shown · ${measuredNote}` +
    (showAverages ? ` · ${hlines.length + references.length} processor averages` : '');

  return {
    main: {
      title,
      subtitle: `${ordered.length} device profiles · ` +
        (showLabels ? 'labels shown · hover a point for details' : 'hover a point to identify it'),
      xLabel,
      yLabel,
      polylines: [],
      dots,
      connectors,
      labels,
      hlines,
      references,
      legend: { items: legend, cols: 1 },
    },
    rank: {
      title: 'Ranking',
      subtitle: higher ? '↑ better' : '↓ better',
      xLabel: rankLabel,
      items: rankItems(ranking, higher, 0),
      offset: state.rankOffsets[state.key] || 0,
    },
  };
}

function socAverageScene(entry) {
  const view = currentView();
  const averages = selectedSocs(entry);
  if (!averages.length) {
    hoverPoints = [];
    const none = !(entry.socAverages || []).length;
    ui.chartNote.textContent = none
      ? 'No precomputed SoC averages in the loaded battery data'
      : 'No processors selected';
    return {
      main: {
        title: none ? 'No processor averages available' : 'Choose processors to compare',
        subtitle: '',
        message: {
          lines: [{
            text: none
              ? 'Collect battery data with --auto-soc'
              : 'Pick processors in the left panel',
          }],
        },
        polylines: [], dots: [],
      },
      rank: {
        title: 'Processor ranking',
        subtitle: none ? 'waiting for SoC data' : '',
        items: [],
        emptyText: 'Nothing to rank yet',
      },
    };
  }

  const powerView = view === 'SoC Average Power Draw';
  const { xKey, yKey, xLabel, yLabel } = socAxes(view);
  const colors = colorMap('socs');

  const dots = [];
  const labels = [];
  const points = [];
  const ranking = [];
  const showLabels = averages.length <= 24;
  averages.forEach((average) => {
    const x = average[xKey];
    const y = average[yKey];
    const color = colors.get(average.soc);
    ranking.push({ label: average.soc, value: y, color });
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    dots.push({ x, y, r: 5.4, fill: color, stroke: Chart.THEME.plot, lineWidth: 1.4 });
    const devices = `${average.deviceCount} device${average.deviceCount === 1 ? '' : 's'}`;
    points.push({
      x, y, color,
      details: [
        `${average.soc} · ${devices}`,
        `Average capacity: ${average.capacityWh.toFixed(2)} Wh`,
        `Average power: ${average.powerW.toFixed(2)} W`,
        `Average efficiency: ${average.minPerWh.toFixed(2)} min/Wh`,
        `Average runtime: ${average.runtimeHours.toFixed(2)} h`,
      ].join('\n'),
    });
    if (showLabels) labels.push({ x, y, text: shorten(average.soc, 25), color: Chart.THEME.text });
  });

  hoverPoints = points;
  const devices = averages.reduce((sum, average) => sum + average.deviceCount, 0);
  const total = (entry.socAverages || []).length;
  ui.chartNote.textContent =
    `${averages.length} of ${total} processor averages · ${devices} device profiles`;

  return {
    main: {
      title: powerView ? 'Processor-average power draw' : 'Processor-average efficiency',
      subtitle: `${averages.length} processors · precomputed across the loaded battery dataset`,
      xLabel,
      yLabel,
      polylines: [],
      dots,
      labels,
      legend: null,
    },
    rank: {
      title: 'Processor ranking',
      subtitle: powerView ? '↓ better' : '↑ better',
      xLabel: powerView ? 'Average power (W)' : 'Average minutes per Wh',
      items: rankItems(ranking, !powerView, 0),
      offset: state.rankOffsets[state.key] || 0,
    },
  };
}

/* Equal-efficiency guides: lines on which the chart's quality ratio is
 * constant. Views whose axes already are that ratio, or whose best value the
 * ranking panel states, get none. References never participate in data
 * bounds, rankings or hover targets. */
function addEfficiencyReference(scene, entry, profiles) {
  ui.referenceNote.hidden = true;
  ui.referenceNote.textContent = '';
  const view = currentView();
  if (!state.showReference || scene.main.message || !REFERENCE_VIEWS.has(view)) return;
  if (entry.kind === 'curve') {
    // Sorted power order makes equal efficiency ties prefer lower power.
    const best = profiles.flatMap((profile) => profile.series.flatMap((series) =>
      series.points.filter((p) => p.every(Number.isFinite) && p[0] > 0 && p[1] > 0)
        .map((p) => ({ power: p[0], score: p[1], source: profile.label }))))
      .sort((a, b) => a.power - b.power || b.score - a.score || a.source.localeCompare(b.source))
      .reduce((top, p) => (!top || p.score / p.power > top.score / top.power ? p : top), null);
    if (!best) return;
    const perWatt = best.score / best.power;
    const digits = perWatt >= 100 ? 0 : perWatt >= 10 ? 1 : 2;
    const levels = GPU_DATASETS.has(entry.key) ? GPU_EFFICIENCY_LEVELS : EFFICIENCY_LEVELS;
    scene.main.references = levels.map((level) => ({
      slope: perWatt * level, intercept: 0, color: Chart.THEME.text,
      width: level === 1 ? 2 : 1.2, dash: level === 1 ? [8, 5] : [3, 4],
      alpha: level === 1 ? 0.9 : 0.45,
      label: level === 1 ? `${fixed(perWatt, digits)}/W` : `${level * 100}%`,
    }));
    const text = `Equal efficiency · best ${fixed(perWatt, digits)} score/W`;
    scene.main.legend ||= { items: [], cols: 1 };
    scene.main.legend.items.push({ text, color: Chart.THEME.text, shape: 'dash' });
    ui.referenceNote.textContent = `Rays through the origin hold score per watt constant. ` +
      `Bold ray: best selected point, ${best.source} (${fixed(best.power, 2)} W, ` +
      `${fixed(best.score, entry.scoreDecimals)} score); fainter rays: ${levels.slice(1).map((level) => `${level * 100}%`).join(', ')} of that efficiency. ` +
      `A curve reaching a higher ray is more efficient there; a curve bending across lower rays shows diminishing returns.`;
    ui.referenceNote.hidden = false;
    return;
  }
  if (view === 'Average power draw' || view === 'SoC Average Power Draw') {
    // Lower-right Pareto frontier: no other selected point has at least as
    // much capacity at no more power. Descending capacity keeps only strict
    // power improvements; ties prefer the larger battery, then the name.
    const soc = view === 'SoC Average Power Draw';
    let bestPower = Infinity;
    const frontier = (soc ? selectedSocs(entry) : profiles)
      .map((p) => ({ source: soc ? p.soc : p.label, capacity: p.capacityWh, power: soc ? p.powerW : p.avgPowerW }))
      .filter((p) => Number.isFinite(p.capacity) && p.capacity > 0 && Number.isFinite(p.power) && p.power > 0)
      .sort((a, b) => b.capacity - a.capacity || a.power - b.power || a.source.localeCompare(b.source))
      .filter((p) => (p.power < bestPower ? (bestPower = p.power, true) : false))
      .reverse();
    if (!frontier.length || !scene.main.dots.length) return;
    const text = 'Lowest draw for its battery size';
    scene.main.references = [{ pts: frontier.map((p) => [p.capacity, p.power]),
      color: Chart.THEME.text, text }, ...(scene.main.references || [])];
    // Ring the frontier points; rings are decoration, not hover targets.
    scene.main.dots.push(...frontier.map((p) => ({ x: p.capacity, y: p.power, r: 8.5,
      stroke: Chart.THEME.text, lineWidth: 1.4, alpha: 0.85 })));
    scene.main.legend ||= { items: [], cols: 1 };
    scene.main.legend.items.push({ text, color: Chart.THEME.text, shape: 'dash' });
    const names = frontier.map((p) => `${p.source} (${fixed(p.capacity, 1)} Wh, ${fixed(p.power, 2)} W)`);
    const kind = soc ? 'processor averages' : 'phones';
    ui.referenceNote.textContent = (frontier.length === 1
      ? `${names[0]} has both the largest battery and the lowest draw of the selected ${kind}, so it beats every other point; there is no trade-off line to draw.`
      : `Pareto frontier of ${frontier.length} selected ${kind}: none of the other selected points has a larger battery at lower power. ` +
        `${names.join(' → ')}. Points above the line are beaten on both. Connects measured points only; no extrapolation.`) +
      (soc ? '' : ' Uses advertised capacity.');
    ui.referenceNote.hidden = false;
    return;
  }
  // Use the advertised-capacity points consistently; measured overlays are
  // a separate estimate and must not silently change the reference basis.
  const best = profiles.filter((p) => Number.isFinite(p.capacityWh) && p.capacityWh > 0)
    .map((p) => ({ source: p.label, value: p.avgPowerW }))
    .filter((p) => Number.isFinite(p.value) && p.value > 0)
    .sort((a, b) => a.value - b.value || a.source.localeCompare(b.source))[0];
  if (!best || !scene.main.dots.length) return;
  const runtime = view === 'Runtime vs capacity';
  const slope = (runtime ? 1 : 60) / best.value;
  const formula = runtime ? `runtime (h) = capacity (Wh) ÷ ${fixed(best.value, 2)} W`
    : `runtime (min) = capacity (Wh) × 60 ÷ ${fixed(best.value, 2)} W`;
  const text = `Best efficiency: ${fixed(best.value, 2)} W`;
  const reference = { slope, intercept: 0, color: Chart.THEME.text, source: best.source, text };
  scene.main.references = [reference, ...(scene.main.references || [])];
  scene.main.legend ||= { items: [], cols: 1 };
  scene.main.legend.items.push({ text: `${text} · ${shorten(best.source, 24)}`,
    color: reference.color, shape: 'dash' });
  ui.referenceNote.textContent = `${text} · ${best.source} (best selected phone · advertised capacity). ${formula}. Constant-efficiency extrapolation; not a measured prediction.`;
  ui.referenceNote.hidden = false;
}

/* ---------- stats ---------- */

function updateStats(entry, chosen) {
  const view = currentView();
  if (!entry.available) {
    ui.statProfiles.textContent = '0 / 0';
    ui.statPoints.textContent = '0';
    ui.statLeader.textContent = '—';
    return;
  }
  if (entry.kind === 'battery' && SOC_VIEWS.has(view)) {
    const shown = selectedSocs(entry);
    ui.statProfiles.textContent = `${shown.length} / ${entry.socAverages.length}`;
    ui.statPoints.textContent = shown.length.toLocaleString();
    const best = shown.reduce((leader, average) => {
      if (!leader) return average;
      if (view === 'SoC Average Power Draw') return average.powerW < leader.powerW ? average : leader;
      return average.minPerWh > leader.minPerWh ? average : leader;
    }, null);
    ui.statLeader.textContent = best ? best.soc : '—';
    return;
  }

  ui.statProfiles.textContent = `${chosen.length} / ${entry.profiles.length}`;
  if (entry.kind === 'curve') {
    const total = chosen.reduce((sum, profile) => sum + profile.count, 0);
    ui.statPoints.textContent = total.toLocaleString();
  } else {
    ui.statPoints.textContent = chosen.reduce((sum, profile) => sum + profile.count, 0).toLocaleString();
  }

  if (!chosen.length) { ui.statLeader.textContent = '—'; return; }
  let leader = chosen[0];
  if (entry.kind === 'curve') {
    const index = view === 'Performance curve' ? 1 : 2;
    const peak = (profile) =>
      Math.max(...profile.series.flatMap((item) => item.points.map((point) => point[index])));
    leader = chosen.reduce((best, profile) => (peak(profile) > peak(best) ? profile : best), chosen[0]);
  } else {
    const key = { 'Runtime vs capacity': 'hours', 'Energy efficiency': 'minPerWh', 'Average power draw': 'avgPowerW' }[view];
    const lowerWins = key === 'avgPowerW';
    leader = chosen.reduce((best, profile) => {
      if (!Number.isFinite(profile[key])) return best;
      if (!Number.isFinite(best[key])) return profile;
      return (lowerWins ? profile[key] < best[key] : profile[key] > best[key]) ? profile : best;
    }, chosen[0]);
  }
  ui.statLeader.textContent = leader.leader || leader.label;
  ui.statLeader.title = ui.statLeader.textContent;
}

/* ---------- the refresh cycle ---------- */

function refresh() {
  const entry = dataset();
  if (!entry) return;
  state.visible = visibleItems();
  applySelectionMode();
  renderListTabs();
  renderCoreFilter();
  renderProfileList();
  syncMeasuredToggle();
  syncOverlayToggle();
  syncReferenceToggle();

  const chosen = selectedProfiles();
  const view = currentView();
  let scene;
  if (!entry.available) {
    hoverPoints = [];
    scene = unavailableScene(entry);
    ui.chartNote.textContent = `Waiting for ${entry.hintFile}`;
  } else if (entry.kind === 'battery' && SOC_VIEWS.has(view)) {
    scene = socAverageScene(entry);
  } else if (!chosen.length) {
    hoverPoints = [];
    scene = emptyScene();
    ui.chartNote.textContent = 'No profiles selected';
  } else if (entry.kind === 'curve') {
    scene = curveScene(entry, chosen);
  } else {
    scene = batteryScene(entry, chosen);
  }

  addEfficiencyReference(scene, entry, chosen);
  const better = !scene.main.message && BETTER[view];
  scene.main.better = better ? better[0] : null;
  ui.chartGuide.textContent = better ? `▲ Better toward the ${better[0]}: ${better[1]}` : '';
  ui.chartGuide.hidden = !better;
  state.lastRank = scene.rank;
  surface.setScene(scene);
  updateStats(entry, chosen);
  Analytics.render(entry, chosen, selectedSocs(entry), state.payload.generatedAt);

  const listed = activeList() === 'socs' ? entry.socAverages.length : entry.profiles.length;
  const picked = selection(state.key).size;
  ui.selectionNote.textContent = activeList() === 'socs'
    ? `${picked} processors · ${state.visible.length} shown · ${listed} available`
    : `${picked} selected · ${state.visible.length} shown · ${listed} available`;
  const total = state.lastRank.items ? state.lastRank.items.length : 0;
  ui.rankHint.textContent = total ? `${total} ranked · scroll over the ranking` : '';
}

function datasetFromHash() {
  const key = decodeURIComponent((location.hash || '').replace(/^#/, ''));
  return state.datasets.has(key) ? key : '';
}

function setDataset(key) {
  const entry = state.datasets.get(key);
  if (!entry) return;
  state.key = key;
  const hash = '#' + encodeURIComponent(key);
  if (location.hash !== hash) history.replaceState(null, '', hash);
  if (!state.modes.devices[key]) state.modes.devices[key] = 'top5';
  // Processor averages start unnarrowed, so the dedicated SoC views still
  // survey the whole dataset until the list is used.
  if (!state.modes.socs[key]) state.modes.socs[key] = 'all';
  ui.kicker.textContent = entry.kicker.toUpperCase();
  ui.title.textContent = entry.title;
  renderTabs();
  renderViews();
  ui.search.value = search(key);
  refresh();
}

/* ---------- data loading ---------- */

async function load(url) {
  ui.reload.disabled = true;
  try {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const payload = await response.json();
    state.payload = payload;
    state.palette = payload.palette;
    state.datasets = new Map(payload.datasets.map((entry) => [entry.key, entry]));
    state.colorMaps.clear();

    // Drop selections for profiles and processors that a reload removed.
    for (const [list, store] of Object.entries(state.selection)) {
      for (const [key, chosen] of Object.entries(store)) {
        const entry = state.datasets.get(key);
        const available = new Set(!entry ? [] : list === 'socs'
          ? (entry.socAverages || []).map((average) => average.soc)
          : entry.profiles.map((profile) => profile.label));
        for (const id of [...chosen]) if (!available.has(id)) chosen.delete(id);
      }
    }

    const sources = payload.datasets.reduce((sum, entry) => sum + entry.sourceCount, 0);
    ui.sourceNote.textContent = `${sources} source file${sources === 1 ? '' : 's'}` +
      (payload.warnings.length ? ` · ${payload.warnings.length} warning(s)` : '');
    ui.sourceNote.title = payload.warnings.join('\n');
    setDataset(state.datasets.has(state.key) ? state.key
      : (datasetFromHash() || payload.initialDataset));
  } catch (error) {
    toast(`Could not load data: ${error.message}`, 'error');
  } finally {
    ui.reload.disabled = false;
  }
}

/* ---------- export ---------- */

function exportName(extension) {
  const entry = dataset();
  const view = currentView();
  const deviceMode = mode(state.key, 'devices');
  const parts = ['socpk', entry.kicker, view, deviceMode];
  parts.push(state.showReference ? 'best-efficiency' : 'no-reference');
  if (entry.coreGroups.length) {
    const filter = cores(state.key);
    parts.push(filter.groups.size ? 'groups-' + [...filter.groups].sort().join('-') : 'all-groups');
    parts.push(filter.names.size ? 'cores-' + [...filter.names].sort().join('-') : 'all-cores');
  }
  const needle = search(state.key, 'devices').trim();
  if (needle) parts.push('search-' + needle);
  if (deviceMode === 'manual') {
    const labels = [...selection(state.key, 'devices')].sort();
    parts.push(`${labels.length}-profiles-${fnv1a(labels.join('\n'))}`);
  }
  if (hasMeasured(entry) && !SOC_VIEWS.has(view)) {
    parts.push(state.showMeasured ? 'measured' : 'no-measured');
  }
  // Processor averages change the chart, so they belong in its filename.
  if (hasSocList(entry)) {
    const socs = [...selection(state.key, 'socs')].sort();
    const overlaid = state.showSocAverages && SOC_TOGGLE_VIEWS.has(view);
    if (overlaid || SOC_VIEWS.has(view)) {
      parts.push(socs.length === entry.socAverages.length
        ? 'all-socs'
        : `${socs.length}-socs-${fnv1a(socs.join('\n'))}`);
    }
    if (overlaid) parts.push('overlay');
  }
  let stem = parts
    .map((part) => part.toLowerCase().replace(/[^\w]+/g, '-').replace(/^-+|-+$/g, ''))
    .join('-');
  if (new TextEncoder().encode(stem).length > 180) {
    stem = stem.slice(0, 160).replace(/-+$/, '') + '-' + fnv1a(stem);
  }
  return `${stem}.${extension}`;
}

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

async function exportChart(format) {
  try {
    if (format === 'svg') {
      download(new Blob([surface.toSvg()], { type: 'image/svg+xml' }), exportName('svg'));
    } else {
      const blob = await surface.toPngBlob(2);
      download(blob, exportName('png'));
    }
    toast(`Exported ${exportName(format)}`);
  } catch (error) {
    toast(`Export failed: ${error.message}`, 'error');
  }
}

/* ---------- internal renderer invocation ----------
 *
 * This is deliberately browser-side: the chart scene is built here from the
 * loaded payload and then passed to the same Chart.Surface used by the page.
 * It gives an agent a deterministic way to request the chart image without
 * creating a second renderer or silently substituting a server-side plot.
 */

function debugList(value, name) {
  if (value === undefined) return null;
  if (!Array.isArray(value) || value.some((item) => typeof item !== 'string')) {
    throw new Error(`[socpkDebug] ${name} must be an array of strings`);
  }
  return [...new Set(value)];
}

function debugMode(value, name) {
  if (value === undefined) return null;
  if (!['manual', 'top5', 'all', 'clear'].includes(value)) {
    throw new Error(`[socpkDebug] ${name} must be manual, top5, all, or clear`);
  }
  return value;
}

function debugSelection(key, list, labels, available, name) {
  const unknown = labels.filter((label) => !available.has(label));
  if (unknown.length) {
    throw new Error(`[socpkDebug] unknown ${name}: ${unknown.join(', ')}`);
  }
  const chosen = selection(key, list);
  chosen.clear();
  labels.forEach((label) => chosen.add(label));
  setMode(key, 'manual', list);
}

function nextFrame() {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

async function waitForDebugRender() {
  // refresh() queues Surface.render(); the second frame also covers a pending
  // ResizeObserver pass after a browser agent changes its viewport.
  await nextFrame();
  await nextFrame();
  if (!surface.scene || surface.size.width < 40 || surface.size.height < 40) {
    throw new Error(
      '[socpkDebug] chart stage has no usable rendered size; open a visible dashboard viewport first',
    );
  }
}

async function blobDataUrl(blob) {
  if (!blob) throw new Error('[socpkDebug] renderer returned no image data');
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  }
  return `data:${blob.type};base64,${btoa(binary)}`;
}

async function invokeDebugChart(spec = {}) {
  if (!spec || typeof spec !== 'object' || Array.isArray(spec)) {
    throw new Error('[socpkDebug] invocation must be an object');
  }
  if (!state.payload) throw new Error('[socpkDebug] dashboard data is not loaded yet');

  const key = spec.dataset;
  if (typeof key !== 'string' || !state.datasets.has(key)) {
    throw new Error(`[socpkDebug] unknown dataset: ${String(key)}`);
  }
  const entry = state.datasets.get(key);
  const view = spec.view;
  if (typeof view !== 'string' || !entry.views.includes(view)) {
    throw new Error(`[socpkDebug] unknown view for ${key}: ${String(view)}`);
  }

  const profiles = debugList(spec.profiles, 'profiles');
  const processors = debugList(spec.processors, 'processors');
  const deviceMode = debugMode(spec.deviceMode, 'deviceMode');
  const processorMode = debugMode(spec.processorMode, 'processorMode');
  const list = spec.list || (SOC_VIEWS.has(view) && processors !== null ? 'socs' : 'devices');
  if (!['devices', 'socs'].includes(list)) {
    throw new Error('[socpkDebug] list must be devices or socs');
  }
  if (list === 'socs' && !hasSocList(entry)) {
    throw new Error(`[socpkDebug] ${key} has no processor-average list`);
  }
  const coreGroups = debugList(spec.coreGroups, 'coreGroups');
  const coreNames = debugList(spec.coreNames, 'coreNames');
  const validGroups = new Set(entry.coreGroups || []);
  const validCoreNames = new Set(entry.coreNames || []);
  if (coreGroups && coreGroups.some((group) => !validGroups.has(group))) {
    throw new Error(`[socpkDebug] unknown core group in ${key}`);
  }
  if (coreNames && coreNames.some((name) => !validCoreNames.has(name))) {
    throw new Error(`[socpkDebug] unknown core name in ${key}`);
  }

  if (spec.format !== undefined && !['png', 'svg'].includes(spec.format)) {
    throw new Error('[socpkDebug] format must be png or svg');
  }
  const format = spec.format || 'png';
  const scale = spec.scale === undefined ? 2 : Number(spec.scale);
  if (!Number.isFinite(scale) || scale <= 0 || scale > 4) {
    throw new Error('[socpkDebug] scale must be a number greater than 0 and no greater than 4');
  }
  if (spec.rankOffset !== undefined &&
      (!Number.isInteger(spec.rankOffset) || spec.rankOffset < 0)) {
    throw new Error('[socpkDebug] rankOffset must be a non-negative integer');
  }

  state.views[key] = view;
  state.lists[key] = list;
  if (coreGroups !== null || coreNames !== null) {
    const filter = cores(key);
    filter.groups = new Set(coreGroups || []);
    filter.names = new Set(coreNames || []);
  }
  if (spec.deviceSearch !== undefined) state.searches.devices[key] = String(spec.deviceSearch);
  if (spec.processorSearch !== undefined) state.searches.socs[key] = String(spec.processorSearch);
  if (spec.showMeasured !== undefined) state.showMeasured = Boolean(spec.showMeasured);
  if (spec.showSocAverages !== undefined) state.showSocAverages = Boolean(spec.showSocAverages);
  if (spec.rankOffset !== undefined) state.rankOffsets[key] = spec.rankOffset;

  if (profiles !== null) {
    if (!entry.available) throw new Error(`[socpkDebug] ${key} has no loaded data`);
    debugSelection(key, 'devices', profiles,
      new Set(entry.profiles.map((profile) => profile.label)), 'profiles');
  } else if (deviceMode) {
    setMode(key, deviceMode, 'devices');
  }
  if (processors !== null) {
    if (!entry.available) throw new Error(`[socpkDebug] ${key} has no loaded data`);
    debugSelection(key, 'socs', processors,
      new Set((entry.socAverages || []).map((average) => average.soc)), 'processors');
  } else if (processorMode) {
    setMode(key, processorMode, 'socs');
  }

  setDataset(key);
  await waitForDebugRender();

  let dataUrl;
  let svg;
  if (format === 'svg') {
    svg = surface.toSvg();
    dataUrl = await blobDataUrl(new Blob([svg], { type: 'image/svg+xml' }));
  } else {
    dataUrl = await blobDataUrl(await surface.toPngBlob(scale));
  }
  return {
    dataset: key,
    view,
    format,
    filename: exportName(format),
    width: surface.size.width,
    height: surface.size.height,
    dataUrl,
    ...(svg === undefined ? {} : { svg }),
  };
}

// Deliberately internal and browser-local. Agents can await `ready`, then
// call `window.__socpkDebug.invoke({...})` in the dashboard page.
window.__socpkDebug = {
  ready: null,
  invoke: invokeDebugChart,
  capture: invokeDebugChart,
};

/* ---------- events ---------- */

ui.view.addEventListener('change', () => {
  state.views[state.key] = ui.view.value;
  state.rankOffsets[state.key] = 0;
  refresh();
});

ui.search.addEventListener('input', () => {
  state.searches[activeList()][state.key] = ui.search.value;
  refresh();
});

ui.listTabs.addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  state.lists[state.key] = button.dataset.list;
  ui.search.value = search(state.key);
  refresh();
});

ui.quick.addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  setMode(state.key, button.dataset.mode);
  state.rankOffsets[state.key] = 0;
  refresh();
  // An automatic pick is usually off-screen in a long list; show it.
  const picked = ui.profileList.querySelector('.row[aria-selected="true"]');
  if (picked) picked.scrollIntoView({ block: 'nearest' });
});

let lastClickedIndex = null;
ui.profileList.addEventListener('click', (event) => {
  const row = event.target.closest('.row');
  if (!row) return;
  const index = Number(row.dataset.index);
  const chosen = selection(state.key);
  setMode(state.key, 'manual');
  if (event.shiftKey && lastClickedIndex != null) {
    const [from, to] = [lastClickedIndex, index].sort((left, right) => left - right);
    const select = !chosen.has(state.visible[index].id);
    for (let step = from; step <= to; step += 1) {
      const id = state.visible[step].id;
      if (select) chosen.add(id); else chosen.delete(id);
    }
  } else {
    const id = state.visible[index].id;
    if (chosen.has(id)) chosen.delete(id); else chosen.add(id);
  }
  lastClickedIndex = index;
  refresh();
});

ui.referenceToggle.addEventListener('change', () => {
  state.showReference = ui.referenceToggle.checked;
  refresh();
});

ui.measuredToggle.addEventListener('change', () => {
  state.showMeasured = ui.measuredToggle.checked;
  refresh();
});

ui.overlayToggle.addEventListener('change', () => {
  state.showSocAverages = ui.overlayToggle.checked;
  refresh();
});

ui.reload.addEventListener('click', () => load('/api/reload'));

function closeMenus(except) {
  for (const [button, menu] of [[ui.exportButton, ui.exportMenu], [ui.coreNamesButton, ui.coreNamesMenu]]) {
    if (menu === except) continue;
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  }
}

function toggleMenu(button, menu) {
  const open = menu.hidden;
  closeMenus(open ? menu : null);
  menu.hidden = !open;
  button.setAttribute('aria-expanded', String(open));
}

ui.exportButton.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleMenu(ui.exportButton, ui.exportMenu);
});
ui.exportMenu.addEventListener('click', (event) => {
  const button = event.target.closest('button');
  if (!button) return;
  closeMenus();
  exportChart(button.dataset.format);
});
ui.coreNamesButton.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleMenu(ui.coreNamesButton, ui.coreNamesMenu);
});
ui.coreNamesMenu.addEventListener('click', (event) => event.stopPropagation());
document.addEventListener('click', () => closeMenus());
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') closeMenus(); });

const stage = element('chart-stage');
stage.addEventListener('mousemove', (event) => {
  const box = stage.getBoundingClientRect();
  const x = event.clientX - box.left;
  const y = event.clientY - box.top;
  surface.drawHover(surface.nearest(hoverPoints, x, y, 24));
});
stage.addEventListener('mouseleave', () => surface.drawHover(null));

stage.addEventListener('wheel', (event) => {
  const box = stage.getBoundingClientRect();
  if (!surface.isOverRank(event.clientX - box.left)) return;
  const rank = state.lastRank;
  const frame = surface.frame && surface.frame.rank;
  if (!rank || !frame || rank.items.length <= frame.pageSize) return;
  event.preventDefault();
  const step = Math.sign(event.deltaY) * Math.max(1, Math.round(Math.abs(event.deltaY) / 40));
  const maximum = rank.items.length - frame.pageSize;
  const next = Math.min(Math.max(0, (state.rankOffsets[state.key] || 0) + step), maximum);
  if (next === (state.rankOffsets[state.key] || 0)) return;
  state.rankOffsets[state.key] = next;
  rank.offset = next;
  surface.setScene(surface.scene);
}, { passive: false });

stage.addEventListener('click', (event) => {
  const box = stage.getBoundingClientRect();
  const hit = surface.nearest(hoverPoints, event.clientX - box.left, event.clientY - box.top, 24);
  if (hit) ui.chartNote.textContent = `Selected chart profile: ${hit.point.details.split('\n')[0]}`;
});

window.addEventListener('keydown', (event) => {
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'SELECT') return;
  const keys = state.payload ? state.payload.datasets.map((entry) => entry.key) : [];
  const index = keys.indexOf(state.key);
  if (event.key === '[' && index > 0) setDataset(keys[index - 1]);
  if (event.key === ']' && index >= 0 && index < keys.length - 1) setDataset(keys[index + 1]);
  if (event.key === '/') { event.preventDefault(); ui.search.focus(); }
});

window.addEventListener('hashchange', () => {
  const key = datasetFromHash();
  if (key && key !== state.key) setDataset(key);
});

window.__socpkDebug.ready = load('/api/data');
