/* Canvas renderer for the comparison and ranking panels.
 *
 * The same draw pass runs against a 2D canvas on screen and against a small
 * SVG shim for vector export, so an exported chart is the chart on screen.
 * Hover lives on a second canvas layered on top, which keeps pointer moves
 * off the main draw path. */
'use strict';

const Chart = (() => {

const FONTS = '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", ' +
  '"Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

const THEME = {
  panel: '#111a2e',
  plot: '#0d1526',
  app: '#070b16',
  text: '#f3f6ff',
  muted: '#8e9bb6',
  faint: '#5b6782',
  grid: '#24304e',
  accent: '#7c6cff',
  cyan: '#38d6d0',
};

const PAD = 8;
const HEADER_H = 30;
const MAIN_FRACTION = 0.683;
const PANEL_GAP = 24;

function font(size, weight) {
  return `${weight || 400} ${size}px ${FONTS}`;
}

/* ---------- scales ---------- */

function niceStep(rough) {
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const normalized = rough / magnitude;
  const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 2.5 ? 2.5
    : normalized <= 5 ? 5 : 10;
  return factor * magnitude;
}

function makeScale(lo, hi, pixelLo, pixelHi, tickCount) {
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi === lo) {
    const spread = Math.abs(hi) || 1;
    lo -= spread * 0.5;
    hi += spread * 0.5;
  }
  const margin = (hi - lo) * 0.05;
  lo -= margin;
  hi += margin;
  const step = niceStep((hi - lo) / Math.max(1, (tickCount || 7) - 1));
  const ticks = [];
  for (let value = Math.ceil(lo / step) * step; value <= hi + step * 1e-6; value += step) {
    ticks.push(Math.round(value / step) * step);
  }
  const span = hi - lo || 1;
  const scale = (value) => pixelLo + ((value - lo) / span) * (pixelHi - pixelLo);
  scale.invert = (pixel) => lo + ((pixel - pixelLo) / (pixelHi - pixelLo || 1)) * span;
  scale.lo = lo;
  scale.hi = hi;
  scale.ticks = ticks;
  scale.step = step;
  return scale;
}

function formatTick(value, step) {
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e6) return (value / 1e6).toFixed(magnitude >= 1e7 ? 0 : 1) + 'M';
  if (magnitude >= 1e4) return (value / 1e3).toFixed(0) + 'k';
  const decimals = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step)));
  return value.toFixed(decimals);
}

function formatValue(value) {
  const magnitude = Math.abs(value);
  if (magnitude >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (magnitude >= 100) return value.toFixed(0);
  return value.toFixed(2);
}

/* ---------- context helpers ---------- */

function rotatedText(ctx, text, x, y, degrees) {
  if (ctx.isSvg) { ctx.rotatedText(text, x, y, degrees); return; }
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate((degrees * Math.PI) / 180);
  ctx.fillText(text, 0, 0);
  ctx.restore();
}

function withClip(ctx, x, y, width, height, draw) {
  if (ctx.isSvg) { ctx.pushClip(x, y, width, height); draw(); ctx.popClip(); return; }
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, width, height);
  ctx.clip();
  draw();
  ctx.restore();
}

function roundRectPath(ctx, x, y, width, height, radius) {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.arc(x + width - r, y + r, r, -Math.PI / 2, 0);
  ctx.lineTo(x + width, y + height - r);
  ctx.arc(x + width - r, y + height - r, r, 0, Math.PI / 2);
  ctx.lineTo(x + r, y + height);
  ctx.arc(x + r, y + height - r, r, Math.PI / 2, Math.PI);
  ctx.lineTo(x, y + r);
  ctx.arc(x + r, y + r, r, Math.PI, Math.PI * 1.5);
  ctx.closePath();
}

function ellipsize(ctx, text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let low = 0;
  let high = text.length;
  while (low < high) {
    const middle = Math.ceil((low + high) / 2);
    if (ctx.measureText(text.slice(0, middle) + '…').width <= maxWidth) low = middle;
    else high = middle - 1;
  }
  return low <= 0 ? '' : text.slice(0, low).trimEnd() + '…';
}

function diamondPath(ctx, x, y, radius) {
  ctx.beginPath();
  ctx.moveTo(x, y - radius);
  ctx.lineTo(x + radius, y);
  ctx.lineTo(x, y + radius);
  ctx.lineTo(x - radius, y);
  ctx.closePath();
}

/* ---------- panel chrome ---------- */

function drawHeader(ctx, panel, title, subtitle) {
  const baseline = panel.y + 15;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = THEME.text;
  ctx.font = font(12.5, 700);
  const titleText = ellipsize(ctx, title || '', panel.width);
  ctx.fillText(titleText, panel.x, baseline);
  if (!subtitle) return;
  const used = ctx.measureText(titleText).width + 12;
  ctx.font = font(10.5, 400);
  ctx.fillStyle = THEME.faint;
  ctx.fillText(ellipsize(ctx, subtitle, Math.max(0, panel.width - used)), panel.x + used, baseline);
}

function measureGutter(ctx, scale, hasAxisLabel) {
  ctx.font = font(10);
  let widest = 0;
  for (const tick of scale.ticks) {
    widest = Math.max(widest, ctx.measureText(formatTick(tick, scale.step)).width);
  }
  return Math.ceil(widest) + 10 + (hasAxisLabel ? 16 : 0);
}

function drawGrid(ctx, plot, xScale, yScale, options) {
  ctx.fillStyle = THEME.plot;
  ctx.fillRect(plot.x, plot.y, plot.width, plot.height);

  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 1;
  ctx.setLineDash([]);
  ctx.beginPath();
  if (xScale && options.vertical !== false) {
    for (const tick of xScale.ticks) {
      const x = Math.round(xScale(tick)) + 0.5;
      if (x < plot.x - 1 || x > plot.x + plot.width + 1) continue;
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.height);
    }
  }
  if (yScale && options.horizontal !== false) {
    for (const tick of yScale.ticks) {
      const y = Math.round(yScale(tick)) + 0.5;
      if (y < plot.y - 1 || y > plot.y + plot.height + 1) continue;
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.width, y);
    }
  }
  ctx.stroke();
}

function drawAxes(ctx, plot, xScale, yScale, xLabel, yLabel) {
  ctx.fillStyle = THEME.muted;
  ctx.font = font(10);
  if (xScale) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const tick of xScale.ticks) {
      const x = xScale(tick);
      if (x < plot.x - 1 || x > plot.x + plot.width + 1) continue;
      ctx.fillText(formatTick(tick, xScale.step), x, plot.y + plot.height + 7);
    }
  }
  if (yScale) {
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (const tick of yScale.ticks) {
      const y = yScale(tick);
      if (y < plot.y - 1 || y > plot.y + plot.height + 1) continue;
      ctx.fillText(formatTick(tick, yScale.step), plot.x - 7, y);
    }
  }
  ctx.fillStyle = THEME.faint;
  ctx.font = font(10.5, 600);
  if (xLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(ellipsize(ctx, xLabel, plot.width),
      plot.x + plot.width / 2, plot.y + plot.height + 34);
  }
  if (yLabel) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    rotatedText(ctx, ellipsize(ctx, yLabel, plot.height), plot.x - 42,
      plot.y + plot.height / 2, -90);
  }
}

function drawMessage(ctx, plot, message) {
  let y = plot.y + plot.height / 2 - (message.lines.length - 1) * 16;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  if (message.title) {
    ctx.font = font(13, 700);
    ctx.fillStyle = THEME.text;
    ctx.fillText(ellipsize(ctx, message.title, plot.width - 20), plot.x + plot.width / 2, y);
    y += 30;
  }
  for (const line of message.lines) {
    ctx.font = line.mono ? `${line.size || 11}px ${MONO}` : font(line.size || 11, line.weight || 400);
    const text = ellipsize(ctx, line.text, plot.width - 30);
    const centre = plot.x + plot.width / 2;
    if (line.box) {
      const width = ctx.measureText(text).width + 24;
      ctx.fillStyle = THEME.app;
      ctx.strokeStyle = THEME.grid;
      ctx.lineWidth = 1;
      roundRectPath(ctx, centre - width / 2, y - 14, width, 28, 8);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = line.color || THEME.muted;
    ctx.fillText(text, centre, y);
    y += 26;
  }
}

/* ---------- legend ---------- */

function drawLegendGlyph(ctx, shape, x, y, color) {
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2.2;
  if (shape === 'dot') {
    ctx.beginPath();
    ctx.arc(x + 9, y, 4, 0, Math.PI * 2);
    ctx.fill();
  } else if (shape === 'ring') {
    ctx.beginPath();
    ctx.arc(x + 9, y, 4.2, 0, Math.PI * 2);
    ctx.fillStyle = THEME.plot;
    ctx.fill();
    ctx.lineWidth = 1.6;
    ctx.stroke();
  } else if (shape === 'diamond') {
    diamondPath(ctx, x + 9, y, 5);
    ctx.fillStyle = THEME.plot;
    ctx.fill();
    ctx.lineWidth = 1.8;
    ctx.stroke();
  } else {
    ctx.setLineDash(shape === 'dash' ? [5, 3] : []);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 18, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawLegend(ctx, plot, legend, occupied) {
  if (!legend || !legend.items.length) return;
  const columns = Math.max(1, legend.cols || 1);
  const rows = Math.ceil(legend.items.length / columns);
  const rowHeight = 15;
  const inset = 9;
  ctx.font = font(10);
  const columnWidth = Math.min(
    Math.max(...legend.items.map((item) => ctx.measureText(item.text).width)) + 28,
    Math.max(90, plot.width / columns - 20)
  );
  const width = columnWidth * columns + inset * 2;
  const height = rows * rowHeight + inset * 2 - 3;

  // Pick the corner that hides the fewest plotted points.
  const margin = 10;
  const corners = [
    { x: plot.x + plot.width - width - margin, y: plot.y + margin },
    { x: plot.x + margin, y: plot.y + margin },
    { x: plot.x + plot.width - width - margin, y: plot.y + plot.height - height - margin },
    { x: plot.x + margin, y: plot.y + plot.height - height - margin },
  ];
  let best = corners[0];
  let bestCost = Infinity;
  for (const corner of corners) {
    let cost = 0;
    for (const point of occupied) {
      if (point[0] >= corner.x - 4 && point[0] <= corner.x + width + 4 &&
          point[1] >= corner.y - 4 && point[1] <= corner.y + height + 4) cost += 1;
    }
    if (cost < bestCost) { bestCost = cost; best = corner; }
    if (cost === 0) break;
  }

  ctx.globalAlpha = 0.93;
  ctx.fillStyle = '#18223a';
  ctx.strokeStyle = THEME.grid;
  ctx.lineWidth = 1;
  roundRectPath(ctx, best.x, best.y, width, height, 9);
  ctx.fill();
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  legend.items.forEach((item, index) => {
    const column = Math.floor(index / rows);
    const row = index % rows;
    const x = best.x + inset + column * columnWidth;
    const y = best.y + inset + row * rowHeight + 5;
    drawLegendGlyph(ctx, item.shape, x, y, item.color);
    ctx.font = font(10);
    ctx.fillStyle = THEME.text;
    ctx.fillText(ellipsize(ctx, item.text, columnWidth - 26), x + 24, y);
  });
}

/* ---------- main panel ---------- */

function drawMain(ctx, panel, main) {
  drawHeader(ctx, panel, main.title, main.subtitle);

  if (main.message) {
    const plot = { x: panel.x, y: panel.y + HEADER_H, width: panel.width, height: panel.height - HEADER_H };
    ctx.fillStyle = THEME.plot;
    ctx.fillRect(plot.x, plot.y, plot.width, plot.height);
    drawMessage(ctx, plot, main.message);
    return null;
  }

  const xs = [];
  const ys = [];
  for (const line of main.polylines) for (const point of line.pts) { xs.push(point[0]); ys.push(point[1]); }
  for (const dot of main.dots) { xs.push(dot.x); ys.push(dot.y); }
  for (const link of main.connectors || []) { xs.push(link.x1, link.x2); ys.push(link.y1, link.y2); }
  for (const rule of main.hlines || []) ys.push(rule.y);
  const xLo = Math.min(...xs);
  const xHi = Math.max(...xs);
  const yLo = Math.min(...ys);
  const yHi = Math.max(...ys);

  const bottom = 46;
  const top = panel.y + HEADER_H;
  const height = panel.height - HEADER_H - bottom;
  const probeY = makeScale(yLo, yHi, top + height, top, 7);
  const gutter = measureGutter(ctx, probeY, Boolean(main.yLabel));
  const plot = { x: panel.x + gutter, y: top, width: panel.width - gutter, height };
  const xScale = makeScale(xLo, xHi, plot.x, plot.x + plot.width, 7);
  const yScale = makeScale(yLo, yHi, plot.y + plot.height, plot.y, 7);

  drawGrid(ctx, plot, xScale, yScale, {});
  drawAxes(ctx, plot, xScale, yScale, main.xLabel, main.yLabel);

  const occupied = [];
  withClip(ctx, plot.x, plot.y, plot.width, plot.height, () => {
    for (const rule of main.hlines || []) {
      const y = yScale(rule.y);
      ctx.strokeStyle = rule.color;
      ctx.lineWidth = rule.width || 1.8;
      ctx.setLineDash(rule.dash || [5, 3]);
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.width, y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    for (const link of main.connectors || []) {
      ctx.strokeStyle = link.color;
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xScale(link.x1), yScale(link.y1));
      ctx.lineTo(xScale(link.x2), yScale(link.y2));
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const line of main.polylines) {
      if (line.pts.length < 2) continue;
      ctx.strokeStyle = line.color;
      ctx.lineWidth = line.width || 2.2;
      ctx.globalAlpha = line.alpha == null ? 1 : line.alpha;
      ctx.setLineDash(line.dash || []);
      ctx.beginPath();
      line.pts.forEach((point, index) => {
        const x = xScale(point[0]);
        const y = yScale(point[1]);
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    for (const dot of main.dots) {
      const x = xScale(dot.x);
      const y = yScale(dot.y);
      occupied.push([x, y]);
      ctx.globalAlpha = dot.alpha == null ? 1 : dot.alpha;
      if (dot.shape === 'diamond') diamondPath(ctx, x, y, dot.r);
      else { ctx.beginPath(); ctx.arc(x, y, dot.r, 0, Math.PI * 2); }
      if (dot.fill) { ctx.fillStyle = dot.fill; ctx.fill(); }
      if (dot.stroke) {
        ctx.strokeStyle = dot.stroke;
        ctx.lineWidth = dot.lineWidth || 1;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    ctx.textBaseline = 'alphabetic';
    ctx.font = font(9.5, 500);
    for (const label of main.labels || []) {
      const anchorX = xScale(label.x);
      const gap = label.dx || 7;
      // Flip a point label inwards rather than letting the plot edge clip it.
      const flip = anchorX + gap + ctx.measureText(label.text).width > plot.x + plot.width - 4;
      ctx.textAlign = flip ? 'right' : 'left';
      ctx.fillStyle = label.color || THEME.text;
      ctx.globalAlpha = 0.92;
      ctx.fillText(label.text, anchorX + (flip ? -gap : gap), yScale(label.y) - (label.dy || 7));
      ctx.globalAlpha = 1;
    }
    ctx.textAlign = 'left';

    for (const rule of main.hlines || []) {
      if (!rule.text) continue;
      ctx.font = font(9.5, 700);
      ctx.textAlign = 'right';
      ctx.textBaseline = 'bottom';
      const width = ctx.measureText(rule.text).width;
      const x = plot.x + plot.width - 6;
      const y = yScale(rule.y) - 2;
      ctx.globalAlpha = 0.78;
      ctx.fillStyle = THEME.plot;
      ctx.fillRect(x - width - 3, y - 12, width + 6, 13);
      ctx.globalAlpha = 1;
      ctx.fillStyle = rule.color;
      ctx.fillText(rule.text, x, y);
    }
  });

  drawLegend(ctx, plot, main.legend, occupied);
  return { plot, xScale, yScale };
}

/* ---------- ranking panel ---------- */

function drawRank(ctx, panel, rank) {
  drawHeader(ctx, panel, rank.title, rank.subtitle);
  const bottom = 46;
  const plot = {
    x: panel.x,
    y: panel.y + HEADER_H,
    width: panel.width,
    height: panel.height - HEADER_H - bottom,
  };

  if (!rank.items.length) {
    ctx.fillStyle = THEME.plot;
    ctx.fillRect(plot.x, plot.y, plot.width, plot.height);
    drawMessage(ctx, plot, { lines: [{ text: rank.emptyText || 'Nothing to rank yet' }] });
    return { plot, rows: [], pageSize: 0, offset: 0, total: 0 };
  }

  const maximum = Math.max(...rank.items.map((item) => item.value), 0);
  const xScale = makeScale(0, maximum + Math.max(maximum, 1) * 0.24, plot.x, plot.x + plot.width, 5);
  xScale.lo = 0;

  drawGrid(ctx, plot, xScale, null, { horizontal: false });
  ctx.fillStyle = THEME.muted;
  ctx.font = font(10);
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (const tick of xScale.ticks) {
    if (tick < 0) continue;
    ctx.fillText(formatTick(tick, xScale.step), xScale(tick), plot.y + plot.height + 7);
  }
  ctx.fillStyle = THEME.faint;
  ctx.font = font(10.5, 600);
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(ellipsize(ctx, rank.xLabel || '', plot.width),
    plot.x + plot.width / 2, plot.y + plot.height + 34);

  const rowHeight = 34;
  const pageSize = Math.max(1, Math.floor(plot.height / rowHeight));
  const offset = Math.min(Math.max(0, rank.offset || 0), Math.max(0, rank.items.length - pageSize));
  const page = rank.items.slice(offset, offset + pageSize);

  const rows = [];
  withClip(ctx, plot.x, plot.y, plot.width, plot.height, () => {
    page.forEach((item, index) => {
      const top = plot.y + index * rowHeight;
      ctx.font = font(10, 600);
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillStyle = THEME.text;
      ctx.fillText(ellipsize(ctx, item.label, plot.width - 12), plot.x + 1, top + 13);

      const barY = top + 18;
      const barHeight = 9;
      ctx.fillStyle = item.color;
      ctx.globalAlpha = 0.9;
      roundRectPath(ctx, plot.x, barY, Math.max(2, xScale(item.value) - plot.x), barHeight, 4);
      ctx.fill();
      ctx.globalAlpha = 1;

      ctx.font = font(10, 700);
      ctx.fillStyle = THEME.text;
      ctx.textBaseline = 'middle';
      ctx.fillText(item.text, xScale(item.value) + 7, barY + barHeight / 2);
      rows.push({ label: item.label, top, height: rowHeight });
    });
  });

  if (rank.items.length > pageSize) {
    const trackX = plot.x + plot.width + 6;
    const visible = pageSize / rank.items.length;
    ctx.fillStyle = THEME.grid;
    roundRectPath(ctx, trackX, plot.y, 3, plot.height, 2);
    ctx.fill();
    ctx.fillStyle = THEME.accent;
    roundRectPath(ctx, trackX, plot.y + (offset / rank.items.length) * plot.height, 3,
      Math.max(18, visible * plot.height), 2);
    ctx.fill();
  }

  return { plot, rows, pageSize, offset, total: rank.items.length };
}

/* ---------- scene ---------- */

function drawScene(ctx, scene, width, height) {
  ctx.fillStyle = THEME.panel;
  ctx.fillRect(0, 0, width, height);

  const usable = width - PAD * 2 - PANEL_GAP;
  const mainWidth = Math.max(220, usable * MAIN_FRACTION);
  const rankWidth = Math.max(150, usable - mainWidth) - 12; // room for the scroll rail
  const panelHeight = height - PAD * 2;

  const mainPanel = { x: PAD, y: PAD, width: mainWidth, height: panelHeight };
  const rankPanel = { x: PAD + mainWidth + PANEL_GAP, y: PAD, width: rankWidth, height: panelHeight };

  const main = drawMain(ctx, mainPanel, scene.main);
  const rank = drawRank(ctx, rankPanel, scene.rank);
  return { main, rank, width, height };
}

/* ---------- SVG backend ---------- */

class SvgContext {
  constructor(width, height) {
    this.isSvg = true;
    this.width = width;
    this.height = height;
    this.parts = [];
    this.defs = [];
    this.stack = [];
    this.clipDepth = 0;
    this.path = [];
    this.fillStyle = '#000';
    this.strokeStyle = '#000';
    this.lineWidth = 1;
    this.globalAlpha = 1;
    this.font = font(10);
    this.textAlign = 'left';
    this.textBaseline = 'alphabetic';
    this.lineJoin = 'miter';
    this.lineCap = 'butt';
    this.dash = [];
    this.probe = document.createElement('canvas').getContext('2d');
  }

  measureText(text) {
    this.probe.font = this.font;
    return this.probe.measureText(text);
  }

  setLineDash(dash) { this.dash = dash || []; }

  save() {
    this.stack.push({
      fillStyle: this.fillStyle, strokeStyle: this.strokeStyle, lineWidth: this.lineWidth,
      globalAlpha: this.globalAlpha, font: this.font, textAlign: this.textAlign,
      textBaseline: this.textBaseline, dash: this.dash,
    });
  }

  restore() { Object.assign(this, this.stack.pop() || {}); }

  beginPath() { this.path = []; }
  moveTo(x, y) { this.path.push(`M${this.n(x)} ${this.n(y)}`); }
  lineTo(x, y) { this.path.push(`L${this.n(x)} ${this.n(y)}`); }
  closePath() { this.path.push('Z'); }

  rect(x, y, width, height) {
    this.path.push(`M${this.n(x)} ${this.n(y)}H${this.n(x + width)}V${this.n(y + height)}H${this.n(x)}Z`);
  }

  arc(cx, cy, radius, start, end) {
    const x0 = cx + radius * Math.cos(start);
    const y0 = cy + radius * Math.sin(start);
    this.path.push(`${this.path.length ? 'L' : 'M'}${this.n(x0)} ${this.n(y0)}`);
    if (Math.abs(end - start) >= Math.PI * 2 - 1e-6) {
      this.path.push(`A${radius} ${radius} 0 1 1 ${this.n(cx - radius * Math.cos(start))} ${this.n(cy - radius * Math.sin(start))}`);
      this.path.push(`A${radius} ${radius} 0 1 1 ${this.n(x0)} ${this.n(y0)}`);
      return;
    }
    const large = Math.abs(end - start) > Math.PI ? 1 : 0;
    this.path.push(`A${radius} ${radius} 0 ${large} ${end > start ? 1 : 0} ` +
      `${this.n(cx + radius * Math.cos(end))} ${this.n(cy + radius * Math.sin(end))}`);
  }

  fill() { this.emitPath(this.fillStyle, 'none'); }
  stroke() { this.emitPath('none', this.strokeStyle); }

  fillRect(x, y, width, height) {
    this.parts.push(`<rect x="${this.n(x)}" y="${this.n(y)}" width="${this.n(width)}" ` +
      `height="${this.n(height)}" fill="${this.fillStyle}"${this.alpha()}/>`);
  }

  fillText(text, x, y) { this.text(text, x, y, ''); }

  rotatedText(text, x, y, degrees) {
    this.text(text, 0, 0, ` transform="translate(${this.n(x)} ${this.n(y)}) rotate(${degrees})"`);
  }

  pushClip(x, y, width, height) {
    const id = `clip${++this.clipDepth}-${this.parts.length}`;
    this.defs.push(`<clipPath id="${id}"><rect x="${this.n(x)}" y="${this.n(y)}" ` +
      `width="${this.n(width)}" height="${this.n(height)}"/></clipPath>`);
    this.parts.push(`<g clip-path="url(#${id})">`);
  }

  popClip() { this.parts.push('</g>'); }

  emitPath(fill, stroke) {
    if (!this.path.length) return;
    const dash = this.dash.length ? ` stroke-dasharray="${this.dash.join(' ')}"` : '';
    const strokeAttrs = stroke === 'none' ? ''
      : ` stroke-width="${this.n(this.lineWidth)}" stroke-linejoin="${this.lineJoin}" ` +
        `stroke-linecap="${this.lineCap}"${dash}`;
    this.parts.push(`<path d="${this.path.join('')}" fill="${fill}" stroke="${stroke}"` +
      `${strokeAttrs}${this.alpha()}/>`);
  }

  text(content, x, y, transform) {
    const anchor = { left: 'start', center: 'middle', right: 'end' }[this.textAlign] || 'start';
    const baseline = { top: 'hanging', middle: 'central', bottom: 'text-after-edge' }[this.textBaseline]
      || 'alphabetic';
    const match = /^(\d+)?\s*([\d.]+)px\s+(.*)$/.exec(this.font) || [];
    this.parts.push(`<text x="${this.n(x)}" y="${this.n(y)}" fill="${this.fillStyle}" ` +
      `font-family="${escapeXml(match[3] || FONTS).replace(/"/g, '&quot;')}" font-size="${match[2] || 10}" ` +
      `font-weight="${match[1] || 400}" text-anchor="${anchor}" ` +
      `dominant-baseline="${baseline}"${this.alpha()}${transform}>${escapeXml(content)}</text>`);
  }

  alpha() { return this.globalAlpha < 1 ? ` opacity="${this.n(this.globalAlpha)}"` : ''; }

  n(value) { return Math.round(value * 100) / 100; }

  toString() {
    return `<?xml version="1.0" encoding="UTF-8"?>\n` +
      `<svg xmlns="http://www.w3.org/2000/svg" width="${this.width}" height="${this.height}" ` +
      `viewBox="0 0 ${this.width} ${this.height}">` +
      (this.defs.length ? `<defs>${this.defs.join('')}</defs>` : '') +
      this.parts.join('') + '</svg>';
  }
}

function escapeXml(value) {
  return String(value).replace(/[&<>]/g, (character) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[character]));
}

/* ---------- surface ---------- */

class Surface {
  constructor(stage, base, overlay) {
    this.stage = stage;
    this.base = base;
    this.overlay = overlay;
    this.scene = null;
    this.frame = null;
    this.size = { width: 0, height: 0 };
    this.pending = false;
    new ResizeObserver(() => this.render()).observe(stage);
  }

  setScene(scene) {
    this.scene = scene;
    this.render();
  }

  render() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      this.draw();
    });
  }

  draw() {
    const width = this.stage.clientWidth;
    const height = this.stage.clientHeight;
    if (!this.scene || width < 40 || height < 40) return;
    const ratio = window.devicePixelRatio || 1;
    for (const canvas of [this.base, this.overlay]) {
      if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
      }
    }
    this.size = { width, height };
    const ctx = this.base.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    this.frame = drawScene(ctx, this.scene, width, height);
    this.clearOverlay();
  }

  clearOverlay() {
    const ratio = window.devicePixelRatio || 1;
    const ctx = this.overlay.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, this.size.width, this.size.height);
    return ctx;
  }

  /** Nearest hover point in pixels, or null when the pointer is too far away. */
  nearest(points, x, y, radius) {
    if (!this.frame || !this.frame.main || !points.length) return null;
    const { xScale, yScale, plot } = this.frame.main;
    if (x < plot.x - 4 || x > plot.x + plot.width + 4 ||
        y < plot.y - 4 || y > plot.y + plot.height + 4) return null;
    let best = null;
    let bestDistance = radius * radius;
    for (const point of points) {
      const dx = xScale(point.x) - x;
      const dy = yScale(point.y) - y;
      const distance = dx * dx + dy * dy;
      if (distance <= bestDistance) { bestDistance = distance; best = point; }
    }
    return best ? { point: best, px: xScale(best.x), py: yScale(best.y) } : null;
  }

  drawHover(hit) {
    const ctx = this.clearOverlay();
    if (!hit) return;
    const { plot } = this.frame.main;
    const { point, px, py } = hit;

    ctx.save();
    ctx.beginPath();
    ctx.rect(plot.x, plot.y, plot.width, plot.height);
    ctx.clip();
    ctx.strokeStyle = THEME.muted;
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(px, plot.y);
    ctx.lineTo(px, plot.y + plot.height);
    ctx.moveTo(plot.x, py);
    ctx.lineTo(plot.x + plot.width, py);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
    ctx.restore();

    ctx.beginPath();
    ctx.arc(px, py, 6, 0, Math.PI * 2);
    ctx.fillStyle = point.color;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.6;
    ctx.stroke();

    const lines = point.details.split('\n');
    ctx.font = font(11, 400);
    const lineHeight = 16;
    const width = Math.max(...lines.map((line) => ctx.measureText(line).width)) + 22;
    const height = lines.length * lineHeight + 16;
    let x = px + 16;
    let y = py + 16;
    if (x + width > this.size.width - 6) x = px - 16 - width;
    if (y + height > this.size.height - 6) y = py - 16 - height;
    x = Math.max(6, x);
    y = Math.max(6, y);

    ctx.globalAlpha = 0.97;
    ctx.fillStyle = THEME.app;
    ctx.strokeStyle = point.color;
    ctx.lineWidth = 1.2;
    roundRectPath(ctx, x, y, width, height, 10);
    ctx.fill();
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    lines.forEach((line, index) => {
      ctx.font = font(11, index === 0 ? 700 : 400);
      ctx.fillStyle = index === 0 ? THEME.text : THEME.muted;
      ctx.fillText(line, x + 11, y + 13 + (index + 1) * lineHeight - 6);
    });
  }

  /** Which ranking row sits under the pointer, if any. */
  rankRowAt(x, y) {
    if (!this.frame || !this.frame.rank) return null;
    const { plot, rows } = this.frame.rank;
    if (x < plot.x - 6 || x > plot.x + plot.width + 12) return null;
    return rows.find((row) => y >= row.top && y < row.top + row.height) || null;
  }

  isOverRank(x) {
    if (!this.frame || !this.frame.rank) return false;
    const { plot } = this.frame.rank;
    return x >= plot.x - PANEL_GAP / 2 && x <= plot.x + plot.width + 14;
  }

  toPngBlob(scale) {
    const canvas = document.createElement('canvas');
    canvas.width = Math.round(this.size.width * scale);
    canvas.height = Math.round(this.size.height * scale);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    drawScene(ctx, this.scene, this.size.width, this.size.height);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
  }

  toSvg() {
    const ctx = new SvgContext(Math.round(this.size.width), Math.round(this.size.height));
    drawScene(ctx, this.scene, this.size.width, this.size.height);
    return ctx.toString();
  }
}

return { Surface, THEME, formatValue, formatTick, makeScale, SvgContext, drawScene };
})();
