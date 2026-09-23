/* Browser-local analytics over published snapshots. No extrapolation or refitting. */
'use strict';
const Analytics = (() => {
  const settings = new Map();
  let context, exported = [];
  const controls = document.getElementById('analysis-controls');
  const results = document.getElementById('analysis-results');
  const fmt = (n, digits = 3) => Number.isFinite(n) ? n.toFixed(digits) : 'Unavailable';
  const mean = a => a.reduce((s, n) => s + n, 0) / a.length;
  const pct = (a, b) => Number.isFinite(a) && b > 0 ? `${fmt(100 * (a / b - 1), 2)}%` : 'Unavailable';
  function node(tag, text, parent, attrs = {}) {
    const el = document.createElement(tag);
    if (text !== null) el.textContent = text;
    Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
    if (parent) parent.append(el);
    return el;
  }
  function note(text) { node('p', text, results, {class: 'analysis-note'}); }
  const collator = new Intl.Collator('en', {numeric: true, sensitivity: 'base'});
  function sortValue(value) {
    const text = String(value ?? '').trim();
    if (['', 'Unavailable', 'None', 'No shared interval'].includes(text)) return null;
    // Numbers, percentages, power ranges and crossover lists compare numerically.
    const numeric = text.replace(/(?:%| W)$/, '');
    if (/^[+-]?\d+(?:\.\d+)?(?:(?:–|, )[-+]?\d+(?:\.\d+)?)*$/.test(numeric)) {
      return numeric.split(/–|, /).map(Number);
    }
    return text;
  }
  function compareValues(left, right) {
    if (Array.isArray(left) && Array.isArray(right)) {
      for (let i = 0; i < Math.min(left.length, right.length); i++) {
        if (left[i] !== right[i]) return left[i] - right[i];
      }
      return left.length - right.length;
    }
    return collator.compare(String(left), String(right));
  }
  function table(title, headings, rows, id) {
    const sorts = settings.get(context.entry.key).sorts;
    const sort = sorts[id];
    const ordered = rows.map((row, index) => ({row, index}));
    if (sort) ordered.sort((a, b) => {
      const left = sortValue(a.row[sort.column]), right = sortValue(b.row[sort.column]);
      // Missing results stay last regardless of direction; ties retain default order.
      if (left === null || right === null) return (left === null) - (right === null) || a.index - b.index;
      return sort.direction * compareValues(left, right) || a.index - b.index;
    });
    const displayed = ordered.map(item => item.row);
    node('h3', title, results);
    const wrap = node('div', null, results, {class: 'analysis-table-wrap'});
    const t = node('table', null, wrap, {id});
    const head = node('tr', null, node('thead', null, t));
    headings.forEach((heading, column) => {
      const active = sort?.column === column;
      const direction = active ? (sort.direction === 1 ? 'ascending' : 'descending') : 'none';
      const th = node('th', null, head, {scope: 'col', 'aria-sort': direction});
      const button = node('button', null, th, {
        type: 'button', class: 'analysis-sort', id: `${id}-sort-${column}`,
        'aria-label': heading,
        title: `Sort ${heading} ${active && sort.direction === 1 ? 'descending' : 'ascending'}`,
      });
      node('span', heading, button);
      node('span', active ? (sort.direction === 1 ? ' ↑' : ' ↓') : ' ↕', button, {'aria-hidden': 'true'});
      button.addEventListener('click', () => {
        sorts[id] = {column, direction: active && sort.direction === 1 ? -1 : 1};
        draw();
        document.getElementById(button.id)?.focus({preventScroll: true});
      });
    });
    const body = node('tbody', null, t);
    displayed.forEach(row => { const tr = node('tr', null, body); row.forEach(c => node('td', String(c), tr)); });
    if (!rows.length) node('td', 'No matching selected profiles', node('tr', null, body), {colspan: headings.length});
    exported.push([title], headings, ...displayed, []);
  }
  function field(label, id, value, attrs, callback, options) {
    const l = node('label', label, controls, {for: id});
    const input = node(options ? 'select' : 'input', null, l, {id, ...attrs});
    if (options) options.forEach(([v, title]) => node('option', title, input, {value: v}));
    input.value = value;
    input.addEventListener(options ? 'change' : 'input', () => { callback(input.value); draw(); });
    return input;
  }
  function curve(profile) {
    if (profile.series.length !== 1) return {error: 'Multiple source series: select a single snapshot'};
    const points = profile.series[0].points.filter(p => p[0] > 0 && p[1] > 0).map(p => p.slice(0, 2)).sort((a,b) => a[0]-b[0]);
    const unique = [];
    for (const p of points) {
      const last = unique.at(-1);
      if (last && last[0] === p[0]) {
        if (last[1] !== p[1]) return {error: 'Conflicting scores at duplicate power'};
      } else unique.push(p);
    }
    return {points: unique, source: profile.series[0].source};
  }
  function evaluate(c, target, axis = 0) {
    if (c.error) return {error: c.error};
    if (!Number.isFinite(target) || target <= 0) return {error: 'Enter a positive target'};
    if (axis === 0) {
      let low = 0, high = c.points.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (c.points[middle][0] < target) low = middle + 1; else high = middle;
      }
      const b = c.points[low], a = c.points[low - 1];
      if (b && Math.abs(b[0] - target) < 1e-9) return {value: b[1], a: b, b, exact: true};
      if (a && b) return {value: a[1] + (b[1]-a[1])*(target-a[0])/(b[0]-a[0]), a, b, exact: false};
      return {error: 'Outside published range'};
    }
    const candidates = [];
    for (const p of c.points) if (Math.abs(p[axis] - target) < 1e-9) candidates.push({value: p[1-axis], a: p, b: p, exact: true});
    for (let i = 1; i < c.points.length; i++) {
      const a = c.points[i-1], b = c.points[i];
      if (target > Math.min(a[axis], b[axis]) && target < Math.max(a[axis], b[axis])) {
        candidates.push({value: a[1-axis] + (b[1-axis]-a[1-axis])*(target-a[axis])/(b[axis]-a[axis]), a, b, exact: false});
      }
    }
    // Nonmonotonic curves may cross a target score more than once.
    candidates.sort((a,b) => a.value-b.value);
    return candidates[0] || {error: 'Outside published range'};
  }
  function evidence(r) {
    if (r.error) return r.error;
    const point = p => `${fmt(p[0])} W / ${fmt(p[1])} score`;
    return r.exact ? `Published: ${point(r.a)}` : `Estimated: ${point(r.a)} → ${point(r.b)}; gap ${fmt(r.b[0]-r.a[0])} W`;
  }
  function plot(series, xLabel, yLabel, id, dotsOnly = false) {
    const points = series.flatMap(s => s.points);
    if (!points.length) return;
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 900 310'); svg.setAttribute('class', 'analysis-plot'); svg.id = id;
    svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${yLabel} by ${xLabel}`); results.append(svg);
    const add = (tag, attrs, text) => { const e = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v)); if(text !== undefined)e.textContent=text;svg.append(e);return e; };
    let xmin=Math.min(...points.map(p=>p[0])), xmax=Math.max(...points.map(p=>p[0]));
    let ymin=Math.min(...points.map(p=>p[1])), ymax=Math.max(...points.map(p=>p[1]));
    if(xmin===xmax){xmin-=0.5;xmax+=0.5;} if(ymin===ymax){ymin-=0.5;ymax+=0.5;}
    const x = v => 75+(v-xmin)/(xmax-xmin)*785, y = v => 250-(v-ymin)/(ymax-ymin)*210;
    add('path',{d:'M75 35 V250 H860',stroke:'#8e9bb6',fill:'none'});
    for(let i=0;i<=4;i++){
      const xv=xmin+(xmax-xmin)*i/4, yv=ymin+(ymax-ymin)*i/4;
      add('text',{x:x(xv),y:270,fill:'#8e9bb6','text-anchor':'middle','font-size':11},fmt(xv,2));
      add('text',{x:68,y:y(yv)+4,fill:'#8e9bb6','text-anchor':'end','font-size':11},fmt(yv,2));
    }
    add('text',{x:450,y:300,fill:'#f3f6ff','text-anchor':'middle','font-size':13},xLabel);
    add('text',{x:75,y:20,fill:'#f3f6ff','font-size':13},yLabel);
    const colors=['#38d6d0','#ffad5a','#a99aff','#ff718b','#63d69d'];
    const legend = node('div', null, results, {class: 'analysis-legend'});
    series.forEach((s,i)=>{
      const color=colors[i%colors.length];
      const label = node('span', s.name, legend); label.style.color = color;
      if(!dotsOnly) { const line=add('polyline',{points:s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' '),stroke:color,fill:'none','stroke-width':2});const t=document.createElementNS(ns,'title');t.textContent=s.name;line.append(t); }
      s.points.forEach(p=>{const dot=add('circle',{cx:x(p[0]),cy:y(p[1]),r:3,fill:color});const t=document.createElementNS(ns,'title');t.textContent=`${p[2] || s.name}: ${fmt(p[0])}, ${fmt(p[1])}`;dot.append(t);});
    });
  }
  function controlsFor(entry, chosen) {
    controls.replaceChildren();
    const s=settings.get(entry.key);
    const modes=entry.kind==='curve' ? [['targets','Equal power / performance'],['tradeoffs','Power trade-offs'],['frontier','Published-point frontier'],['baseline','Baseline / generation comparison']] : [['groups','Phone power grouped by processor'],['runtime','Runtime / generation comparison']];
    field('Analysis','analysis-mode',s.mode,{},v=>{s.mode=v;controlsFor(entry,chosen);},modes);
    field('Baseline','analysis-baseline',s.baseline,{},v=>s.baseline=v, chosen.map(p=>[p.label,p.label]));
    if(entry.kind==='curve') {
      field('Power target (W)','analysis-power',s.power,{type:'number',min:'0',step:'0.1'},v=>{s.power=v; document.getElementById('analysis-power-slider').value=v;});
      const all=chosen.flatMap(p=>p.series.flatMap(c=>c.points.map(p=>p[0])));
      field('Adjust power target','analysis-power-slider',s.power,{type:'range',min:Math.min(...all,0.1),max:Math.max(...all,10),step:'0.01'},v=>{s.power=v;document.getElementById('analysis-power').value=v;});
      field('Target score','analysis-score',s.score,{type:'number',min:'0',step:'any'},v=>s.score=v);
    } else {
      field('Brand','analysis-brand',s.brand,{},v=>s.brand=v,[['','All brands'],...Array.from(new Set(entry.profiles.map(p=>p.brand))).sort().map(b=>[b,b])]);
      field('Minimum screen (in)','analysis-screen',s.screen,{type:'number',min:'0',step:'0.1'},v=>s.screen=v);
      field('Maximum screen (in)','analysis-screen-max',s.screenMax,{type:'number',min:'0',step:'0.1'},v=>s.screenMax=v);
      field('Listed refresh (Hz)','analysis-hz',s.hz,{type:'number',min:'0',step:'1'},v=>s.hz=v);
      field('Common battery (Wh)','analysis-wh',s.wh,{type:'number',min:'0',step:'0.5'},v=>s.wh=v);
    }
    node('button', 'Clear sort', controls, {type: 'button', class: 'btn', id: 'analysis-clear-sort', title: 'Restore default order for every analysis table in this dataset'}).addEventListener('click', () => { s.sorts = {}; draw(); });
    node('button','Export analysis CSV',controls,{type:'button',class:'btn',id:'analysis-export'}).addEventListener('click',exportCsv);
  }
  function targets(profiles,s) {
    const base=profiles.find(p=>p.label===s.baseline), bc=base&&curve(base);
    const bp=bc?evaluate(bc,Number(s.power)): {}, bs=bc?evaluate(bc,Number(s.score),1):{};
    table('At equal power', ['Profile','Score','Score/W','Performance vs baseline','Evidence','Source'],profiles.map(p=>{
      const c=curve(p),r=evaluate(c,Number(s.power)); return [p.label,fmt(r.value),fmt(r.value/Number(s.power)),pct(r.value,bp.value),evidence(r),c.source||''];
    }),'analysis-target-power');
    table('At equal performance',['Profile','Minimum power (W)','Power saved vs baseline','Evidence'],profiles.map(p=>{
      const r=evaluate(curve(p),Number(s.score),1);return [p.label,fmt(r.value),r.value>0&&bs.value>0?`${fmt(100*(1-r.value/bs.value),2)}%`:'Unavailable',evidence(r)];
    }),'analysis-target-score');
  }
  function tradeoffs(profiles) {
    table('Power needed for a fraction of each published peak',['Profile','Peak score','80% (W)','90% (W)','95% (W)','Peak (W)','Final 10% extra W','Evidence'],profiles.map(p=>{
      const c=curve(p);if(c.error||!c.points.length)return[p.label,'Unavailable','','','','','',c.error||'No positive points'];
      const peak=Math.max(...c.points.map(p=>p[1])), r=[.8,.9,.95,1].map(f=>evaluate(c,peak*f,1));
      return[p.label,fmt(peak),...r.map(v=>fmt(v.value)),fmt(r[3].value-r[1].value),r.map((v,i)=>`${[80,90,95,100][i]}%: ${evidence(v)}`).join('; ')];
    }),'analysis-tradeoffs');
    note('Each fraction refers to that profile’s own published peak, not an equal-performance target. Thresholds below the published range are unavailable. These curves do not establish sustained performance or thermal limits.');
  }
  function frontier(profiles) {
    const all=profiles.flatMap(p=>p.series.flatMap(s=>s.points.map(v=>({name:p.label,power:v[0],score:v[1],source:s.source}))));
    const sorted=all.slice().sort((a,b)=>a.power-b.power||b.score-a.score);
    let best=-Infinity, bestPower=-Infinity;const front=[];
    for(const p of sorted){if(p.score>best || (p.score===best && p.power===bestPower)){front.push(p);best=p.score;bestPower=p.power;}}
    plot([{name:'Published frontier',points:front.map(p=>[p.power,p.score])}], 'Board power (W)','Published score','analysis-frontier-plot',true);
    table('Non-dominated published points',['Profile','Power (W)','Score','Source'],front.map(p=>[p.name,fmt(p.power),fmt(p.score),p.source]),'analysis-frontier');
    note(`${front.length} of ${all.length} published points have no selected point with at least as much performance and no greater power, with one strict improvement. Ties are retained. The frontier is discrete; it does not infer intermediate operating points.`);
  }
  function baseline(profiles,s) {
    const base=profiles.find(p=>p.label===s.baseline);if(!base){note('Select a baseline profile.');return;}
    const bc=curve(base), series=[],rows=[];
    for(const p of profiles.filter(p=>p!==base)) {
      const c=curve(p);
      if(c.error||bc.error||!c.points.length||!bc.points.length){rows.push([p.label,'Unavailable','','',c.error||bc.error||'No points']);continue;}
      const lo=Math.max(c.points[0][0],bc.points[0][0]),hi=Math.min(c.points.at(-1)[0],bc.points.at(-1)[0]);
      if(lo>=hi){rows.push([p.label,'No shared interval','','','']);continue;}
      const knots=[...new Set([lo,hi,...c.points.map(p=>p[0]),...bc.points.map(p=>p[0])].filter(x=>x>=lo&&x<=hi))].sort((a,b)=>a-b);
      const crosses=[],samples=[],ties=[];
      const delta=x=>evaluate(c,x).value-evaluate(bc,x).value;
      for(let i=1;i<knots.length;i++){
        const a=knots[i-1],b=knots[i],da=delta(a),db=delta(b);
        if(Math.abs(da)<1e-8&&Math.abs(db)<1e-8)ties.push(`${fmt(a)}–${fmt(b)} W`);
        else {if(Math.abs(da)<1e-8)crosses.push(a);if(Math.abs(db)<1e-8)crosses.push(b);if(da*db<0)crosses.push(a-da*(b-a)/(db-da));}
        for(let j=0;j<8;j++){const x=a+(b-a)*j/8;samples.push([x,100*(evaluate(c,x).value/evaluate(bc,x).value-1)]);}
      }
      samples.push([hi,100*(evaluate(c,hi).value/evaluate(bc,hi).value-1)]);
      series.push({name:p.label,points:samples});
      rows.push([p.label,`${fmt(lo)}–${fmt(hi)} W`,pct(evaluate(c,Number(s.power)).value,evaluate(bc,Number(s.power)).value),[...new Set(crosses.map(x=>fmt(x)))].join(', ')||'None',ties.length?`Tied intervals: ${ties.join(', ')}`:'Piecewise-linear estimates']);
    }
    plot(series,'Board power (W)',`Performance gain vs ${base.label} (%)`,'analysis-baseline-plot');
    table('Baseline-relative comparison',['Profile','Shared power range','Gain at target power','Equal-score crossings (W)','Method'],rows,'analysis-baseline-results');
    note('Compare generations by choosing the older profile as baseline. Curves use only shared power ranges; sparse gaps remain estimates. Hover a plotted point for its profile and values.');
  }
  function battery(profiles,s) {
    const active = profiles.filter(p=>(!s.brand||p.brand===s.brand)&&(!s.screen||(p.screenSize!==null&&p.screenSize>=Number(s.screen)))&&(!s.screenMax||(p.screenSize!==null&&p.screenSize<=Number(s.screenMax)))&&(!s.hz||p.refreshHz===Number(s.hz)));
    note(`${active.length} selected device profiles match the analysis filters. These filters apply to this panel. Missing specifications are excluded when their filter is active. Listed refresh is not the test’s actual refresh rate.`);
    if(s.mode==='runtime') {
      const b=active.find(p=>p.label===s.baseline);
      table('Runtime and capacity decomposition',['Phone','Runtime (h)','Capacity ratio','Inverse power ratio','Runtime ratio','Common battery runtime (h)','OS / source'],active.map(p=>[p.label,fmt(p.hours),fmt(b?p.capacityWh/b.capacityWh:NaN),fmt(b?(b.capacityWh/b.hours)/(p.capacityWh/p.hours):NaN),fmt(b?p.hours/b.hours:NaN),fmt(Number(s.wh)>0?Number(s.wh)*p.hours/p.capacityWh:NaN),`${p.os} / ${(p.sources||[]).join('; ')}`]),'analysis-runtime');
      note(b?`Baseline: ${b.label}. Runtime ratio = capacity ratio × inverse power ratio. For profiles combining rows, decomposition power is mean capacity divided by mean runtime. Common-battery runtime is a scenario holding each phone’s estimated consumption constant, not a measurement. Match product family and size manually for generation comparisons; software and other hardware still differ.`:'Baseline is not in the filtered selection; ratio comparisons are unavailable.');
      return;
    }
    const selected=new Set(context.socs.map(p=>p.soc));
    const groups=new Map();active.forEach(p=>{if(p.soc&&selected.has(p.soc)){if(!groups.has(p.soc))groups.set(p.soc,[]);groups.get(p.soc).push(p);}});
    const rows=[],omissions=[],dots=[];
    for(const [soc,phones] of groups){
      const a=phones.map(p=>p.avgPowerW).sort((a,b)=>a-b),brands=[...new Set(phones.map(p=>p.brand))];
      const med=(a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;
      rows.push([soc,phones.length,brands.length,fmt(mean(a)),fmt(med),`${fmt(a[0])}–${fmt(a.at(-1))}`,phones.length===1?'Single device profile':'Observed device spread']);
      dots.push({name:soc,points:phones.map(p=>[p.capacityWh,p.avgPowerW,p.label])});
      for(const brand of brands){const rest=phones.filter(p=>p.brand!==brand).map(p=>p.avgPowerW);omissions.push([soc,brand,rest.length,fmt(mean(rest)),fmt(mean(rest)-mean(a)),rest.length?'Descriptive sensitivity':'No remaining devices']);}
    }
    plot(dots,'Battery capacity (Wh)','Estimated whole-phone power (W)','analysis-distribution',true);
    table('Phone power grouped by processor',['Processor','Device profiles','Brands','Mean W','Median W','Range W','Interpretation'],rows,'analysis-groups');
    table('Remove one brand at a time',['Processor','Omitted brand','Remaining profiles','Mean W','Change W','Coverage'],omissions,'analysis-omissions');
    table('Individual device profiles',['Phone','Processor','Power W','Screen in','Listed Hz','Review / provenance'],active.map(p=>[p.label,p.soc||'Unassigned',fmt(p.avgPowerW),fmt(p.screenSize,2),fmt(p.refreshHz,0),[p.metadataReview,p.metadataSource,p.metadataReviewed].filter(Boolean).join(' · ')||'Snapshot metadata']),'analysis-devices');
    note('Means weight each selected device profile equally. Repeated OS profiles are not independent phones. Brand counts use source brand labels, not independent manufacturers. Spread is descriptive, not a confidence interval or repeated-run error. All power estimates retain battery-imprint Wh; measured-mAh overlays are excluded.');
  }
  function draw() {
    if(!context)return;
    results.replaceChildren(); exported=[];
    document.getElementById('analysis-clear-sort').disabled = !Object.keys(settings.get(context.entry.key).sorts).length;
    const {entry,profiles}=context,s=settings.get(entry.key);
    if(!entry.available||!profiles.length){note('Select profiles to analyse.');return;}
    if(entry.kind==='curve') {
      note('Uses published points, which may already be fitted upstream. Estimates join adjacent power-ordered points; no extrapolation. Multiple source series or conflicting duplicate powers are unavailable for interpolation. Comparisons stay within this benchmark and assume a compatible board-power measurement basis.');
      if(s.mode==='targets')targets(profiles,s);
      if(s.mode==='tradeoffs')tradeoffs(profiles);
      if(s.mode==='frontier')frontier(profiles);
      if(s.mode==='baseline')baseline(profiles,s);
    } else battery(profiles,s);
  }
  function exportCsv() {
    const s=settings.get(context.entry.key);
    const rows=[['Dataset',context.entry.key],['Snapshot loaded at',context.generated],['Settings',JSON.stringify(s)],['Selected profiles',context.profiles.map(p=>p.label).join('; ')],['Selected processors',context.socs.map(p=>p.soc).join('; ')],['Sources',context.profiles.flatMap(p=>p.sources||p.series.map(s=>s.source)).join('; ')],['Method','Published snapshots; linear interpolation only within brackets; no extrapolation. Battery scenarios use imprint energy and constant estimated power.'],[],...exported];
    const quote=v=>'"'+String(v??'').replace(/^[=+@\-]/,"'$&").replaceAll('"','""')+'"';
    const blob=new Blob(['\ufeff'+rows.map(r=>r.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
    const url=URL.createObjectURL(blob),a=node('a',null,document.body,{href:url,download:`socpk-${context.entry.key.replaceAll(' ','-')}-${s.mode}-analysis.csv`});a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function render(entry, profiles, socs, generated) {
    if(!settings.has(entry.key))settings.set(entry.key,{sorts:{},mode:entry.kind==='curve'?'targets':'groups',baseline:'',power:'4',score:entry.scoreDecimals===3?'2':'3000',wh:'20',brand:'',screen:'',screenMax:'',hz:''});
    const s=settings.get(entry.key);
    if(!profiles.some(p=>p.label===s.baseline))s.baseline=profiles[0]?.label||'';
    context={entry,profiles,socs,generated}; controlsFor(entry,profiles);draw();
  }
  return {render};
})();
