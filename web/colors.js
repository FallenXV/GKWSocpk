/* Spatial colour assignment.
 *
 * A profile's colour must not depend on what else is selected, or toggling one
 * device recolours the whole chart.  So the assignment is computed once per
 * figure over every profile in the dataset, and selection only decides what is
 * drawn.
 *
 * With more profiles than palette entries some colours have to repeat.  Which
 * ones repeat is chosen here: profiles that run close together on the figure
 * get colours that are far apart in OKLab, so a repeat lands between two marks
 * that are already far apart on screen.  Best effort — dense charts overlap
 * more than twelve colours can separate. */
'use strict';

const Colors = (() => {

const SAMPLES = 24;      // per profile, enough to shape a curve
const SWEEPS = 4;        // refinement passes after the greedy pass
// Weight is 1/(separation + FLOOR), so a pair that nearly touches outweighs
// many comfortable ones and the optimiser spends its colours where it counts.
// 0.06 measured best across the project's figures: it lifts the closest
// same-coloured pair above a naive round-robin on every one of them.
const FLOOR = 0.06;

/* ---------- perceptual colour distance ---------- */

function linearise(channel) {
  return channel <= 0.04045 ? channel / 12.92 : Math.pow((channel + 0.055) / 1.055, 2.4);
}

/** sRGB hex to OKLab, whose Euclidean distance tracks perceived difference. */
function oklab(hex) {
  const red = linearise(parseInt(hex.slice(1, 3), 16) / 255);
  const green = linearise(parseInt(hex.slice(3, 5), 16) / 255);
  const blue = linearise(parseInt(hex.slice(5, 7), 16) / 255);
  const long = Math.cbrt(0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue);
  const medium = Math.cbrt(0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue);
  const short = Math.cbrt(0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue);
  return [
    0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
    1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
    0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
  ];
}

/** Pairwise similarity in [0, 1]; 1 is the same colour. */
function similarityMatrix(palette) {
  const labs = palette.map(oklab);
  const distances = labs.map((left) => labs.map((right) => Math.hypot(
    left[0] - right[0], left[1] - right[1], left[2] - right[2])));
  const widest = Math.max(...distances.flat()) || 1;
  return distances.map((row) => row.map((distance) => 1 - distance / widest));
}

/* ---------- figure geometry ---------- */

function normalise(items, extent) {
  if (!extent) {
    const xs = items.flatMap((item) => item.points.map((point) => point[0])).filter(Number.isFinite);
    const ys = items.flatMap((item) => item.points.map((point) => point[1])).filter(Number.isFinite);
    extent = xs.length
      ? { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) }
      : { x0: 0, x1: 1, y0: 0, y1: 1 };
  }
  const spanX = (extent.x1 - extent.x0) || 1;
  const spanY = (extent.y1 - extent.y0) || 1;
  return items.map((item) => {
    const points = item.points.filter(
      (point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
    // Evenly thin a long curve; its shape survives, the pairwise cost does not.
    const step = Math.max(1, Math.ceil(points.length / SAMPLES));
    const kept = points.filter((_, index) => index % step === 0);
    if (points.length && kept[kept.length - 1] !== points[points.length - 1]) {
      kept.push(points[points.length - 1]);
    }
    return {
      id: item.id,
      kind: item.kind || 'samples',
      samples: kept.map((point) => [
        (point[0] - extent.x0) / spanX,
        (point[1] - extent.y0) / spanY,
      ]),
    };
  });
}

/** Mean distance from each of `from`'s samples to the nearest of `to`'s. */
function chamfer(from, to) {
  let total = 0;
  for (const a of from.samples) {
    let closest = Infinity;
    for (const b of to.samples) {
      const distance = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
      if (distance < closest) closest = distance;
    }
    total += Math.sqrt(closest);
  }
  return total / (from.samples.length || 1);
}

/**
 * How strongly two profiles compete for distinguishable colours.
 *
 * Closest approach is useless for curves, which cross each other constantly;
 * what matters is how closely they run overall, so this is a symmetric
 * Chamfer distance. For single points it is just the distance between them.
 */
function proximity(left, right) {
  if (!left.samples.length || !right.samples.length) return 0;
  let separation;
  if (left.kind === 'hline' || right.kind === 'hline') {
    // A horizontal rule occupies every x position. Its visual distance from a
    // point or curve is therefore vertical only; treating just its endpoints
    // as samples would miss marks that sit directly on the middle of it.
    const line = left.kind === 'hline' ? left : right;
    const other = left.kind === 'hline' ? right : left;
    const y = line.samples[0][1];
    separation = other.samples.reduce((sum, point) => sum + Math.abs(point[1] - y), 0)
      / other.samples.length;
  } else {
    separation = (chamfer(left, right) + chamfer(right, left)) / 2;
  }
  return 1 / (separation + FLOOR);
}

/* ---------- assignment ---------- */

/**
 * Map each item id to a palette colour.
 *
 * `items` are `{id, points}` in figure units; `extent` is the figure's data
 * range. Deterministic: the same input always yields the same mapping.
 */
function assign(items, palette, extent) {
  const colors = new Map();
  if (!items.length || !palette.length) return colors;

  const shaped = normalise(items, extent || null);
  const count = shaped.length;
  const near = Array.from({ length: count }, () => new Float64Array(count));
  for (let i = 0; i < count; i += 1) {
    for (let j = i + 1; j < count; j += 1) {
      near[i][j] = near[j][i] = proximity(shaped[i], shaped[j]);
    }
  }

  const similar = similarityMatrix(palette);
  const assigned = new Int32Array(count).fill(-1);
  const load = new Int32Array(palette.length);

  const cost = (index, choice) => {
    let total = 0;
    for (let other = 0; other < count; other += 1) {
      if (other === index || assigned[other] < 0) continue;
      total += near[index][other] * similar[choice][assigned[other]];
    }
    return total;
  };

  // Place the most crowded profiles first; they have the least freedom later.
  const order = shaped
    .map((item, index) => ({ index, crowding: near[index].reduce((sum, value) => sum + value, 0) }))
    .sort((left, right) => right.crowding - left.crowding
      || String(shaped[left.index].id).localeCompare(String(shaped[right.index].id)))
    .map((entry) => entry.index);

  for (const index of order) {
    let choice = -1;
    let lowest = Infinity;
    // Keep every palette entry within one use of every other. This exhausts
    // the palette before any colour repeats, while cost still decides which
    // of the currently least-used colours best separates this profile.
    const minimumLoad = Math.min(...load);
    for (let candidate = 0; candidate < palette.length; candidate += 1) {
      if (load[candidate] !== minimumLoad) continue;
      const score = cost(index, candidate);
      if (score < lowest - 1e-12) { lowest = score; choice = candidate; }
    }
    assigned[index] = choice < 0 ? 0 : choice;
    load[assigned[index]] += 1;
  }

  // Swapping two profiles' colours keeps the balance, so refinement only has
  // to decide whether the swap separates them better.
  const delta = (i, j) => {
    const ci = assigned[i];
    const cj = assigned[j];
    if (ci === cj) return 0;
    let change = 0;
    for (let k = 0; k < count; k += 1) {
      if (k === i || k === j) continue;
      const ck = assigned[k];
      change += near[i][k] * (similar[cj][ck] - similar[ci][ck])
        + near[j][k] * (similar[ci][ck] - similar[cj][ck]);
    }
    return change;
  };

  for (let sweep = 0; sweep < SWEEPS; sweep += 1) {
    let moved = false;
    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        if (delta(i, j) < -1e-12) {
          const swap = assigned[i];
          assigned[i] = assigned[j];
          assigned[j] = swap;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  shaped.forEach((item, index) => colors.set(item.id, palette[assigned[index]]));
  return colors;
}

return { assign, oklab, similarityMatrix };
})();
