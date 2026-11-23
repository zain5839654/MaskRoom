/* Runs U2-Net-small (exported to a plain layer list) on the CPU with no libraries.
   Layers used by the model: Conv, Relu, MaxPool, Resize (bilinear), Concat, Add, Sigmoid. */
'use strict';

/* 3x3 convolution, four output channels per pass so each input value is read once for 12 multiplies */
function conv3(x, Wt, wo, bo, Co, Ci, d) {
  var H = x.h, Wd = x.w, HW = H * Wd, xin = x.d, out = new Float32Array(Co * HW);
  var xs = d, xe = Wd - d, co = 0;
  for (; co + 3 < Co; co += 4) {
    var ob = co * HW, b0 = Wt[bo + co], b1 = Wt[bo + co + 1], b2 = Wt[bo + co + 2], b3 = Wt[bo + co + 3];
    for (var y = 0; y < H; y++) {
      var o0 = ob + y * Wd, o1 = o0 + HW, o2 = o1 + HW, o3 = o2 + HW, xx;
      for (xx = 0; xx < Wd; xx++) { out[o0 + xx] = b0; out[o1 + xx] = b1; out[o2 + xx] = b2; out[o3 + xx] = b3; }
      for (var ci = 0; ci < Ci; ci++) {
        var ib = ci * HW, q0 = wo + (co * Ci + ci) * 9, q1 = q0 + Ci * 9, q2 = q1 + Ci * 9, q3 = q2 + Ci * 9;
        for (var ky = 0; ky < 3; ky++) {
          var iy = y + (ky - 1) * d;
          if (iy < 0 || iy >= H) continue;
          var r = ib + iy * Wd, k = ky * 3;
          var a0 = Wt[q0 + k], a1 = Wt[q0 + k + 1], a2 = Wt[q0 + k + 2];
          var c0 = Wt[q1 + k], c1 = Wt[q1 + k + 1], c2 = Wt[q1 + k + 2];
          var e0 = Wt[q2 + k], e1 = Wt[q2 + k + 1], e2 = Wt[q2 + k + 2];
          var g0 = Wt[q3 + k], g1 = Wt[q3 + k + 1], g2 = Wt[q3 + k + 2];
          var p, pl, pr;
          if (xe < xs) {
            for (xx = 0; xx < Wd; xx++) {
              p = xin[r + xx]; pl = xx - d >= 0 ? xin[r + xx - d] : 0; pr = xx + d < Wd ? xin[r + xx + d] : 0;
              out[o0 + xx] += a0 * pl + a1 * p + a2 * pr;
              out[o1 + xx] += c0 * pl + c1 * p + c2 * pr;
              out[o2 + xx] += e0 * pl + e1 * p + e2 * pr;
              out[o3 + xx] += g0 * pl + g1 * p + g2 * pr;
            }
          } else {
            for (xx = 0; xx < xs; xx++) {
              p = xin[r + xx]; pr = xin[r + xx + d];
              out[o0 + xx] += a1 * p + a2 * pr;
              out[o1 + xx] += c1 * p + c2 * pr;
              out[o2 + xx] += e1 * p + e2 * pr;
              out[o3 + xx] += g1 * p + g2 * pr;
            }
            for (xx = xs; xx < xe; xx++) {
              pl = xin[r + xx - d]; p = xin[r + xx]; pr = xin[r + xx + d];
              out[o0 + xx] += a0 * pl + a1 * p + a2 * pr;
              out[o1 + xx] += c0 * pl + c1 * p + c2 * pr;
              out[o2 + xx] += e0 * pl + e1 * p + e2 * pr;
              out[o3 + xx] += g0 * pl + g1 * p + g2 * pr;
            }
            for (xx = xe; xx < Wd; xx++) {
              pl = xin[r + xx - d]; p = xin[r + xx];
              out[o0 + xx] += a0 * pl + a1 * p;
              out[o1 + xx] += c0 * pl + c1 * p;
              out[o2 + xx] += e0 * pl + e1 * p;
              out[o3 + xx] += g0 * pl + g1 * p;
            }
          }
        }
      }
    }
  }
  if (co < Co) conv3Rest(x, Wt, wo, bo, Co, Ci, d, out, co);
  return { d: out, c: Co, h: H, w: Wd };
}

function conv3Rest(x, Wt, wo, bo, Co, Ci, d, out, coStart) {
  var H = x.h, Wd = x.w, HW = H * Wd, xin = x.d;
  var xs = d, xe = Wd - d;
  for (var co = coStart; co < Co; co++) {
    var ob = co * HW, bias = Wt[bo + co];
    for (var y = 0; y < H; y++) {
      var orow = ob + y * Wd, xx;
      for (xx = 0; xx < Wd; xx++) out[orow + xx] = bias;
      for (var ci = 0; ci < Ci; ci++) {
        var wb = wo + (co * Ci + ci) * 9, ib = ci * HW;
        for (var ky = 0; ky < 3; ky++) {
          var iy = y + (ky - 1) * d;
          if (iy < 0 || iy >= H) continue;
          var r = ib + iy * Wd, w0 = Wt[wb + ky * 3], w1 = Wt[wb + ky * 3 + 1], w2 = Wt[wb + ky * 3 + 2], s;
          if (xe < xs) {
            for (xx = 0; xx < Wd; xx++) {
              s = w1 * xin[r + xx];
              if (xx - d >= 0) s += w0 * xin[r + xx - d];
              if (xx + d < Wd) s += w2 * xin[r + xx + d];
              out[orow + xx] += s;
            }
          } else {
            for (xx = 0; xx < xs; xx++) out[orow + xx] += w1 * xin[r + xx] + w2 * xin[r + xx + d];
            for (xx = xs; xx < xe; xx++) out[orow + xx] += w0 * xin[r + xx - d] + w1 * xin[r + xx] + w2 * xin[r + xx + d];
            for (xx = xe; xx < Wd; xx++) out[orow + xx] += w0 * xin[r + xx - d] + w1 * xin[r + xx];
          }
        }
      }
    }
  }
}

function convGeneric(x, Wt, wo, bo, Co, Ci, kh, kw, pads, dil) {
  var H = x.h, Wd = x.w, HW = H * Wd, xin = x.d, out = new Float32Array(Co * HW);
  var pt = pads[0], pl = pads[1], dy = dil[0], dx = dil[1];
  for (var co = 0; co < Co; co++) {
    for (var y = 0; y < H; y++) {
      for (var xx = 0; xx < Wd; xx++) {
        var s = Wt[bo + co];
        for (var ci = 0; ci < Ci; ci++) {
          for (var ky = 0; ky < kh; ky++) {
            var iy = y - pt + ky * dy;
            if (iy < 0 || iy >= H) continue;
            for (var kx = 0; kx < kw; kx++) {
              var ix = xx - pl + kx * dx;
              if (ix < 0 || ix >= Wd) continue;
              s += Wt[wo + ((co * Ci + ci) * kh + ky) * kw + kx] * xin[ci * HW + iy * Wd + ix];
            }
          }
        }
        out[co * HW + y * Wd + xx] = s;
      }
    }
  }
  return { d: out, c: Co, h: H, w: Wd };
}

function maxpool(x, k, s, ceil) {
  var H = x.h, Wd = x.w;
  var oh = ceil ? Math.ceil((H - k[0]) / s[0]) + 1 : Math.floor((H - k[0]) / s[0]) + 1;
  var ow = ceil ? Math.ceil((Wd - k[1]) / s[1]) + 1 : Math.floor((Wd - k[1]) / s[1]) + 1;
  var out = new Float32Array(x.c * oh * ow);
  for (var c = 0; c < x.c; c++) {
    for (var y = 0; y < oh; y++) {
      for (var xx = 0; xx < ow; xx++) {
        var m = -Infinity;
        for (var ky = 0; ky < k[0]; ky++) {
          var iy = y * s[0] + ky;
          if (iy >= H) continue;
          for (var kx = 0; kx < k[1]; kx++) {
            var ix = xx * s[1] + kx;
            if (ix >= Wd) continue;
            var v = x.d[c * H * Wd + iy * Wd + ix];
            if (v > m) m = v;
          }
        }
        out[c * oh * ow + y * ow + xx] = m;
      }
    }
  }
  return { d: out, c: x.c, h: oh, w: ow };
}

function resize(x, oh, ow) {
  var H = x.h, Wd = x.w, out = new Float32Array(x.c * oh * ow);
  var y0 = new Int32Array(oh), y1 = new Int32Array(oh), fy = new Float32Array(oh);
  var x0 = new Int32Array(ow), x1 = new Int32Array(ow), fx = new Float32Array(ow), i;
  for (i = 0; i < oh; i++) {
    var sy = oh > 1 ? (i + 0.5) * (H / oh) - 0.5 : 0;
    sy = sy < 0 ? 0 : sy > H - 1 ? H - 1 : sy;
    y0[i] = Math.floor(sy); y1[i] = Math.min(y0[i] + 1, H - 1); fy[i] = sy - y0[i];
  }
  for (i = 0; i < ow; i++) {
    var sx = ow > 1 ? (i + 0.5) * (Wd / ow) - 0.5 : 0;
    sx = sx < 0 ? 0 : sx > Wd - 1 ? Wd - 1 : sx;
    x0[i] = Math.floor(sx); x1[i] = Math.min(x0[i] + 1, Wd - 1); fx[i] = sx - x0[i];
  }
  for (var c = 0; c < x.c; c++) {
    var ib = c * H * Wd, ob = c * oh * ow;
    for (var y = 0; y < oh; y++) {
      var ra = ib + y0[y] * Wd, rb = ib + y1[y] * Wd, wy = fy[y];
      for (var xx = 0; xx < ow; xx++) {
        var a = x.d[ra + x0[xx]] * (1 - fx[xx]) + x.d[ra + x1[xx]] * fx[xx];
        var b = x.d[rb + x0[xx]] * (1 - fx[xx]) + x.d[rb + x1[xx]] * fx[xx];
        out[ob + y * ow + xx] = a * (1 - wy) + b * wy;
      }
    }
  }
  return { d: out, c: x.c, h: oh, w: ow };
}

/* input: Float32Array of 3*S*S, already normalised. Returns Float32Array(S*S) in 0..1. */
function runModel(graph, Wt, input, onProgress) {
  var S = graph.size, T = Object.create(null), last = Object.create(null), nodes = graph.nodes, i, j;
  T[graph.input] = { d: input, c: 3, h: S, w: S };
  for (i = 0; i < nodes.length; i++) for (j = 0; j < nodes[i].in.length; j++) last[nodes[i].in[j]] = i;
  for (j = 0; j < graph.outputs.length; j++) last[graph.outputs[j]] = Infinity;
  for (i = 0; i < nodes.length; i++) {
    var n = nodes[i], res, x = T[n.in[0]];
    if (n.op === 'Conv') {
      var wt = graph.tensors[n.w], bt = graph.tensors[n.b];
      var Co = wt.shape[0], Ci = wt.shape[1];
      if (n.k[0] === 3 && n.k[1] === 3 && n.dil[0] === n.dil[1] && n.pads[0] === n.dil[0] && n.pads[1] === n.dil[1] && n.pads[2] === n.dil[0] && n.pads[3] === n.dil[1]) {
        res = conv3(x, Wt, wt.off, bt.off, Co, Ci, n.dil[0]);
      } else {
        res = convGeneric(x, Wt, wt.off, bt.off, Co, Ci, n.k[0], n.k[1], n.pads, n.dil);
      }
    } else if (n.op === 'Relu') {
      var inPlace = last[n.in[0]] === i && x.d !== input;
      var rd = inPlace ? x.d : new Float32Array(x.d.length);
      for (j = 0; j < rd.length; j++) { var v = x.d[j]; rd[j] = v > 0 ? v : 0; }
      res = { d: rd, c: x.c, h: x.h, w: x.w };
    } else if (n.op === 'MaxPool') {
      res = maxpool(x, n.k, n.s, n.ceil);
    } else if (n.op === 'Resize') {
      res = resize(x, n.size[0], n.size[1]);
    } else if (n.op === 'Concat') {
      var parts = n.in.map(function (nm) { return T[nm]; }), tc = 0, p;
      for (p = 0; p < parts.length; p++) tc += parts[p].c;
      var cd = new Float32Array(tc * x.h * x.w), o = 0;
      for (p = 0; p < parts.length; p++) { cd.set(parts[p].d, o); o += parts[p].d.length; }
      res = { d: cd, c: tc, h: x.h, w: x.w };
    } else if (n.op === 'Add') {
      var y2 = T[n.in[1]], ad = last[n.in[0]] === i && x.d !== input ? x.d : new Float32Array(x.d.length);
      for (j = 0; j < ad.length; j++) ad[j] = x.d[j] + y2.d[j];
      res = { d: ad, c: x.c, h: x.h, w: x.w };
    } else if (n.op === 'Sigmoid') {
      var sd = new Float32Array(x.d.length);
      for (j = 0; j < sd.length; j++) sd[j] = 1 / (1 + Math.exp(-x.d[j]));
      res = { d: sd, c: x.c, h: x.h, w: x.w };
    } else {
      throw new Error('Unsupported layer: ' + n.op);
    }
    T[n.out[0]] = res;
    for (j = 0; j < n.in.length; j++) if (last[n.in[j]] === i) delete T[n.in[j]];
    if (onProgress && (i % 6 === 0 || i === nodes.length - 1)) onProgress((i + 1) / nodes.length);
  }
  return T[graph.outputs[0]].d;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { runModel: runModel };
}

/* web worker side */
if (typeof importScripts === 'function') {
  var cache = null;
  self.onmessage = function (e) {
    var msg = e.data;
    if (msg.type !== 'run') return;
    var load = cache ? Promise.resolve(cache) : Promise.all([
      fetch(msg.graphUrl).then(function (r) { if (!r.ok) throw new Error('graph ' + r.status); return r.json(); }),
      fetch(msg.weightsUrl).then(function (r) { if (!r.ok) throw new Error('weights ' + r.status); return r.text(); })
    ]).then(function (a) {
      /* weights are stored as base64 text because the host only serves text-like files */
      var bin = atob(a[1].trim()), u8 = new Uint8Array(bin.length);
      for (var k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
      cache = { g: a[0], w: new Float32Array(u8.buffer) };
      return cache;
    });
    load.then(function (m) {
      if (m.g.size !== msg.size) throw new Error('model size mismatch');
      var out = runModel(m.g, m.w, msg.input, function (f) { self.postMessage({ type: 'progress', frac: f }); });
      self.postMessage({ type: 'done', mask: out }, [out.buffer]);
    }).catch(function (err) {
      self.postMessage({ type: 'error', message: String(err && err.message || err) });
    });
  };
}
