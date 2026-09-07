import fastify from "fastify";
import cors from "@fastify/cors";
import fetch from "node-fetch";

const PORT = 3000;

const API_MD5 = "https://taixiumd5.maksh3979madfw.com/api/md5luckydice/GetSoiCau";
const API_HU = "https://taixiu.maksh3979madfw.com/api/luckydice/GetSoiCau";

const app = fastify({ logger: false });
await app.register(cors, { origin: "*" });

function fetchData(api) {
    return fetch(api, {
        headers: { "User-Agent": "Mozilla/5.0" }
    }).then(r => r.json());
}

function parseData(data) {
    if (!data || !data.length) return [];
    return data.map(item => ({
        session: item.SessionId || parseInt(item.sid),
        dice: [item.FirstDice || item.d1, item.SecondDice || item.d2, item.ThirdDice || item.d3],
        total: (item.FirstDice || item.d1) + (item.SecondDice || item.d2) + (item.ThirdDice || item.d3),
        result: ((item.FirstDice || item.d1) + (item.SecondDice || item.d2) + (item.ThirdDice || item.d3)) > 10 ? "Tài" : "Xỉu",
        tx: ((item.FirstDice || item.d1) + (item.SecondDice || item.d2) + (item.ThirdDice || item.d3)) > 10 ? "T" : "X"
    })).sort((a, b) => a.session - b.session);
}

function countIn(arr, val, n) {
    let c = 0, m = Math.min(n || arr.length, arr.length);
    for (let i = 0; i < m; i++) if (arr[i] === val) c++;
    return c;
}

function streak(arr) {
    if (!arr.length) return 0;
    let s = 1;
    for (let i = 1; i < arr.length; i++) {
        if (arr[i] === arr[i - 1]) s++;
        else break;
    }
    return s;
}

function bayesP(a, b) {
    return (a + 1) / (b + 2);
}

function average(nums) {
    if (!nums.length) return 0;
    return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function stddev(nums) {
    if (nums.length < 2) return 0;
    let mean = average(nums);
    let variance = average(nums.map(n => Math.pow(n - mean, 2)));
    return Math.sqrt(variance);
}

function entropy(arr) {
    if (!arr.length) return 0;
    let freq = {};
    for (let v of arr) freq[v] = (freq[v] || 0) + 1;
    let e = 0, n = arr.length;
    for (let k in freq) {
        let p = freq[k] / n;
        e -= p * Math.log2(p);
    }
    return e;
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

function markovProb(h, k) {
    let n = h.length;
    if (n <= k) return null;
    let ctx = h.slice(0, k).join('|');
    let ctxC = 0, tarC = 0;
    for (let i = 0; i < n - k; i++) {
        if (h.slice(i, i + k).join('|') === ctx) {
            ctxC++;
            if (h[i + k] === 'T') tarC++;
        }
    }
    if (ctxC < 2) return null;
    return bayesP(tarC, ctxC);
}

function msPatternMatch(h) {
    let n = Math.min(h.length, 200);
    if (n < 5) return null;
    let ps = [], ws = [];
    let maxW = Math.min(7, Math.max(3, Math.floor(n / 2)));
    for (let w = 3; w <= maxW; w++) {
        let ctx = h.slice(0, w).join('');
        let a = 0, b = 0;
        for (let i = 1; i <= n - w - 1; i++) {
            let c = h.slice(i, i + w).join('');
            let m = 0;
            for (let j = 0; j < w; j++) if (ctx[j] === c[j]) m++;
            if (m >= w - 1) {
                if (h[i + w] === 'T') a++;
                else b++;
            }
        }
        if (a + b >= 2) { ps.push(bayesP(a, b)); ws.push(a + b); }
    }
    if (!ps.length) return null;
    let sp = 0, sw = 0;
    for (let i = 0; i < ps.length; i++) { sp += ps[i] * ws[i]; sw += ws[i]; }
    return sw > 0 ? sp / sw : null;
}

function hmmTrain(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return null;
    let S = [];
    for (let i = 0; i < n; i++) {
        let s = Math.max(0, i - 5), e2 = Math.min(n, i + 6);
        let tC = 0;
        for (let j = s; j < e2; j++) if (h[j] === 'T') tC++;
        let p = tC / (e2 - s);
        S.push(p > 0.8 ? 1 : 0);
    }
    let A = [[0, 0], [0, 0]], B = [[0, 0], [0, 0]];
    for (let i = 0; i < n - 1; i++) {
        A[S[i]][S[i + 1]]++;
        B[S[i]][h[i] === 'T' ? 0 : 1]++;
    }
    B[S[n - 1]][h[n - 1] === 'T' ? 0 : 1]++;
    for (let s = 0; s < 2; s++) {
        let sA = A[s][0] + A[s][1];
        if (sA > 0) { A[s][0] /= sA; A[s][1] /= sA; } else { A[s][0] = 0.5; A[s][1] = 0.5; }
        let sB = B[s][0] + B[s][1];
        if (sB > 0) { B[s][0] /= sB; B[s][1] /= sB; } else { B[s][0] = 0.5; B[s][1] = 0.5; }
    }
    return { A, B, ls: S[n - 1] };
}

function hmmProb(m) {
    if (!m) return null;
    return m.A[m.ls][0] * m.B[0][0] + m.A[m.ls][1] * m.B[1][0];
}

function stumpEnsemble(h) {
    let n = Math.min(h.length, 200);
    if (n < 3) return 0.5;
    let lt = h[0] === 'T' ? 1 : 0;
    let s = streak(h);
    let tC = countIn(h, 'T', n);
    let bal = tC / n;
    let local = h.slice(0, Math.min(10, n));
    let tL = 0;
    for (let i = 0; i < local.length; i++) if (local[i] === 'T') tL++;
    let pL = tL / local.length;
    let ent = -(pL > 0.01 ? pL * Math.log2(pL) : 0) - ((1 - pL) > 0.01 ? (1 - pL) * Math.log2(1 - pL) : 0);
    let alt = 0;
    for (let i = 1; i < Math.min(n, 30); i++) if (h[i] !== h[i - 1]) alt++;
    let altR = alt / Math.min(n - 1, 29);
    let r = [
        lt === 1 ? 0.56 : 0.44,
        s >= 4 ? (lt === 1 ? 0.38 : 0.62) : (lt === 1 ? 0.52 : 0.48),
        bal > 0.54 ? 0.57 : bal < 0.46 ? 0.43 : 0.50,
        ent > 0.9 ? 0.50 : (lt === 1 ? 0.54 : 0.46),
        altR > 0.55 ? (lt === 1 ? 0.44 : 0.56) : (s >= 3 ? (lt === 1 ? 0.56 : 0.44) : 0.5),
        tL >= 6 ? 0.56 : tL <= 4 ? 0.44 : 0.5
    ];
    return r.reduce((a, b) => a + b, 0) / r.length;
}

function meanRevProb(h) {
    let n = Math.min(h.length, 200);
    let curS = streak(h);
    let curV = h[0];
    if (curS < 3) return curV === 'T' ? 0.52 : 0.48;
    let rev = 0, revTot = 0;
    for (let i = 0; i < n - 1; i++) {
        if (i + curS <= n) {
            let same = true;
            for (let j = 0; j < curS && i + j < n; j++) {
                if (h[i + j] !== h[i]) { same = false; break; }
            }
            if (same && i + curS < n) {
                revTot++;
                if (h[i + curS] !== h[i]) rev++;
            }
        }
    }
    if (revTot < 2) return curV === 'T' ? 0.45 : 0.55;
    return curV === 'T' ? 1 - rev / revTot : rev / revTot;
}

function detectRegime(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 'UNKNOWN';
    let tC = countIn(h, 'T', n);
    let bal = tC / n;
    let maxS = 1, cS = 1;
    for (let i = 1; i < n; i++) {
        if (h[i] === h[i - 1]) { cS++; if (cS > maxS) maxS = cS; } else cS = 1;
    }
    let alt = 0;
    for (let i = 1; i < n; i++) if (h[i] !== h[i - 1]) alt++;
    let altR = alt / (n - 1);
    let bOC = Math.abs(bal - 0.5);
    if (bOC > 0.15 && maxS >= 8) return 'STRONG_TREND';
    if (bOC > 0.10 && maxS >= 5) return 'TREND';
    if (bOC < 0.04 && maxS < 5 && altR > 0.35) return 'BALANCED';
    if (altR > 0.65) return 'HIGH_ENTROPY';
    if (altR < 0.15) return 'LOW_ENTROPY';
    return 'MIXED';
}

function adaptiveEWMA(h) {
    let n = Math.min(h.length, 200);
    if (n < 5) return 0.5;
    let ema = 0.5, emaVol = 0.2, alpha0 = 0.3, lambda = 2.0;
    for (let i = 0; i < n; i++) {
        let v = h[i] === 'T' ? 1 : 0;
        let a = alpha0 * Math.exp(-lambda * emaVol);
        a = Math.min(Math.max(a, 0.02), 0.5);
        ema = a * v + (1 - a) * ema;
        emaVol = 0.1 * Math.abs(v - ema) + 0.9 * emaVol;
    }
    return Math.min(Math.max(ema, 0.01), 0.99);
}

function cusumDetect(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let target = countIn(h, 'T', n) / n;
    let cusum = 0, maxCusum = 0, drift = 0.05;
    for (let i = 0; i < Math.min(n, 50); i++) {
        let val = h[i] === 'T' ? 1 : 0;
        cusum = Math.max(0, cusum + (val - target) - drift);
        if (cusum > maxCusum) maxCusum = cusum;
    }
    if (maxCusum > 2.5) {
        let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
        return bayesP(Math.round(recent * 20), 20);
    }
    let rec = countIn(h, 'T', Math.min(15, n)) / Math.min(15, n);
    return 0.3 + 0.4 * rec;
}

function slidingWinEnsemble(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let wins = [5, 10, 15, 20, 30, 50], ps = [], ws = [];
    for (let w of wins) {
        if (w > n) continue;
        let bias = countIn(h, 'T', w) / w;
        let errBias = Math.abs(bias - 0.6) > 0.05 ? bias : 0.5;
        ps.push(bayesP(Math.round(errBias * w), w));
        ws.push(Math.sqrt(w));
    }
    if (!ps.length) return 0.5;
    let sp = 0, sw = 0;
    for (let i = 0; i < ps.length; i++) { sp += ps[i] * ws[i]; sw += ws[i]; }
    return Math.min(Math.max(sp / sw, 0.01), 0.99);
}

function baggEnsemble(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let bags = 20, ps = [];
    for (let b = 0; b < bags; b++) {
        let a = 1, bCnt = 1;
        for (let i = 0; i < n; i++) {
            let idx = Math.floor(Math.random() * n);
            if (h[idx] === 'T') a++;
            else bCnt++;
        }
        ps.push((a - 1) / (a + bCnt - 2));
    }
    let sp = 0;
    for (let p of ps) sp += p;
    return Math.min(Math.max(sp / bags, 0.01), 0.99);
}

function boostPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let learners = 12, alpha = 1.0, probs = [];
    let weights = new Array(n).fill(1 / n);
    for (let l = 0; l < learners; l++) {
        let pred = [];
        for (let i = 0; i < n; i++) {
            let ctx = h.slice(Math.max(0, i - 2), i).join('');
            let aC = 0, bC = 0;
            for (let j = 0; j < n; j++) {
                if (j === i) continue;
                if (h.slice(Math.max(0, j - 2), j).join('') === ctx) {
                    if (h[j] === 'T') aC++;
                    else bC++;
                }
            }
            pred.push(aC + bC > 0 ? bayesP(aC, aC + bC) : 0.5);
        }
        let err = 0;
        for (let i = 0; i < n; i++) err += weights[i] * Math.abs(pred[i] - y[i]);
        if (err < 0.01 || err > 0.99) break;
        alpha = 0.5 * Math.log((1 - err) / Math.max(err, 0.001));
        for (let i = 0; i < n; i++) weights[i] *= Math.exp(-alpha * (2 * y[i] - 1) * (2 * pred[i] - 1));
        let wSum2 = weights.reduce((a, b) => a + b, 0);
        if (wSum2 > 0) for (let i = 0; i < n; i++) weights[i] /= wSum2;
        let lastW = 0;
        for (let i = Math.max(0, n - 5); i < n; i++) lastW += weights[i];
        probs.push(alpha * lastW / Math.min(5, n));
    }
    if (!probs.length) return 0.5;
    let sp = 0, sw = 0;
    for (let p of probs) { sp += p; sw++; }
    let prob = 0.5 + sp / sw * 0.3;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function cauBet(h) {
    let n = Math.min(h.length, 200);
    if (n < 4) return 0.5;
    let stream = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) stream++; else break; }
    let curV = h[0];
    let tC = countIn(h, 'T', n) / n;
    if (stream >= 8) return curV === 'T' ? Math.min(0.5 + tC * 0.5, 0.92) : Math.min(0.5 + (1 - tC) * 0.5, 0.92);
    if (stream >= 5) return curV === 'T' ? 0.5 + tC * 0.4 : 0.5 + (1 - tC) * 0.4;
    if (stream >= 3) return curV === 'T' ? 0.55 + tC * 0.2 : 0.55 + (1 - tC) * 0.2;
    if (stream >= 2) return curV === 'T' ? 0.52 : 0.48;
    return curV === 'T' ? 0.48 : 0.52;
}

function cau11(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    let alt = 0;
    for (let i = 1; i < Math.min(n, 30); i++) if (h[i] !== h[i - 1]) alt++;
    let altRate = alt / Math.min(n - 1, 29);
    if (altRate < 0.6) return null;
    let lastRun = 1;
    for (let i = 1; i < Math.min(n, 20); i++) { if (h[i] === h[i - 1]) lastRun++; else break; }
    if (lastRun >= 4) return { p: h[0] === 'T' ? 0.42 : 0.58 };
    if (lastRun === 1 && n >= 3) {
        if (h[0] !== h[1] && h[1] !== h[2]) {
            let pred = h[2] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.55 : 0.45 };
        }
        if (h[0] !== h[1]) {
            let pred = h[1] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.53 : 0.47 };
        }
        return null;
    }
    if (lastRun <= 3 && lastRun >= 2 && n > lastRun) {
        let streamV = h[0];
        if (countIn(h, streamV, lastRun) === lastRun) {
            let pred = streamV === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.5 + lastRun * 0.03 : 0.5 - lastRun * 0.03 };
        }
    }
    return null;
}

function cau3Nhip(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    for (let i = 0; i <= Math.min(3, n - 6); i++) {
        let a = h.slice(i, i + 6);
        if (a[0] !== a[1] && a[1] === a[2] && a[2] !== a[3]) {
            if (n >= i + 7) {
                let pred = a[3] === 'T' ? 'X' : 'T';
                return { p: pred === 'T' ? 0.6 : 0.4 };
            }
        }
        if (a[0] === a[1] && a[1] === a[2] && a[2] !== a[3] && a[3] === a[4] && a[4] !== a[5]) {
            if (n >= i + 7) {
                let pred = a[5] === 'T' ? 'X' : 'T';
                return { p: pred === 'T' ? 0.58 : 0.42 };
            }
        }
    }
    return null;
}

function cauDao(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS < 2) return null;
    let curV = h[0];
    let revCount = 0, revTotal = 0;
    for (let i = 1; i < n; i++) {
        if (h[i] !== h[i - 1]) {
            revTotal++;
            if (i + 1 < n && h[i + 1] === curV) revCount++;
        }
    }
    if (revTotal < 2) return null;
    let revRate = revCount / revTotal;
    let prob = curV === 'T' ? (0.5 - revRate * 0.3) : (0.5 + revRate * 0.3);
    return { p: Math.min(Math.max(prob, 0.01), 0.99) };
}

function cauTong(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let tC = countIn(h, 'T', n);
    let tR = tC / n;
    if (tR > 0.6) return { p: Math.min(tR + 0.05, 0.88) };
    if (tR < 0.4) return { p: Math.min(1 - tR + 0.05, 0.88) };
    let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
    let recentBias = countIn(h, 'T', Math.min(15, n)) / Math.min(15, n);
    let diff = recentBias - tR;
    if (Math.abs(diff) > 0.1) {
        let prob = recentBias + diff * 0.3;
        return { p: Math.min(Math.max(prob, 0.01), 0.99) };
    }
    return null;
}

function cauCat(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 2) return null;
    let totalCat = 0, afterCat = 0;
    for (let i = curS; i < n - 1; i++) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        if (s >= curS - 1 && i + s < n) {
            totalCat++;
            if (h[i + s] !== h[i + s - 1]) afterCat++;
        }
        i += Math.max(1, s - 1);
    }
    if (totalCat < 2) return null;
    let catRate = afterCat / totalCat;
    let prob = curV === 'T' ? 0.5 - catRate * 0.2 : 0.5 + catRate * 0.2;
    return { p: Math.min(Math.max(prob, 0.01), 0.99) };
}

function cauRongHo(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS >= 5) {
        let tC = countIn(h, 'T', n) / n;
        if (curV === 'T' && tC > 0.55) return { p: Math.min(tC + 0.1, 0.92) };
        if (curV === 'X' && tC < 0.45) return { p: Math.min(1 - tC + 0.1, 0.92) };
    }
    if (curS >= 3) {
        return { p: curV === 'T' ? 0.58 : 0.42 };
    }
    return null;
}

function cau12(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    let p = [];
    for (let i = 0; i <= n - 6; i++) {
        let pat = h.slice(i, i + 6);
        if (pat[0] !== pat[1] && pat[1] === pat[2] && pat[2] !== pat[3]) {
            p.push({ pred: pat[3] === 'T' ? 'X' : 'T', conf: 0.55 });
        }
        if (pat[0] === pat[1] && pat[1] !== pat[2] && pat[2] !== pat[3]) {
            p.push({ pred: pat[3] === 'T' ? 'X' : 'T', conf: 0.53 });
        }
        if (pat[0] !== pat[1] && pat[1] !== pat[2] && pat[2] === pat[3]) {
            p.push({ pred: pat[3] === 'T' ? 'X' : 'T', conf: 0.52 });
        }
    }
    if (!p.length) return null;
    let best = p.reduce((a, b) => a.conf > b.conf ? a : b);
    return { p: best.pred === 'T' ? best.conf : 1 - best.conf };
}

function cau212(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    for (let i = 0; i <= n - 8; i++) {
        let pat = h.slice(i, i + 8);
        if (pat[0] === pat[2] && pat[2] !== pat[1] && pat[3] === pat[5] && pat[5] !== pat[4] && pat[6] === pat[7]) {
            let pred = pat[7] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.58 : 0.42 };
        }
        if (pat[0] !== pat[1] && pat[1] === pat[2] && pat[2] !== pat[3] && pat[3] !== pat[4] && pat[4] === pat[5] && pat[5] !== pat[6] && pat[6] !== pat[7]) {
            let pred = pat[7] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.56 : 0.44 };
        }
    }
    return null;
}

function cau33(h) {
    let n = Math.min(h.length, 200);
    if (n < 9) return null;
    for (let i = 0; i <= n - 9; i++) {
        let pat = h.slice(i, i + 9);
        let r1 = pat[0] === pat[1] && pat[1] === pat[2];
        let r2 = pat[3] === pat[4] && pat[4] === pat[5];
        let r3 = pat[6] === pat[7] && pat[7] === pat[8];
        if (r1 && r2 && r3 && pat[0] !== pat[3] && pat[3] !== pat[6]) {
            let pred = pat[8] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.6 : 0.4 };
        }
        if (r1 && r2 && pat[0] !== pat[3] && pat[3] !== pat[6] && pat[6] === pat[7] && pat[7] === pat[8]) {
            let pred = pat[8] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.57 : 0.43 };
        }
    }
    return null;
}

function cau42(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    for (let i = 0; i <= n - 8; i++) {
        let pat = h.slice(i, i + 8);
        if (pat[0] === pat[1] && pat[1] === pat[2] && pat[2] === pat[3] && pat[4] !== pat[5] && pat[5] === pat[6] && pat[6] === pat[7]) {
            let pred = pat[7] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.62 : 0.38 };
        }
        if (pat[0] === pat[1] && pat[1] === pat[2] && pat[2] === pat[3] && pat[3] !== pat[4] && pat[4] === pat[5] && pat[5] === pat[6] && pat[6] === pat[7]) {
            let pred = pat[7] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.65 : 0.35 };
        }
    }
    return null;
}

function cauDao1122(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    for (let i = 0; i <= n - 9; i++) {
        let pat = h.slice(i, i + 9);
        if (pat[0] !== pat[1] && pat[1] !== pat[2] && pat[2] === pat[3] && pat[3] === pat[4] && pat[4] !== pat[5] && pat[5] !== pat[6] && pat[6] === pat[7] && pat[7] === pat[8]) {
            let pred = pat[8] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.55 : 0.45 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS >= 2) {
        let afterStream = 0, totalAfter = 0;
        for (let i = curS; i < n - 1; i++) {
            let s = 1;
            while (i + s < n && h[i + s] === h[i + s - 1]) s++;
            if (s >= 2 && i + s < n) { totalAfter++; if (h[i + s] !== curV) afterStream++; }
        }
        if (totalAfter >= 2) {
            let pRev = afterStream / totalAfter;
            return { p: curV === 'T' ? 0.5 - pRev * 0.3 : 0.5 + pRev * 0.3 };
        }
    }
    return null;
}

function cauGay(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return 0.5;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS >= 2 && curS <= 5) {
        let totalBreak = 0, breakContinue = 0;
        for (let i = 1; i < n - 1; i++) {
            let s = 1;
            while (i + s < n && h[i + s] === h[i + s - 1]) s++;
            if (s === curS && i + s < n) {
                totalBreak++;
                if (h[i + s] === h[i + s - 1]) breakContinue++;
                else { i += s + 1; }
            }
        }
        if (totalBreak >= 2) {
            let breakRate = breakContinue / totalBreak;
            if (breakRate > 0.6) {
                return curV === 'T' ? Math.min(0.55 + breakRate * 0.15, 0.85) : Math.min(0.55 + (1 - breakRate) * 0.15, 0.85);
            }
        }
    }
    if (curS >= 6) {
        let recent5 = h.slice(1, Math.min(6, n));
        let alt = 0;
        for (let i = 1; i < recent5.length; i++) if (recent5[i] !== recent5[i - 1]) alt++;
        if (alt >= 2) {
            return curV === 'T' ? 0.55 : 0.45;
        }
    }
    return 0.5;
}

function cauThong(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 2) return null;
    let match = 0, total = 0;
    for (let i = 0; i < n - curS; i += 1) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        if (s >= curS + 2 && i + s < n - 1) { total++; if (h[i + s] === curV) match++; }
        i += Math.max(1, s - 1);
    }
    if (total < 2) return null;
    let contRate = match / total;
    let prob = curV === 'T' ? 0.5 + (contRate - 0.5) * 0.4 : 0.5 - (contRate - 0.5) * 0.4;
    return { p: Math.min(Math.max(prob, 0.01), 0.99) };
}

function cau2Nhip(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    let best = null;
    for (let i = 0; i <= n - 6; i++) {
        let pat = h.slice(i, i + 6);
        if (pat[0] === pat[1] && pat[2] === pat[3] && pat[4] === pat[5] && pat[0] !== pat[2] && pat[2] !== pat[4]) {
            let pred = pat[5] === 'T' ? 'X' : 'T';
            let pPred = pred === 'T' ? 0.6 : 0.4;
            if (!best || pPred > Math.abs(best.p - 0.5)) best = { p: pPred };
        }
        if (pat[0] !== pat[1] && pat[2] !== pat[3] && pat[4] !== pat[5] && pat[0] === pat[2] && pat[2] === pat[4]) {
            let pred = pat[5] === 'T' ? 'X' : 'T';
            let pPred = pred === 'T' ? 0.58 : 0.42;
            if (!best || pPred > Math.abs(best.p - 0.5)) best = { p: pPred };
        }
    }
    if (best && best.p) {
        let last4 = h.slice(0, 4).join('');
        if (last4[0] === last4[1] && last4[2] === last4[3] && last4[0] !== last4[2]) {
            return best;
        }
    }
    return null;
}

function cau4Nhip(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    for (let i = 0; i <= n - 10; i++) {
        let pat = h.slice(i, i + 10);
        if (pat[0] !== pat[1] && pat[1] === pat[2] && pat[2] !== pat[3] && pat[3] !== pat[4] && pat[4] === pat[5] && pat[5] !== pat[6] && pat[6] !== pat[7] && pat[7] === pat[8] && pat[8] !== pat[9]) {
            let pred = pat[9] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.54 : 0.46 };
        }
    }
    return null;
}

function cau3_2(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    for (let i = 0; i <= n - 10; i++) {
        let pat = h.slice(i, i + 10);
        let r1 = pat[0] === pat[1] && pat[1] === pat[2];
        let r2 = pat[3] === pat[4];
        let r3 = pat[5] === pat[6] && pat[6] === pat[7];
        let r4 = pat[8] === pat[9];
        if (r1 && r2 && pat[0] !== pat[3] && pat[3] === pat[4] && pat[4] !== pat[5] && r3 && pat[5] !== pat[8] && r4) {
            return { p: pat[9] === 'T' ? 0.58 : 0.42 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS === 3) {
        let after32 = 0, total32 = 0;
        for (let i = curS; i < n - 2; i++) {
            if (h.slice(i, i + 2).every(v => v !== curV) && h.slice(i + 2, i + 5).every(v => v === curV)) { total32++; if (h[i + 4] === h[i + 2]) after32++; }
        }
        if (total32 >= 2) {
            let rate = after32 / total32;
            return { p: curV === 'T' ? 0.5 + rate * 0.15 : 0.5 - rate * 0.15 };
        }
    }
    if (curS === 2) {
        let after23 = 0, total23 = 0;
        for (let i = curS; i < n - 3; i++) {
            if (h.slice(i, i + 3).every(v => v !== curV) && h.slice(i + 3, i + 5).every(v => v === curV)) { total23++; if (h[i + 4] === h[i + 2]) after23++; }
        }
        if (total23 >= 2) {
            let rate = after23 / total23;
            return { p: curV === 'T' ? 0.5 + rate * 0.12 : 0.5 - rate * 0.12 };
        }
    }
    return null;
}

function cau1_2_3(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return null;
    for (let i = 0; i <= n - 12; i++) {
        let pat = h.slice(i, i + 12);
        if (pat[0] !== pat[1] && pat[1] === pat[2] && pat[2] !== pat[3] && pat[3] === pat[4] && pat[4] === pat[5] && pat[5] !== pat[6] && pat[6] === pat[7] && pat[7] === pat[8] && pat[8] === pat[9]) {
            let pred = pat[9] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.6 : 0.4 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS >= 1 && curS <= 3) {
        let streamLens = [];
        let pos = 0;
        while (pos < Math.min(n, 20)) {
            let s = 1;
            for (let j = pos + 1; j < Math.min(n, 20); j++) { if (h[j] === h[j - 1]) s++; else break; }
            streamLens.push(s);
            pos += s;
            if (streamLens.length >= 3) break;
        }
        if (streamLens.length >= 3 && streamLens[0] < streamLens[1] && streamLens[1] < streamLens[2]) {
            return { p: curV === 'T' ? 0.46 : 0.54 };
        }
    }
    return null;
}

function cauBac(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    let totalBac = 0, sameBac = 0;
    for (let i = curS; i < n - 1; i++) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        if (s === 1 && i + 1 < n) {
            totalBac++;
            if (h[i + 1] === curV) sameBac++;
        }
        i += Math.max(1, s - 1);
    }
    if (totalBac < 2) return null;
    let bacRate = sameBac / totalBac;
    let prob = curV === 'T' ? 0.5 - bacRate * 0.25 : 0.5 + bacRate * 0.25;
    return { p: Math.min(Math.max(prob, 0.01), 0.99) };
}

function cau112(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 1) return null;
    let total112 = 0, match112 = 0;
    for (let i = curS; i < n - 3; i++) {
        if (h.slice(i, i + 4).join('') === curV + (curV === 'T' ? 'X' : 'T') + (curV === 'T' ? 'X' : 'T')) {
            total112++;
            if (i + 4 < n && h[i + 4] === curV) match112++;
        }
        i++;
    }
    if (total112 < 2) return null;
    let rate = match112 / total112;
    return { p: curV === 'T' ? 0.5 + rate * 0.15 : 0.5 - rate * 0.15 };
}

function cau221(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 2) return null;
    let total221 = 0, match221 = 0;
    for (let i = curS; i < n - 4; i++) {
        if (h.slice(i, i + 5).join('') === curV + curV + (curV === 'T' ? 'X' : 'T') + (curV === 'T' ? 'X' : 'T') + curV) {
            total221++;
            if (i + 5 < n && h[i + 5] === (curV === 'T' ? 'X' : 'T')) match221++;
        }
        i++;
    }
    if (total221 < 2) return null;
    let rate = match221 / total221;
    return { p: curV === 'T' ? 0.5 - rate * 0.15 : 0.5 + rate * 0.15 };
}

function cauDao22(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let total22 = 0, match22 = 0;
    for (let i = 0; i <= n - 5; i++) {
        let pat = h.slice(i, i + 4).join('');
        if (pat === 'TTXX' || pat === 'XXTT') {
            total22++;
            if (h[i + 4] === h[i]) match22++;
        }
        i++;
    }
    if (total22 < 2) return null;
    let rate = match22 / total22;
    let lastP = h.slice(0, 4).join('');
    let isTX = lastP === 'TTXX' || lastP === 'XXTT';
    if (!isTX) return { p: 0.5 };
    return { p: h[0] === 'T' ? 0.5 + rate * 0.12 : 0.5 - rate * 0.12 };
}

function cauCham(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 2) return null;
    let chamCount = 0, matchCham = 0;
    for (let i = curS; i < n - 2; i++) {
        if (h[i] !== h[i - 1] && h[i] !== h[i + 1]) {
            chamCount++;
            if (chamCount > 1 && h[i] === curV) matchCham++;
        }
    }
    if (chamCount < 3) return null;
    let rate = matchCham / (chamCount || 1);
    return { p: curV === 'T' ? 0.5 + rate * 0.15 : 0.5 - rate * 0.15 };
}

function cauKep(h) {
    let n = Math.min(h.length, 200);
    if (n < 6) return null;
    for (let i = 0; i <= n - 5; i++) {
        let p = h.slice(i, i + 5).join('');
        if (p[0] === p[1] && p[3] === p[4] && p[1] !== p[2] && p[2] !== p[3] && p[1] === p[3]) {
            return { p: p[4] === 'T' ? 0.58 : 0.42 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS >= 3) {
        return { p: h[0] === 'T' ? 0.55 : 0.45 };
    }
    return null;
}

function cauPhanXa(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curV = h[0];
    let totalPX = 0, matchPX = 0;
    for (let len = 2; len <= 5; len++) {
        for (let i = 1; i <= Math.min(n - len - 1, 40); i++) {
            let pa = h.slice(0, len).join('');
            let pb = h.slice(i, i + len).join('');
            if (pa === pb) {
                totalPX++;
                if (i + len < n && h[i + len] === (curV === 'T' ? 'X' : 'T')) matchPX++;
            }
        }
    }
    if (totalPX < 3) return null;
    let rate = matchPX / totalPX;
    return { p: curV === 'T' ? 0.5 - rate * 0.2 : 0.5 + rate * 0.2 };
}

function cauLoRoi(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS < 3) return null;
    let totalR = 0, matchR = 0;
    for (let i = curS; i < n - 2; i++) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        if (s >= curS && i + s < n) {
            totalR++;
            if (h[i + s] !== h[i]) matchR++;
        }
        i += s;
    }
    if (totalR < 2) return null;
    let rate = matchR / totalR;
    return { p: h[0] === 'T' ? 0.5 - rate * 0.18 : 0.5 + rate * 0.18 };
}

function cauSongHanh(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return null;
    let pairs = [];
    for (let i = 0; i <= n - 4; i += 2) {
        let p = h.slice(i, i + 2).join('');
        pairs.push(p);
    }
    if (pairs.length < 3) return null;
    let match = 0, total = 0;
    for (let i = 2; i < pairs.length; i++) {
        if (pairs[i - 2] === pairs[i - 1]) {
            total++;
            if (pairs[i] === pairs[i - 1]) match++;
        }
    }
    if (total < 2) return { p: 0.5 };
    let rate = match / total;
    let lastPair = pairs[pairs.length - 1];
    return { p: lastPair === 'TT' || lastPair === 'XX' ? 0.5 + rate * 0.12 : 0.5 - rate * 0.12 };
}

function cauGiaoNhau(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    let gnCount = 0, matchGN = 0;
    for (let i = curS; i < n - 2; i++) {
        let s = streak(h.slice(i));
        if (s === 1 && i + 2 < n && h[i + 1] === curV && h[i + 2] === (curV === 'T' ? 'X' : 'T')) {
            gnCount++;
            if (i + 3 < n && h[i + 3] === curV) matchGN++;
        }
    }
    if (gnCount < 2) return null;
    let rate = matchGN / gnCount;
    return { p: curV === 'T' ? 0.5 + rate * 0.12 : 0.5 - rate * 0.12 };
}

function cauBet12(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    if (curS < 1 || curS > 2) return null;
    let total = 0, match = 0;
    for (let i = curS; i < n - 2; i++) {
        let s = streak(h.slice(i));
        if (s === 2 && h[i] === curV) {
            total++;
            if (i + 2 < n && h[i + 2] === (curV === 'T' ? 'X' : 'T')) match++;
        }
        i += Math.max(1, s - 1);
    }
    if (total < 2) return null;
    let rate = match / total;
    return { p: curV === 'T' ? 0.5 + rate * 0.13 : 0.5 - rate * 0.13 };
}

function cauXien22(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let totalX = 0, matchX = 0;
    for (let i = 0; i <= n - 6; i += 2) {
        let p = h.slice(i, i + 4).join('');
        if (p[0] === p[1] && p[2] === p[3] && p[0] !== p[2]) {
            totalX++;
            if (i + 4 < n && h[i + 4] === h[i + 2]) matchX++;
        }
    }
    if (totalX < 2) return null;
    let rate = matchX / totalX;
    let last4 = h.slice(0, 4).join('');
    if (!(last4[0] === last4[1] && last4[2] === last4[3] && last4[0] !== last4[2])) return { p: 0.5 };
    return { p: h[0] === 'T' ? 0.5 + rate * 0.1 : 0.5 - rate * 0.1 };
}

function cau331(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    for (let i = 0; i <= n - 9; i++) {
        let p = h.slice(i, i + 7).join('');
        if (p[0] === p[1] && p[1] === p[2] && p[3] === p[4] && p[4] === p[5] && p[2] !== p[3] && p[5] === p[6]) {
            return { p: p[6] === 'T' ? 0.55 : 0.45 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS === 3) {
        let total33 = 0, match33 = 0;
        for (let i = curS; i < n - 4; i++) {
            let s = streak(h.slice(i));
            if (s === 3 && h[i] !== h[0]) {
                total33++;
                if (i + 3 < n && h[i + 3] === h[0]) match33++;
            }
            i += s;
        }
        if (total33 >= 2) {
            let r = match33 / total33;
            return { p: h[0] === 'T' ? 0.5 + r * 0.12 : 0.5 - r * 0.12 };
        }
    }
    return null;
}

function cau133(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    for (let i = 0; i <= n - 9; i++) {
        let p = h.slice(i, i + 7).join('');
        if (p[0] === p[1] && p[2] === p[3] && p[3] === p[4] && p[4] === p[5] && p[0] !== p[2] && p[5] === p[6]) {
            return { p: p[6] === 'T' ? 0.55 : 0.45 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS === 1) {
        let total13 = 0, match13 = 0;
        for (let i = 1; i < n - 5; i++) {
            let s = streak(h.slice(i));
            if (s === 3 && h[i] !== h[0]) {
                total13++;
                if (i + 3 < n && h[i + 3] === h[0]) match13++;
            }
            i += s;
        }
        if (total13 >= 2) {
            let r = match13 / total13;
            return { p: h[0] === 'T' ? 0.5 + r * 0.15 : 0.5 - r * 0.15 };
        }
    }
    return null;
}

function cauNhayCoc(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    let skipCount = 0, matchSkip = 0;
    for (let i = curS; i < n - 2; i += 2) {
        if (h[i] === curV) {
            skipCount++;
            if (i + 1 < n && h[i + 1] === (curV === 'T' ? 'X' : 'T')) matchSkip++;
        }
    }
    if (skipCount < 2) return null;
    let rate = matchSkip / skipCount;
    return { p: curV === 'T' ? 0.5 + rate * 0.12 : 0.5 - rate * 0.12 };
}

function cau421(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return null;
    for (let i = 0; i <= n - 10; i++) {
        let p = h.slice(i, i + 7).join('');
        if (p[0] === p[1] && p[1] === p[2] && p[2] === p[3] && p[4] === p[5] && p[3] !== p[4] && p[5] === p[6]) {
            return { p: p[6] === 'T' ? 0.54 : 0.46 };
        }
    }
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    if (curS === 4) {
        let total = 0, match = 0;
        for (let i = curS; i < n - 4; i++) {
            let s = streak(h.slice(i));
            if (s === 2 && h[i] !== h[0]) {
                total++;
                if (i + 2 < n && h[i + 2] === (h[0] === 'T' ? 'X' : 'T')) match++;
            }
            i += s;
        }
        if (total >= 2) {
            let r = match / total;
            return { p: h[0] === 'T' ? 0.5 + r * 0.12 : 0.5 - r * 0.12 };
        }
    }
    return null;
}

function cauDoiXung(h) {
    let n = Math.min(h.length, 200);
    if (n < 14) return null;
    let symCount = 0, matchSym = 0;
    let half = Math.min(6, Math.floor(n / 2));
    for (let i = half; i < n - half - 2; i++) {
        let sym = true;
        for (let j = 0; j < half; j++) {
            if (h[i - j - 1] !== h[i + j]) { sym = false; break; }
        }
        if (sym && i + half < n) {
            symCount++;
            let next = h[i + half];
            let mirror = h[i - half];
            if (next === mirror) matchSym++;
        }
    }
    if (symCount < 2) return null;
    let rate = matchSym / symCount;
    return { p: rate > 0.55 ? 0.52 : 0.48 };
}

function cauXenKe(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return null;
    let xkCount = 0, matchXK = 0;
    for (let i = 0; i <= n - 5; i++) {
        let p = h.slice(i, i + 4).join('');
        if (p[0] !== p[1] && p[1] !== p[2] && p[2] !== p[3] && p[0] === p[2] && p[1] === p[3]) {
            xkCount++;
            if (i + 4 < n && h[i + 4] === p[1]) matchXK++;
        }
    }
    if (xkCount < 2) return null;
    let rate = matchXK / xkCount;
    let lastP = h.slice(0, 4).join('');
    if (!(lastP[0] !== lastP[1] && lastP[1] !== lastP[2] && lastP[2] !== lastP[3] && lastP[0] === lastP[2] && lastP[1] === lastP[3])) {
        return { p: 0.48 };
    }
    return { p: h[0] === 'T' ? 0.5 + rate * 0.1 : 0.5 - rate * 0.1 };
}

function cauThep(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let curS = 1;
    for (let i = 1; i < n; i++) { if (h[i] === h[i - 1]) curS++; else break; }
    let curV = h[0];
    let tpCount = 0, matchTP = 0;
    for (let i = curS; i < n - 2; i++) {
        if (h[i] === (curV === 'T' ? 'X' : 'T') && h[i + 1] === curV && h[i + 2] === (curV === 'T' ? 'X' : 'T')) {
            tpCount++;
            if (i + 3 < n && h[i + 3] === curV) matchTP++;
        }
        i += 2;
    }
    if (tpCount < 2) return null;
    let rate = matchTP / tpCount;
    return { p: curV === 'T' ? 0.5 + rate * 0.11 : 0.5 - rate * 0.11 };
}

function cauNhipTang(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    for (let j = 0; j <= runs.length - 4; j++) {
        let r0 = runs[j], r1 = runs[j + 1], r2 = runs[j + 2], r3 = runs[j + 3];
        if (r0.v !== r1.v && r1.v !== r2.v && r2.v !== r3.v) continue;
        if (r0.len < r1.len && r1.len < r2.len && r2.len < r3.len) {
            let isUp = r0.v === 'T';
            let nextRun = runs[j + 4];
            if (nextRun) {
                let match = nextRun.v === r0.v;
                return { p: isUp ? (match ? 0.56 : 0.44) : (match ? 0.44 : 0.56) };
            }
        }
        if (r0.len > r1.len && r1.len > r2.len && r2.len > r3.len) {
            let isDown = r0.v === 'T';
            let nextRun = runs[j + 4];
            if (nextRun) {
                let match = nextRun.v !== r0.v;
                return { p: isDown ? (match ? 0.56 : 0.44) : (match ? 0.44 : 0.56) };
            }
        }
    }
    return null;
}

function cau343(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return null;
    for (let i = 0; i <= n - 10; i++) {
        let p = h.slice(i, i + 10).join('');
        if (p[0] === p[1] && p[1] === p[2] && p[3] === p[4] && p[4] === p[5] && p[6] === p[7] && p[7] === p[8] && p[8] === p[9] && p[2] !== p[3] && p[5] !== p[6]) {
            let pred = h[9] === 'T' ? 'X' : 'T';
            return { p: pred === 'T' ? 0.56 : 0.44 };
        }
    }
    return null;
}

function cauXienCheo(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return null;
    let xcCount = 0, matchXC = 0;
    for (let i = 0; i <= n - 4; i++) {
        let p = h.slice(i, i + 4).join('');
        if (p[0] === p[2] && p[1] === p[3] && p[0] !== p[1]) {
            xcCount++;
            if (i + 4 < n && h[i + 4] === p[1]) matchXC++;
        }
    }
    if (xcCount < 3) return null;
    let rate = matchXC / xcCount;
    return { p: rate > 0.55 ? 0.52 : 0.48 };
}

function lstmGatedPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let hDim = 8;
    let hState = new Array(hDim).fill(0), cState = new Array(hDim).fill(0);
    let Wf = [], Wi = [], Wo = [], Wc = [];
    for (let i = 0; i < hDim; i++) { Wf[i] = Math.random() * 0.1 - 0.05; Wi[i] = Math.random() * 0.1 - 0.05; Wo[i] = Math.random() * 0.1 - 0.05; Wc[i] = Math.random() * 0.1 - 0.05; }
    for (let t = 0; t < n; t++) {
        let x = y[t] * 2 - 1;
        let f = sigmoid(Wf.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let iGate = sigmoid(Wi.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let o = sigmoid(Wo.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let c = tanh(Wc.reduce((s, w, i) => s + w * x + cState[i] * 0.5, 0));
        cState = cState.map((cs, i) => f * cs + iGate * c);
        hState = cState.map(cs => o * tanh(cs));
    }
    let out = 0;
    for (let i = 0; i < hDim; i++) out += hState[i];
    let prob = sigmoid(out / hDim);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.35 + recent * 0.65, 0.01), 0.99);
}

function bilstmPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let hDim = 6;
    let hFw = new Array(hDim).fill(0), cFw = new Array(hDim).fill(0);
    let hBw = new Array(hDim).fill(0), cBw = new Array(hDim).fill(0);
    let Wf = [], Wi = [], Wo = [], Wc = [];
    for (let i = 0; i < hDim; i++) { Wf[i] = Math.random() * 0.1 - 0.05; Wi[i] = Math.random() * 0.1 - 0.05; Wo[i] = Math.random() * 0.1 - 0.05; Wc[i] = Math.random() * 0.1 - 0.05; }
    for (let t = 0; t < n; t++) {
        let x = y[t] * 2 - 1;
        let f = sigmoid(Wf.reduce((s, w, i) => s + w * x + hFw[i] * 0.5 + 0.1, 0));
        let iGate = sigmoid(Wi.reduce((s, w, i) => s + w * x + hFw[i] * 0.5 + 0.1, 0));
        let o = sigmoid(Wo.reduce((s, w, i) => s + w * x + hFw[i] * 0.5 + 0.1, 0));
        let c = tanh(Wc.reduce((s, w, i) => s + w * x + cFw[i] * 0.5, 0));
        cFw = cFw.map((cs, i) => f * cs + iGate * c);
        hFw = cFw.map(cs => o * tanh(cs));
    }
    for (let t = n - 1; t >= 0; t--) {
        let x = y[t] * 2 - 1;
        let f = sigmoid(Wf.reduce((s, w, i) => s + w * x + hBw[i] * 0.5 + 0.1, 0));
        let iGate = sigmoid(Wi.reduce((s, w, i) => s + w * x + hBw[i] * 0.5 + 0.1, 0));
        let o = sigmoid(Wo.reduce((s, w, i) => s + w * x + hBw[i] * 0.5 + 0.1, 0));
        let c = tanh(Wc.reduce((s, w, i) => s + w * x + cBw[i] * 0.5, 0));
        cBw = cBw.map((cs, i) => f * cs + iGate * c);
        hBw = cBw.map(cs => o * tanh(cs));
    }
    let out = 0;
    for (let i = 0; i < hDim; i++) out += hFw[i] + hBw[i];
    let prob = sigmoid(out / (2 * hDim));
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.35 + recent * 0.65, 0.01), 0.99);
}

function gruGatedPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let hDim = 8;
    let hState = new Array(hDim).fill(0);
    let Wz = [], Wr = [], Wh = [];
    for (let i = 0; i < hDim; i++) { Wz[i] = Math.random() * 0.1 - 0.05; Wr[i] = Math.random() * 0.1 - 0.05; Wh[i] = Math.random() * 0.1 - 0.05; }
    for (let t = 0; t < n; t++) {
        let x = y[t] * 2 - 1;
        let z = sigmoid(Wz.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.05, 0));
        let r = sigmoid(Wr.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.05, 0));
        let hh = tanh(Wh.reduce((s, w, i) => s + w * x + r * hState[i] * 0.5 + 0.05, 0));
        hState = hState.map((hs, i) => (1 - z) * hs + z * hh);
    }
    let out = 0;
    for (let i = 0; i < hDim; i++) out += hState[i];
    let prob = sigmoid(out / hDim);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.35 + recent * 0.65, 0.01), 0.99);
}

function transformerEncoderPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let dModel = 8, nHeads = 2, seqLen = Math.min(10, n);
    let pos = Array.from({ length: seqLen }, (_, i) => {
        let pe = new Array(dModel).fill(0);
        for (let j = 0; j < dModel; j += 2) {
            pe[j] = Math.sin(i / Math.pow(10000, j / dModel));
            pe[j + 1] = Math.cos(i / Math.pow(10000, j / dModel));
        }
        return pe;
    });
    let tokens = [];
    for (let i = 0; i < seqLen; i++) {
        let tok = new Array(dModel).fill(y[i] * 0.5);
        tokens.push(tok);
    }
    let Wq = Array.from({ length: nHeads }, () => Array.from({ length: dModel }, () => Math.random() * 0.1 - 0.05));
    let Wk = Array.from({ length: nHeads }, () => Array.from({ length: dModel }, () => Math.random() * 0.1 - 0.05));
    let Wv = Array.from({ length: nHeads }, () => Array.from({ length: dModel }, () => Math.random() * 0.1 - 0.05));
    for (let head = 0; head < nHeads; head++) {
        let Q = tokens.map(t => { let s = 0; for (let i = 0; i < dModel; i++) s += Wq[head][i] * t[i]; return s; });
        let K = tokens.map(t => { let s = 0; for (let i = 0; i < dModel; i++) s += Wk[head][i] * t[i]; return s; });
        let V = tokens.map(t => { let s = 0; for (let i = 0; i < dModel; i++) s += Wv[head][i] * t[i]; return s; });
        let scores = Q.map((q, i) => {
            let expS = K.map(k => Math.exp(Math.min(q * k / Math.sqrt(dModel), 50)));
            let sumExp = expS.reduce((a, b) => a + b, 1e-10);
            let out = 0;
            for (let j = 0; j < V.length; j++) out += (expS[j] / sumExp) * V[j];
            return out;
        });
        tokens = tokens.map((t, i) => t.map((v, j) => v + scores[i] * 0.1));
    }
    let lastTok = tokens[tokens.length - 1];
    let out = lastTok.reduce((a, b) => a + b, 0) / dModel;
    let prob = sigmoid(out);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.35 + recent * 0.65, 0.01), 0.99);
}

function mlpPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return 0.5;
    let hn = 12, lr = 0.02, pl = 3;
    let W1 = [], b1 = [], W2 = new Array(hn).fill(0), b2 = 0;
    for (let i = 0; i < hn; i++) {
        W1[i] = [];
        for (let j = 0; j < pl; j++) W1[i][j] = Math.random() * 0.2 - 0.1;
        b1[i] = 0;
    }
    for (let i = 0; i < hn; i++) W2[i] = Math.random() * 0.2 - 0.1;
    for (let ep = 0; ep < 8; ep++) {
        for (let i = pl; i < n - 1; i++) {
            let x = [];
            for (let j = 0; j < pl; j++) x.push(h[i - pl + j] === 'T' ? 1 : 0);
            let h1 = [];
            for (let j = 0; j < hn; j++) {
                let s = b1[j];
                for (let k = 0; k < pl; k++) s += W1[j][k] * x[k];
                h1.push(tanh(s));
            }
            let p = b2;
            for (let j = 0; j < hn; j++) p += W2[j] * h1[j];
            p = Math.min(Math.max(p, 0), 1);
            let e = p - (h[i] === 'T' ? 1 : 0);
            for (let j = 0; j < hn; j++) {
                let g = e * W2[j] * (1 - h1[j] * h1[j]);
                for (let k = 0; k < pl; k++) W1[j][k] -= lr * g * x[k];
                b1[j] -= lr * g;
            }
            for (let j = 0; j < hn; j++) W2[j] -= lr * e * h1[j];
            b2 -= lr * e;
        }
    }
    let x = [];
    for (let j = 0; j < pl; j++) x.push(h[n - pl + j] === 'T' ? 1 : 0);
    let h1 = [];
    for (let j = 0; j < hn; j++) {
        let s = b1[j];
        for (let k = 0; k < pl; k++) s += W1[j][k] * x[k];
        h1.push(tanh(s));
    }
    let p = b2;
    for (let j = 0; j < hn; j++) p += W2[j] * h1[j];
    p = Math.min(Math.max(p, 0.01), 0.99);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(p * 0.35 + recent * 0.65, 0.01), 0.99);
}

function cnnLstmPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let kernelSize = 3, nFilters = 4, hDim = 4;
    let convW = Array.from({ length: nFilters }, () => new Array(kernelSize).fill(Math.random() * 0.1 - 0.05));
    let convB = new Array(nFilters).fill(0);
    let convOut = new Array(n).fill(0);
    for (let f = 0; f < nFilters; f++) {
        for (let i = 0; i < n; i++) {
            let s = convB[f];
            for (let k = 0; k < kernelSize; k++) {
                let idx = i + k - Math.floor(kernelSize / 2);
                if (idx >= 0 && idx < n) s += convW[f][k] * y[idx];
            }
            convOut[i] += Math.max(0, s) / nFilters;
        }
    }
    let hState = new Array(hDim).fill(0), cState = new Array(hDim).fill(0);
    let Wf_lstm = [], Wi_lstm = [], Wo_lstm = [], Wc_lstm = [];
    for (let i = 0; i < hDim; i++) {
        Wf_lstm[i] = Math.random() * 0.1 - 0.05;
        Wi_lstm[i] = Math.random() * 0.1 - 0.05;
        Wo_lstm[i] = Math.random() * 0.1 - 0.05;
        Wc_lstm[i] = Math.random() * 0.1 - 0.05;
    }
    for (let t = 0; t < n; t++) {
        let x = convOut[t] * 2 - 1;
        let f = sigmoid(Wf_lstm.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let iGate = sigmoid(Wi_lstm.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let o = sigmoid(Wo_lstm.reduce((s, w, i) => s + w * x + hState[i] * 0.5 + 0.1, 0));
        let c = tanh(Wc_lstm.reduce((s, w, i) => s + w * x + cState[i] * 0.5, 0));
        cState = cState.map((cs, i) => f * cs + iGate * c);
        hState = cState.map(cs => o * tanh(cs));
    }
    let out = 0;
    for (let i = 0; i < hDim; i++) out += hState[i];
    let prob = sigmoid(out / hDim);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.3 + recent * 0.7, 0.01), 0.99);
}

function wavenetLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let channels = 4, kernelSize2 = 3, dilations = [1, 2, 4];
    let x = y.slice(0, n);
    let Wconv = Array.from({ length: channels }, () => new Array(kernelSize2).fill(Math.random() * 0.1 - 0.05));
    let bconv = new Array(channels).fill(0);
    let Wout2 = new Array(channels).fill(Math.random() * 0.1 - 0.05);
    for (let d of dilations) {
        let conv = new Array(channels).fill(0);
        for (let c = 0; c < channels; c++) {
            for (let k = 0; k < kernelSize2; k++) {
                let idx = n - 1 - k * d;
                if (idx >= 0) conv[c] += Wconv[c][k] * x[idx];
            }
            conv[c] = tanh(conv[c] + bconv[c]);
        }
        let out = 0;
        for (let c = 0; c < channels; c++) out += Wout2[c] * conv[c];
        x = [sigmoid(out), ...x.slice(0, n - 1)];
    }
    let prob = x[0];
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.3 + recent * 0.7, 0.01), 0.99);
}

function xgboostLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let ntrees = 15, lr = 0.1;
    let preds = new Array(n).fill(0.5), residuals = [];
    let features = [];
    for (let i = 0; i < n; i++) {
        let f = [1];
        if (i > 0) f.push(y[i - 1]);
        if (i > 1) f.push(y[i - 2]);
        if (i > 2) f.push(y[i - 3]);
        f.push(countIn(h, 'T', Math.min(i + 1, 10)) / (Math.min(i + 1, 10) || 1));
        f.push(streak(h.slice(0, Math.min(i + 1, n))));
        features.push(f);
    }
    for (let t = 0; t < ntrees; t++) {
        residuals = [];
        for (let i = 0; i < n; i++) residuals.push(y[i] - preds[i]);
        let bestGain = 0, bestF = 0, bestTh = 0, bestPred = 0;
        for (let f = 0; f < features[0].length; f++) {
            for (let th = 0; th < 10; th++) {
                let thVal = 0.1 * th, gain = 0, leftSum = 0, leftN = 0, rightSum = 0, rightN = 0;
                for (let i = 0; i < n; i++) {
                    if (features[i][f] <= thVal) { leftSum += residuals[i]; leftN++; }
                    else { rightSum += residuals[i]; rightN++; }
                }
                if (leftN < 2 || rightN < 2) continue;
                let leftAvg = leftSum / leftN, rightAvg = rightSum / rightN;
                gain = leftSum * leftAvg + rightSum * rightAvg;
                if (gain > bestGain) { bestGain = gain; bestF = f; bestTh = thVal; bestPred = leftAvg; }
            }
        }
        if (bestGain < 0.001) break;
        for (let i = 0; i < n; i++) {
            if (features[i][bestF] <= bestTh) preds[i] += lr * bestPred;
        }
    }
    let lastPred = preds[n - 1] || 0.5;
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(lastPred * 0.35 + recent * 0.65, 0.01), 0.99);
}

function nBeatsLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let nStacks = 4;
    let trendPolys = [];
    for (let s = 0; s < nStacks; s++) {
        let coefs = [];
        for (let i = 0; i < 3; i++) coefs.push(Math.random() * 0.02 - 0.01);
        trendPolys.push(coefs);
    }
    let backcast = new Array(n).fill(0), forecast = 0;
    for (let s = 0; s < nStacks; s++) {
        let res = new Array(n).fill(0);
        for (let i = 0; i < n; i++) {
            let poly = 0;
            for (let p = 0; p < 3; p++) poly += trendPolys[s][p] * Math.pow(i / (n || 1), p);
            res[i] = y[i] - backcast[i];
            backcast[i] += poly;
        }
        let lastRes = res.slice(0, Math.min(5, n)).reduce((a, b) => a + b, 0) / Math.min(5, n);
        forecast += lastRes * 0.25;
    }
    let base = countIn(h, 'T', n) / n;
    let prob = base + forecast;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function bridgePatternDetect(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return 0.5;
    let pat = {};
    for (let l = 2; l <= 5; l++) {
        for (let i = 0; i <= n - l - 1; i++) {
            let p = h.slice(i, i + l).join('');
            if (!pat[p]) pat[p] = { t: 0, x: 0 };
            if (i + l < n) { h[i + l] === 'T' ? pat[p].t++ : pat[p].x++; }
        }
    }
    let bestP = 0.5, bestScore = 0;
    for (let p in pat) {
        let e = pat[p], tot = e.t + e.x;
        if (tot >= 3) {
            let prob = bayesP(e.t, tot), score = tot * Math.abs(prob - 0.5);
            if (score > bestScore) { bestScore = score; bestP = prob; }
        }
    }
    return bestP;
}

function randomForestPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let nTrees = 12, ps = [], indices = Array.from({ length: n }, (_, i) => i);
    for (let t = 0; t < nTrees; t++) {
        let a = 1, b = 1, subSize = Math.max(5, Math.floor(n * 0.7));
        for (let s = 0; s < subSize; s++) {
            let idx = indices[(t * subSize * 7 + s * 13) % n];
            if (h[idx] === 'T') a++;
            else b++;
        }
        ps.push((a - 1) / (a + b - 2));
    }
    ps.sort((a, b) => a - b);
    return Math.min(Math.max(ps[Math.floor(ps.length / 2)], 0.01), 0.99);
}

function trendStrengthPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return 0.5;
    let sCount = [], curS = 1;
    for (let i = 1; i < n; i++) {
        if (h[i] === h[i - 1]) curS++;
        else { sCount.push(curS); curS = 1; }
    }
    sCount.push(curS);
    if (sCount.length < 2) return 0.5;
    let avgS = sCount.reduce((a, b) => a + b, 0) / sCount.length;
    let longT = countIn(h, 'T', n) / n;
    let shortT = countIn(h, 'T', Math.min(5, n)) / Math.min(5, n);
    let streakBias = streak(h) / Math.max(avgS, 1);
    let prob = longT * 0.4 + shortT * 0.3 + (h[0] === 'T' ? 0.5 + streakBias * 0.1 : 0.5 - streakBias * 0.1);
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function thetaPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let v = h.map(x => x === 'T' ? 1 : 0), th = 2.0, m = v.reduce((a, b) => a + b, 0) / n;
    let t = Array.from({ length: n }, (_, i) => i), tm = (n - 1) / 2, num = 0, den = 0;
    for (let i = 0; i < n; i++) { num += (v[i] - m) * (t[i] - tm); den += (t[i] - tm) ** 2; }
    let sl = den > 0 ? num / den : 0, ic = m - sl * tm;
    let tl = v.map((x, i) => x + th * (sl * (i - tm) + ic - x));
    return Math.min(Math.max(tl[tl.length - 1], 0.01), 0.99);
}

function corrPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let v = h.map(x => x === 'T' ? 1 : 0), m = v.reduce((a, b) => a + b, 0) / n;
    let vr = 0;
    for (let x of v) vr += (x - m) ** 2;
    if (vr < 1e-10) return 0.5;
    let bc = 0, bl = 0;
    for (let l = 1; l <= Math.min(20, Math.floor(n / 3)); l++) {
        let num = 0;
        for (let i = l; i < n; i++) num += (v[i] - m) * (v[i - l] - m);
        if (Math.abs(num / vr) > Math.abs(bc)) { bc = num / vr; bl = l; }
    }
    return Math.abs(bc) < 0.05 ? 0.5 : Math.min(Math.max(0.5 + bc * 0.3, 0.01), 0.99);
}

function dampedTrendPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 8) return 0.5;
    let v = h.map(x => x === 'T' ? 1 : 0), l = v[0], t = 0, a = 0.2, b = 0.05, phi = 0.95;
    for (let i = 1; i < n; i++) {
        let p = l;
        l = a * v[i] + (1 - a) * (l + phi * t);
        t = b * (l - p) + (1 - b) * phi * t;
    }
    return Math.min(Math.max(l + phi * t, 0.01), 0.99);
}

function spectralFFTPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return null;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let meanVal = y.reduce((a, b) => a + b, 0) / n;
    let padded = y.slice();
    while (padded.length & (padded.length - 1)) padded.push(meanVal);
    let N = padded.length;

    function fft(sig) {
        if (sig.length <= 1) return sig;
        let even = fft(sig.filter((_, i) => i % 2 === 0));
        let odd = fft(sig.filter((_, i) => i % 2 === 1));
        let res = new Array(sig.length).fill(0);
        for (let k = 0; k < sig.length / 2; k++) {
            let t = -2 * Math.PI * k / sig.length;
            let c = Math.cos(t), s = Math.sin(t);
            let re = odd[k] * c - 0 * s, im = 0 * c + odd[k] * s;
            res[k] = even[k] + re;
            res[k + sig.length / 2] = even[k] - re;
        }
        return res.map(x => Math.abs(x));
    }
    let spectrum = fft(padded);
    let bestFreq = 0, bestPow = 0;
    for (let f = 2; f <= Math.min(30, Math.floor(N / 3)); f++) {
        if (spectrum[f] > bestPow) { bestPow = spectrum[f]; bestFreq = f; }
    }
    if (bestFreq < 2) return null;
    let phase = n % bestFreq;
    let refIdx = n - 1 - phase >= 0 ? n - 1 - phase : 0;
    let match = 0, total = 0;
    for (let i = refIdx + bestFreq; i < n; i += bestFreq) { total++; if (h[i] === h[refIdx]) match++; }
    if (total < 2) return null;
    let phaseProb = match / total;
    return Math.min(Math.max(phaseProb * 0.6 + (meanVal > 0.5 ? meanVal : 1 - meanVal) * 0.4, 0.01), 0.99);
}

function mutualInfoPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return null;
    let bestLag = 0, bestMI = -Infinity;
    for (let lag = 1; lag <= Math.min(15, Math.floor(n / 3)); lag++) {
        let a = 0, b = 0, c = 0, d = 0;
        for (let i = lag; i < n; i++) {
            if (h[i - lag] === 'T' && h[i] === 'T') a++;
            else if (h[i - lag] === 'T' && h[i] === 'X') b++;
            else if (h[i - lag] === 'X' && h[i] === 'T') c++;
            else if (h[i - lag] === 'X' && h[i] === 'X') d++;
        }
        let total = a + b + c + d;
        if (total < 4) continue;
        let pT = (a + c) / total, pX = (b + d) / total;
        let pT_lag = (a + b) / total, pX_lag = (c + d) / total;
        let pTT = a / total, pTX = b / total, pXT = c / total, pXX = d / total;
        let mi = 0;
        if (pTT > 0 && pT > 0 && pT_lag > 0) mi += pTT * Math.log2(pTT / (pT * pT_lag));
        if (pTX > 0 && pX > 0 && pT_lag > 0) mi += pTX * Math.log2(pTX / (pX * pT_lag));
        if (pXT > 0 && pT > 0 && pX_lag > 0) mi += pXT * Math.log2(pXT / (pT * pX_lag));
        if (pXX > 0 && pX > 0 && pX_lag > 0) mi += pXX * Math.log2(pXX / (pX * pX_lag));
        if (mi > bestMI) { bestMI = mi; bestLag = lag; }
    }
    if (bestLag === 0 || bestMI < 0.05) return null;
    let a2 = 0, b2 = 0;
    for (let i = bestLag; i < n; i++) {
        if (h[i - bestLag] === h[0]) { (h[i] === 'T') ? a2++ : b2++; }
    }
    if (a2 + b2 < 3) return null;
    return Math.min(Math.max(bayesP(a2, b2), 0.01), 0.99);
}

function copulaPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return null;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let tau = 0, cnt = 0;
    for (let i = 0; i < n - 1; i++) {
        for (let j = i + 1; j < Math.min(i + 10, n); j++) {
            let ci = y[i] > y[j] ? 1 : (y[i] < y[j] ? -1 : 0);
            let cj = y[i + 1] > (j + 1 < n ? y[j + 1] : y[j]) ? 1 : (y[i + 1] < (j + 1 < n ? y[j + 1] : y[j]) ? -1 : 0);
            if (ci * cj > 0) tau++;
            else if (ci * cj < 0) tau--;
            cnt++;
        }
    }
    if (cnt < 5) return null;
    let kendall = tau / cnt;
    let p = 0.5 + kendall * 0.3;
    return Math.min(Math.max(p, 0.01), 0.99);
}

function ftrlPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let w = [0, 0, 0], z = [0, 0, 0], nVec = [0, 0, 0];
    let alpha = 0.1, beta = 1.0, l1 = 0.01, l2 = 0.01;
    let feats = [
        () => 1,
        (i) => countIn(h, 'T', Math.min(i + 1, n)) / (Math.min(i + 1, n) || 1) - 0.5,
        (i) => { let s = streak(h.slice(0, Math.min(i + 1, n))); return Math.min(s, 10) / 10; }
    ];
    for (let i = 0; i < n - 1; i++) {
        let x = feats.map(f => f(i));
        let wx = 0;
        for (let j = 0; j < 3; j++) {
            let s = 0;
            if (Math.abs(z[j]) <= l1) s = 0;
            else s = -(z[j] - (z[j] > 0 ? l1 : -l1)) / ((beta + Math.sqrt(nVec[j])) / alpha + l2);
            wx += s * x[j];
        }
        let p = Math.min(Math.max(1 / (1 + Math.exp(-wx)), 0.01), 0.99);
        let y = h[i + 1] === 'T' ? 1 : 0;
        let g = (p - y);
        for (let j = 0; j < 3; j++) {
            let sigma = (Math.sqrt(nVec[j] + g * g) - Math.sqrt(nVec[j])) / alpha;
            z[j] += g - sigma * w[j];
            nVec[j] += g * g;
            w[j] = (Math.abs(z[j]) <= l1) ? 0 : -(z[j] - (z[j] > 0 ? l1 : -l1)) / ((beta + Math.sqrt(nVec[j])) / alpha + l2);
        }
    }
    let xLast = feats.map(f => f(n - 1));
    let wxLast = 0;
    for (let j = 0; j < 3; j++) {
        let s = 0;
        if (Math.abs(z[j]) <= l1) s = 0;
        else s = -(z[j] - (z[j] > 0 ? l1 : -l1)) / ((beta + Math.sqrt(nVec[j])) / alpha + l2);
        wxLast += s * xLast[j];
    }
    let prob = 1 / (1 + Math.exp(-wxLast));
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.3 + recent * 0.7, 0.01), 0.99);
}

function variationBayesPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 10) return 0.5;
    let aN = 1, bN = 1;
    for (let ep = 0; ep < 5; ep++) {
        let tC = countIn(h, 'T', n);
        let xC = n - tC;
        let ELogP = Math.log(aN) - 1 / (2 * aN) - (Math.log(aN + bN) - 1 / (2 * (aN + bN)));
        let ELog1mP = Math.log(bN) - 1 / (2 * bN) - (Math.log(aN + bN) - 1 / (2 * (aN + bN)));
        aN = 1 + tC;
        bN = 1 + xC;
        aN = Math.max(aN, 0.1);
        bN = Math.max(bN, 0.1);
    }
    let qMean = aN / (aN + bN);
    let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
    let prob = qMean * 0.6 + recent * 0.4;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function ntkPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let sigma = 1.0;

    function kernel(xi, xj) { return Math.exp(-((xi - xj) ** 2) / (2 * sigma * sigma)); }
    let K = [];
    for (let i = 0; i < n; i++) { K[i] = []; for (let j = 0; j < n; j++) K[i][j] = kernel(i, j); }
    let target = y.slice(0, n);
    let Kx = [];
    for (let i = 0; i < n; i++) Kx[i] = kernel(n, i);
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let j = 0; j < n; j++) rowSum += K[i][j];
        if (rowSum > 0) { num += Kx[i] * target[i]; den += Kx[i]; }
    }
    if (den < 1e-10) return 0.5;
    let mu = num / den;
    let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
    let prob = mu * 0.4 + recent * 0.6;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function lightGbmLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 25) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let nLeaves = 4, lr = 0.05, nRounds = 10;
    let p = [], feat = [];
    for (let i = 0; i < n; i++) {
        let f = [
            countIn(h, 'T', Math.min(i + 1, n)) / (Math.min(i + 1, n) || 1),
            streak(h.slice(0, Math.min(i + 1, n))) / 10,
            i > 0 ? (h[i - 1] === 'T' ? 1 : 0) : 0.5,
            i > 1 ? (h[i - 2] === 'T' ? 1 : 0) : 0.5,
            i > 2 ? (h[i - 3] === 'T' ? 1 : 0) : 0.5
        ];
        feat.push(f);
        p.push(0.5);
    }
    let gradients = [], hessians = [];
    for (let r = 0; r < nRounds; r++) {
        for (let i = 0; i < n; i++) {
            let pp = Math.min(Math.max(p[i], 0.001), 0.999);
            let g = y[i] - pp;
            let hh = pp * (1 - pp);
            gradients.push(g);
            hessians.push(hh);
        }
        for (let f = 0; f < feat[0].length; f++) {
            let sorted = Array.from({ length: n }, (_, i) => i).sort((a, b) => feat[a][f] - feat[b][f]);
            let bestGain = 0, bestTh = 0;
            for (let t = 0; t < n - 1; t++) {
                if (feat[sorted[t]][f] === feat[sorted[t + 1]][f]) continue;
                let cut = (feat[sorted[t]][f] + feat[sorted[t + 1]][f]) / 2;
                let gLeft = 0, hLeft = 0, gRight = 0, hRight = 0;
                for (let i = 0; i < n; i++) {
                    if (feat[i][f] <= cut) { gLeft += gradients[i];
                        hLeft += hessians[i]; } else { gRight += gradients[i];
                        hRight += hessians[i]; }
                }
                if (hLeft < 0.1 || hRight < 0.1) continue;
                let gain = (gLeft * gLeft) / (hLeft + 0.01) + (gRight * gRight) / (hRight + 0.01);
                if (gain > bestGain) { bestGain = gain;
                    bestTh = cut; }
            }
            if (bestGain > 0) {
                for (let i = 0; i < n; i++) {
                    if (feat[i][f] <= bestTh) {
                        let leafScore = gradients[i] / (hessians[i] + 0.01);
                        p[i] += lr * Math.min(Math.max(leafScore, -0.5), 0.5);
                    }
                }
            }
        }
        gradients.length = 0;
        hessians.length = 0;
    }
    return Math.min(Math.max(p[n - 1] || 0.5, 0.01), 0.99);
}

function deepStatePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let d = 4;
    let mu = new Array(d).fill(0.5),
        P = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < d; i++) P[i][i] = 1;
    let Q = 0.01,
        R = 0.2,
        F = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => i === j ? 0.95 : 0));
    let H = new Array(d).fill(0.5);
    for (let t = 0; t < n; t++) {
        let muPred = new Array(d).fill(0);
        let PPred = Array.from({ length: d }, () => new Array(d).fill(0));
        for (let i = 0; i < d; i++) { muPred[i] = 0; for (let j = 0; j < d; j++) PPred[i][j] = 0; }
        for (let i = 0; i < d; i++) {
            for (let j = 0; j < d; j++) {
                muPred[i] += F[i][j] * mu[j];
                for (let k = 0; k < d; k++) PPred[i][j] += F[i][k] * P[k][j] * F[j][k];
            }
        }
        let yPred = 0;
        for (let i = 0; i < d; i++) yPred += H[i] * muPred[i];
        let s = 0;
        for (let i = 0; i < d; i++)
            for (let j = 0; j < d; j++) s += H[i] * PPred[i][j] * H[j];
        s += R;
        let K = new Array(d).fill(0);
        for (let i = 0; i < d; i++) { K[i] = 0; for (let j = 0; j < d; j++) K[i] += PPred[i][j] * H[j];
            K[i] /= s; }
        let innov = y[t] - yPred;
        for (let i = 0; i < d; i++) mu[i] = muPred[i] + K[i] * innov;
        for (let i = 0; i < d; i++)
            for (let j = 0; j < d; j++) P[i][j] = PPred[i][j] - K[i] * H[j] * s;
    }
    let yNext = 0;
    for (let i = 0; i < d; i++) yNext += H[i] * mu[i];
    return Math.min(Math.max(yNext, 0.01), 0.99);
}

function tcnPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let channels = 6,
        kernelSize = 3,
        dilations = [1, 2, 4];
    let x = y.slice(0, n);
    let W1 = Array.from({ length: channels }, () => new Array(kernelSize).fill(Math.random() * 0.1 - 0.05));
    let b1 = new Array(channels).fill(0);
    let W2 = Array.from({ length: 1 }, () => new Array(channels).fill(Math.random() * 0.1 - 0.05));
    let b2 = 0;
    for (let d of dilations) {
        let conv = new Array(channels).fill(0);
        for (let c = 0; c < channels; c++) {
            for (let k = 0; k < kernelSize; k++) {
                let idx = n - 1 - k * d;
                if (idx >= 0) conv[c] += W1[c][k] * x[idx];
            }
            conv[c] = tanh(conv[c] + b1[c]);
        }
        let out = b2;
        for (let c = 0; c < channels; c++) out += W2[0][c] * conv[c];
        x = [Math.min(Math.max(sigmoid(out), 0.01), 0.99), ...x.slice(0, n - 1)];
    }
    return Math.min(Math.max(x[0], 0.01), 0.99);
}

function hierarchicalBayesPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return 0.5;
    let groups = [Math.min(5, n), Math.min(10, n), Math.min(20, n), n];
    let alphas = groups.map(g => { let tC = countIn(h, 'T', g); return tC + 1; });
    let betas = groups.map(g => { let tC = countIn(h, 'T', g); return g - tC + 1; });
    let muAlpha = alphas.reduce((a, b) => a + b, 0) / alphas.length;
    let muBeta = betas.reduce((a, b) => a + b, 0) / betas.length;
    let globalPrior = muAlpha / (muAlpha + muBeta);
    let tau = 1 / (1 + n / 50);
    let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
    let prob = globalPrior * (1 - tau) + recent * tau;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function garchLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let mu = y.reduce((a, b) => a + b, 0) / n;
    let omega = 0.01,
        alpha = 0.1,
        beta = 0.8;
    let sigma2 = 0.1;
    for (let i = 0; i < n; i++) {
        let eps = y[i] - mu;
        sigma2 = omega + alpha * eps * eps + beta * sigma2;
    }
    let vol = Math.sqrt(sigma2);
    let bias = mu > 0.5 ? Math.min(mu + vol * 0.2, 0.95) : Math.max(mu - vol * 0.2, 0.05);
    return Math.min(Math.max(bias, 0.01), 0.99);
}

function prophetLikePredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let trend = y.reduce((a, b) => a + b, 0) / n;
    let changepoints = [];
    let nCp = Math.min(5, Math.floor(n / 5));
    for (let i = 0; i < nCp; i++) {
        let idx = Math.floor((i + 1) * n / (nCp + 1));
        let before = countIn(h, 'T', idx) / idx;
        let after = countIn(h.slice(idx), 'T', n - idx) / (n - idx);
        if (Math.abs(before - after) > 0.15) changepoints.push({ idx, delta: after - before });
    }
    if (changepoints.length > 0) {
        let avgDelta = changepoints.reduce((a, c) => a + c.delta, 0) / changepoints.length;
        trend += avgDelta * 0.3;
    }
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    let prob = trend * 0.5 + recent * 0.5;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function nHitsPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let nBlocks = 3,
        nPools = 2;
    let x = y.slice(-Math.min(10, n));
    let preds = [];
    for (let b = 0; b < nBlocks; b++) {
        let W = Array.from({ length: nPools }, () => Array.from({ length: x.length }, () => Math.random() * 0.1 - 0.05));
        let bW = new Array(nPools).fill(0);
        let poolOuts = [];
        for (let p = 0; p < nPools; p++) {
            let out = bW[p];
            for (let i = 0; i < x.length; i++) out += W[p][i] * x[i];
            poolOuts.push(tanh(out));
        }
        let maxPool = Math.max(...poolOuts);
        let avgPool = poolOuts.reduce((a, b) => a + b, 0) / nPools;
        let blockOut = maxPool * 0.5 + avgPool * 0.5;
        preds.push(blockOut);
    }
    let base = y.reduce((a, b) => a + b, 0) / n;
    let forecast = preds.reduce((a, b) => a + b, 0) / nBlocks * 0.1;
    let prob = base + forecast;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function patchTstPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let patchLen = 4,
        stride = 2,
        dModel2 = 6;
    let patches = [];
    for (let i = 0; i + patchLen <= n; i += stride) {
        let p = y.slice(i, i + patchLen);
        let avg = p.reduce((a, b) => a + b, 0) / patchLen;
        patches.push(avg);
    }
    if (patches.length < 2) return 0.5;
    let W = Array.from({ length: dModel2 }, () => Array.from({ length: patches.length }, () => Math.random() * 0.1 - 0.05));
    let b = new Array(dModel2).fill(0);
    let proj = new Array(dModel2).fill(0);
    for (let d = 0; d < dModel2; d++) { proj[d] = b[d]; for (let i = 0; i < patches.length; i++) proj[d] += W[d][i] * patches[i]; }
    let out = proj.reduce((a, b) => a + b, 0) / dModel2;
    let prob = sigmoid(out);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    return Math.min(Math.max(prob * 0.3 + recent * 0.7, 0.01), 0.99);
}

function timesNetPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 20) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let periods = [2, 3, 5, 7, 10];
    let bestScore = 0,
        bestPeriod = 2;
    for (let p of periods) {
        if (p >= n) continue;
        let score = 0,
            cnt = 0;
        for (let i = p; i < n; i++) {
            if (Math.abs(y[i] - y[i - p]) < 0.5) score++;
            cnt++;
        }
        let s = cnt > 0 ? score / cnt : 0;
        if (s > bestScore) { bestScore = s;
            bestPeriod = p; }
    }
    let match = 0,
        total = 0;
    for (let i = bestPeriod; i < n; i++) { total++; if (y[i] === y[i - bestPeriod]) match++; }
    let periodProb = total > 0 ? match / total : 0.5;
    let trend = y.slice(0, Math.min(10, n)).reduce((a, b) => a + b, 0) / Math.min(10, n);
    let prob = periodProb * 0.4 + trend * 0.6;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function dlinearPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 12) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let movingAvg = Math.min(7, Math.floor(n / 3));
    let trend = [];
    for (let i = 0; i < n; i++) {
        let s = Math.max(0, i - Math.floor(movingAvg / 2));
        let e = Math.min(n, i + Math.floor(movingAvg / 2) + 1);
        let sum = 0;
        for (let j = s; j < e; j++) sum += y[j];
        trend.push(sum / (e - s));
    }
    let seasonal = y.map((v, i) => v - trend[i]);
    let tSlope = (trend[n - 1] - trend[Math.max(0, n - 10)]) / Math.min(9, n - 1);
    let trendNext = trend[n - 1] + tSlope;
    let recentSeasonal = seasonal.slice(-Math.min(4, n)).reduce((a, b) => a + b, 0) / Math.min(4, n);
    let prob = trendNext + recentSeasonal;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function conformalPredict(h) {
    let n = Math.min(h.length, 200);
    if (n < 15) return 0.5;
    let y = h.map(v => v === 'T' ? 1 : 0);
    let calSize = Math.max(5, Math.floor(n * 0.3));
    let calY = y.slice(n - calSize);
    let trainY = y.slice(0, n - calSize);
    let trainBias = trainY.length > 0 ? trainY.reduce((a, b) => a + b, 0) / trainY.length : 0.5;
    let nonconf = calY.map((v, i) => Math.abs(v - trainBias));
    nonconf.sort((a, b) => a - b);
    let alpha = 0.2,
        qIdx = Math.ceil((1 - alpha) * (calSize + 1)) - 1;
    let q = nonconf[Math.min(qIdx, nonconf.length - 1)] || 0.5;
    let pred = trainBias;
    let low = Math.max(0, pred - q),
        high = Math.min(1, pred + q);
    let recent = countIn(h, 'T', Math.min(8, n)) / Math.min(8, n);
    let prob = recent * 0.5 + (low + high) / 2 * 0.5;
    return Math.min(Math.max(prob, 0.01), 0.99);
}

function computePrediction(h) {
    let d = h.slice(0, Math.min(h.length, 200));
    let n = d.length;
    if (n < 3) return { prediction: Math.random() >= 0.5 ? 'T' : 'X', confidence: 50 };

    let stratList = [];
    let curV = d[0];

    let pPOP = bayesP(countIn(d, 'T', n), n);
    stratList.push({ n: 'POP', p: pPOP, b: 1.0 });

    for (let k = 1; k <= 5; k++) {
        let mp = markovProb(d, k);
        if (mp !== null) stratList.push({ n: 'MC' + k, p: mp, b: 0.9 });
    }

    let pMSP = msPatternMatch(d);
    if (pMSP !== null) stratList.push({ n: 'MSP', p: pMSP, b: 0.8 });

    let hmm = hmmTrain(d);
    let pHMM = hmmProb(hmm);
    if (pHMM !== null) stratList.push({ n: 'HMM', p: pHMM, b: 0.85 });

    let pSTP = stumpEnsemble(d);
    stratList.push({ n: 'STP', p: pSTP, b: 0.7 });

    let pMRV = meanRevProb(d);
    stratList.push({ n: 'MRV', p: pMRV, b: 0.6 });

    let regime = detectRegime(d);
    let pREG = 0.5;
    if (regime === 'STRONG_TREND') pREG = curV === 'T' ? 0.65 : 0.35;
    else if (regime === 'TREND') pREG = curV === 'T' ? 0.58 : 0.42;
    else if (regime === 'SWITCHING') pREG = curV === 'T' ? 0.42 : 0.58;
    else if (regime === 'LOW_ENTROPY') pREG = curV === 'T' ? 0.57 : 0.43;
    else if (regime === 'HIGH_ENTROPY') pREG = 0.5;
    stratList.push({ n: 'REG', p: pREG, b: 0.65 });

    let pEWMA = adaptiveEWMA(d);
    stratList.push({ n: 'EWMA', p: pEWMA, b: 0.6 });

    let pCUSUM = cusumDetect(d);
    if (pCUSUM !== null) stratList.push({ n: 'CUSUM', p: pCUSUM, b: 0.6 });

    let pSWE = slidingWinEnsemble(d);
    stratList.push({ n: 'SWE', p: pSWE, b: 0.65 });

    let pBAGG = baggEnsemble(d);
    stratList.push({ n: 'BAGG', p: pBAGG, b: 0.65 });

    let pBOOST = boostPredict(d);
    stratList.push({ n: 'BOOST', p: pBOOST, b: 0.6 });

    let pcBet = cauBet(d);
    stratList.push({ n: 'CAU_BET', p: pcBet, b: 0.55 });

    let pc11 = cau11(d);
    if (pc11) stratList.push({ n: 'CAU_11', p: pc11.p, b: 0.55 });

    let pc3n = cau3Nhip(d);
    if (pc3n) stratList.push({ n: 'CAU_3NHIP', p: pc3n.p, b: 0.52 });

    let pcDao = cauDao(d);
    if (pcDao) stratList.push({ n: 'CAU_DAO', p: pcDao.p, b: 0.55 });

    let pcTong = cauTong(d);
    if (pcTong) stratList.push({ n: 'CAU_TONG', p: pcTong.p, b: 0.6 });

    let pcCat = cauCat(d);
    if (pcCat) stratList.push({ n: 'CAU_CAT', p: pcCat.p, b: 0.55 });

    let pcRong = cauRongHo(d);
    if (pcRong) stratList.push({ n: 'CAU_RONGHO', p: pcRong.p, b: 0.58 });

    let pc12 = cau12(d);
    if (pc12) stratList.push({ n: 'CAU_12', p: pc12.p, b: 0.5 });

    let pc212 = cau212(d);
    if (pc212) stratList.push({ n: 'CAU_212', p: pc212.p, b: 0.5 });

    let pc33 = cau33(d);
    if (pc33) stratList.push({ n: 'CAU_33', p: pc33.p, b: 0.55 });

    let pc42 = cau42(d);
    if (pc42) stratList.push({ n: 'CAU_42', p: pc42.p, b: 0.55 });

    let pcDao1122 = cauDao1122(d);
    if (pcDao1122) stratList.push({ n: 'CAU_DAO1122', p: pcDao1122.p, b: 0.5 });

    let pcGay = cauGay(d);
    stratList.push({ n: 'CAU_GAY', p: pcGay, b: 0.5 });

    let pcThong = cauThong(d);
    if (pcThong) stratList.push({ n: 'CAU_THONG', p: pcThong.p, b: 0.55 });

    let pc2Nhip = cau2Nhip(d);
    if (pc2Nhip) stratList.push({ n: 'CAU_2NHIP', p: pc2Nhip.p, b: 0.5 });

    let pc4Nhip = cau4Nhip(d);
    if (pc4Nhip) stratList.push({ n: 'CAU_4NHIP', p: pc4Nhip.p, b: 0.48 });

    let pc3_2 = cau3_2(d);
    if (pc3_2) stratList.push({ n: 'CAU_3_2', p: pc3_2.p, b: 0.52 });

    let pc1_2_3 = cau1_2_3(d);
    if (pc1_2_3) stratList.push({ n: 'CAU_1_2_3', p: pc1_2_3.p, b: 0.5 });

    let pcBac = cauBac(d);
    if (pcBac) stratList.push({ n: 'CAU_BAC', p: pcBac.p, b: 0.48 });

    let pc112 = cau112(d);
    if (pc112) stratList.push({ n: 'CAU_112', p: pc112.p, b: 0.5 });

    let pc221 = cau221(d);
    if (pc221) stratList.push({ n: 'CAU_221', p: pc221.p, b: 0.5 });

    let pcDao22 = cauDao22(d);
    if (pcDao22) stratList.push({ n: 'CAU_DAO22', p: pcDao22.p, b: 0.5 });

    let pcCham = cauCham(d);
    if (pcCham) stratList.push({ n: 'CAU_CHAM', p: pcCham.p, b: 0.5 });

    let pcKep = cauKep(d);
    if (pcKep) stratList.push({ n: 'CAU_KEP', p: pcKep.p, b: 0.52 });

    let pcPhanXa = cauPhanXa(d);
    if (pcPhanXa) stratList.push({ n: 'CAU_PHANXA', p: pcPhanXa.p, b: 0.5 });

    let pcLoRoi = cauLoRoi(d);
    if (pcLoRoi) stratList.push({ n: 'CAU_LOROI', p: pcLoRoi.p, b: 0.5 });

    let pcSongHanh = cauSongHanh(d);
    if (pcSongHanh) stratList.push({ n: 'CAU_SONGHANH', p: pcSongHanh.p, b: 0.48 });

    let pcGiaoNhau = cauGiaoNhau(d);
    if (pcGiaoNhau) stratList.push({ n: 'CAU_GIAONHAU', p: pcGiaoNhau.p, b: 0.5 });

    let pcBet12 = cauBet12(d);
    if (pcBet12) stratList.push({ n: 'CAU_BET12', p: pcBet12.p, b: 0.5 });

    let pcXien22 = cauXien22(d);
    if (pcXien22) stratList.push({ n: 'CAU_XIEN22', p: pcXien22.p, b: 0.48 });

    let pc331 = cau331(d);
    if (pc331) stratList.push({ n: 'CAU_331', p: pc331.p, b: 0.5 });

    let pc133 = cau133(d);
    if (pc133) stratList.push({ n: 'CAU_133', p: pc133.p, b: 0.5 });

    let pcNhayCoc = cauNhayCoc(d);
    if (pcNhayCoc) stratList.push({ n: 'CAU_NHAYCOC', p: pcNhayCoc.p, b: 0.5 });

    let pc421 = cau421(d);
    if (pc421) stratList.push({ n: 'CAU_421', p: pc421.p, b: 0.5 });

    let pcDoiXung = cauDoiXung(d);
    if (pcDoiXung) stratList.push({ n: 'CAU_DOIXUNG', p: pcDoiXung.p, b: 0.48 });

    let pcXenKe = cauXenKe(d);
    if (pcXenKe) stratList.push({ n: 'CAU_XENKE', p: pcXenKe.p, b: 0.48 });

    let pcThep = cauThep(d);
    if (pcThep) stratList.push({ n: 'CAU_THEP', p: pcThep.p, b: 0.5 });

    let pcNhipTang = cauNhipTang(d);
    if (pcNhipTang) stratList.push({ n: 'CAU_NHIPTANG', p: pcNhipTang.p, b: 0.5 });

    let pc343 = cau343(d);
    if (pc343) stratList.push({ n: 'CAU_343', p: pc343.p, b: 0.52 });

    let pcXienCheo = cauXienCheo(d);
    if (pcXienCheo) stratList.push({ n: 'CAU_XIENCHEO', p: pcXienCheo.p, b: 0.48 });

    let pLSTM = lstmGatedPredict(d);
    stratList.push({ n: 'LSTM', p: pLSTM, b: 0.55 });

    let pBiLSTM = bilstmPredict(d);
    stratList.push({ n: 'BILSTM', p: pBiLSTM, b: 0.55 });

    let pGRU = gruGatedPredict(d);
    stratList.push({ n: 'GRU', p: pGRU, b: 0.55 });

    let pTransformer = transformerEncoderPredict(d);
    stratList.push({ n: 'TRANSFORMER', p: pTransformer, b: 0.55 });

    let pMLP = mlpPredict(d);
    stratList.push({ n: 'MLP', p: pMLP, b: 0.5 });

    let pCNN = cnnLstmPredict(d);
    stratList.push({ n: 'CNN_LSTM', p: pCNN, b: 0.5 });

    let pWaveNet = wavenetLikePredict(d);
    stratList.push({ n: 'WAVENET', p: pWaveNet, b: 0.5 });

    let pXGB = xgboostLikePredict(d);
    stratList.push({ n: 'XGBOOST', p: pXGB, b: 0.5 });

    let pNBeats = nBeatsLikePredict(d);
    stratList.push({ n: 'NBEATS', p: pNBeats, b: 0.5 });

    let pBridge = bridgePatternDetect(d);
    stratList.push({ n: 'BRIDGE', p: pBridge, b: 0.55 });

    let pRF = randomForestPredict(d);
    stratList.push({ n: 'RDFOREST', p: pRF, b: 0.5 });

    let pTS = trendStrengthPredict(d);
    stratList.push({ n: 'TRENDSTR', p: pTS, b: 0.55 });

    let pTheta = thetaPredict(d);
    stratList.push({ n: 'THETA', p: pTheta, b: 0.5 });

    let pCorr = corrPredict(d);
    stratList.push({ n: 'CORR', p: pCorr, b: 0.48 });

    let pDamped = dampedTrendPredict(d);
    stratList.push({ n: 'DAMPED', p: pDamped, b: 0.55 });

    let pSpectral = spectralFFTPredict(d);
    if (pSpectral) stratList.push({ n: 'SPECTRAL', p: pSpectral, b: 0.55 });

    let pMI = mutualInfoPredict(d);
    if (pMI) stratList.push({ n: 'MI', p: pMI, b: 0.48 });

    let pCopula = copulaPredict(d);
    if (pCopula) stratList.push({ n: 'COPULA', p: pCopula, b: 0.45 });

    let pFTRL = ftrlPredict(d);
    stratList.push({ n: 'FTRL', p: pFTRL, b: 0.5 });

    let pVB = variationBayesPredict(d);
    stratList.push({ n: 'VARBAYES', p: pVB, b: 0.55 });

    let pNTK = ntkPredict(d);
    stratList.push({ n: 'NTK', p: pNTK, b: 0.45 });

    let pLGBM = lightGbmLikePredict(d);
    stratList.push({ n: 'LIGHTGBM', p: pLGBM, b: 0.5 });

    let pDeepState = deepStatePredict(d);
    stratList.push({ n: 'DEEPSTATE', p: pDeepState, b: 0.5 });

    let pTCN = tcnPredict(d);
    stratList.push({ n: 'TCN', p: pTCN, b: 0.48 });

    let pHier = hierarchicalBayesPredict(d);
    stratList.push({ n: 'HIERBAYES', p: pHier, b: 0.55 });

    let pGARCH = garchLikePredict(d);
    stratList.push({ n: 'GARCH', p: pGARCH, b: 0.45 });

    let pProphet = prophetLikePredict(d);
    stratList.push({ n: 'PROPHET', p: pProphet, b: 0.52 });

    let pNHits = nHitsPredict(d);
    stratList.push({ n: 'NHITS', p: pNHits, b: 0.5 });

    let pPatchTST = patchTstPredict(d);
    stratList.push({ n: 'PATCHTST', p: pPatchTST, b: 0.48 });

    let pTimesNet = timesNetPredict(d);
    stratList.push({ n: 'TIMESNET', p: pTimesNet, b: 0.5 });

    let pDLinear = dlinearPredict(d);
    stratList.push({ n: 'DLINEAR', p: pDLinear, b: 0.52 });

    let pConformal = conformalPredict(d);
    stratList.push({ n: 'CONFORMAL', p: pConformal, b: 0.5 });

    let sorted = stratList.sort((a, b) => {
        let sa = Math.abs(a.p - 0.5) * 0.7 + 0.3;
        let sb = Math.abs(b.p - 0.5) * 0.7 + 0.3;
        return sb - sa;
    });

    let topK = Math.min(sorted.length, Math.max(25, Math.floor(sorted.length * 0.5)));
    let topStrats = sorted.slice(0, topK);

    let sumP = 0,
        sumW = 0;
    let uniqVotes = { T: 0, total: 0 };
    for (let s of topStrats) {
        let accBoost = 1 + Math.abs(s.p - 0.5) * 0.5;
        s.b = s.b * accBoost;
        let dir = s.p >= 0.5 ? 'T' : 'X';
        uniqVotes[dir] = (uniqVotes[dir] || 0) + s.b;
        uniqVotes.total += s.b;
        sumP += s.p * s.b;
        sumW += s.b;
    }

    let finalP = sumW > 0 ? sumP / sumW : 0.5;
    let agreeP = uniqVotes.T / uniqVotes.total;
    finalP = finalP * 0.5 + agreeP * 0.5;

    let finalDecision = finalP >= 0.5 ? 'T' : 'X';
    let confidence = Math.min(98, Math.max(50, Math.round(Math.abs(finalP - 0.5) * 200)));

    // KHÔNG SKIP - Luôn trả về dự đoán
    let skip = false;

    return {
        prediction: finalDecision,
        confidence: confidence,
        skip: skip,
        regime: regime,
        strats: stratList.length
    };
}

async function getGameData(api) {
    try {
        let data = await fetchData(api);
        if (!data || !data.length) return null;
        let parsed = parseData(data);
        if (!parsed.length) return null;
        return parsed;
    } catch (e) {
        return null;
    }
}

async function handlePrediction(api, gameName) {
    let data = await getGameData(api);
    if (!data) {
        return { error: "Khong the lay du lieu tu API" };
    }

    let h = data.map(r => r.tx);
    let result = computePrediction(h);
    let latest = data.at(-1);

    let response = {
        Id: "@ZukaNoPro2",
        Game: gameName,
        Phien_truoc: latest.session,
        Xuc_xac: `${latest.dice[0]} ${latest.dice[1]} ${latest.dice[2]}`,
        Ket_qua: latest.result,
        Phien_nay: latest.session + 1,
        Du_doan: result.prediction === "T" ? "T" : "X",
        Do_tin_cay: `${result.confidence}%`
    };

    return response;
}

app.get("/tx/md5", async (request, reply) => {
    let result = await handlePrediction(API_MD5, "max789 md5");
    if (result.error) {
        return reply.status(503).send({ error: result.error });
    }
    return result;
});

app.get("/tx/hu", async (request, reply) => {
    let result = await handlePrediction(API_HU, "max789 hu");
    if (result.error) {
        return reply.status(503).send({ error: result.error });
    }
    return result;
});

app.get("/", async () => {
    return {
        status: "active",
        service: "Max789 Prediction API",
        author: "@cskhvilong1",
        endpoints: {
            md5: "/tx/md5",
            hu: "/tx/hu"
        }
    };
});

const start = async () => {
    try {
        await app.listen({ port: PORT, host: "0.0.0.0" });
        console.log(`Server running on port ${PORT}`);
    } catch (err) {
        console.error("Error starting server:", err.message);
        process.exit(1);
    }
};

start();
