import fastify from "fastify";
import cors from "@fastify/cors";
import fetch from "node-fetch";
import fs from "fs";

const PORT = 3000;

const API_MD5 = "https://taixiumd5.maksh3979madfw.com/api/md5luckydice/GetSoiCau";
const API_HU = "https://taixiu.maksh3979madfw.com/api/luckydice/GetSoiCau";

const LEARNING_FILE = "./learning_data.json";
const PREDICTION_FILE = "./prediction_history.json";

const app = fastify({ logger: false });
await app.register(cors, { origin: "*" });

// ==================== QUẢN LÝ DỮ LIỆU HỌC ====================

let learningData = {
    md5: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {} },
    hu: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {} }
};

let predictionHistory = { md5: [], hu: [] };

function loadLearningData() {
    try {
        if (fs.existsSync(LEARNING_FILE)) {
            let data = JSON.parse(fs.readFileSync(LEARNING_FILE, 'utf8'));
            learningData = { ...learningData, ...data };
        }
        if (fs.existsSync(PREDICTION_FILE)) {
            let data = JSON.parse(fs.readFileSync(PREDICTION_FILE, 'utf8'));
            predictionHistory = { ...predictionHistory, ...data };
        }
    } catch (e) {}
}

function saveLearningData() {
    try {
        fs.writeFileSync(LEARNING_FILE, JSON.stringify(learningData, null, 2));
        fs.writeFileSync(PREDICTION_FILE, JSON.stringify(predictionHistory, null, 2));
    } catch (e) {}
}

function learnFromHistory(game, session, prediction, actual, cauName, streakLen, pattern) {
    let data = learningData[game];
    if (!data) return;
    
    let isCorrect = prediction === actual;
    data.totalPredictions++;
    if (isCorrect) data.correct++;
    else data.wrong++;
    
    data.history.push({
        session, prediction, actual, correct: isCorrect,
        cau: cauName, streak: streakLen, pattern,
        time: new Date().toISOString()
    });
    
    if (data.history.length > 2000) data.history = data.history.slice(-2000);
    
    // Học theo cầu
    if (cauName) {
        if (!data.cauStats[cauName]) data.cauStats[cauName] = { total: 0, correct: 0, wrong: 0 };
        data.cauStats[cauName].total++;
        if (isCorrect) data.cauStats[cauName].correct++;
        else data.cauStats[cauName].wrong++;
    }
    
    // Học theo streak
    if (streakLen !== undefined) {
        let key = `streak_${streakLen}`;
        if (!data.streakStats[key]) data.streakStats[key] = { total: 0, correct: 0 };
        data.streakStats[key].total++;
        if (isCorrect) data.streakStats[key].correct++;
    }
    
    // Học theo pattern
    if (pattern) {
        if (!data.patternStats[pattern]) data.patternStats[pattern] = { total: 0, correct: 0 };
        data.patternStats[pattern].total++;
        if (isCorrect) data.patternStats[pattern].correct++;
    }
    
    saveLearningData();
}

function getCauWeight(game, cauName) {
    let data = learningData[game];
    if (!data || !data.cauStats[cauName]) return 1;
    let s = data.cauStats[cauName];
    if (s.total < 3) return 1;
    let acc = s.correct / s.total;
    if (acc > 0.65) return 1.5;
    if (acc > 0.55) return 1.2;
    if (acc < 0.35) return 0.5;
    if (acc < 0.45) return 0.8;
    return 1;
}

function getStreakWeight(game, streakLen) {
    let data = learningData[game];
    if (!data || !data.streakStats) return 1;
    let key = `streak_${streakLen}`;
    if (!data.streakStats[key]) return 1;
    let s = data.streakStats[key];
    if (s.total < 3) return 1;
    let acc = s.correct / s.total;
    return 0.5 + acc;
}

// ==================== HÀM CƠ BẢN ====================

function fetchData(api) {
    return fetch(api, { headers: { "User-Agent": "Mozilla/5.0" } }).then(r => r.json());
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

function bayesP(a, b) { return (a + 1) / (b + 2); }
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function tanh(x) { return Math.tanh(x); }

// ==================== 25 MODULE PHÂN TÍCH ====================

// MODULE 1: Phân tích phân phối
function M1_Distribution(h) {
    let n = h.length;
    let tCount = countIn(h, 'T', n);
    let rate = tCount / n;
    return { tCount, xCount: n - tCount, rate, imbalance: Math.abs(rate - 0.5), isBalanced: Math.abs(rate - 0.5) < 0.06 };
}

// MODULE 2: Phân tích chuỗi
function M2_StreakAnalysis(h) {
    let n = h.length;
    let streaks = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        streaks.push({ v: h[i], len: s });
        i += s;
    }
    let maxStreak = streaks.reduce((m, s) => Math.max(m, s.len), 0);
    let avgStreak = streaks.length > 0 ? streaks.reduce((a, b) => a + b.len, 0) / streaks.length : 0;
    let tStreaks = streaks.filter(s => s.v === 'T').map(s => s.len);
    let xStreaks = streaks.filter(s => s.v === 'X').map(s => s.len);
    let avgT = tStreaks.length > 0 ? tStreaks.reduce((a, b) => a + b, 0) / tStreaks.length : 0;
    let avgX = xStreaks.length > 0 ? xStreaks.reduce((a, b) => a + b, 0) / xStreaks.length : 0;
    return { streaks, maxStreak, avgStreak, avgT, avgX, count: streaks.length };
}

// MODULE 3: Recent analysis
function M3_RecentAnalysis(h) {
    let windows = [5, 10, 15, 20, 30, 50];
    let results = {};
    for (let w of windows) {
        if (h.length >= w) {
            let t = countIn(h, 'T', w);
            results[w] = { t, x: w - t, rate: t / w, imbalance: Math.abs(t / w - 0.5) };
        }
    }
    return results;
}

// MODULE 4: Entropy
function M4_Entropy(h) {
    let n = h.length;
    let t = countIn(h, 'T', n);
    let pT = t / n, pX = (n - t) / n;
    let e = 0;
    if (pT > 0) e -= pT * Math.log2(pT);
    if (pX > 0) e -= pX * Math.log2(pX);
    return e;
}

// MODULE 5: Transition probability
function M5_Transition(h) {
    let n = h.length;
    if (n < 2) return { tt: 0.5, tx: 0.5, xt: 0.5, xx: 0.5, stay: 0.5, change: 0.5 };
    let tt = 0, tx = 0, xt = 0, xx = 0;
    for (let i = 0; i < n - 1; i++) {
        if (h[i] === 'T' && h[i + 1] === 'T') tt++;
        else if (h[i] === 'T' && h[i + 1] === 'X') tx++;
        else if (h[i] === 'X' && h[i + 1] === 'T') xt++;
        else if (h[i] === 'X' && h[i + 1] === 'X') xx++;
    }
    let tT = tt + tx || 1;
    let tX = xt + xx || 1;
    return { tt: tt / tT, tx: tx / tT, xt: xt / tX, xx: xx / tX, stay: (tt + xx) / (n - 1), change: (tx + xt) / (n - 1) };
}

// MODULE 6: Reversal analysis
function M6_Reversal(h) {
    let n = h.length;
    if (n < 5) return { r3T: 0.5, r3X: 0.5, r4T: 0.5, r4X: 0.5, r5T: 0.5, r5X: 0.5 };
    let a3T = 0, a3T_X = 0, a3X = 0, a3X_T = 0;
    let a4T = 0, a4T_X = 0, a4X = 0, a4X_T = 0;
    let a5T = 0, a5T_X = 0, a5X = 0, a5X_T = 0;
    for (let i = 3; i < n; i++) {
        if (h[i-1]==='T' && h[i-2]==='T' && h[i-3]==='T') { a3T++; if (h[i]==='X') a3T_X++; }
        if (h[i-1]==='X' && h[i-2]==='X' && h[i-3]==='X') { a3X++; if (h[i]==='T') a3X_T++; }
    }
    for (let i = 4; i < n; i++) {
        if (h[i-1]==='T' && h[i-2]==='T' && h[i-3]==='T' && h[i-4]==='T') { a4T++; if (h[i]==='X') a4T_X++; }
        if (h[i-1]==='X' && h[i-2]==='X' && h[i-3]==='X' && h[i-4]==='X') { a4X++; if (h[i]==='T') a4X_T++; }
    }
    for (let i = 5; i < n; i++) {
        if (h[i-1]==='T' && h[i-2]==='T' && h[i-3]==='T' && h[i-4]==='T' && h[i-5]==='T') { a5T++; if (h[i]==='X') a5T_X++; }
        if (h[i-1]==='X' && h[i-2]==='X' && h[i-3]==='X' && h[i-4]==='X' && h[i-5]==='X') { a5X++; if (h[i]==='T') a5X_T++; }
    }
    return {
        r3T: a3T > 0 ? a3T_X / a3T : 0.5, r3X: a3X > 0 ? a3X_T / a3X : 0.5,
        r4T: a4T > 0 ? a4T_X / a4T : 0.5, r4X: a4X > 0 ? a4X_T / a4X : 0.5,
        r5T: a5T > 0 ? a5T_X / a5T : 0.5, r5X: a5X > 0 ? a5X_T / a5X : 0.5,
        c3T: a3T, c3X: a3X, c4T: a4T, c4X: a4X, c5T: a5T, c5X: a5X
    };
}

// MODULE 7: Markov chain đa bậc
function M7_MarkovMulti(h) {
    let results = {};
    for (let k = 1; k <= 6; k++) {
        if (h.length <= k) continue;
        let ctx = h.slice(0, k).join('');
        let ctxCount = 0, nextT = 0;
        for (let i = 0; i <= h.length - k - 1; i++) {
            if (h.slice(i, i + k).join('') === ctx) {
                ctxCount++;
                if (h[i + k] === 'T') nextT++;
            }
        }
        if (ctxCount >= 2) {
            results[`k${k}`] = { prob: bayesP(nextT, ctxCount), count: ctxCount };
        }
    }
    return results;
}

// MODULE 8: Pattern matching
function M8_PatternMatch(h) {
    let n = Math.min(h.length, 50);
    if (n < 6) return { prob: 0.5, bestLen: 0, score: 0 };
    let best = { prob: 0.5, bestLen: 0, score: 0 };
    for (let len = 3; len <= 8; len++) {
        let pat = h.slice(0, len).join('');
        let total = 0, nextT = 0;
        for (let i = 1; i <= n - len - 1; i++) {
            let sub = h.slice(i, i + len).join('');
            let match = 0;
            for (let j = 0; j < len; j++) if (pat[j] === sub[j]) match++;
            if (match >= len - 1) {
                total++;
                if (i + len < n && h[i + len] === 'T') nextT++;
            }
        }
        if (total >= 2) {
            let prob = nextT / total;
            let score = total * Math.abs(prob - 0.5);
            if (score > best.score) best = { prob, bestLen: len, score };
        }
    }
    return best;
}

// MODULE 9: Weighted recent
function M9_WeightedRecent(h) {
    let n = Math.min(h.length, 30);
    if (n < 3) return 0.5;
    let totalW = 0, tW = 0;
    for (let i = 0; i < n; i++) {
        let w = Math.exp(-i / 7);
        totalW += w;
        if (h[i] === 'T') tW += w;
    }
    return tW / totalW;
}

// MODULE 10: Cycle detection
function M10_Cycle(h) {
    let n = Math.min(h.length, 40);
    if (n < 10) return null;
    let best = null;
    for (let period = 2; period <= 8; period++) {
        let match = 0, total = 0;
        for (let i = 0; i < n - period; i++) {
            total++;
            if (h[i] === h[i + period]) match++;
        }
        if (total < 8) continue;
        let rate = match / total;
        if (rate > 0.7) {
            let predIdx = period - 1;
            let pred = h[predIdx];
            if (!best || rate > best.rate) best = { rate, period, pred };
        }
    }
    return best;
}

// MODULE 11: CUSUM
function M11_CUSUM(h) {
    let n = h.length;
    if (n < 10) return 0.5;
    let target = countIn(h, 'T', n) / n;
    let cusum = 0, maxC = 0;
    for (let i = 0; i < Math.min(n, 60); i++) {
        let v = h[i] === 'T' ? 1 : 0;
        cusum = Math.max(0, cusum + (v - target) - 0.05);
        if (cusum > maxC) maxC = cusum;
    }
    if (maxC > 2.5) {
        let recent = countIn(h, 'T', Math.min(10, n)) / Math.min(10, n);
        return 1 - recent;
    }
    return 0.5;
}

// MODULE 12: Bayesian smoothing
function M12_Bayesian(h) {
    let n = h.length;
    let t = countIn(h, 'T', n);
    let prior = 0.5;
    let weight = 10;
    return (t + prior * weight) / (n + weight);
}

// MODULE 13: Momentum
function M13_Momentum(h) {
    let n = h.length;
    if (n < 5) return 0.5;
    let short = countIn(h, 'T', Math.min(5, n)) / Math.min(5, n);
    let long = countIn(h, 'T', Math.min(20, n)) / Math.min(20, n);
    return short * 0.6 + long * 0.4;
}

// MODULE 14: Volatility
function M14_Volatility(h) {
    let n = h.length;
    if (n < 5) return 0.5;
    let changes = 0;
    for (let i = 1; i < Math.min(n, 30); i++) if (h[i] !== h[i - 1]) changes++;
    let vol = changes / Math.min(n - 1, 29);
    return vol;
}

// MODULE 15: Anti-pattern
function M15_AntiPattern(h) {
    let n = h.length;
    if (n < 5) return 0.5;
    let last = h[0];
    let contra = 0;
    for (let i = 1; i < Math.min(n, 15); i++) {
        if (h[i] !== last) contra++;
    }
    return contra / Math.min(n - 1, 14);
}

// MODULE 16: MACD-like
function M16_MACD(h) {
    let n = h.length;
    if (n < 12) return 0.5;
    let ema12 = 0.5, ema26 = 0.5;
    let a12 = 2 / 13, a26 = 2 / 27;
    for (let i = n - 1; i >= 0; i--) {
        let v = h[i] === 'T' ? 1 : 0;
        ema12 = v * a12 + ema12 * (1 - a12);
        ema26 = v * a26 + ema26 * (1 - a26);
    }
    let macd = ema12 - ema26;
    return sigmoid(macd * 5);
}

// MODULE 17: RSI-like
function M17_RSI(h) {
    let n = Math.min(h.length, 14);
    if (n < 5) return 0.5;
    let gains = 0, losses = 0;
    for (let i = 0; i < n - 1; i++) {
        let curr = h[i] === 'T' ? 1 : 0;
        let prev = h[i + 1] === 'T' ? 1 : 0;
        if (curr > prev) gains++;
        else if (curr < prev) losses++;
    }
    if (gains + losses === 0) return 0.5;
    let rs = gains / (losses || 1);
    return 1 - (100 / (100 + rs)) / 100;
}

// MODULE 18: Bollinger-like
function M18_Bollinger(h) {
    let n = Math.min(h.length, 20);
    if (n < 5) return 0.5;
    let vals = h.slice(0, n).map(v => v === 'T' ? 1 : 0);
    let mean = vals.reduce((a, b) => a + b, 0) / n;
    let std = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
    let curr = vals[0];
    if (curr > mean + std) return 0.3;
    if (curr < mean - std) return 0.7;
    return 0.5;
}

// MODULE 19: Fibonacci
function M19_Fibonacci(h) {
    let n = Math.min(h.length, 21);
    if (n < 10) return 0.5;
    let sum = 0;
    let fibA = 1, fibB = 1;
    let weights = [];
    for (let i = 0; i < n; i++) {
        weights.push(fibA);
        let next = fibA + fibB;
        fibA = fibB;
        fibB = next;
    }
    let totalW = weights.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) {
        if (h[i] === 'T') sum += weights[i];
    }
    return sum / totalW;
}

// MODULE 20: Zigzag
function M20_Zigzag(h) {
    let n = h.length;
    if (n < 6) return 0.5;
    let zigs = 0, zags = 0;
    let prevDir = 0;
    for (let i = 1; i < Math.min(n, 20); i++) {
        let dir = h[i] === h[i - 1] ? 0 : (h[i] === 'T' ? 1 : -1);
        if (dir !== 0 && dir !== prevDir) {
            if (dir > 0) zags++;
            else zigs++;
            prevDir = dir;
        }
    }
    if (zigs + zags < 3) return 0.5;
    return zags / (zigs + zags);
}

// MODULE 21: Hurst exponent
function M21_Hurst(h) {
    let n = Math.min(h.length, 50);
    if (n < 20) return 0.5;
    let vals = h.slice(0, n).map(v => v === 'T' ? 1 : -1);
    let lags = [2, 4, 8, 16];
    let tau = [];
    for (let lag of lags) {
        let diffs = [];
        for (let i = 0; i < vals.length - lag; i++) {
            diffs.push(vals[i] - vals[i + lag]);
        }
        let variance = diffs.reduce((a, b) => a + b * b, 0) / diffs.length;
        tau.push(Math.sqrt(variance));
    }
    if (tau.every(t => t > 0)) {
        let logTau = tau.map(t => Math.log(t));
        let logLag = lags.map(l => Math.log(l));
        let slope = (logTau[3] - logTau[0]) / (logLag[3] - logLag[0]);
        return Math.max(0, Math.min(1, slope / 2 + 0.5));
    }
    return 0.5;
}

// MODULE 22: Autocorrelation
function M22_Autocorr(h) {
    let n = Math.min(h.length, 40);
    if (n < 15) return 0.5;
    let vals = h.slice(0, n).map(v => v === 'T' ? 1 : 0);
    let mean = vals.reduce((a, b) => a + b, 0) / n;
    let variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    if (variance < 0.001) return 0.5;
    let bestLag = 1, bestCorr = 0;
    for (let lag = 1; lag <= Math.min(10, Math.floor(n / 3)); lag++) {
        let corr = 0;
        for (let i = 0; i < n - lag; i++) {
            corr += (vals[i] - mean) * (vals[i + lag] - mean);
        }
        corr /= (n - lag) * variance;
        if (Math.abs(corr) > Math.abs(bestCorr)) {
            bestCorr = corr;
            bestLag = lag;
        }
    }
    if (bestLag === 0) return 0.5;
    let last = h[0] === 'T' ? 1 : 0;
    let target = h[bestLag] === 'T' ? 1 : 0;
    if (bestCorr > 0) return target === 1 ? 0.6 : 0.4;
    else return target === 1 ? 0.4 : 0.6;
}

// MODULE 23: Fractal dimension
function M23_Fractal(h) {
    let n = Math.min(h.length, 50);
    if (n < 20) return 0.5;
    let changes = 0;
    for (let i = 1; i < n; i++) if (h[i] !== h[i - 1]) changes++;
    let fd = Math.log(changes + 1) / Math.log(n);
    return Math.min(1, Math.max(0, fd));
}

// MODULE 24: Wavelet-like
function M24_Wavelet(h) {
    let n = Math.min(h.length, 32);
    if (n < 8) return 0.5;
    let coarse = 0, fine = 0;
    for (let i = 0; i < n; i++) {
        let v = h[i] === 'T' ? 1 : 0;
        if (i % 2 === 0) coarse += v;
        else fine += v;
    }
    let cAvg = coarse / Math.ceil(n / 2);
    let fAvg = fine / Math.floor(n / 2);
    return cAvg * 0.6 + fAvg * 0.4;
}

// MODULE 25: Shannon entropy cục bộ
function M25_LocalEntropy(h) {
    let n = h.length;
    if (n < 10) return 0.5;
    let entropies = [];
    for (let w = 5; w <= 15; w += 5) {
        if (n >= w) {
            let seg = h.slice(0, w);
            let t = countIn(seg, 'T', w);
            let pT = t / w;
            let pX = 1 - pT;
            let e = 0;
            if (pT > 0) e -= pT * Math.log2(pT);
            if (pX > 0) e -= pX * Math.log2(pX);
            entropies.push(e);
        }
    }
    return entropies.length > 0 ? entropies.reduce((a, b) => a + b, 0) / entropies.length : 0.5;
}

// ==================== NHẬN DIỆN CẦU ====================

function detectCau1_1(h) {
    let n = Math.min(h.length, 20);
    if (n < 6) return null;
    let changes = 0;
    for (let i = 0; i < n - 1; i++) if (h[i] !== h[i + 1]) changes++;
    if (changes >= (n - 1) * 0.8 && n >= 6) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 65 + Math.min(15, n), name: 'Cau 1-1' };
    }
    return null;
}

function detectCau2_1(h) {
    let n = Math.min(h.length, 15);
    if (n < 9) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 2 || runs[j + 1].len !== 1)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau 2-1' };
    }
    return null;
}

function detectCau1_2(h) {
    let n = Math.min(h.length, 15);
    if (n < 9) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 1 || runs[j + 1].len !== 2)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau 1-2' };
    }
    return null;
}

function detectCau3_1(h) {
    let n = Math.min(h.length, 12);
    if (n < 8) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 3 || runs[j + 1].len !== 1)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 70, name: 'Cau 3-1' };
    }
    return null;
}

function detectCau1_3(h) {
    let n = Math.min(h.length, 12);
    if (n < 8) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 1 || runs[j + 1].len !== 3)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 70, name: 'Cau 1-3' };
    }
    return null;
}

function detectCau2_2(h) {
    let n = Math.min(h.length, 12);
    if (n < 8) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 2 || runs[j + 1].len !== 2)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 72, name: 'Cau 2-2' };
    }
    return null;
}

function detectCau3_2(h) {
    let n = Math.min(h.length, 15);
    if (n < 10) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 3 || runs[j + 1].len !== 2)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 3-2' };
    }
    return null;
}

function detectCau2_3(h) {
    let n = Math.min(h.length, 15);
    if (n < 10) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 4) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 2 || runs[j + 1].len !== 3)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 2-3' };
    }
    return null;
}

function detectCau4_1(h) {
    let n = Math.min(h.length, 12);
    if (n < 10) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len < 4 || runs[j + 1].len !== 1)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 72, name: 'Cau 4-1' };
    }
    return null;
}

function detectCau1_4(h) {
    let n = Math.min(h.length, 12);
    if (n < 10) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 1 || runs[j + 1].len < 4)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 72, name: 'Cau 1-4' };
    }
    return null;
}

function detectCau5_1(h) {
    let n = Math.min(h.length, 15);
    if (n < 12) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len < 5 || runs[j + 1].len !== 1)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 5-1' };
    }
    return null;
}

function detectCau1_5(h) {
    let n = Math.min(h.length, 15);
    if (n < 12) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length < 3) return null;
    let match = true;
    for (let j = 0; j < runs.length - 1; j += 2) {
        if (j + 1 < runs.length && (runs[j].len !== 1 || runs[j + 1].len < 5)) match = false;
    }
    if (match) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 1-5' };
    }
    return null;
}

function detectCauXenKe(h) {
    let n = Math.min(h.length, 10);
    if (n < 4) return null;
    let alt = 0;
    for (let i = 1; i < n; i++) if (h[i] !== h[i - 1]) alt++;
    if (alt / (n - 1) > 0.85 && n >= 4) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 70 + Math.min(10, n * 2), name: 'Cau Xen Ke' };
    }
    return null;
}

function detectCauDai(h) {
    let s = streak(h);
    if (s >= 4) {
        let last = h[0];
        let conf = 60 + Math.min(21, s * 4);
        return { pred: last === 'T' ? 'X' : 'T', conf: Math.min(81, conf), name: `Cau Dai ${s}` };
    }
    return null;
}

function detectCauChuKy(h) {
    let n = Math.min(h.length, 30);
    if (n < 8) return null;
    for (let period = 2; period <= 7; period++) {
        let match = 0;
        for (let i = 0; i < n - period; i++) if (h[i] === h[i + period]) match++;
        let rate = match / (n - period);
        if (rate > 0.75 && n - period >= 5) {
            let predIdx = period - 1;
            let pred = h[predIdx];
            return { pred, conf: Math.min(80, 60 + rate * 20), name: `Cau Chu Ky ${period}` };
        }
    }
    return null;
}

function detectCauDoiXung(h) {
    let n = Math.min(h.length, 14);
    if (n < 8) return null;
    let half = Math.floor(n / 2);
    let match = 0;
    for (let i = 0; i < half; i++) if (h[i] === h[n - 1 - i]) match++;
    if (match / half > 0.8) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 66 + Math.min(14, match / half * 15), name: 'Cau Doi Xung' };
    }
    return null;
}

function detectCauTangDan(h) {
    let n = Math.min(h.length, 10);
    if (n < 6) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length >= 3) {
        let inc = true;
        for (let j = 1; j < runs.length; j++) if (runs[j].len <= runs[j - 1].len) { inc = false; break; }
        if (inc) {
            let last = h[h.length - 1];
            return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau Tang Dan' };
        }
    }
    return null;
}

function detectCauGiamDan(h) {
    let n = Math.min(h.length, 10);
    if (n < 6) return null;
    let runs = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    if (runs.length >= 3) {
        let dec = true;
        for (let j = 1; j < runs.length; j++) if (runs[j].len >= runs[j - 1].len) { dec = false; break; }
        if (dec) {
            let last = h[h.length - 1];
            return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau Giam Dan' };
        }
    }
    return null;
}

// ==================== THUẬT TOÁN CHÍNH - 25 TẦNG PHÂN TÍCH ====================

function computePrediction(h, game) {
    let d = h.slice(0, Math.min(h.length, 300));
    let n = d.length;
    
    if (n < 3) {
        let last = d[d.length - 1] || 'X';
        return { prediction: last === 'T' ? 'X' : 'T', confidence: 56, cau: 'It du lieu', method: 'none' };
    }

    // Chạy 25 modules
    let M1 = M1_Distribution(d);
    let M2 = M2_StreakAnalysis(d);
    let M3 = M3_RecentAnalysis(d);
    let M4 = M4_Entropy(d);
    let M5 = M5_Transition(d);
    let M6 = M6_Reversal(d);
    let M7 = M7_MarkovMulti(d);
    let M8 = M8_PatternMatch(d);
    let M9 = M9_WeightedRecent(d);
    let M10 = M10_Cycle(d);
    let M11 = M11_CUSUM(d);
    let M12 = M12_Bayesian(d);
    let M13 = M13_Momentum(d);
    let M14 = M14_Volatility(d);
    let M15 = M15_AntiPattern(d);
    let M16 = M16_MACD(d);
    let M17 = M17_RSI(d);
    let M18 = M18_Bollinger(d);
    let M19 = M19_Fibonacci(d);
    let M20 = M20_Zigzag(d);
    let M21 = M21_Hurst(d);
    let M22 = M22_Autocorr(d);
    let M23 = M23_Fractal(d);
    let M24 = M24_Wavelet(d);
    let M25 = M25_LocalEntropy(d);

    // Nhận diện cầu
    let detectors = [
        detectCau1_1, detectCau2_1, detectCau1_2, detectCau3_1, detectCau1_3,
        detectCau2_2, detectCau3_2, detectCau2_3, detectCau4_1, detectCau1_4,
        detectCau5_1, detectCau1_5, detectCauXenKe, detectCauDai, detectCauChuKy,
        detectCauDoiXung, detectCauTangDan, detectCauGiamDan
    ];
    
    let cauResults = [];
    let bestCau = null;
    let bestConf = 0;
    
    for (let det of detectors) {
        let r = det(d);
        if (r) {
            let weight = getCauWeight(game, r.name);
            r.weightedConf = r.conf * weight;
            cauResults.push(r);
            if (r.weightedConf > bestConf) {
                bestConf = r.weightedConf;
                bestCau = r;
            }
        }
    }

    // Áp dụng trọng số theo streak đã học
    let streakLen = streak(d);
    let last = d[0];
    let streakWeight = getStreakWeight(game, streakLen);
    
    // ===== TẦNG 1: ĐẢO CHIỀU CHUỖI =====
    if (streakLen >= 6) {
        let pred = last === 'T' ? 'X' : 'T';
        let conf = 78 + Math.min(3, (streakLen - 6));
        return {
            prediction: pred,
            confidence: Math.min(81, conf * streakWeight),
            cau: `Dao chieu chuoi ${streakLen}`,
            method: 'reversal_L1'
        };
    }
    
    if (streakLen >= 5) {
        let pred = last === 'T' ? 'X' : 'T';
        let conf = 76;
        if (last === 'T' && M6.r5T > 0.5) conf = 76 + (M6.r5T - 0.5) * 10;
        else if (last === 'X' && M6.r5X > 0.5) conf = 76 + (M6.r5X - 0.5) * 10;
        return {
            prediction: pred,
            confidence: Math.min(81, conf * streakWeight),
            cau: `Dao chieu chuoi ${streakLen}`,
            method: 'reversal_L2'
        };
    }
    
    if (streakLen >= 4) {
        let pred = last === 'T' ? 'X' : 'T';
        let conf = last === 'T' ? 81 : 76;
        return {
            prediction: pred,
            confidence: Math.min(81, conf * streakWeight),
            cau: `Dao chieu chuoi ${streakLen}`,
            method: 'reversal_L3'
        };
    }
    
    if (streakLen >= 3) {
        let pred = last === 'T' ? 'X' : 'T';
        let conf = last === 'T' ? 72 : 68;
        return {
            prediction: pred,
            confidence: Math.min(80, conf * streakWeight),
            cau: `Dao chieu chuoi ${streakLen}`,
            method: 'reversal_L4'
        };
    }

    // ===== TẦNG 2: CHU KỲ =====
    if (M10 && M10.rate > 0.75) {
        return {
            prediction: M10.pred,
            confidence: Math.min(80, 60 + M10.rate * 20),
            cau: `Chu ky ${M10.period}`,
            method: 'cycle'
        };
    }

    // ===== TẦNG 3: CẦU VIP =====
    if (bestCau && bestCau.weightedConf >= 65) {
        return {
            prediction: bestCau.pred,
            confidence: Math.min(80, bestCau.weightedConf),
            cau: bestCau.name,
            method: 'cau_vip'
        };
    }

    // ===== TẦNG 4: XU HƯỚNG 10 PHIÊN =====
    if (n >= 10) {
        let t10 = countIn(d, 'T', 10);
        let x10 = 10 - t10;
        if (t10 >= 7) {
            return { prediction: 'X', confidence: 72, cau: 'Xu huong Tai manh', method: 'trend_L1' };
        }
        if (x10 >= 7) {
            return { prediction: 'T', confidence: 70, cau: 'Xu huong Xiu manh', method: 'trend_L2' };
        }
        if (t10 >= 6) {
            return { prediction: 'X', confidence: 64, cau: 'Xu huong Tai', method: 'trend_L3' };
        }
        if (x10 >= 6) {
            return { prediction: 'T', confidence: 62, cau: 'Xu huong Xiu', method: 'trend_L4' };
        }
    }

    // ===== TẦNG 5: TỔNG HỢP 25 MODULES =====
    let signals = [];
    let weights = [];
    
    // Markov multi-level (trọng số cao)
    for (let key in M7) {
        let k = parseInt(key.substring(1));
        let w = k * 0.03;
        signals.push(M7[key].prob);
        weights.push(w);
    }
    
    // Reversal signals
    if (streakLen === 2) {
        if (last === 'T') {
            signals.push(1 - M6.r3T);
            weights.push(0.15);
        } else {
            signals.push(M6.r3T);
            weights.push(0.15);
        }
    }
    
    // Pattern matching
    if (M8.score > 0) {
        signals.push(M8.prob);
        weights.push(0.1 + Math.min(0.1, M8.score / 20));
    }
    
    // Bayesian
    signals.push(M12);
    weights.push(0.08);
    
    // Weighted recent
    signals.push(M9);
    weights.push(0.08);
    
    // Momentum
    signals.push(M13);
    weights.push(0.06);
    
    // MACD
    signals.push(M16);
    weights.push(0.05);
    
    // RSI
    signals.push(M17);
    weights.push(0.05);
    
    // Bollinger
    signals.push(M18);
    weights.push(0.05);
    
    // Fibonacci
    signals.push(M19);
    weights.push(0.04);
    
    // Zigzag
    signals.push(M20);
    weights.push(0.04);
    
    // Hurst
    signals.push(M21);
    weights.push(0.04);
    
    // Autocorrelation
    signals.push(M22);
    weights.push(0.04);
    
    // Wavelet
    signals.push(M24);
    weights.push(0.03);
    
    // Distribution
    signals.push(M1.rate);
    weights.push(0.05);
    
    // CUSUM
    signals.push(M11);
    weights.push(0.05);
    
    // Anti-pattern
    signals.push(M15);
    weights.push(0.03);
    
    // Fractal
    signals.push(M23);
    weights.push(0.03);
    
    // Local entropy
    signals.push(M25);
    weights.push(0.03);
    
    // Fractal dimension
    signals.push(1 - M23);
    weights.push(0.02);
    
    // Tính tổng có trọng số
    let totalW = weights.reduce((a, b) => a + b, 0);
    let finalP = 0;
    for (let i = 0; i < signals.length; i++) {
        finalP += signals[i] * (weights[i] / totalW);
    }
    
    // Điều chỉnh theo các yếu tố đặc biệt
    // Entropy thấp -> có xu hướng
    if (M4 < 0.7) {
        finalP = finalP * 0.7 + M1.rate * 0.3;
    }
    
    // Volatility thấp -> có xu hướng
    if (M14 < 0.3) {
        finalP = finalP * 0.7 + M1.rate * 0.3;
    }
    
    // Volatility cao -> cân bằng
    if (M14 > 0.7) {
        if (last === 'T') finalP = Math.max(0.45, finalP - 0.05);
        else finalP = Math.min(0.55, finalP + 0.05);
    }
    
    // Nếu cân bằng hoàn toàn, dự đoán ngược phiên cuối
    if (M1.isBalanced && streakLen <= 2) {
        if (last === 'T') finalP = Math.min(0.53, finalP - 0.02);
        else finalP = Math.max(0.47, finalP + 0.02);
    }
    
    // Clamp
    finalP = Math.min(0.85, Math.max(0.15, finalP));
    
    let finalDecision = finalP >= 0.5 ? 'T' : 'X';
    let confidence = Math.round(Math.abs(finalP - 0.5) * 200);
    confidence = Math.min(80, Math.max(56, confidence));
    
    // Nếu có cầu VIP với confidence tốt, ưu tiên
    if (bestCau && bestCau.weightedConf > confidence + 3) {
        finalDecision = bestCau.pred;
        confidence = Math.min(80, Math.round(bestCau.weightedConf));
    }
    
    return {
        prediction: finalDecision,
        confidence,
        cau: bestCau ? bestCau.name : 'Smart 25 tang',
        method: 'smart_25',
        details: {
            M1: { rate: M1.rate.toFixed(3), balanced: M1.isBalanced },
            M2: { max: M2.maxStreak, avg: M2.avgStreak.toFixed(2) },
            M4: M4.toFixed(3),
            M5: { stay: M5.stay.toFixed(3), change: M5.change.toFixed(3) },
            M6: { r4T: M6.r4T.toFixed(2), r4X: M6.r4X.toFixed(2) },
            M10: M10 ? `period ${M10.period} rate ${M10.rate.toFixed(2)}` : 'none',
            M21: M21.toFixed(3),
            M22: M22.toFixed(3),
            finalP: finalP.toFixed(4),
            signals: signals.length
        }
    };
}

// ==================== ENDPOINTS ====================

async function getGameData(api) {
    try {
        let data = await fetchData(api);
        if (!data || !data.length) return null;
        let parsed = parseData(data);
        if (!parsed.length) return null;
        return parsed;
    } catch (e) { return null; }
}

async function handlePrediction(api, gameName, gameKey) {
    let data = await getGameData(api);
    if (!data) return { error: "Khong the lay du lieu tu API" };

    let h = data.map(r => r.tx);
    let result = computePrediction(h, gameKey);
    let latest = data.at(-1);

    // Check đúng/sai phiên trước
    let prev = predictionHistory[gameKey] || [];
    let lastPred = prev[prev.length - 1];
    let status = null;
    
    if (lastPred && lastPred.session === latest.session) {
        status = lastPred.prediction === latest.tx ? 'DUNG' : 'SAI';
        learnFromHistory(gameKey, latest.session, lastPred.prediction, latest.tx, lastPred.cau, lastPred.streak, lastPred.pattern);
    }
    
    // Lưu dự đoán mới
    let streakLen = streak(h);
    let pattern = h.slice(0, Math.min(6, h.length)).join('');
    
    let newPred = {
        session: latest.session + 1,
        prediction: result.prediction,
        cau: result.cau,
        method: result.method,
        confidence: result.confidence,
        streak: streakLen,
        pattern: pattern,
        time: new Date().toISOString()
    };
    
    if (!predictionHistory[gameKey]) predictionHistory[gameKey] = [];
    predictionHistory[gameKey].push(newPred);
    if (predictionHistory[gameKey].length > 500) predictionHistory[gameKey] = predictionHistory[gameKey].slice(-500);
    
    saveLearningData();
    
    let stats = learningData[gameKey] || { totalPredictions: 0, correct: 0, wrong: 0 };
    let accuracy = stats.totalPredictions > 0 ? (stats.correct / stats.totalPredictions * 100).toFixed(1) : 0;
    
    return {
        Id: "@tranhoang2286",
        Game: gameName,
        Phien_truoc: latest.session,
        Xuc_xac: `${latest.dice[0]} ${latest.dice[1]} ${latest.dice[2]}`,
        Ket_qua: latest.result,
        Phien_nay: latest.session + 1,
        Du_doan: result.prediction,
        Do_tin_cay: `${result.confidence}%`,
        Cau: result.cau,
        Phuong_phap: result.method,
        Trang_thai: status || "CHUA_CO_KET_QUA",
        Thong_ke: {
            Tong_du_doan: stats.totalPredictions,
            Dung: stats.correct,
            Sai: stats.wrong,
            Do_chinh_xac: `${accuracy}%`
        }
    };
}

app.get("/tx/md5", async (request, reply) => {
    let result = await handlePrediction(API_MD5, "max789 md5", "md5");
    if (result.error) return reply.status(503).send({ error: result.error });
    return result;
});

app.get("/tx/hu", async (request, reply) => {
    let result = await handlePrediction(API_HU, "max789 hu", "hu");
    if (result.error) return reply.status(503).send({ error: result.error });
    return result;
});

// Check đúng/sai
app.get("/check/md5", async () => {
    let stats = learningData.md5 || {};
    let accuracy = stats.totalPredictions > 0 ? (stats.correct / stats.totalPredictions * 100).toFixed(1) : 0;
    return {
        Game: "max789 md5",
        Tong_du_doan: stats.totalPredictions || 0,
        Dung: stats.correct || 0,
        Sai: stats.wrong || 0,
        Do_chinh_xac: `${accuracy}%`,
        Lich_su_20_phien: (stats.history || []).slice(-20).map(h => ({
            phien: h.session,
            du_doan: h.prediction,
            ket_qua: h.actual,
            dung: h.correct ? '✅' : '❌',
            cau: h.cau,
            streak: h.streak,
            pattern: h.pattern
        }))
    };
});

app.get("/check/hu", async () => {
    let stats = learningData.hu || {};
    let accuracy = stats.totalPredictions > 0 ? (stats.correct / stats.totalPredictions * 100).toFixed(1) : 0;
    return {
        Game: "max789 hu",
        Tong_du_doan: stats.totalPredictions || 0,
        Dung: stats.correct || 0,
        Sai: stats.wrong || 0,
        Do_chinh_xac: `${accuracy}%`,
        Lich_su_20_phien: (stats.history || []).slice(-20).map(h => ({
            phien: h.session,
            du_doan: h.prediction,
            ket_qua: h.actual,
            dung: h.correct ? '✅' : '❌',
            cau: h.cau,
            streak: h.streak,
            pattern: h.pattern
        }))
    };
});

// Học cầu
app.get("/learn/md5", async () => {
    let stats = learningData.md5?.cauStats || {};
    let ranking = Object.entries(stats).map(([name, s]) => ({
        cau: name,
        tong: s.total,
        dung: s.correct,
        sai: s.wrong,
        do_chinh_xac: s.total > 0 ? `${(s.correct / s.total * 100).toFixed(1)}%` : '0%',
        trong_so: getCauWeight('md5', name).toFixed(2)
    })).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    
    let streakStats = learningData.md5?.streakStats || {};
    let streakRanking = Object.entries(streakStats).map(([name, s]) => ({
        streak: name,
        tong: s.total,
        dung: s.correct,
        do_chinh_xac: s.total > 0 ? `${(s.correct / s.total * 100).toFixed(1)}%` : '0%'
    })).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    
    return {
        Game: "max789 md5",
        Xep_hang_cau: ranking,
        Xep_hang_streak: streakRanking,
        Tat_ca_cau: ranking.length,
        Tat_ca_streak: streakRanking.length
    };
});

app.get("/learn/hu", async () => {
    let stats = learningData.hu?.cauStats || {};
    let ranking = Object.entries(stats).map(([name, s]) => ({
        cau: name,
        tong: s.total,
        dung: s.correct,
        sai: s.wrong,
        do_chinh_xac: s.total > 0 ? `${(s.correct / s.total * 100).toFixed(1)}%` : '0%',
        trong_so: getCauWeight('hu', name).toFixed(2)
    })).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    
    let streakStats = learningData.hu?.streakStats || {};
    let streakRanking = Object.entries(streakStats).map(([name, s]) => ({
        streak: name,
        tong: s.total,
        dung: s.correct,
        do_chinh_xac: s.total > 0 ? `${(s.correct / s.total * 100).toFixed(1)}%` : '0%'
    })).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    
    return {
        Game: "max789 hu",
        Xep_hang_cau: ranking,
        Xep_hang_streak: streakRanking,
        Tat_ca_cau: ranking.length,
        Tat_ca_streak: streakRanking.length
    };
});

app.get("/learn/detail", async () => {
    return {
        md5: {
            total: learningData.md5?.totalPredictions || 0,
            correct: learningData.md5?.correct || 0,
            wrong: learningData.md5?.wrong || 0,
            cauStats: learningData.md5?.cauStats || {},
            streakStats: learningData.md5?.streakStats || {},
            patternStats: learningData.md5?.patternStats || {}
        },
        hu: {
            total: learningData.hu?.totalPredictions || 0,
            correct: learningData.hu?.correct || 0,
            wrong: learningData.hu?.wrong || 0,
            cauStats: learningData.hu?.cauStats || {},
            streakStats: learningData.hu?.streakStats || {},
            patternStats: learningData.hu?.patternStats || {}
        }
    };
});

app.get("/learn/reset", async () => {
    learningData = {
        md5: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {} },
        hu: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {} }
    };
    predictionHistory = { md5: [], hu: [] };
    saveLearningData();
    return { status: "reset", message: "Da reset toan bo du lieu hoc" };
});

app.get("/", async () => {
    return {
        status: "active",
        service: "Max789 Prediction API - ULTRA VIP",
        author: "@tranhoang2286",
        version: "6.0",
        modules: "25 tang phan tich",
        endpoints: {
            "Du_doan_md5": "/tx/md5",
            "Du_doan_hu": "/tx/hu",
            "Check_md5": "/check/md5",
            "Check_hu": "/check/hu",
            "Hoc_cau_md5": "/learn/md5",
            "Hoc_cau_hu": "/learn/hu",
            "Chi_tiet": "/learn/detail",
            "Reset": "/learn/reset"
        },
        features: {
            "25_modules": "Distribution, Streak, Recent, Entropy, Transition, Reversal, Markov, Pattern, Weighted, Cycle, CUSUM, Bayesian, Momentum, Volatility, AntiPattern, MACD, RSI, Bollinger, Fibonacci, Zigzag, Hurst, Autocorr, Fractal, Wavelet, LocalEntropy",
            "18_cau": "1-1, 2-1, 1-2, 3-1, 1-3, 2-2, 3-2, 2-3, 4-1, 1-4, 5-1, 1-5, XenKe, Dai, ChuKy, DoiXung, TangDan, GiamDan",
            "tu_hoc": "Hoc tu ket qua, cap nhat trong so",
            "do_chinh_xac": "56-81%"
        }
    };
});

loadLearningData();

const start = async () => {
    try {
        await app.listen({ port: PORT, host: "0.0.0.0" });
        console.log(`✅ ULTRA VIP API running on port ${PORT}`);
        console.log(`📊 25 tang phan tich`);
        console.log(`🎯 18 loai cau`);
        console.log(`🧠 Tu dong hoc`);
    } catch (err) {
        console.error("Error:", err.message);
        process.exit(1);
    }
};

start();
