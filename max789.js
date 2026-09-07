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

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function tanh(x) {
    return Math.tanh(x);
}

// ==================== PHÂN TÍCH THỐNG KÊ NÂNG CAO ====================

function analyzeDistribution(h) {
    let n = h.length;
    let tCount = countIn(h, 'T', n);
    let xCount = n - tCount;
    let rate = tCount / n;
    let imbalance = Math.abs(rate - 0.5);
    let isBalanced = imbalance < 0.08;
    let isTiltedT = rate > 0.55;
    let isTiltedX = rate < 0.45;
    
    return { tCount, xCount, rate, imbalance, isBalanced, isTiltedT, isTiltedX };
}

function analyzeStreaks(h) {
    let n = h.length;
    let streaks = [];
    let i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        streaks.push({ value: h[i], length: s });
        i += s;
    }
    let maxStreak = 0;
    let avgStreak = 0;
    let tStreaks = [];
    let xStreaks = [];
    for (let st of streaks) {
        if (st.length > maxStreak) maxStreak = st.length;
        if (st.value === 'T') tStreaks.push(st.length);
        else xStreaks.push(st.length);
    }
    if (streaks.length > 0) {
        avgStreak = streaks.reduce((a, b) => a + b.length, 0) / streaks.length;
    }
    let avgTStreak = tStreaks.length > 0 ? tStreaks.reduce((a, b) => a + b, 0) / tStreaks.length : 0;
    let avgXStreak = xStreaks.length > 0 ? xStreaks.reduce((a, b) => a + b, 0) / xStreaks.length : 0;
    
    return { streaks, maxStreak, avgStreak, avgTStreak, avgXStreak, streakCount: streaks.length };
}

function analyzeRecent(h, windowSizes = [5, 10, 15, 20, 30]) {
    let results = {};
    for (let w of windowSizes) {
        if (h.length >= w) {
            let recent = h.slice(0, w);
            let tCount = countIn(recent, 'T', w);
            let xCount = w - tCount;
            let rate = tCount / w;
            results[w] = { tCount, xCount, rate, imbalance: Math.abs(rate - 0.5) };
        }
    }
    return results;
}

function calculateEntropy(h) {
    let n = h.length;
    let tCount = countIn(h, 'T', n);
    let xCount = n - tCount;
    let pT = tCount / n;
    let pX = xCount / n;
    let entropy = 0;
    if (pT > 0) entropy -= pT * Math.log2(pT);
    if (pX > 0) entropy -= pX * Math.log2(pX);
    return entropy;
}

function calculateTransitionProb(h) {
    let n = h.length;
    if (n < 2) return { tt: 0.5, tx: 0.5, xt: 0.5, xx: 0.5 };
    let tt = 0, tx = 0, xt = 0, xx = 0;
    for (let i = 0; i < n - 1; i++) {
        if (h[i] === 'T' && h[i + 1] === 'T') tt++;
        else if (h[i] === 'T' && h[i + 1] === 'X') tx++;
        else if (h[i] === 'X' && h[i + 1] === 'T') xt++;
        else if (h[i] === 'X' && h[i + 1] === 'X') xx++;
    }
    let totalT = tt + tx || 1;
    let totalX = xt + xx || 1;
    return {
        tt: tt / totalT,
        tx: tx / totalT,
        xt: xt / totalX,
        xx: xx / totalX,
        stay: (tt + xx) / (n - 1),
        change: (tx + xt) / (n - 1)
    };
}

// ==================== NHẬN DIỆN CẦU TOÀN DIỆN ====================

function detectCau1_1(h) {
    let n = Math.min(h.length, 20);
    if (n < 6) return null;
    let pattern = [];
    for (let i = 0; i < n - 1; i++) {
        if (h[i] !== h[i + 1]) pattern.push(1);
        else pattern.push(0);
    }
    let ones = pattern.filter(p => p === 1).length;
    if (ones >= pattern.length * 0.8 && pattern.length >= 6) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 65 + Math.min(15, pattern.length), name: 'Cau 1-1' };
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 2 || runs[j + 1].len !== 1) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 4) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 1 || runs[j + 1].len !== 2) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 4) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 3 || runs[j + 1].len !== 1) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 1 || runs[j + 1].len !== 3) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 2 || runs[j + 1].len !== 2) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 4) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 3 || runs[j + 1].len !== 2) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 4) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 2 || runs[j + 1].len !== 3) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 4) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len < 4 || runs[j + 1].len !== 1) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 1 || runs[j + 1].len < 4) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 72, name: 'Cau 1-4' };
    }
    return null;
}

function detectCau5_1(h) {
    let n = Math.min(h.length, 12);
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
        if (j + 1 < runs.length) {
            if (runs[j].len < 5 || runs[j + 1].len !== 1) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 5-1' };
    }
    return null;
}

function detectCau1_5(h) {
    let n = Math.min(h.length, 12);
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
        if (j + 1 < runs.length) {
            if (runs[j].len !== 1 || runs[j + 1].len < 5) {
                match = false;
                break;
            }
        }
    }
    if (match && runs.length >= 3) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 74, name: 'Cau 1-5' };
    }
    return null;
}

function detectCauXenKe(h) {
    let n = Math.min(h.length, 10);
    if (n < 4) return null;
    let alt = 0;
    for (let i = 1; i < n; i++) {
        if (h[i] !== h[i - 1]) alt++;
    }
    let altRate = alt / (n - 1);
    if (altRate > 0.85 && n >= 4) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 70 + Math.min(10, n * 2), name: 'Cau Xen Ke' };
    }
    return null;
}

function detectCauDai(h) {
    let s = streak(h);
    if (s >= 5) {
        let last = h[0];
        let conf = 60 + Math.min(20, s * 3);
        return { pred: last === 'T' ? 'X' : 'T', conf: Math.min(80, conf), name: `Cau Dai ${s}` };
    }
    return null;
}

function detectCauGanDay(h) {
    let n = Math.min(h.length, 15);
    let recent = h.slice(0, n);
    let tCount = countIn(recent, 'T', n);
    let xCount = n - tCount;
    
    if (tCount > xCount + 3) {
        let last = h[h.length - 1];
        let conf = 62 + Math.min(15, (tCount - xCount) * 2);
        return { pred: last === 'T' ? 'X' : 'T', conf: Math.min(80, conf), name: 'Cau Gan Day' };
    } else if (xCount > tCount + 3) {
        let last = h[h.length - 1];
        let conf = 62 + Math.min(15, (xCount - tCount) * 2);
        return { pred: last === 'T' ? 'X' : 'T', conf: Math.min(80, conf), name: 'Cau Gan Day' };
    }
    return null;
}

function detectCauChuKy(h) {
    let n = Math.min(h.length, 30);
    if (n < 8) return null;
    
    for (let period = 2; period <= 6; period++) {
        let match = true;
        let count = 0;
        for (let i = 0; i < n - period; i++) {
            if (h[i] === h[i + period]) count++;
        }
        let matchRate = count / (n - period);
        if (matchRate > 0.75 && n - period >= 5) {
            let lastIdx = h.length - 1;
            let predIdx = lastIdx - period;
            if (predIdx >= 0) {
                let next = h[predIdx];
                let conf = 60 + Math.min(18, matchRate * 20);
                return { pred: next, conf: Math.min(80, conf), name: `Cau Chu Ky ${period}` };
            }
        }
    }
    return null;
}

function detectCauDoiXung(h) {
    let n = Math.min(h.length, 14);
    if (n < 8) return null;
    let half = Math.floor(n / 2);
    let match = 0;
    for (let i = 0; i < half; i++) {
        if (h[i] === h[n - 1 - i]) match++;
    }
    let matchRate = match / half;
    if (matchRate > 0.8) {
        let last = h[h.length - 1];
        let next = last === 'T' ? 'X' : 'T';
        return { pred: next, conf: 66 + Math.min(14, matchRate * 15), name: 'Cau Doi Xung' };
    }
    return null;
}

function detectCauThuanNghich(h) {
    let n = Math.min(h.length, 10);
    if (n < 6) return null;
    let first = h.slice(0, 3);
    let last = h.slice(-3);
    let match = 0;
    for (let i = 0; i < 3; i++) {
        if (first[i] !== last[i]) match++;
    }
    if (match >= 2) {
        let next = h[2] === 'T' ? 'X' : 'T';
        return { pred: next, conf: 65, name: 'Cau Thuan Nghich' };
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
        let increasing = true;
        for (let j = 1; j < runs.length; j++) {
            if (runs[j].len <= runs[j - 1].len) {
                increasing = false;
                break;
            }
        }
        if (increasing) {
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
        let decreasing = true;
        for (let j = 1; j < runs.length; j++) {
            if (runs[j].len >= runs[j - 1].len) {
                decreasing = false;
                break;
            }
        }
        if (decreasing) {
            let last = h[h.length - 1];
            return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau Giam Dan' };
        }
    }
    return null;
}

function detectCauGiaoThoa(h) {
    let n = Math.min(h.length, 12);
    if (n < 8) return null;
    let tPositions = [];
    let xPositions = [];
    for (let i = 0; i < n; i++) {
        if (h[i] === 'T') tPositions.push(i);
        else xPositions.push(i);
    }
    if (tPositions.length > 0 && xPositions.length > 0) {
        let tGap = tPositions.length > 1 ? tPositions[1] - tPositions[0] : 0;
        let xGap = xPositions.length > 1 ? xPositions[1] - xPositions[0] : 0;
        if (tGap === xGap && tGap > 0 && tGap <= 3) {
            let last = h[h.length - 1];
            return { pred: last === 'T' ? 'X' : 'T', conf: 70, name: 'Cau Giao Thoa' };
        }
    }
    return null;
}

function detectCauBienThien(h) {
    let n = Math.min(h.length, 8);
    if (n < 5) return null;
    let changes = 0;
    for (let i = 1; i < n; i++) {
        if (h[i] !== h[i - 1]) changes++;
    }
    if (changes >= 4) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 64, name: 'Cau Bien Thien' };
    }
    return null;
}

function detectCauNhipNhay(h) {
    let n = Math.min(h.length, 8);
    if (n < 6) return null;
    let pattern = [];
    for (let i = 0; i < n - 1; i++) {
        if (h[i] !== h[i + 1]) pattern.push(1);
        else pattern.push(0);
    }
    let ones = pattern.filter(p => p === 1).length;
    let zeros = pattern.length - ones;
    if (ones > 0 && zeros > 0 && Math.abs(ones - zeros) <= 1) {
        let last = h[h.length - 1];
        let next = last === 'T' ? 'X' : 'T';
        let conf = 62 + Math.min(8, Math.min(ones, zeros) * 2);
        return { pred: next, conf: Math.min(80, conf), name: 'Cau Nhip Nhay' };
    }
    return null;
}

// ==================== MÔ HÌNH MÁY HỌC ĐƠN GIẢN ====================

function simpleMarkov(h) {
    let n = h.length;
    if (n < 5) return 0.5;
    let trans = calculateTransitionProb(h);
    let last = h[h.length - 1];
    if (last === 'T') {
        return trans.tt;
    } else {
        return trans.xt;
    }
}

function weightedRecent(h) {
    let n = Math.min(h.length, 20);
    let weights = [];
    let totalWeight = 0;
    let tWeight = 0;
    for (let i = 0; i < n; i++) {
        let w = Math.exp(-i / 5);
        weights.push(w);
        totalWeight += w;
        if (h[i] === 'T') tWeight += w;
    }
    let prob = tWeight / totalWeight;
    return prob;
}

function patternMatching(h) {
    let n = Math.min(h.length, 30);
    if (n < 6) return 0.5;
    let bestMatch = 0.5;
    let bestScore = 0;
    for (let len = 3; len <= 6; len++) {
        let pattern = h.slice(0, len);
        let matches = 0;
        let total = 0;
        let nextT = 0;
        for (let i = 1; i <= n - len - 1; i++) {
            let sub = h.slice(i, i + len);
            let match = 0;
            for (let j = 0; j < len; j++) {
                if (pattern[j] === sub[j]) match++;
            }
            if (match >= len - 1) {
                total++;
                if (i + len < n && h[i + len] === 'T') nextT++;
            }
        }
        if (total >= 2) {
            let prob = nextT / total;
            let score = total * Math.abs(prob - 0.5);
            if (score > bestScore) {
                bestScore = score;
                bestMatch = prob;
            }
        }
    }
    return bestMatch;
}

// ==================== HÀM DỰ ĐOÁN CHÍNH ====================

function computePrediction(h) {
    let d = h.slice(0, Math.min(h.length, 200));
    let n = d.length;
    
    if (n < 3) {
        let last = d[d.length - 1] || 'X';
        return { 
            prediction: last === 'T' ? 'X' : 'T', 
            confidence: 56,
            cau: 'Khong du du lieu',
            skip: false,
            details: {}
        };
    }

    // ===== PHÂN TÍCH TOÀN DIỆN =====
    let distribution = analyzeDistribution(d);
    let streakAnalysis = analyzeStreaks(d);
    let recentAnalysis = analyzeRecent(d);
    let entropy = calculateEntropy(d);
    let transProb = calculateTransitionProb(d);
    
    // ===== NHẬN DIỆN CẦU =====
    let detectors = [
        detectCau1_1, detectCau2_1, detectCau1_2, detectCau3_1, detectCau1_3,
        detectCau2_2, detectCau3_2, detectCau2_3, detectCau4_1, detectCau1_4,
        detectCau5_1, detectCau1_5, detectCauXenKe, detectCauDai, detectCauGanDay,
        detectCauChuKy, detectCauDoiXung, detectCauThuanNghich, detectCauTangDan,
        detectCauGiamDan, detectCauGiaoThoa, detectCauBienThien, detectCauNhipNhay
    ];
    
    let cauResults = [];
    let bestCau = null;
    let bestConf = 0;
    
    for (let det of detectors) {
        let result = det(d);
        if (result) {
            cauResults.push(result);
            if (result.conf > bestConf) {
                bestConf = result.conf;
                bestCau = result;
            }
        }
    }
    
    // ===== MÔ HÌNH MÁY HỌC =====
    let markovProb = simpleMarkov(d);
    let weightedProb = weightedRecent(d);
    let patternProb = patternMatching(d);
    
    // ===== TỔNG HỢP DỰ ĐOÁN =====
    let predictions = [];
    let weights = [];
    
    // Cầu VIP
    if (bestCau && bestConf >= 60) {
        let p = bestCau.pred === 'T' ? 0.7 : 0.3;
        predictions.push(p);
        weights.push(bestConf / 100);
    }
    
    // Markov
    predictions.push(markovProb);
    weights.push(0.15);
    
    // Weighted Recent
    predictions.push(weightedProb);
    weights.push(0.2);
    
    // Pattern Matching
    predictions.push(patternProb);
    weights.push(0.15);
    
    // Distribution
    let distProb = distribution.rate;
    predictions.push(distProb);
    weights.push(0.1);
    
    // Transition
    let transProbValue = transProb.change > 0.5 ? 0.4 : 0.6;
    predictions.push(transProbValue);
    weights.push(0.1);
    
    // Streak Analysis
    let streakProb = 0.5;
    if (streakAnalysis.maxStreak >= 4) {
        let last = d[0];
        streakProb = last === 'T' ? 0.35 : 0.65;
    }
    predictions.push(streakProb);
    weights.push(0.1);
    
    // Entropy
    let entropyProb = entropy > 0.9 ? 0.5 : (distribution.rate);
    predictions.push(entropyProb);
    weights.push(0.05);
    
    // ===== TÍNH TOÁN CUỐI CÙNG =====
    let totalWeight = weights.reduce((a, b) => a + b, 0);
    let finalP = 0;
    for (let i = 0; i < predictions.length; i++) {
        finalP += predictions[i] * (weights[i] / totalWeight);
    }
    
    // ===== ĐIỀU CHỈNH LINH HOẠT =====
    let streakLen = streak(d);
    let last = d[0];
    
    // Nếu chuỗi quá dài -> đảo chiều mạnh
    if (streakLen >= 6) {
        finalP = last === 'T' ? 0.3 : 0.7;
        bestCau = { name: `Dao chieu chuoi ${streakLen}`, conf: 75 };
        bestConf = 75;
    } else if (streakLen >= 4) {
        let adjustment = 0.08 * (streakLen - 3);
        if (last === 'T') finalP -= adjustment;
        else finalP += adjustment;
    }
    
    // Điều chỉnh theo entropy
    if (entropy < 0.7 && streakLen < 3) {
        // Low entropy -> có xu hướng
        if (distribution.rate > 0.55) finalP = Math.min(0.7, finalP + 0.05);
        else if (distribution.rate < 0.45) finalP = Math.max(0.3, finalP - 0.05);
    }
    
    // Điều chỉnh theo độ cân bằng
    if (distribution.isBalanced && streakLen < 3) {
        // Nếu cân bằng, dự đoán ngược với phiên cuối
        if (last === 'T') finalP = Math.min(0.55, finalP - 0.02);
        else finalP = Math.max(0.45, finalP + 0.02);
    }
    
    // Đảm bảo trong khoảng 0.01 - 0.99
    finalP = Math.min(0.99, Math.max(0.01, finalP));
    
    // ===== QUYẾT ĐỊNH =====
    let finalDecision = finalP >= 0.5 ? 'T' : 'X';
    let confidence = Math.min(80, Math.max(56, Math.round(Math.abs(finalP - 0.5) * 200)));
    
    // Nếu có cầu VIP confidence cao, ưu tiên
    if (bestCau && bestConf > confidence + 5) {
        finalDecision = bestCau.pred;
        confidence = bestConf;
    }
    
    // Đảm bảo confidence 56-80
    confidence = Math.min(80, Math.max(56, confidence));
    
    let cauName = bestCau ? bestCau.name : 'Khong nhan dien';
    
    return {
        prediction: finalDecision,
        confidence: confidence,
        skip: false,
        cau: cauName,
        details: {
            distribution: distribution,
            streak: streakAnalysis,
            entropy: entropy,
            transProb: transProb,
            cauCount: cauResults.length,
            finalP: finalP
        }
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
        Id: "@tranhoang2286",
        Game: gameName,
        Phien_truoc: latest.session,
        Xuc_xac: `${latest.dice[0]} ${latest.dice[1]} ${latest.dice[2]}`,
        Ket_qua: latest.result,
        Phien_nay: latest.session + 1,
        Du_doan: result.prediction === "T" ? "T" : "X",
        Do_tin_cay: `${result.confidence}%`,
        Cau: result.cau || "Khong xac dinh",
        So_cau: result.details.cauCount || 0
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
        service: "Max789 Prediction API - VIP",
        author: "@tranhoang2286",
        version: "3.0",
        features: {
            cau_nhan_dien: "23 loai cau",
            phan_tich: "Thong ke, Markov, Pattern, Entropy",
            do_tin_cay: "56-80%"
        },
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
        console.log(`✅ Prediction API VIP started!`);
        console.log(`📊 23 loai cau duoc nhan dien`);
        console.log(`🎯 Do tin cay: 56-80%`);
    } catch (err) {
        console.error("Error starting server:", err.message);
        process.exit(1);
    }
};

start();
