import fastify from "fastify";
import cors from "@fastify/cors";
import fetch from "node-fetch";
import fs from "fs";

const PORT = 3000;

const API_MD5 = "https://taixiumd5.maksh3979madfw.com/api/md5luckydice/GetSoiCau";
const API_HU = "https://taixiu.maksh3979madfw.com/api/luckydice/GetSoiCau";

const LEARNING_FILE = "./learning_data.json";
const PREDICTION_FILE = "./prediction_history.json";
const PATTERN_DB_FILE = "./pattern_database.json";
const CAU_KNOWLEDGE_FILE = "./cau_knowledge.json";

const app = fastify({ logger: false });
await app.register(cors, { origin: "*" });

// ==================== KNOWLEDGE BASE - BỘ NÃO HỌC CẦU ====================

let cauKnowledge = {
    md5: {
        // Kiến thức về từng cầu: độ tin cậy, ngữ cảnh tốt/xấu
        cauProfiles: {},
        // Chuỗi hành vi: cầu nào hoạt động tốt trong điều kiện nào
        contexts: {
            byStreak: {},      // Theo độ dài chuỗi
            byEntropy: {},     // Theo entropy
            byVolatility: {},  // Theo độ biến động
            byBalance: {},     // Theo độ cân bằng
            byHour: {},        // Theo giờ
            byDay: {}          // Theo ngày
        },
        // Lịch sử học
        learningLog: [],
        // Trọng số đã học
        adaptiveWeights: {},
        // Cầu nào đang bị "quên" (sai nhiều)
        deprecatedCau: [],
        // Cầu nào đang "hot" (đúng nhiều)
        hotCau: []
    },
    hu: {
        cauProfiles: {},
        contexts: {
            byStreak: {},
            byEntropy: {},
            byVolatility: {},
            byBalance: {},
            byHour: {},
            byDay: {}
        },
        learningLog: [],
        adaptiveWeights: {},
        deprecatedCau: [],
        hotCau: []
    }
};

let learningData = {
    md5: {
        totalPredictions: 0, correct: 0, wrong: 0,
        history: [], cauStats: {}, streakStats: {}, patternStats: {},
        methodStats: {}, hourStats: {}, dayStats: {},
        confidenceStats: {}, consecutiveWrong: 0, consecutiveCorrect: 0
    },
    hu: {
        totalPredictions: 0, correct: 0, wrong: 0,
        history: [], cauStats: {}, streakStats: {}, patternStats: {},
        methodStats: {}, hourStats: {}, dayStats: {},
        confidenceStats: {}, consecutiveWrong: 0, consecutiveCorrect: 0
    }
};

let predictionHistory = { md5: [], hu: [] };
let patternDB = { md5: {}, hu: {} };

// ==================== LOAD/SAVE ====================

function loadAllData() {
    try {
        if (fs.existsSync(LEARNING_FILE)) {
            let d = JSON.parse(fs.readFileSync(LEARNING_FILE, 'utf8'));
            learningData = { ...learningData, ...d };
        }
        if (fs.existsSync(PREDICTION_FILE)) {
            predictionHistory = JSON.parse(fs.readFileSync(PREDICTION_FILE, 'utf8'));
        }
        if (fs.existsSync(PATTERN_DB_FILE)) {
            patternDB = JSON.parse(fs.readFileSync(PATTERN_DB_FILE, 'utf8'));
        }
        if (fs.existsSync(CAU_KNOWLEDGE_FILE)) {
            let d = JSON.parse(fs.readFileSync(CAU_KNOWLEDGE_FILE, 'utf8'));
            cauKnowledge = { ...cauKnowledge, ...d };
        }
    } catch (e) {}
}

function saveAllData() {
    try {
        fs.writeFileSync(LEARNING_FILE, JSON.stringify(learningData, null, 2));
        fs.writeFileSync(PREDICTION_FILE, JSON.stringify(predictionHistory, null, 2));
        fs.writeFileSync(PATTERN_DB_FILE, JSON.stringify(patternDB, null, 2));
        fs.writeFileSync(CAU_KNOWLEDGE_FILE, JSON.stringify(cauKnowledge, null, 2));
    } catch (e) {}
}

// ==================== HỆ THỐNG HỌC CẦU ====================

// Phân tích ngữ cảnh hiện tại
function analyzeContext(h) {
    let n = h.length;
    let streakLen = streak(h);
    let tCount = countIn(h, 'T', n);
    let rate = tCount / n;
    
    // Entropy
    let pT = rate, pX = 1 - rate;
    let entropy = 0;
    if (pT > 0) entropy -= pT * Math.log2(pT);
    if (pX > 0) entropy -= pX * Math.log2(pX);
    
    // Volatility
    let changes = 0;
    for (let i = 1; i < Math.min(n, 30); i++) if (h[i] !== h[i-1]) changes++;
    let volatility = changes / Math.min(n - 1, 29);
    
    // Balance
    let balance = Math.abs(rate - 0.5);
    let balanceLevel = balance < 0.05 ? 'balanced' : balance < 0.15 ? 'slight' : 'strong';
    
    // Streak level
    let streakLevel = streakLen >= 5 ? 'very_long' : streakLen >= 4 ? 'long' : streakLen >= 3 ? 'medium' : 'short';
    
    // Entropy level
    let entropyLevel = entropy > 0.9 ? 'high' : entropy > 0.7 ? 'medium' : 'low';
    
    // Volatility level
    let volLevel = volatility > 0.65 ? 'high' : volatility > 0.4 ? 'medium' : 'low';
    
    let hour = new Date().getHours();
    let day = new Date().getDay();
    
    return {
        streakLen, streakLevel,
        entropy, entropyLevel,
        volatility, volLevel,
        rate, balance, balanceLevel,
        hour, day,
        pattern6: h.slice(0, Math.min(6, n)).join(''),
        pattern8: h.slice(0, Math.min(8, n)).join('')
    };
}

// HỌC CẦU: Khi có kết quả, cập nhật knowledge
function learnCau(game, cauName, prediction, actual, context, confidence) {
    if (!cauName) return;
    
    let kb = cauKnowledge[game];
    let isCorrect = prediction === actual;
    
    // 1. Cập nhật profile của cầu
    if (!kb.cauProfiles[cauName]) {
        kb.cauProfiles[cauName] = {
            total: 0, correct: 0, wrong: 0,
            recentResults: [],     // 50 kết quả gần nhất
            byStreak: {},          // Hiệu suất theo streak
            byEntropy: {},         // Hiệu suất theo entropy
            byVolatility: {},      // Hiệu suất theo volatility
            byBalance: {},         // Hiệu suất theo balance
            byHour: {},            // Hiệu suất theo giờ
            byDay: {},             // Hiệu suất theo ngày
            byConfidence: {},      // Hiệu suất theo confidence
            bestContexts: [],      // Ngữ cảnh tốt nhất
            worstContexts: [],     // Ngữ cảnh xấu nhất
            lastUpdated: new Date().toISOString(),
            status: 'active'       // active, hot, deprecated
        };
    }
    
    let profile = kb.cauProfiles[cauName];
    profile.total++;
    if (isCorrect) profile.correct++;
    else profile.wrong++;
    
    // Lưu kết quả gần đây (max 50)
    profile.recentResults.push(isCorrect ? 1 : 0);
    if (profile.recentResults.length > 50) {
        profile.recentResults = profile.recentResults.slice(-50);
    }
    
    // 2. Học theo ngữ cảnh
    function updateContext(contextKey, contextValue) {
        let key = `${contextKey}_${contextValue}`;
        if (!profile[contextKey]) profile[contextKey] = {};
        if (!profile[contextKey][contextValue]) {
            profile[contextKey][contextValue] = { total: 0, correct: 0, wrong: 0 };
        }
        profile[contextKey][contextValue].total++;
        if (isCorrect) profile[contextKey][contextValue].correct++;
        else profile[contextKey][contextValue].wrong++;
    }
    
    updateContext('byStreak', context.streakLevel);
    updateContext('byEntropy', context.entropyLevel);
    updateContext('byVolatility', context.volLevel);
    updateContext('byBalance', context.balanceLevel);
    updateContext('byHour', context.hour);
    updateContext('byDay', context.day);
    
    // Học confidence range
    let confRange = Math.floor(confidence / 5) * 5;
    if (!profile.byConfidence[confRange]) profile.byConfidence[confRange] = { total: 0, correct: 0, wrong: 0 };
    profile.byConfidence[confRange].total++;
    if (isCorrect) profile.byConfidence[confRange].correct++;
    else profile.byConfidence[confRange].wrong++;
    
    // 3. Cập nhật contexts chung
    function updateGlobalContext(catKey, contextValue) {
        let cat = kb.contexts[catKey];
        if (!cat[contextValue]) cat[contextValue] = { total: 0, correct: 0, wrong: 0 };
        cat[contextValue].total++;
        if (isCorrect) cat[contextValue].correct++;
        else cat[contextValue].wrong++;
    }
    
    updateGlobalContext('byStreak', context.streakLevel);
    updateGlobalContext('byEntropy', context.entropyLevel);
    updateGlobalContext('byVolatility', context.volLevel);
    updateGlobalContext('byBalance', context.balanceLevel);
    updateGlobalContext('byHour', context.hour);
    updateGlobalContext('byDay', context.day);
    
    // 4. Cập nhật trọng số thích ứng
    let recentAcc = profile.recentResults.length > 0 
        ? profile.recentResults.reduce((a, b) => a + b, 0) / profile.recentResults.length 
        : 0.5;
    
    // Trọng số = kết hợp độ chính xác tổng + gần đây
    let totalAcc = profile.total > 0 ? profile.correct / profile.total : 0.5;
    let adaptiveWeight = (totalAcc * 0.4 + recentAcc * 0.6);
    
    // Penalty nếu sai liên tiếp
    let last5 = profile.recentResults.slice(-5);
    let last5Wrong = last5.filter(r => r === 0).length;
    if (last5Wrong >= 4) adaptiveWeight *= 0.3;
    else if (last5Wrong >= 3) adaptiveWeight *= 0.6;
    
    // Bonus nếu đúng liên tiếp
    let last5Correct = last5.filter(r => r === 1).length;
    if (last5Correct >= 4) adaptiveWeight *= 1.5;
    else if (last5Correct >= 3) adaptiveWeight *= 1.2;
    
    kb.adaptiveWeights[cauName] = adaptiveWeight;
    
    // 5. Cập nhật trạng thái cầu
    if (profile.total >= 10) {
        if (totalAcc >= 0.7 && recentAcc >= 0.6) {
            profile.status = 'hot';
            if (!kb.hotCau.includes(cauName)) kb.hotCau.push(cauName);
            let idx = kb.deprecatedCau.indexOf(cauName);
            if (idx >= 0) kb.deprecatedCau.splice(idx, 1);
        } else if (totalAcc <= 0.35 || last5Wrong >= 4) {
            profile.status = 'deprecated';
            if (!kb.deprecatedCau.includes(cauName)) kb.deprecatedCau.push(cauName);
            let idx = kb.hotCau.indexOf(cauName);
            if (idx >= 0) kb.hotCau.splice(idx, 1);
        } else {
            profile.status = 'active';
            let idx1 = kb.hotCau.indexOf(cauName);
            if (idx1 >= 0) kb.hotCau.splice(idx1, 1);
            let idx2 = kb.deprecatedCau.indexOf(cauName);
            if (idx2 >= 0) kb.deprecatedCau.splice(idx2, 1);
        }
    }
    
    // 6. Tìm best/worst contexts
    let contextsWithData = [];
    ['byStreak', 'byEntropy', 'byVolatility', 'byBalance'].forEach(cat => {
        if (profile[cat]) {
            for (let key in profile[cat]) {
                let c = profile[cat][key];
                if (c.total >= 3) {
                    contextsWithData.push({
                        category: cat,
                        value: key,
                        acc: c.correct / c.total,
                        total: c.total
                    });
                }
            }
        }
    });
    
    contextsWithData.sort((a, b) => b.acc - a.acc);
    profile.bestContexts = contextsWithData.slice(0, 3);
    profile.worstContexts = contextsWithData.slice(-3).reverse();
    
    profile.lastUpdated = new Date().toISOString();
    
    // 7. Log việc học
    kb.learningLog.push({
        time: new Date().toISOString(),
        cau: cauName,
        correct: isCorrect,
        context: {
            streak: context.streakLevel,
            entropy: context.entropyLevel,
            volatility: context.volLevel,
            balance: context.balanceLevel
        },
        adaptiveWeight: adaptiveWeight.toFixed(3)
    });
    
    if (kb.learningLog.length > 500) kb.learningLog = kb.learningLog.slice(-500);
    
    saveAllData();
}

// Lấy trọng số đã học của cầu
function getLearnedCauWeight(game, cauName, currentContext) {
    let kb = cauKnowledge[game];
    if (!kb || !kb.cauProfiles[cauName]) return 1;
    
    let profile = kb.cauProfiles[cauName];
    
    // Base weight từ adaptive
    let weight = kb.adaptiveWeights[cauName] || 0.5;
    
    // Boost nếu context hiện tại phù hợp với best context
    if (profile.bestContexts.length > 0) {
        for (let bc of profile.bestContexts) {
            let ctxValue = currentContext[bc.category.replace('by', '').toLowerCase()];
            if (bc.category === 'byStreak' && bc.value === currentContext.streakLevel) {
                weight *= 1.3;
                break;
            }
            if (bc.category === 'byEntropy' && bc.value === currentContext.entropyLevel) {
                weight *= 1.3;
                break;
            }
            if (bc.category === 'byVolatility' && bc.value === currentContext.volLevel) {
                weight *= 1.3;
                break;
            }
            if (bc.category === 'byBalance' && bc.value === currentContext.balanceLevel) {
                weight *= 1.3;
                break;
            }
        }
    }
    
    // Penalty nếu context hiện tại nằm trong worst context
    if (profile.worstContexts.length > 0) {
        for (let wc of profile.worstContexts) {
            if (wc.category === 'byStreak' && wc.value === currentContext.streakLevel) {
                weight *= 0.5;
                break;
            }
            if (wc.category === 'byEntropy' && wc.value === currentContext.entropyLevel) {
                weight *= 0.5;
                break;
            }
            if (wc.category === 'byVolatility' && wc.value === currentContext.volLevel) {
                weight *= 0.5;
                break;
            }
            if (wc.category === 'byBalance' && wc.value === currentContext.balanceLevel) {
                weight *= 0.5;
                break;
            }
        }
    }
    
    // Status bonus/penalty
    if (profile.status === 'hot') weight *= 1.5;
    if (profile.status === 'deprecated') weight *= 0.2;
    
    return Math.max(0.1, Math.min(3, weight));
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

// ==================== 40 MODULES ====================

function M1_Dist(h) { let n = h.length, t = countIn(h, 'T', n); let r = t/n; return { t, x: n-t, rate: r, imbalance: Math.abs(r-0.5), isBalanced: Math.abs(r-0.5) < 0.06 }; }
function M2_Streak(h) { let n = h.length, s = [], i = 0; while (i < n) { let x = 1; while (i+x < n && h[i+x] === h[i+x-1]) x++; s.push({v:h[i],len:x}); i+=x; } return { streaks: s, maxS: s.reduce((m,x)=>Math.max(m,x.len),0), avgS: s.length>0?s.reduce((a,b)=>a+b.len,0)/s.length:0, count: s.length }; }
function M3_Recent(h) { let r = {}; for (let w of [5,10,15,20,30,50,100]) if (h.length >= w) { let t = countIn(h,'T',w); r[w] = {t,x:w-t,rate:t/w,imbalance:Math.abs(t/w-0.5)}; } return r; }
function M4_Entropy(h) { let n = h.length, t = countIn(h,'T',n); let pT = t/n, pX = (n-t)/n, e = 0; if (pT>0) e -= pT*Math.log2(pT); if (pX>0) e -= pX*Math.log2(pX); return e; }
function M5_Trans(h) { let n = h.length; if (n<2) return {tt:0.5,tx:0.5,xt:0.5,xx:0.5,stay:0.5,change:0.5}; let tt=0,tx=0,xt=0,xx=0; for (let i=0;i<n-1;i++){if(h[i]==='T'&&h[i+1]==='T')tt++;else if(h[i]==='T'&&h[i+1]==='X')tx++;else if(h[i]==='X'&&h[i+1]==='T')xt++;else xx++;} let tT=tt+tx||1,tX=xt+xx||1; return {tt:tt/tT,tx:tx/tT,xt:xt/tX,xx:xx/tX,stay:(tt+xx)/(n-1),change:(tx+xt)/(n-1)}; }
function M6_Rev(h) { let n = h.length; if (n<5) return {r3T:0.5,r3X:0.5,r4T:0.5,r4X:0.5,r5T:0.5,r5X:0.5}; let a3T=0,a3TX=0,a3X=0,a3XT=0,a4T=0,a4TX=0,a4X=0,a4XT=0,a5T=0,a5TX=0,a5X=0,a5XT=0; for(let i=3;i<n;i++){if(h[i-1]==='T'&&h[i-2]==='T'&&h[i-3]==='T'){a3T++;if(h[i]==='X')a3TX++;}if(h[i-1]==='X'&&h[i-2]==='X'&&h[i-3]==='X'){a3X++;if(h[i]==='T')a3XT++;}} for(let i=4;i<n;i++){if(h[i-1]==='T'&&h[i-2]==='T'&&h[i-3]==='T'&&h[i-4]==='T'){a4T++;if(h[i]==='X')a4TX++;}if(h[i-1]==='X'&&h[i-2]==='X'&&h[i-3]==='X'&&h[i-4]==='X'){a4X++;if(h[i]==='T')a4XT++;}} for(let i=5;i<n;i++){if(h[i-1]==='T'&&h[i-2]==='T'&&h[i-3]==='T'&&h[i-4]==='T'&&h[i-5]==='T'){a5T++;if(h[i]==='X')a5TX++;}if(h[i-1]==='X'&&h[i-2]==='X'&&h[i-3]==='X'&&h[i-4]==='X'&&h[i-5]==='X'){a5X++;if(h[i]==='T')a5XT++;}} return {r3T:a3T>0?a3TX/a3T:0.5,r3X:a3X>0?a3XT/a3X:0.5,r4T:a4T>0?a4TX/a4T:0.5,r4X:a4X>0?a4XT/a4X:0.5,r5T:a5T>0?a5TX/a5T:0.5,r5X:a5X>0?a5XT/a5X:0.5}; }
function M7_Markov(h) { let r = {}; for (let k=1;k<=8;k++){ if(h.length<=k)continue; let ctx=h.slice(0,k).join(''); let c=0,nt=0; for(let i=0;i<=h.length-k-1;i++){if(h.slice(i,i+k).join('')===ctx){c++;if(h[i+k]==='T')nt++;}} if(c>=2)r[`k${k}`]={prob:bayesP(nt,c),count:c}; } return r; }
function M8_Pattern(h) { let n=Math.min(h.length,60); if(n<6)return{prob:0.5,bestLen:0,score:0}; let best={prob:0.5,bestLen:0,score:0}; for(let len=3;len<=10;len++){let pat=h.slice(0,len).join('');let total=0,nextT=0;for(let i=1;i<=n-len-1;i++){let sub=h.slice(i,i+len).join('');let m=0;for(let j=0;j<len;j++)if(pat[j]===sub[j])m++;if(m>=len-1){total++;if(i+len<n&&h[i+len]==='T')nextT++;}}if(total>=2){let prob=nextT/total;let score=total*Math.abs(prob-0.5);if(score>best.score)best={prob,bestLen:len,score};}} return best; }
function M9_Weighted(h) { let n=Math.min(h.length,50); if(n<3)return 0.5; let tW=0,tT=0; for(let i=0;i<n;i++){let w=Math.exp(-i/10);tW+=w;if(h[i]==='T')tT+=w;} return tT/tW; }
function M10_Cycle(h) { let n=Math.min(h.length,50); if(n<12)return null; let best=null; for(let p=2;p<=10;p++){let m=0,t=0;for(let i=0;i<n-p;i++){t++;if(h[i]===h[i+p])m++;}if(t<10)continue;let r=m/t;if(r>0.7){let pred=h[p-1];if(!best||r>best.rate)best={rate:r,period:p,pred};}} return best; }
function M11_CUSUM(h) { let n=h.length; if(n<10)return 0.5; let target=countIn(h,'T',n)/n; let c=0,mC=0; for(let i=0;i<Math.min(n,60);i++){let v=h[i]==='T'?1:0;c=Math.max(0,c+(v-target)-0.05);if(c>mC)mC=c;} if(mC>2.5)return 1-countIn(h,'T',Math.min(10,n))/Math.min(10,n); return 0.5; }
function M12_Bayes(h) { let n=h.length,t=countIn(h,'T',n); return (t+5)/(n+10); }
function M13_Mom(h) { let n=h.length; if(n<5)return 0.5; let s=countIn(h,'T',Math.min(5,n))/Math.min(5,n); let l=countIn(h,'T',Math.min(20,n))/Math.min(20,n); return s*0.6+l*0.4; }
function M14_Vol(h) { let n=h.length; if(n<5)return 0.5; let c=0; for(let i=1;i<Math.min(n,30);i++)if(h[i]!==h[i-1])c++; return c/Math.min(n-1,29); }
function M15_Anti(h) { let n=h.length; if(n<5)return 0.5; let last=h[0],c=0; for(let i=1;i<Math.min(n,15);i++)if(h[i]!==last)c++; return c/Math.min(n-1,14); }
function M16_MACD(h) { let n=h.length; if(n<12)return 0.5; let e12=0.5,e26=0.5,a12=2/13,a26=2/27; for(let i=n-1;i>=0;i--){let v=h[i]==='T'?1:0;e12=v*a12+e12*(1-a12);e26=v*a26+e26*(1-a26);} return sigmoid((e12-e26)*5); }
function M17_RSI(h) { let n=Math.min(h.length,14); if(n<5)return 0.5; let g=0,l=0; for(let i=0;i<n-1;i++){let c=h[i]==='T'?1:0,p=h[i+1]==='T'?1:0;if(c>p)g++;else if(c<p)l++;} if(g+l===0)return 0.5; let rs=g/(l||1); return 1-(100/(100+rs))/100; }
function M18_Boll(h) { let n=Math.min(h.length,20); if(n<5)return 0.5; let v=h.slice(0,n).map(x=>x==='T'?1:0); let m=v.reduce((a,b)=>a+b,0)/n; let s=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/n); let c=v[0]; if(c>m+s)return 0.3; if(c<m-s)return 0.7; return 0.5; }
function M19_Fib(h) { let n=Math.min(h.length,21); if(n<10)return 0.5; let sum=0,a=1,b=1,w=[]; for(let i=0;i<n;i++){w.push(a);let nx=a+b;a=b;b=nx;} let tW=w.reduce((x,y)=>x+y,0); for(let i=0;i<n;i++)if(h[i]==='T')sum+=w[i]; return sum/tW; }
function M20_Zig(h) { let n=h.length; if(n<6)return 0.5; let zg=0,za=0,pd=0; for(let i=1;i<Math.min(n,25);i++){let d=h[i]===h[i-1]?0:(h[i]==='T'?1:-1);if(d!==0&&d!==pd){if(d>0)za++;else zg++;pd=d;}} if(zg+za<3)return 0.5; return za/(zg+za); }
function M21_Hurst(h) { let n=Math.min(h.length,50); if(n<20)return 0.5; let v=h.slice(0,n).map(x=>x==='T'?1:-1); let lags=[2,4,8,16]; let tau=[]; for(let lag of lags){let d=[];for(let i=0;i<v.length-lag;i++)d.push(v[i]-v[i+lag]);let vr=d.reduce((a,b)=>a+b*b,0)/d.length;tau.push(Math.sqrt(vr));} if(tau.every(t=>t>0)){let lT=tau.map(t=>Math.log(t)),lL=lags.map(l=>Math.log(l));let sl=(lT[3]-lT[0])/(lL[3]-lL[0]);return Math.max(0,Math.min(1,sl/2+0.5));} return 0.5; }
function M22_Auto(h) { let n=Math.min(h.length,40); if(n<15)return 0.5; let v=h.slice(0,n).map(x=>x==='T'?1:0); let m=v.reduce((a,b)=>a+b,0)/n; let va=v.reduce((a,b)=>a+(b-m)**2,0)/n; if(va<0.001)return 0.5; let bl=1,bc=0; for(let lag=1;lag<=Math.min(10,Math.floor(n/3));lag++){let c=0;for(let i=0;i<n-lag;i++)c+=(v[i]-m)*(v[i+lag]-m);c/=(n-lag)*va;if(Math.abs(c)>Math.abs(bc)){bc=c;bl=lag;}} let last=h[0]==='T'?1:0; let target=h[bl]==='T'?1:0; if(bc>0)return target===1?0.6:0.4; return target===1?0.4:0.6; }
function M23_Frac(h) { let n=Math.min(h.length,50); if(n<20)return 0.5; let c=0; for(let i=1;i<n;i++)if(h[i]!==h[i-1])c++; return Math.min(1,Math.max(0,Math.log(c+1)/Math.log(n))); }
function M24_Wave(h) { let n=Math.min(h.length,32); if(n<8)return 0.5; let c=0,f=0; for(let i=0;i<n;i++){let v=h[i]==='T'?1:0;if(i%2===0)c+=v;else f+=v;} return (c/Math.ceil(n/2))*0.6+(f/Math.floor(n/2))*0.4; }
function M25_LocalEnt(h) { let n=h.length; if(n<10)return 0.5; let es=[]; for(let w=5;w<=20;w+=5){if(n>=w){let seg=h.slice(0,w);let t=countIn(seg,'T',w);let pT=t/w,pX=1-pT;let e=0;if(pT>0)e-=pT*Math.log2(pT);if(pX>0)e-=pX*Math.log2(pX);es.push(e);}} return es.length>0?es.reduce((a,b)=>a+b,0)/es.length:0.5; }
function M26_Skew(h) { let n=h.length; if(n<10)return 0; let v=h.map(x=>x==='T'?1:0); let m=v.reduce((a,b)=>a+b,0)/n; let s=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/n); if(s<0.001)return 0; return v.reduce((a,b)=>a+((b-m)/s)**3,0)/n; }
function M27_Kurt(h) { let n=h.length; if(n<10)return 0; let v=h.map(x=>x==='T'?1:0); let m=v.reduce((a,b)=>a+b,0)/n; let s=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/n); if(s<0.001)return 0; return v.reduce((a,b)=>a+((b-m)/s)**4,0)/n-3; }
function M28_RunL(h) { let n=h.length; if(n<5)return 0.5; let runs=[],i=0; while(i<n){let s=1;while(i+s<n&&h[i+s]===h[i+s-1])s++;runs.push(s);i+=s;} let m=runs.reduce((a,b)=>a+b,0)/runs.length; let vr=runs.reduce((a,b)=>a+(b-m)**2,0)/runs.length; return m/(1+Math.sqrt(vr)); }
function M29_PatFreq(h) { let n=h.length; if(n<10)return 0.5; let f={}; for(let len=2;len<=4;len++){for(let i=0;i<n-len;i++){let p=h.slice(i,i+len).join('');f[p]=(f[p]||0)+1;}} let mF=0,mP=''; for(let p in f){if(f[p]>mF){mF=f[p];mP=p;}} if(!mP)return 0.5; let tC=0; for(let c of mP)if(c==='T')tC++; return tC/mP.length; }
function M30_RecBias(h) { let n=Math.min(h.length,30); if(n<3)return 0.5; let rec=h.slice(0,5),old=h.slice(5,15); let rT=countIn(rec,'T',rec.length)/(rec.length||1); let oT=countIn(old,'T',old.length)/(old.length||1); return rT*0.7+oT*0.3; }
function M31_SupRes(h) { let n=Math.min(h.length,50); if(n<10)return 0.5; let r=countIn(h,'T',n)/n; if(r>0.65)return 0.35; if(r<0.35)return 0.65; return 0.5; }
function M32_Stoch(h) { let n=Math.min(h.length,14); if(n<5)return 0.5; return countIn(h,'T',n)/n; }
function M33_ADX(h) { let n=Math.min(h.length,14); if(n<8)return 0.5; let c=0; for(let i=1;i<n;i++)if(h[i]!==h[i-1])c++; return c/(n-1); }
function M34_WillR(h) { let n=Math.min(h.length,14); if(n<5)return 0.5; return 1-countIn(h,'T',n)/n; }
function M35_CCI(h) { let n=Math.min(h.length,20); if(n<10)return 0.5; let v=h.slice(0,n).map(x=>x==='T'?1:0); let m=v.reduce((a,b)=>a+b,0)/n; let d=v.reduce((a,b)=>a+Math.abs(b-m),0)/n; if(d<0.001)return 0.5; return sigmoid((v[0]-m)/(0.015*d)); }
function M36_ATR(h) { let n=Math.min(h.length,14); if(n<5)return 0.5; let c=0; for(let i=1;i<n;i++)if(h[i]!==h[i-1])c++; return c/(n-1); }
function M37_Vortex(h) { let n=Math.min(h.length,14); if(n<5)return 0.5; let u=0,d=0; for(let i=1;i<n;i++){if(h[i]==='T'&&h[i-1]==='X')u++;if(h[i]==='X'&&h[i-1]==='T')d++;} if(u+d===0)return 0.5; return u/(u+d); }
function M38_Trix(h) { let n=Math.min(h.length,15); if(n<9)return 0.5; let e1=0.5,e2=0.5,e3=0.5,a=2/4; for(let i=n-1;i>=0;i--){let v=h[i]==='T'?1:0;e1=v*a+e1*(1-a);e2=e1*a+e2*(1-a);e3=e2*a+e3*(1-a);} return e3; }
function M39_UO(h) { let n=Math.min(h.length,28); if(n<10)return 0.5; let r=countIn(h,'T',n)/n; let s=countIn(h,'T',Math.min(7,n))/Math.min(7,n); let l=countIn(h,'T',Math.min(14,n))/Math.min(14,n); return (s*2+l+r)/4; }
function M40_MeanRev(h) { let n=Math.min(h.length,50); if(n<10)return 0.5; let m=countIn(h,'T',n)/n; let last=h[0]==='T'?1:0; if(last>m+0.1)return 0.35; if(last<m-0.1)return 0.65; return 0.5; }

// ==================== NHẬN DIỆN CẦU ====================

function checkRuns(runs, pattern) {
    if (runs.length < pattern.length) return false;
    for (let i = 0; i < pattern.length; i++) if (runs[i].len !== pattern[i]) return false;
    return true;
}

function getRuns(h, n) {
    let runs = [], i = 0;
    while (i < n) {
        let s = 1;
        while (i + s < n && h[i + s] === h[i + s - 1]) s++;
        runs.push({ v: h[i], len: s });
        i += s;
    }
    return runs;
}

function makeDetector(name, patterns, confidence) {
    return function(h) {
        let n = Math.min(h.length, 20);
        if (n < 8) return null;
        let runs = getRuns(h, n);
        for (let pat of patterns) {
            if (checkRuns(runs, pat)) {
                let last = h[h.length - 1];
                return { pred: last === 'T' ? 'X' : 'T', conf: confidence, name };
            }
        }
        return null;
    };
}

let detectors = [
    makeDetector('Cau 1-1', [[1,1]], 65),
    makeDetector('Cau 2-1', [[2,1],[2,1]], 68),
    makeDetector('Cau 1-2', [[1,2],[1,2]], 68),
    makeDetector('Cau 3-1', [[3,1],[3,1]], 70),
    makeDetector('Cau 1-3', [[1,3],[1,3]], 70),
    makeDetector('Cau 2-2', [[2,2],[2,2]], 72),
    makeDetector('Cau 3-2', [[3,2],[3,2]], 74),
    makeDetector('Cau 2-3', [[2,3],[2,3]], 74),
    makeDetector('Cau 4-1', [[4,1],[4,1]], 72),
    makeDetector('Cau 1-4', [[1,4],[1,4]], 72),
    makeDetector('Cau 5-1', [[5,1],[5,1]], 74),
    makeDetector('Cau 1-5', [[1,5],[1,5]], 74),
    makeDetector('Cau 4-2', [[4,2],[4,2]], 74),
    makeDetector('Cau 2-4', [[2,4],[2,4]], 74),
    makeDetector('Cau 3-3', [[3,3],[3,3]], 76),
    makeDetector('Cau 1-1-2', [[1,1,2]], 66),
    makeDetector('Cau 2-1-1', [[2,1,1]], 66),
    makeDetector('Cau 1-2-1', [[1,2,1]], 68),
];

function detectCauXenKe(h) {
    let n = Math.min(h.length, 10);
    if (n < 4) return null;
    let alt = 0;
    for (let i = 1; i < n; i++) if (h[i] !== h[i-1]) alt++;
    if (alt / (n - 1) > 0.85) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 70 + Math.min(10, n * 2), name: 'Cau Xen Ke' };
    }
    return null;
}

function detectCauDai(h) {
    let s = streak(h);
    if (s >= 4) {
        let last = h[0];
        return { pred: last === 'T' ? 'X' : 'T', conf: Math.min(81, 60 + s * 4), name: `Cau Dai ${s}` };
    }
    return null;
}

function detectCauChuKy(h) {
    let n = Math.min(h.length, 40);
    if (n < 8) return null;
    for (let p = 2; p <= 10; p++) {
        let m = 0;
        for (let i = 0; i < n - p; i++) if (h[i] === h[i + p]) m++;
        let rate = m / (n - p);
        if (rate > 0.75 && n - p >= 8) {
            return { pred: h[p - 1], conf: Math.min(80, 60 + rate * 20), name: `Cau Chu Ky ${p}` };
        }
    }
    return null;
}

function detectCauDoiXung(h) {
    let n = Math.min(h.length, 14);
    if (n < 8) return null;
    let half = Math.floor(n / 2);
    let m = 0;
    for (let i = 0; i < half; i++) if (h[i] === h[n - 1 - i]) m++;
    if (m / half > 0.8) {
        let last = h[h.length - 1];
        return { pred: last === 'T' ? 'X' : 'T', conf: 66 + Math.min(14, m / half * 15), name: 'Cau Doi Xung' };
    }
    return null;
}

function detectCauTangDan(h) {
    let n = Math.min(h.length, 12);
    if (n < 6) return null;
    let runs = getRuns(h, n);
    if (runs.length >= 3) {
        let inc = true;
        for (let j = 1; j < runs.length; j++) if (runs[j].len <= runs[j-1].len) { inc = false; break; }
        if (inc) { let last = h[h.length - 1]; return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau Tang Dan' }; }
    }
    return null;
}

function detectCauGiamDan(h) {
    let n = Math.min(h.length, 12);
    if (n < 6) return null;
    let runs = getRuns(h, n);
    if (runs.length >= 3) {
        let dec = true;
        for (let j = 1; j < runs.length; j++) if (runs[j].len >= runs[j-1].len) { dec = false; break; }
        if (dec) { let last = h[h.length - 1]; return { pred: last === 'T' ? 'X' : 'T', conf: 68, name: 'Cau Giam Dan' }; }
    }
    return null;
}

detectors.push(detectCauXenKe, detectCauDai, detectCauChuKy, detectCauDoiXung, detectCauTangDan, detectCauGiamDan);

// ==================== THUẬT TOÁN CHÍNH VỚI HỌC CẦU ====================

function computePrediction(h, game) {
    let d = h.slice(0, Math.min(h.length, 300));
    let n = d.length;
    
    if (n < 3) {
        let last = d[d.length - 1] || 'X';
        return { prediction: last === 'T' ? 'X' : 'T', confidence: 56, cau: 'It du lieu', method: 'none', context: null };
    }

    // Phân tích ngữ cảnh hiện tại
    let context = analyzeContext(d);
    
    // Chạy modules
    let M1 = M1_Dist(d), M2 = M2_Streak(d), M4 = M4_Entropy(d);
    let M6 = M6_Rev(d), M7 = M7_Markov(d), M8 = M8_Pattern(d);
    let M9 = M9_Weighted(d), M10 = M10_Cycle(d), M11 = M11_CUSUM(d);
    let M12 = M12_Bayes(d), M13 = M13_Mom(d), M14 = M14_Vol(d);
    let M16 = M16_MACD(d), M17 = M17_RSI(d), M18 = M18_Boll(d);
    let M19 = M19_Fib(d), M20 = M20_Zig(d), M21 = M21_Hurst(d);
    let M22 = M22_Auto(d), M24 = M24_Wave(d), M25 = M25_LocalEnt(d);
    let M28 = M28_RunL(d), M29 = M29_PatFreq(d), M30 = M30_RecBias(d);
    let M31 = M31_SupRes(d), M32 = M32_Stoch(d), M33 = M33_ADX(d);
    let M35 = M35_CCI(d), M37 = M37_Vortex(d), M38 = M38_Trix(d);
    let M39 = M39_UO(d), M40 = M40_MeanRev(d);

    // Nhận diện cầu - ÁP DỤNG TRỌNG SỐ ĐÃ HỌC
    let cauResults = [];
    for (let det of detectors) {
        let r = det(d);
        if (r) {
            // Lấy trọng số đã học từ knowledge base
            let learnedWeight = getLearnedCauWeight(game, r.name, context);
            
            // Kết hợp với trọng số từ learning data
            let dataWeight = getCauWeight(game, r.name);
            
            r.weightedConf = r.conf * learnedWeight * dataWeight;
            r.learnedWeight = learnedWeight;
            r.dataWeight = dataWeight;
            cauResults.push(r);
        }
    }
    
    // Sort theo weightedConf
    cauResults.sort((a, b) => b.weightedConf - a.weightedConf);
    let bestCau = cauResults[0] || null;

    let streakLen = streak(d);
    let last = d[0];
    
    // TẦNG 1: ĐẢO CHIỀU CHUỖI
    if (streakLen >= 6) {
        return { prediction: last === 'T' ? 'X' : 'T', confidence: 81, cau: `Dao chieu chuoi ${streakLen}`, method: 'reversal_L1', context };
    }
    if (streakLen >= 5) {
        return { prediction: last === 'T' ? 'X' : 'T', confidence: 78, cau: `Dao chieu chuoi ${streakLen}`, method: 'reversal_L2', context };
    }
    if (streakLen >= 4) {
        return { prediction: last === 'T' ? 'X' : 'T', confidence: last === 'T' ? 81 : 76, cau: `Dao chieu chuoi ${streakLen}`, method: 'reversal_L3', context };
    }
    if (streakLen >= 3) {
        return { prediction: last === 'T' ? 'X' : 'T', confidence: last === 'T' ? 72 : 68, cau: `Dao chieu chuoi ${streakLen}`, method: 'reversal_L4', context };
    }

    // TẦNG 2: CHU KỲ
    if (M10 && M10.rate > 0.75) {
        return { prediction: M10.pred, confidence: Math.min(80, 60 + M10.rate * 20), cau: `Chu ky ${M10.period}`, method: 'cycle', context };
    }

    // TẦNG 3: CẦU VIP ĐÃ HỌC
    if (bestCau && bestCau.weightedConf >= 50) {
        return {
            prediction: bestCau.pred,
            confidence: Math.min(80, Math.round(bestCau.weightedConf)),
            cau: bestCau.name,
            method: 'cau_hoc_vip',
            context
        };
    }

    // TẦNG 4: XU HƯỚNG
    if (n >= 10) {
        let t10 = countIn(d, 'T', 10), x10 = 10 - t10;
        if (t10 >= 7) return { prediction: 'X', confidence: 72, cau: 'Xu huong Tai manh', method: 'trend_L1', context };
        if (x10 >= 7) return { prediction: 'T', confidence: 70, cau: 'Xu huong Xiu manh', method: 'trend_L2', context };
    }

    // TẦNG 5: TỔNG HỢP 40 MODULES
    let signals = [], weights = [];
    
    for (let key in M7) {
        let k = parseInt(key.substring(1));
        signals.push(M7[key].prob);
        weights.push(k * 0.025);
    }
    
    if (streakLen === 2) {
        if (last === 'T') { signals.push(1 - M6.r3T); weights.push(0.12); }
        else { signals.push(M6.r3T); weights.push(0.12); }
    }
    
    if (M8.score > 0) { signals.push(M8.prob); weights.push(0.08 + Math.min(0.1, M8.score / 20)); }
    
    signals.push(M12, M9, M13, M16, M17, M18, M19, M20, M21, M22, M24);
    weights.push(0.07, 0.07, 0.05, 0.05, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.03);
    
    signals.push(M1.rate, M11, M28, M29, M30, M31, M32, M33, M35, M37, M38, M39, M40);
    weights.push(0.04, 0.04, 0.02, 0.02, 0.03, 0.02, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03);
    
    let totalW = weights.reduce((a, b) => a + b, 0);
    let finalP = 0;
    for (let i = 0; i < signals.length; i++) {
        finalP += signals[i] * (weights[i] / totalW);
    }
    
    // Điều chỉnh
    if (M4 < 0.7) finalP = finalP * 0.7 + M1.rate * 0.3;
    if (M14 < 0.3) finalP = finalP * 0.7 + M1.rate * 0.3;
    if (M14 > 0.7) {
        if (last === 'T') finalP = Math.max(0.45, finalP - 0.05);
        else finalP = Math.min(0.55, finalP + 0.05);
    }
    if (M1.isBalanced && streakLen <= 2) {
        if (last === 'T') finalP = Math.min(0.53, finalP - 0.02);
        else finalP = Math.max(0.47, finalP + 0.02);
    }
    
    finalP = Math.min(0.85, Math.max(0.15, finalP));
    
    let finalDecision = finalP >= 0.5 ? 'T' : 'X';
    let confidence = Math.round(Math.abs(finalP - 0.5) * 200);
    confidence = Math.min(80, Math.max(56, confidence));
    
    if (bestCau && bestCau.weightedConf > confidence + 3) {
        finalDecision = bestCau.pred;
        confidence = Math.min(80, Math.round(bestCau.weightedConf));
    }
    
    return {
        prediction: finalDecision,
        confidence,
        cau: bestCau ? bestCau.name : 'Smart 40 tang',
        method: 'smart_40',
        context
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
    
    // Build pattern DB
    buildPatternDB(gameKey, h.slice(0, Math.min(h.length, 100)));
    
    let result = computePrediction(h, gameKey);
    let latest = data.at(-1);

    // Check đúng/sai VÀ HỌC CẦU
    let prev = predictionHistory[gameKey] || [];
    let lastPred = prev[prev.length - 1];
    let status = null;
    let learningResult = null;
    
    if (lastPred && lastPred.session === latest.session) {
        status = lastPred.prediction === latest.tx ? 'DUNG' : 'SAI';
        
        // Cập nhật learning data
        learnFromHistory(gameKey, latest.session, lastPred.prediction, latest.tx,
            lastPred.cau, lastPred.streak, lastPred.pattern, lastPred.method, lastPred.confidence);
        
        // HỌC CẦU - Cập nhật knowledge base
        if (lastPred.context) {
            learnCau(gameKey, lastPred.cau, lastPred.prediction, latest.tx, lastPred.context, lastPred.confidence);
            learningResult = {
                cau_hoc: lastPred.cau,
                ket_qua: lastPred.prediction === latest.tx ? 'DUNG' : 'SAI',
                trong_so_moi: (cauKnowledge[gameKey].adaptiveWeights[lastPred.cau] || 0).toFixed(3),
                trang_thai: cauKnowledge[gameKey].cauProfiles[lastPred.cau]?.status || 'new'
            };
        }
    }
    
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
        context: result.context,
        time: new Date().toISOString()
    };
    
    if (!predictionHistory[gameKey]) predictionHistory[gameKey] = [];
    predictionHistory[gameKey].push(newPred);
    if (predictionHistory[gameKey].length > 1000) predictionHistory[gameKey] = predictionHistory[gameKey].slice(-1000);
    
    saveAllData();
    
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
        Hoc_cau: learningResult,
        Ngu_canh: result.context ? {
            Chuoi: result.context.streakLevel,
            Entropy: result.context.entropyLevel,
            Bien_dong: result.context.volLevel,
            Can_bang: result.context.balanceLevel
        } : null,
        Thong_ke: {
            Tong_du_doan: stats.totalPredictions,
            Dung: stats.correct,
            Sai: stats.wrong,
            Do_chinh_xac: `${accuracy}%`,
            Dung_lien_tiep: stats.consecutiveCorrect || 0,
            Sai_lien_tiep: stats.consecutiveWrong || 0
        }
    };
}

// ==================== HỌC CẦU FUNCTIONS ====================

function learnFromHistory(game, session, prediction, actual, cauName, streakLen, pattern, method, confidence) {
    let data = learningData[game];
    if (!data) return;
    
    let isCorrect = prediction === actual;
    data.totalPredictions++;
    if (isCorrect) { data.correct++; data.consecutiveWrong = 0; data.consecutiveCorrect = (data.consecutiveCorrect || 0) + 1; }
    else { data.wrong++; data.consecutiveCorrect = 0; data.consecutiveWrong = (data.consecutiveWrong || 0) + 1; }
    
    data.history.push({ session, prediction, actual, correct: isCorrect, cau: cauName, streak: streakLen, pattern, method, confidence, time: new Date().toISOString() });
    if (data.history.length > 5000) data.history = data.history.slice(-5000);
    
    if (cauName) {
        if (!data.cauStats[cauName]) data.cauStats[cauName] = { total: 0, correct: 0, wrong: 0, consecutiveWrong: 0 };
        data.cauStats[cauName].total++;
        if (isCorrect) { data.cauStats[cauName].correct++; data.cauStats[cauName].consecutiveWrong = 0; }
        else { data.cauStats[cauName].wrong++; data.cauStats[cauName].consecutiveWrong = (data.cauStats[cauName].consecutiveWrong || 0) + 1; }
    }
    
    if (method) {
        if (!data.methodStats[method]) data.methodStats[method] = { total: 0, correct: 0, wrong: 0 };
        data.methodStats[method].total++;
        if (isCorrect) data.methodStats[method].correct++;
        else data.methodStats[method].wrong++;
    }
    
    saveAllData();
}

function getCauWeight(game, cauName) {
    let data = learningData[game];
    if (!data || !data.cauStats[cauName]) return 1;
    let s = data.cauStats[cauName];
    if (s.total < 3) return 1;
    let acc = s.correct / s.total;
    let w = 1;
    if (acc > 0.75) w = 1.8;
    else if (acc > 0.65) w = 1.5;
    else if (acc > 0.55) w = 1.2;
    else if (acc < 0.35) w = 0.4;
    else if (acc < 0.45) w = 0.7;
    if (s.consecutiveWrong >= 3) w *= 0.5;
    else if (s.consecutiveWrong >= 2) w *= 0.75;
    return w;
}

function buildPatternDB(game, history) {
    for (let len = 2; len <= 8; len++) {
        for (let i = 0; i < history.length - len; i++) {
            let pattern = history.slice(i, i + len).join('');
            let next = history[i + len];
            if (!patternDB[game]) patternDB[game] = {};
            if (!patternDB[game][pattern]) patternDB[game][pattern] = { T: 0, X: 0 };
            patternDB[game][pattern][next]++;
        }
    }
}

// ==================== API ROUTES ====================

app.get("/tx/md5", async (req, rep) => {
    let r = await handlePrediction(API_MD5, "max789 md5", "md5");
    if (r.error) return rep.status(503).send({ error: r.error });
    return r;
});

app.get("/tx/hu", async (req, rep) => {
    let r = await handlePrediction(API_HU, "max789 hu", "hu");
    if (r.error) return rep.status(503).send({ error: r.error });
    return r;
});

// CHECK ĐÚNG/SAI
app.get("/check/md5", async () => analyzeCheck("md5", "max789 md5"));
app.get("/check/hu", async () => analyzeCheck("hu", "max789 hu"));

function analyzeCheck(game, gameName) {
    let stats = learningData[game] || {};
    let accuracy = stats.totalPredictions > 0 ? (stats.correct / stats.totalPredictions * 100).toFixed(1) : 0;
    
    return {
        Game: gameName,
        Tong_du_doan: stats.totalPredictions || 0,
        Dung: stats.correct || 0,
        Sai: stats.wrong || 0,
        Do_chinh_xac: `${accuracy}%`,
        Dung_lien_tiep: stats.consecutiveCorrect || 0,
        Sai_lien_tiep: stats.consecutiveWrong || 0,
        Lich_su_30_phien: (stats.history || []).slice(-30).map(h => ({
            phien: h.session,
            du_doan: h.prediction,
            ket_qua: h.actual,
            dung: h.correct ? '✅' : '❌',
            cau: h.cau,
            method: h.method,
            confidence: h.confidence
        }))
    };
}

// ==================== HỌC CẦU - ENDPOINTS ====================

// Xem kiến thức học được
app.get("/learn/md5", async () => analyzeLearn("md5", "max789 md5"));
app.get("/learn/hu", async () => analyzeLearn("hu", "max789 hu"));

function analyzeLearn(game, gameName) {
    let kb = cauKnowledge[game];
    if (!kb) return { error: "Chua co du lieu hoc" };
    
    let profiles = Object.entries(kb.cauProfiles || {}).map(([name, p]) => {
        let acc = p.total > 0 ? (p.correct / p.total * 100).toFixed(1) : 0;
        let recentAcc = p.recentResults.length > 0
            ? (p.recentResults.reduce((a, b) => a + b, 0) / p.recentResults.length * 100).toFixed(1)
            : 0;
        
        return {
            cau: name,
            tong: p.total,
            dung: p.correct,
            sai: p.wrong,
            do_chinh_xac: `${acc}%`,
            do_chinh_xac_gan_day: `${recentAcc}%`,
            trong_so_hoc: (kb.adaptiveWeights[name] || 0).toFixed(3),
            trang_thai: p.status,
            ngu_canh_tot: p.bestContexts.map(c => `${c.category}:${c.value}(${c.acc.toFixed(0)}%)`),
            ngu_canh_xau: p.worstContexts.map(c => `${c.category}:${c.value}(${c.acc.toFixed(0)}%)`),
            cap_nhat: p.lastUpdated
        };
    }).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    
    // Context chung
    let contextStats = {};
    for (let cat in kb.contexts) {
        contextStats[cat] = Object.entries(kb.contexts[cat]).map(([k, v]) => ({
            gia_tri: k,
            tong: v.total,
            dung: v.correct,
            do_chinh_xac: v.total > 0 ? `${(v.correct / v.total * 100).toFixed(1)}%` : '0%'
        })).sort((a, b) => parseFloat(b.do_chinh_xac) - parseFloat(a.do_chinh_xac));
    }
    
    return {
        Game: gameName,
        Tong_cau_da_hoc: profiles.length,
        Cau_hot: kb.hotCau || [],
        Cau_bi_quen: kb.deprecatedCau || [],
        Xep_hang_cau: profiles,
        Thong_ke_ngu_canh: contextStats,
        Log_hoc_gan_day: (kb.learningLog || []).slice(-20)
    };
}

// Xem chi tiết 1 cầu
app.get("/learn/cau/:name", async (req) => {
    let name = req.params.name;
    return {
        md5: cauKnowledge.md5?.cauProfiles?.[name] || null,
        hu: cauKnowledge.hu?.cauProfiles?.[name] || null
    };
});

// Pattern DB
app.get("/pattern/md5", async () => analyzePatternDB("md5", "max789 md5"));
app.get("/pattern/hu", async () => analyzePatternDB("hu", "max789 hu"));

function analyzePatternDB(game, gameName) {
    let db = patternDB[game] || {};
    let patterns = [];
    
    for (let pat in db) {
        let p = db[pat];
        let total = p.T + p.X;
        if (total >= 3) {
            patterns.push({
                pattern: pat,
                do_dai: pat.length,
                tong: total,
                T: p.T,
                X: p.X,
                du_doan: p.T > p.X ? 'T' : 'X',
                do_tin_cay: `${(Math.abs(p.T - p.X) / total * 100).toFixed(1)}%`
            });
        }
    }
    
    patterns.sort((a, b) => parseFloat(b.do_tin_cay) - parseFloat(a.do_tin_cay));
    
    return {
        Game: gameName,
        Tong_pattern: patterns.length,
        Top_50: patterns.slice(0, 50)
    };
}

app.get("/learn/reset", async () => {
    learningData = {
        md5: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {}, methodStats: {}, hourStats: {}, dayStats: {}, confidenceStats: {}, consecutiveWrong: 0, consecutiveCorrect: 0 },
        hu: { totalPredictions: 0, correct: 0, wrong: 0, history: [], cauStats: {}, streakStats: {}, patternStats: {}, methodStats: {}, hourStats: {}, dayStats: {}, confidenceStats: {}, consecutiveWrong: 0, consecutiveCorrect: 0 }
    };
    predictionHistory = { md5: [], hu: [] };
    patternDB = { md5: {}, hu: {} };
    cauKnowledge = {
        md5: { cauProfiles: {}, contexts: { byStreak: {}, byEntropy: {}, byVolatility: {}, byBalance: {}, byHour: {}, byDay: {} }, learningLog: [], adaptiveWeights: {}, deprecatedCau: [], hotCau: [] },
        hu: { cauProfiles: {}, contexts: { byStreak: {}, byEntropy: {}, byVolatility: {}, byBalance: {}, byHour: {}, byDay: {} }, learningLog: [], adaptiveWeights: {}, deprecatedCau: [], hotCau: [] }
    };
    saveAllData();
    return { status: "reset", message: "Da reset toan bo he thong hoc" };
});

app.get("/", async () => ({
    status: "active",
    service: "Max789 Prediction API - LEARNING AI",
    author: "@tranhoang2286",
    version: "8.0",
    features: {
        "40_modules": "Phan tich da tang",
        "24_cau": "Nhan dien cau",
        "hoc_cau": "He thong hoc cau thong minh",
        "cau_profiles": "Profile rieng cho tung cau",
        "context_learning": "Hoc theo ngu canh (streak, entropy, volatility, balance)",
        "adaptive_weights": "Trong so thich ung theo thoi gian",
        "hot_deprecated": "Danh dau cau hot/bi quen"
    },
    endpoints: {
        "Du_doan_md5": "/tx/md5",
        "Du_doan_hu": "/tx/hu",
        "Check_md5": "/check/md5",
        "Check_hu": "/check/hu",
        "Hoc_cau_md5": "/learn/md5",
        "Hoc_cau_hu": "/learn/hu",
        "Xem_cau_cu_the": "/learn/cau/:name",
        "Pattern_md5": "/pattern/md5",
        "Pattern_hu": "/pattern/hu",
        "Reset": "/learn/reset"
    }
}));

loadAllData();

const start = async () => {
    try {
        await app.listen({ port: PORT, host: "0.0.0.0" });
        console.log(`✅ LEARNING AI API running on port ${PORT}`);
        console.log(`🧠 He thong hoc cau thong minh`);
        console.log(`📊 40 modules + 24 cau`);
        console.log(`🎯 Adaptive weights theo ngu canh`);
    } catch (err) {
        console.error("Error:", err.message);
        process.exit(1);
    }
};

start();
