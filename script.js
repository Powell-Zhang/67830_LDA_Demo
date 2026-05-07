// ── palette ───────────────────────────────────────────────────────────────────
const COLORS = [
  "210,78%,62%","25,84%,58%","152,58%,46%","278,62%,63%","338,68%,59%",
  "52,88%,52%","192,68%,49%","12,74%,56%","124,53%,46%","262,58%,66%",
  "36,78%,54%","172,63%,45%","302,53%,59%","82,63%,47%","222,68%,56%",
  "2,68%,56%","144,58%,49%","312,58%,61%","62,84%,49%","202,73%,53%",
];

function digamma(x) {
  let r = 0;
  while (x < 6) { r -= 1 / x; x++; }
  r += Math.log(x) - 0.5/x - 1/(12*x*x) + 1/(120*x*x*x*x) - 1/(252*x*x*x*x*x*x);
  return r;
}

// ── stopwords ─────────────────────────────────────────────────────────────────
const STOPWORDS = new Set([
  "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
  "yourself","yourselves","he","him","his","himself","she","her","hers",
  "herself","it","its","itself","they","them","their","theirs","themselves",
  "what","which","who","whom","this","that","these","those","am","is","are",
  "was","were","be","been","being","have","has","had","having","do","does",
  "did","doing","a","an","the","and","but","if","or","because","as","until",
  "while","of","at","by","for","with","about","against","between","into",
  "through","during","before","after","above","below","to","from","up","down",
  "in","out","on","off","over","under","again","further","then","once","here",
  "there","when","where","why","how","all","both","each","few","more","most",
  "other","some","such","no","nor","not","only","own","same","so","than",
  "too","very","s","t","can","will","just","don","should","now","d","ll",
  "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
  "isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
  "won","wouldn","also","one","two","three","four","five","first","second",
  "third","last","next","new","old","many","much","well","often","would",
  "could","may","might","must","shall","said","known","used","made","became",
  "include","including","however","although","since","later","early","large",
  "small","different","another","several","various","number","part","set",
  "use","uses","using","de","b","see","references","held",
]);

// ── tokenizer ─────────────────────────────────────────────────────────────────
function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z'\s]/g, " ")     
    .replace(/'\s|^\s*'|'\s*$/g, " ") 
    .split(/\s+/)
    .filter(t => t.length > 2 && !STOPWORDS.has(t));
}

// ── variational inference ─────────────────────────────────────────────────────
function infer(uniqueIds, counts, beta, alpha, maxIter = 100, tol = 1e-3) {
  const k       = alpha.length;
  const nUnique = uniqueIds.length;

  const phi = uniqueIds.map(wid => {
    const row = new Float64Array(k);
    let mx = -Infinity;
    for (let j = 0; j < k; j++) {
      row[j] = Math.log((beta[j][wid] || 0) + 1e-12);
      if (row[j] > mx) mx = row[j];
    }
    let s = 0;
    for (let j = 0; j < k; j++) { row[j] = Math.exp(row[j] - mx); s += row[j]; }
    for (let j = 0; j < k; j++) row[j] /= s;
    return row;
  });

  const gamma = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    gamma[j] = alpha[j];
    for (let n = 0; n < nUnique; n++) gamma[j] += phi[n][j] * counts[n];
  }

  for (let iter = 0; iter < maxIter; iter++) {
    const gammaOld = gamma.slice();
    const dgSum    = digamma(gamma.reduce((a, b) => a + b, 0));
    const logTheta = Array.from(gamma, g => digamma(g) - dgSum);

    for (let n = 0; n < nUnique; n++) {
      const wid = uniqueIds[n];
      let mx = -Infinity, s = 0;
      for (let j = 0; j < k; j++) {
        phi[n][j] = Math.log((beta[j][wid] || 0) + 1e-12) + logTheta[j];
        if (phi[n][j] > mx) mx = phi[n][j];
      }
      for (let j = 0; j < k; j++) { phi[n][j] = Math.exp(phi[n][j] - mx); s += phi[n][j]; }
      for (let j = 0; j < k; j++) phi[n][j] /= s;
    }

    for (let j = 0; j < k; j++) {
      gamma[j] = alpha[j];
      for (let n = 0; n < nUnique; n++) gamma[j] += phi[n][j] * counts[n];
    }

    let delta = 0;
    for (let j = 0; j < k; j++) delta += Math.abs(gamma[j] - gammaOld[j]);
    if (delta / k < tol) break;
  }

  return {gamma, phi, uniqueIds};
}

// ── model ─────────────────────────────────────────────────────────────────────
let MODEL = null;

async function loadModel(filename = "wiki_100k_100.json.gz") {
  const statusWrap = document.getElementById("model-status");
  const statusTxt  = document.getElementById("model-status-text");
  const select = document.getElementById("model-trigger-btn");
  const analyzeBtn = document.getElementById("analyze-btn");

  MODEL = null;
  select.disabled = true;
  analyzeBtn.disabled = true;
  statusWrap.className = "";
  statusTxt.textContent = "Loading model…";
  resetResults();

  try {
    const baseFilename = filename; 
    const responses = [];
    const mainRes = await fetch(baseFilename);

    if (mainRes.ok) {
      responses.push(mainRes);
    } else {
      let partIndex = 1;
      
      while (true) {
        const partRes = await fetch(`${baseFilename}.part${partIndex}`);
        
        if (!partRes.ok) {
          if (partIndex === 1) throw new Error(`Model not found at ${baseFilename} or as parts.`);
          break; 
        }
        
        responses.push(partRes);
        partIndex++;
      }
    }

    let total = 0;
    for (const res of responses) {
      const contentLength = res.headers.get("Content-Length");
      if (contentLength) total += parseInt(contentLength);
    }

    let loaded = 0;
    const chunks = [];

    for (const res of responses) {
      const reader = res.body.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        chunks.push(value);
        loaded += value.byteLength;
        
        if (total) {
          const pct = Math.round((loaded / total) * 100);
          statusTxt.textContent = `Loading model… ${pct}%`;
        }
      }
    }

    const blob = new Blob(chunks);
    const decompressed = blob.stream().pipeThrough(new DecompressionStream("gzip"));
    const text = await new Response(decompressed).text();
    const data = JSON.parse(text);
    
    MODEL = {
      alpha:      data.alpha,
      beta:       data.beta,
      vocab:      data.vocab,
      wordToId:   Object.fromEntries(data.vocab.map((w, i) => [w, i])),
      topicDescs: data.topic_descs || [],
    };
    
    statusWrap.className = "ready";
    statusTxt.textContent = `${MODEL.beta.length} topics · ${MODEL.vocab.length.toLocaleString()} words`;
    document.getElementById("analyze-btn").disabled = false;
    select.disabled = false;

  } catch (e) {
    statusWrap.className = "error";
    statusTxt.textContent = "model not found";
    if (typeof select !== 'undefined') select.disabled = false;
    console.error(e);
  }
}

// ── run analysis ──────────────────────────────────────────────────────────────
function runAnalysis(text) {
  const tokens = tokenize(text);
  const docIds = tokens.map(t => MODEL.wordToId[t]).filter(id => id !== undefined);
  if (!docIds.length) return null;

  const countMap = {};
  for (const id of docIds) countMap[id] = (countMap[id] || 0) + 1;
  const uniqueIds = Object.keys(countMap).map(Number);
  const counts    = uniqueIds.map(id => countMap[id]);

  const { gamma, phi, uniqueIds: inferredIds } = infer(uniqueIds, counts, MODEL.beta, MODEL.alpha);
  const phiByWid = {};
  for (let n = 0; n < inferredIds.length; n++) {
    phiByWid[inferredIds[n]] = phi[n];
  }
  const gammaSum = gamma.reduce((a, b) => a + b, 0);
  const theta    = Array.from(gamma, g => g / gammaSum);
  const evidence = Array.from(gamma, (g, j) => g - MODEL.alpha[j]);

  const pTopics = evidence
    .map((e, i) => ({ id: i, evidence: e, weight: theta[i] }))
    .filter(t => t.evidence >= 0.05)
    .sort((a, b) => b.evidence - a.evidence)
    .map(t => {
      const desc     = MODEL.topicDescs[t.id] || {};
      const sorted   = Object.entries(desc).sort((a, b) => b[1] - a[1]);
      const label    = sorted.slice(0, 3).map(([w]) => w).join(" · ") || `Topic ${t.id}`;
      const topWords = sorted.slice(0, 10);
      return {
        ...t,
        label,
        topWords,
        color: null,
      };
    });

  const sorted = pTopics.sort((a, b) => b.weight - a.weight);
  const gaps = sorted.slice(1).map((t, i) => sorted[i].weight - t.weight);
  const elbowIdx = Math.max(gaps.indexOf(Math.max(...gaps)), 4);
  const topics = sorted.slice(0, elbowIdx + 1);
  topics.forEach((obj, i) => {
    obj.color = COLORS[i % COLORS.length];
  });

  if (!topics.length) return null;
  const activeIds = topics.map(t => t.id);

  const words = text.match(/\$?[0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[a-z]+(?:'[a-z]+)*|\s+|[^\w\s]/gi).map(token => {
    const wid = MODEL.wordToId[token.toLowerCase()];
    if (wid === undefined || !phiByWid[wid]) return { text: token, topic: null, color: null };

    const phiRow = phiByWid[wid]; 
    const scores = activeIds.map(tid => phiRow[tid]);
    const total_score = scores.reduce((a,c) => a+c, 0)
    const qScores   = scores.map(score => score / total_score);
    const bestIdx   = qScores.indexOf(Math.max(...qScores));
    const bestQ     = qScores[bestIdx];
    const bestTopic = activeIds[bestIdx];

    if (bestQ <= 0.7) return { text: token, topic: null, color: null, qScores, activeIds};
    return { text: token, topic: bestTopic, color: COLORS[bestIdx % COLORS.length], qScores, activeIds};
  });
  return { topics, words };
}

// ── render helpers ────────────────────────────────────────────────────────────
let activeFilter = null;
let lastWords    = [];
let lastTopics   = [];

function renderBars(topics) {
  const el  = document.getElementById("topic-bars");
  el.innerHTML = "";
  const max = Math.max(...topics.map(t => t.weight));
  topics.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "topic-bar-row";
    row.style.animationDelay = `${i * 40}ms`;
    row.title = "Top words: " + t.topWords.map(([w]) => w).join(", ");
    row.dataset.id = t.id;
    row.title = t.topWords.map(([w]) => w).join(", ");
    row.style.cssText = `border-color:hsla(${t.color},0.4); background:hsla(${t.color},var(--bg-alpha);`;
    row.innerHTML = `
      <div class="bar-label">${t.label}</div>
      <div class="bar-track"><div class="bar-fill" style="background:hsl(${t.color})"></div></div>
      <div class="bar-pct">${(t.weight * 100).toFixed(1)}%</div>`;
    row.addEventListener("click", () => {
      const wasActive = activeFilter === t.id;
      activeFilter = wasActive ? null : t.id;
      document.querySelectorAll(".topic-bar-row").forEach(c => {
        c.classList.toggle("dimmed",   activeFilter !== null && parseInt(c.dataset.id) !== activeFilter);
        c.classList.toggle("selected", parseInt(c.dataset.id) === activeFilter);
      });
      renderWords(lastWords, activeFilter);
      if (!wasActive) {
        renderDetailPanel(t);
      } else {
        hideDetailPanel();
      }
    });
    el.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const fill = row.querySelector(".bar-fill");
      fill.style.transition = `transform 0.55s cubic-bezier(0.22,1,0.36,1) ${i * 50}ms`;
      fill.style.transform  = `scaleX(${t.weight / max})`;
    }));
  });
}

function renderWords(words, filter) {
  const el = document.getElementById("output-text");
  el.innerHTML = "";
  
  words.forEach(w => {
    const span = document.createElement("span");
    span.textContent = w.text;
    if (w.topic === null && !w.qScores) {
      span.className = "word oop";
    } else if (w.topic === null) {
      // out-of-palette but has qscores (below threshold) — still clickable
      span.className = "word oop word-clickable";
      span.style.cursor = "pointer";
      span.addEventListener("click", () => showWordDetail(w, span));
    } else {
      const dim = filter !== null && w.topic !== filter;
      span.className = `word has-topic${dim ? " dimmed" : ""}`;
      span.style.background = `hsla(${w.color},${dim ? 0 : 0.22})`;
      span.style.color      = dim ? "" : `hsl(${w.color})`;
      const desc   = MODEL.topicDescs[w.topic] || {};
      const sorted = Object.entries(desc).sort((a, b) => b[1] - a[1]);
      span.title   = sorted.slice(0, 3).map(([wd]) => wd).join(" · ") || `Topic ${w.topic}`;
      span.style.cursor = "pointer";
      span.addEventListener("click", () => showWordDetail(w, span));
    }

    el.appendChild(span);
    
  });
}

function showWordDetail(word, spanEl) {
  activeFilter = null;
  document.querySelectorAll(".topic-chip").forEach(c => {
    c.classList.remove("dimmed", "selected");
  });
  renderWords(lastWords, null);

  const panel = document.getElementById("topic-detail");
  const dot   = document.getElementById("detail-dot");
  const title = document.getElementById("detail-title");
  const sub   = document.getElementById("detail-subtitle");
  const bars  = document.getElementById("detail-word-bars");

  dot.style.background = word.color ? `hsl(${word.color})` : `var(--muted)`;
  title.textContent    = `"${word.text.replace(/[^a-zA-Z']/g, "")}"`;
  sub.textContent      = "topic q-scores";

  const scored = word.activeIds.map((tid, i) => ({ tid, q: word.qScores[i] }))
    .sort((a, b) => b.q - a.q);

  bars.innerHTML = "";
  const maxQ = scored[0]?.q || 1;

  scored.forEach(({ tid, q }, i) => {
    const topicObj = lastTopics.find(t => t.id === tid);
    const label    = topicObj ? topicObj.label : `Topic ${tid}`;
    const color    = topicObj ? topicObj.color : COLORS[i % COLORS.length];

    const row = document.createElement("div");
    row.className = "word-bar-row";
    row.style.animationDelay = `${i * 25}ms`;
    row.innerHTML = `
      <div class="word-bar-label" title="${label}">${label}</div>
      <div class="word-bar-track"><div class="word-bar-fill" style="background:hsl(${color})"></div></div>
      <div class="word-bar-val">${(q * 100).toFixed(1)}%</div>`;
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const fill = row.querySelector(".word-bar-fill");
      fill.style.transition = `transform 0.45s cubic-bezier(0.22,1,0.36,1) ${i * 30}ms`;
      fill.style.transform  = `scaleX(${q / maxQ})`;
    }));
  });

  panel.classList.add("visible");
}

function renderDetailPanel(topic) {
  const panel = document.getElementById("topic-detail");
  const dot   = document.getElementById("detail-dot");
  const title = document.getElementById("detail-title");
  const sub   = document.getElementById("detail-subtitle");
  const bars  = document.getElementById("detail-word-bars");

  dot.style.background = `hsl(${topic.color})`;
  title.textContent    = topic.label;
  sub.textContent      = `${(topic.weight * 100).toFixed(1)}% of document`;

  bars.innerHTML = "";
  const maxW = topic.topWords.length ? topic.topWords[0][1] : 1;

  topic.topWords.forEach(([word, weight], i) => {
    const row = document.createElement("div");
    row.className = "word-bar-row";
    row.style.animationDelay = `${i * 30}ms`;
    row.innerHTML = `
      <div class="word-bar-label">${word}</div>
      <div class="word-bar-track"><div class="word-bar-fill" style="background:hsl(${topic.color})"></div></div>
      <div class="word-bar-val">${weight.toExponential(2)}</div>`;
    bars.appendChild(row);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const fill = row.querySelector(".word-bar-fill");
      fill.style.transition = `transform 0.45s cubic-bezier(0.22,1,0.36,1) ${i * 35}ms`;
      fill.style.transform  = `scaleX(${weight / maxW})`;
    }));
  });

  panel.classList.add("visible");
}

function hideDetailPanel() {
  const panel = document.getElementById("topic-detail");
  panel.classList.remove("visible");
}

// ── analyze button ────────────────────────────────────────────────────────────
document.getElementById("analyze-btn").addEventListener("click", () => {
  const text = document.getElementById("input-text").value.trim();
  if (!text || !MODEL) return;

  const btn    = document.getElementById("analyze-btn");
  const status = document.getElementById("status");
  btn.disabled = true;
  activeFilter = null;
  hideDetailPanel();
  status.innerHTML = '<span class="spinner"></span>';

  setTimeout(() => {
    const t0     = performance.now();
    const result = runAnalysis(text);
    const ms     = (performance.now() - t0).toFixed(0);

    if (!result) {
      status.textContent = "No known vocabulary found.";
      btn.disabled = false;
      return;
    }
    lastWords  = result.words;
    lastTopics = result.topics;
    renderBars(result.topics);
    renderWords(result.words, null);
    status.textContent = `${result.topics.length} topics · ${result.words.length} tokens · ${ms}ms`;
    btn.disabled = false;
  }, 30);
});

// ── reset results UI ──────────────────────────────────────────────────────────
function resetResults() {
  activeFilter = null;
  lastWords    = [];
  lastTopics   = [];
  hideDetailPanel();
  document.getElementById("topic-bars").innerHTML  = '<p class="empty">Run analysis to see topics.</p>';
  document.getElementById("output-text").innerHTML = '<p class="empty">Words will be coloured by their dominant topic.</p>';
  document.getElementById("status").textContent    = "";
}

// ── boot ──────────────────────────────────────────────────────────────────────

function scaleBody() {
  console.log("scaling");
  const body = document.body;
  const width = document.documentElement.clientWidth;
  if (width < 480) {
    const scale = width / 480;
    body.style.width = '480px';
    body.style.transform = `scale(${scale})`;
  } 
}
scaleBody();

const backdrop    = document.getElementById('model-backdrop');
const triggerBtn  = document.getElementById('model-trigger-btn');
const analyzeBtn  = document.getElementById("analyze-btn");
const triggerText = document.getElementById('model-trigger-text');
const triggerMobile = document.getElementById('model-trigger-mobile');
const confirmBtn  = document.getElementById('mp-confirm');
const cancelBtn   = document.getElementById('mp-cancel');
const topics200   = document.getElementById('topics-200');
const yearRow = document.getElementById('year-row');
const yearBtns = document.querySelectorAll('.mp-year-btn');


let selCorpus = 'wiki', selTopics = '100', selArticles = '100k', selYear='';
topics200.disabled=true;
yearRow.classList.toggle('visible', false);


function modelValue() {
  if (selCorpus === 'nyt') return `model_weights/${selCorpus}${selYear}_${selArticles}_${selTopics}.json.gz`;
  return `model_weights/${selCorpus}_${selArticles}_${selTopics}.json.gz`;
}

function updateConfirm() {
  confirmBtn.classList.toggle('active', !!(selCorpus && selTopics && selArticles && (selCorpus == 'wiki' || selYear)));
}
function update200() {
  topics200.disabled = (selCorpus !== 'wiki' || selArticles === '100k');
  if (topics200.disabled && selTopics === '200') {
    selTopics = null;
    document.querySelectorAll('.mp-topic-btn').forEach(b => b.classList.remove('selected'));
  }
}

document.querySelectorAll('.mp-option').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mp-option').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selCorpus = btn.dataset.corpus;
    update200();
    if (selCorpus !== "nyt") yearRow.classList.remove('visible');
    else yearRow.classList.add('visible');
    updateConfirm();
  });
});

document.querySelectorAll('.mp-topic-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.disabled) return;
    document.querySelectorAll('.mp-topic-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selTopics = btn.dataset.topics;
    updateConfirm();
  });
});

document.querySelectorAll('.mp-article-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mp-article-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selArticles = btn.dataset.articles;
    update200();
    updateConfirm();
  });
});

document.querySelectorAll('.mp-year-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mp-year-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selYear = btn.dataset.year;
    updateConfirm();
  });
});

triggerBtn.addEventListener('click', () => backdrop.classList.add('open'));
cancelBtn.addEventListener('click',  () => backdrop.classList.remove('open'));
backdrop.addEventListener('click', e => { if (e.target === backdrop) backdrop.classList.remove('open'); });

confirmBtn.addEventListener('click', () => {
  if (!selCorpus || !selTopics) return;
  const corpus = selCorpus.startsWith('wiki') ? 'Wikipedia'
  : selCorpus.startsWith('nyt') ? `NYT ${selYear}` : '';
  const label = `${corpus} · ${selTopics} topics · ${selArticles} articles`;
  const mobileLabel = `${corpus}`;
  
 
  triggerText.textContent = label;
  triggerMobile.textContent = mobileLabel;
  backdrop.classList.remove('open');
  loadModel(modelValue()); 
});

updateConfirm();
backdrop.classList.add('open');
analyzeBtn.disabled=true;

document.getElementById("input-text").value =
  "The William Randolph Hearst Foundation will give $1.25 million to Lincoln Center, Metropolitan " +
  "Opera Co., New York Philharmonic and Juilliard School. \"Our board felt that we had a " +
  "real opportunity to make a mark on the future of the performing arts with these grants an act " +
  "every bit as important as our traditional areas of support in health, medical research, education " +
  "and the social services,\" Hearst Foundation President Randolph A. Hearst said Monday in " +
  "announcing the grants. Lincoln Center's share will be $200,000 for its new building, which " +
  "will house young artists and provide new public facilities. The Metropolitan Opera Co. and " +
  "New York Philharmonic will receive $400,000 each. The Juilliard School, where music and " +
  "the performing arts are taught, will get $250,000. The Hearst Foundation, a leading supporter " +
  "of the Lincoln Center Consolidated Corporate Fund, will make its usual annual $100,000 donation, too. ";