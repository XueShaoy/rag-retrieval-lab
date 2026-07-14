"use client";

import { useEffect, useMemo, useState } from "react";

type Chunk = {
  id: string;
  docId: string;
  source: string;
  text: string;
  start: number;
  end: number;
  prevId?: string;
  nextId?: string;
};

type RankedChunk = Chunk & {
  vector: number;
  bm25: number;
  vectorNorm: number;
  bm25Norm: number;
  hybrid: number;
  rerank: number;
  coverage: number;
  reasons: string[];
};

const DEFAULT_CORPUS = `[访谈 A｜新用户｜上海]
我第一次用这个旅行平台是为了预订周末酒店。搜索结果很多，但我最在意的是最终价格是否包含早餐和取消费用。进入付款页后价格比列表页高了七十多元，我不确定是税费还是服务费，所以没有继续付款。后来客服解释得很清楚，但如果价格明细能提前展示，我当时就会直接下单。

[访谈 B｜高频用户｜北京]
我每个月出差两三次，通常会同时比较几个平台。筛选速度和发票政策对我很重要。最近一次订单改期时，机器人一直重复标准答案，找人工客服需要点击很多层。人工客服接入后五分钟就解决了问题，因此我希望复杂售后可以更快转人工。

[访谈 C｜家庭出游｜杭州]
带孩子出行时，我会重点看房型面积、早餐人数和是否有儿童用品。酒店图片很多，但关键信息散落在不同位置。我收藏了三个房型，最后因为无法快速比较早餐和退改规则，回到电脑上整理后才下单。移动端如果有对比功能会很有帮助。

[访谈 D｜价格敏感用户｜成都]
我经常先看低价日历，再决定出行日期。优惠券很多，但是领取条件不直观。有一次显示可以减一百元，结算时才发现只适用于指定银行卡。我放弃付款不是因为价格贵，而是觉得优惠信息前后不一致，对最终金额没有信心。

[访谈 E｜会员用户｜深圳]
会员权益里我最常用延迟退房和免费早餐。预订时看到了会员价，但权益是否适用于当前酒店并不明确。我希望系统在订单确认前用一张清单告诉我已经获得哪些权益、哪些需要到店确认，这样会更安心。`;

const STEPS = [
  { id: "chunk", index: "01", name: "文本分块", short: "Chunk" },
  { id: "vector", index: "02", name: "向量检索", short: "Vector" },
  { id: "bm25", index: "03", name: "BM25", short: "Keyword" },
  { id: "fusion", index: "04", name: "混合融合", short: "Hybrid" },
  { id: "rerank", index: "05", name: "交叉重排", short: "Rerank" },
  { id: "answer", index: "06", name: "答案生成", short: "Answer" },
];

const EXAMPLE_QUERIES = [
  "用户为什么会在付款前放弃订单？",
  "哪些体验会让用户更信任最终价格？",
  "复杂售后场景中，用户期待什么？",
];

function tokenize(value: string) {
  const normalized = value.toLowerCase().replace(/[，。！？；：、（）【】“”‘’]/g, " ");
  const english = normalized.match(/[a-z0-9]+/g) ?? [];
  const chineseRuns = normalized.match(/[\u4e00-\u9fff]+/g) ?? [];
  const chinese: string[] = [];

  chineseRuns.forEach((run) => {
    const chars = [...run];
    chinese.push(...chars);
    for (let i = 0; i < chars.length - 1; i += 1) {
      chinese.push(chars[i] + chars[i + 1]);
    }
  });

  return [...english, ...chinese].filter((token) => token.trim().length > 0);
}

function parseCorpus(corpus: string) {
  return corpus
    .split(/\n\s*\n/)
    .map((block, index) => {
      const lines = block.trim().split("\n").filter(Boolean);
      const hasHeader = /^\[.+\]$/.test(lines[0] ?? "");
      return {
        id: `D${index + 1}`,
        source: hasHeader ? lines[0].slice(1, -1) : `文档 ${index + 1}`,
        text: (hasHeader ? lines.slice(1) : lines).join(" ").trim(),
      };
    })
    .filter((doc) => doc.text.length > 0);
}

function createChunks(corpus: string, chunkSize: number, overlap: number) {
  const docs = parseCorpus(corpus);
  const result: Chunk[] = [];

  docs.forEach((doc) => {
    const local: Chunk[] = [];
    let start = 0;
    let index = 0;

    while (start < doc.text.length) {
      const hardEnd = Math.min(start + chunkSize, doc.text.length);
      let end = hardEnd;
      if (hardEnd < doc.text.length) {
        const nearby = doc.text.slice(start + Math.floor(chunkSize * 0.62), hardEnd);
        const punctuation = Math.max(
          nearby.lastIndexOf("。"),
          nearby.lastIndexOf("！"),
          nearby.lastIndexOf("？"),
          nearby.lastIndexOf("；"),
        );
        if (punctuation >= 0) end = start + Math.floor(chunkSize * 0.62) + punctuation + 1;
      }

      local.push({
        id: `${doc.id}-C${index + 1}`,
        docId: doc.id,
        source: doc.source,
        text: doc.text.slice(start, end),
        start,
        end,
      });

      if (end >= doc.text.length) break;
      start = Math.max(end - overlap, start + 1);
      index += 1;
    }

    local.forEach((chunk, index) => {
      chunk.prevId = local[index - 1]?.id;
      chunk.nextId = local[index + 1]?.id;
    });
    result.push(...local);
  });

  return result;
}

function hashToken(token: string, dimensions = 96) {
  let hash = 2166136261;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash) % dimensions;
}

function embedding(value: string, dimensions = 96) {
  const vector = new Array(dimensions).fill(0);
  tokenize(value).forEach((token) => {
    vector[hashToken(token, dimensions)] += token.length > 1 ? 1.35 : 0.45;
  });
  const magnitude = Math.sqrt(vector.reduce((sum, item) => sum + item * item, 0)) || 1;
  return vector.map((item) => item / magnitude);
}

function cosine(a: number[], b: number[]) {
  return a.reduce((sum, item, index) => sum + item * b[index], 0);
}

function normalize(values: number[]) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (!Number.isFinite(min) || max === min) return values.map(() => 0.5);
  return values.map((value) => (value - min) / (max - min));
}

function rankChunks(chunks: Chunk[], query: string, alpha: number): RankedChunk[] {
  if (!chunks.length) return [];
  const rawQueryTokens = [...new Set(tokenize(query))];
  const informativeTokens = rawQueryTokens.filter((token) => token.length > 1 || /^[a-z0-9]+$/.test(token));
  const queryTokens = informativeTokens.length ? informativeTokens : rawQueryTokens;
  const chunkTokens = chunks.map((chunk) => tokenize(chunk.text));
  const avgLength = chunkTokens.reduce((sum, tokens) => sum + tokens.length, 0) / chunks.length || 1;
  const queryVector = embedding(query);
  const vectors = chunks.map((chunk) => Math.max(0, cosine(queryVector, embedding(chunk.text))));
  const k1 = 1.5;
  const b = 0.75;

  const bm25s = chunkTokens.map((tokens) => {
    const frequencies = new Map<string, number>();
    tokens.forEach((token) => frequencies.set(token, (frequencies.get(token) ?? 0) + 1));
    return queryTokens.reduce((score, term) => {
      const documentFrequency = chunkTokens.filter((items) => items.includes(term)).length;
      const idf = Math.log(1 + (chunks.length - documentFrequency + 0.5) / (documentFrequency + 0.5));
      const frequency = frequencies.get(term) ?? 0;
      const denominator = frequency + k1 * (1 - b + b * (tokens.length / avgLength));
      return score + idf * ((frequency * (k1 + 1)) / (denominator || 1));
    }, 0);
  });

  const vectorNorms = normalize(vectors);
  const bm25Norms = normalize(bm25s);

  return chunks.map((chunk, index) => {
    const uniqueChunkTokens = new Set(chunkTokens[index]);
    const matched = queryTokens.filter((token) => uniqueChunkTokens.has(token));
    const longMatched = matched.filter((token) => token.length > 1);
    const coverage = matched.length / Math.max(queryTokens.length, 1);
    const semantic = vectorNorms[index];
    const hybrid = alpha * semantic + (1 - alpha) * bm25Norms[index];
    const sourceOverlap = tokenize(chunk.source).filter((token) => queryTokens.includes(token)).length > 0 ? 1 : 0;
    const exactPhraseBoost = longMatched.length / Math.max(queryTokens.filter((token) => token.length > 1).length, 1);
    const rerank = Math.min(1, hybrid * 0.52 + coverage * 0.26 + exactPhraseBoost * 0.17 + sourceOverlap * 0.05);
    const reasons = [
      coverage > 0.24 ? "查询覆盖高" : "补充语义",
      exactPhraseBoost > 0.18 ? "关键短语命中" : semantic > 0.55 ? "语义接近" : "候选补充",
      chunk.prevId || chunk.nextId ? "可扩展上下文" : "独立片段",
    ];

    return {
      ...chunk,
      vector: vectors[index],
      bm25: bm25s[index],
      vectorNorm: vectorNorms[index],
      bm25Norm: bm25Norms[index],
      hybrid,
      rerank,
      coverage,
      reasons,
    };
  });
}

function bestSentence(text: string, query: string) {
  const rawQueryTokens = [...new Set(tokenize(query))];
  const informativeTokens = rawQueryTokens.filter((token) => token.length > 1 || /^[a-z0-9]+$/.test(token));
  const queryTokens = new Set(informativeTokens.length ? informativeTokens : rawQueryTokens);
  const sentences = text.split(/(?<=[。！？；])/).filter(Boolean);
  return sentences
    .map((sentence) => ({
      sentence,
      score: tokenize(sentence).filter((token) => queryTokens.has(token)).length,
    }))
    .sort((a, b) => b.score - a.score)[0]?.sentence ?? text;
}

function formatScore(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "0.000";
}

export default function Home() {
  const [query, setQuery] = useState(EXAMPLE_QUERIES[0]);
  const [corpus, setCorpus] = useState(DEFAULT_CORPUS);
  const [chunkSize, setChunkSize] = useState(92);
  const [overlap, setOverlap] = useState(22);
  const [topK, setTopK] = useState(4);
  const [alpha, setAlpha] = useState(0.58);
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [selectedChunkId, setSelectedChunkId] = useState<string | null>(null);
  const [showCorpus, setShowCorpus] = useState(false);

  const docs = useMemo(() => parseCorpus(corpus), [corpus]);
  const chunks = useMemo(
    () => createChunks(corpus, chunkSize, Math.min(overlap, chunkSize - 8)),
    [corpus, chunkSize, overlap],
  );
  const ranked = useMemo(() => rankChunks(chunks, query, alpha), [chunks, query, alpha]);
  const vectorRank = useMemo(() => [...ranked].sort((a, b) => b.vector - a.vector), [ranked]);
  const bm25Rank = useMemo(() => [...ranked].sort((a, b) => b.bm25 - a.bm25), [ranked]);
  const hybridRank = useMemo(() => [...ranked].sort((a, b) => b.hybrid - a.hybrid), [ranked]);
  const rerankRank = useMemo(() => [...ranked].sort((a, b) => b.rerank - a.rerank), [ranked]);
  const selectedChunk = chunks.find((chunk) => chunk.id === selectedChunkId) ?? chunks[0];
  const answerSources = rerankRank.slice(0, topK);
  const answerSentences = answerSources
    .slice(0, 3)
    .map((chunk) => ({ id: chunk.id, sentence: bestSentence(chunk.text, query) }))
    .filter((item, index, list) => list.findIndex((other) => other.sentence === item.sentence) === index);

  useEffect(() => {
    if (!isPlaying) return;
    if (activeStep >= STEPS.length - 1) {
      const done = window.setTimeout(() => setIsPlaying(false), 400);
      return () => window.clearTimeout(done);
    }
    const timer = window.setTimeout(() => setActiveStep((step) => step + 1), 720);
    return () => window.clearTimeout(timer);
  }, [activeStep, isPlaying]);

  useEffect(() => {
    if (selectedChunkId && !chunks.some((chunk) => chunk.id === selectedChunkId)) {
      setSelectedChunkId(chunks[0]?.id ?? null);
    }
  }, [chunks, selectedChunkId]);

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
        event.preventDefault();
        setActiveStep(0);
        setIsPlaying(true);
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, []);

  function runPipeline() {
    setActiveStep(0);
    setIsPlaying(true);
  }

  function stepForward() {
    setIsPlaying(false);
    setActiveStep((step) => (step + 1) % STEPS.length);
  }

  return (
    <main className="site-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <header className="topbar">
        <a className="brand" href="#top" aria-label="RAG Flow Studio 首页">
          <span className="brand-mark"><i /><i /><i /></span>
          <span>RAG FLOW STUDIO</span>
        </a>
        <div className="local-badge"><span /> LOCAL · ZERO API</div>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>INTERACTIVE LAB</span><b>完整链路 · 可解释 · 可调参</b></div>
        <h1>把一次 RAG 检索，<br /><em>拆开看懂。</em></h1>
        <p>
          从客户访谈原文到带引用的回答。调整参数、逐步执行，观察每个片段如何被召回、融合和重新排序。
        </p>
        <div className="hero-actions">
          <button className="primary-button" onClick={runPipeline}>
            <span>{isPlaying ? "运行中" : "运行完整流程"}</span>
            <b>{isPlaying ? "•••" : "→"}</b>
          </button>
          <button className="ghost-button" onClick={stepForward}>单步执行 <span>⌘ ↵</span></button>
        </div>
      </section>

      <section className="lab-frame" aria-label="RAG 交互实验室">
        <div className="lab-header">
          <div>
            <span className="window-dot coral" />
            <span className="window-dot amber" />
            <span className="window-dot mint" />
          </div>
          <div className="lab-title">rag-pipeline / customer-interviews</div>
          <div className="runtime">● READY</div>
        </div>

        <nav className="pipeline" aria-label="处理流程">
          {STEPS.map((step, index) => (
            <button
              key={step.id}
              onClick={() => { setActiveStep(index); setIsPlaying(false); }}
              className={`pipeline-step ${index === activeStep ? "active" : ""} ${index < activeStep ? "complete" : ""}`}
            >
              <span className="step-index">{index < activeStep ? "✓" : step.index}</span>
              <span><b>{step.name}</b><small>{step.short}</small></span>
            </button>
          ))}
          <div className="pipeline-progress" style={{ width: `${(activeStep / (STEPS.length - 1)) * 100}%` }} />
        </nav>

        <div className="workspace">
          <aside className="control-panel">
            <div className="panel-section query-section">
              <label htmlFor="query">检索问题</label>
              <textarea
                id="query"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                rows={3}
              />
              <div className="query-presets" aria-label="示例问题">
                {EXAMPLE_QUERIES.map((item, index) => (
                  <button
                    key={item}
                    className={query === item ? "selected" : ""}
                    onClick={() => setQuery(item)}
                    title={item}
                  >
                    Q{index + 1}
                  </button>
                ))}
              </div>
            </div>

            <div className="panel-section controls-section">
              <div className="section-heading"><span>参数控制</span><small>实时生效</small></div>
              <ControlSlider label="CHUNK SIZE" value={chunkSize} min={60} max={160} step={4} suffix=" 字" onChange={setChunkSize} />
              <ControlSlider label="OVERLAP" value={overlap} min={0} max={48} step={2} suffix=" 字" onChange={setOverlap} />
              <ControlSlider label="TOP K" value={topK} min={2} max={6} step={1} suffix="" onChange={setTopK} />
              <ControlSlider label="VECTOR WEIGHT" value={Math.round(alpha * 100)} min={0} max={100} step={5} suffix="%" onChange={(value) => setAlpha(value / 100)} />
            </div>

            <div className="panel-section corpus-section">
              <button className="corpus-toggle" onClick={() => setShowCorpus((value) => !value)}>
                <span><b>示例语料</b><small>{docs.length} 篇访谈 · {corpus.length} 字</small></span>
                <i>{showCorpus ? "−" : "+"}</i>
              </button>
              {showCorpus && (
                <textarea
                  className="corpus-editor"
                  aria-label="语料编辑器"
                  value={corpus}
                  onChange={(event) => setCorpus(event.target.value)}
                  rows={15}
                />
              )}
            </div>

            <div className="local-note">
              <span>↳</span>
              <p><b>完全在浏览器内计算</b><br />语料不会上传；向量与重排使用可解释的教学模拟。</p>
            </div>
          </aside>

          <section className="result-panel" aria-live="polite">
            <div className="result-heading">
              <div>
                <span className="stage-label">STAGE {STEPS[activeStep].index}</span>
                <h2>{STEPS[activeStep].name}</h2>
              </div>
              <div className="result-stats">
                <Stat value={String(docs.length)} label="DOCS" />
                <Stat value={String(chunks.length)} label="CHUNKS" />
                <Stat value={`${Math.round(chunks.reduce((sum, chunk) => sum + chunk.text.length, 0) / Math.max(chunks.length, 1))}`} label="AVG LEN" />
              </div>
            </div>

            {activeStep === 0 && (
              <ChunkStage
                chunks={chunks}
                selected={selectedChunk}
                onSelect={setSelectedChunkId}
                overlap={overlap}
              />
            )}
            {activeStep === 1 && (
              <RankingStage
                title="语义相似度排名"
                description="将查询与每个片段映射到同一向量空间，使用余弦相似度寻找“意思接近”的内容。"
                items={vectorRank}
                scoreKey="vector"
                accent="violet"
                formula="cos(q, d) = q · d / (‖q‖ × ‖d‖)"
                onSelect={setSelectedChunkId}
              />
            )}
            {activeStep === 2 && (
              <RankingStage
                title="关键词相关度排名"
                description="BM25 奖励稀有查询词的命中，同时通过文档长度归一化避免长片段天然占优。"
                items={bm25Rank}
                scoreKey="bm25"
                accent="orange"
                formula="IDF × tf(k₁ + 1) / (tf + k₁(1 − b + bL/avgL))"
                onSelect={setSelectedChunkId}
              />
            )}
            {activeStep === 3 && (
              <FusionStage items={hybridRank} alpha={alpha} onSelect={setSelectedChunkId} />
            )}
            {activeStep === 4 && (
              <RerankStage before={hybridRank} after={rerankRank} topK={topK} onSelect={setSelectedChunkId} />
            )}
            {activeStep === 5 && (
              <AnswerStage
                query={query}
                sentences={answerSentences}
                sources={answerSources}
                onSelect={(id) => { setSelectedChunkId(id); setActiveStep(0); }}
              />
            )}
          </section>
        </div>
      </section>

      <footer>
        <p>RAG FLOW STUDIO <span>·</span> 一次看见检索系统内部发生了什么</p>
        <a href="#top">回到顶部 ↑</a>
      </footer>
    </main>
  );
}

function ControlSlider({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  const percentage = ((value - min) / (max - min)) * 100;
  return (
    <div className="slider-control">
      <div><label>{label}</label><output>{value}{suffix}</output></div>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={value}
        style={{ "--range": `${percentage}%` } as React.CSSProperties}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return <div><b>{value}</b><span>{label}</span></div>;
}

function ChunkStage({
  chunks,
  selected,
  overlap,
  onSelect,
}: {
  chunks: Chunk[];
  selected?: Chunk;
  overlap: number;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="stage-content chunk-stage">
      <div className="stage-intro">
        <div><span className="intro-icon">⌗</span><p><b>按语义边界切分原文</b><br />优先在标点处截断，并保留 {overlap} 字重叠，减少跨边界信息丢失。</p></div>
        <span className="formula-chip">size ≈ {Math.round(chunks.reduce((s, c) => s + c.text.length, 0) / Math.max(chunks.length, 1))} · overlap = {overlap}</span>
      </div>
      <div className="chunk-layout">
        <div className="chunk-list">
          {chunks.map((chunk, index) => (
            <button
              key={chunk.id}
              className={`chunk-card ${selected?.id === chunk.id ? "selected" : ""}`}
              onClick={() => onSelect(chunk.id)}
            >
              <span className="chunk-order">{String(index + 1).padStart(2, "0")}</span>
              <span className="chunk-body"><b>{chunk.id}</b><small>{chunk.source}</small><p>{chunk.text}</p></span>
              <span className="chunk-length">{chunk.text.length}c</span>
            </button>
          ))}
        </div>
        {selected && (
          <aside className="chunk-inspector">
            <span className="inspector-label">CHUNK INSPECTOR</span>
            <h3>{selected.id}</h3>
            <p className="source-name">{selected.source}</p>
            <div className="position-map">
              <span style={{ left: `${Math.min(84, (selected.start / Math.max(selected.end, 1)) * 75)}%`, width: "22%" }} />
            </div>
            <dl>
              <div><dt>字符范围</dt><dd>{selected.start} → {selected.end}</dd></div>
              <div><dt>前一片段</dt><dd>{selected.prevId ?? "—"}</dd></div>
              <div><dt>后一片段</dt><dd>{selected.nextId ?? "—"}</dd></div>
            </dl>
            <div className="context-rail">
              <span className={selected.prevId ? "linked" : ""}>PREV</span>
              <i />
              <b>CURRENT</b>
              <i />
              <span className={selected.nextId ? "linked" : ""}>NEXT</span>
            </div>
            <p className="inspector-tip">检索命中后，可按 prev / next 关系补齐上下文，再交给生成模型。</p>
          </aside>
        )}
      </div>
    </div>
  );
}

function RankingStage({
  title,
  description,
  items,
  scoreKey,
  accent,
  formula,
  onSelect,
}: {
  title: string;
  description: string;
  items: RankedChunk[];
  scoreKey: "vector" | "bm25";
  accent: "violet" | "orange";
  formula: string;
  onSelect: (id: string) => void;
}) {
  const max = Math.max(...items.map((item) => item[scoreKey]), 0.001);
  return (
    <div className={`stage-content ranking-stage ${accent}`}>
      <div className="stage-intro">
        <div><span className="intro-icon">{scoreKey === "vector" ? "◌" : "Aa"}</span><p><b>{title}</b><br />{description}</p></div>
        <span className="formula-chip">{formula}</span>
      </div>
      <div className="ranking-table">
        <div className="ranking-head"><span>RANK</span><span>CHUNK / MATCHED TEXT</span><span>SCORE</span></div>
        {items.slice(0, 7).map((item, index) => (
          <button className="ranking-row" key={item.id} onClick={() => onSelect(item.id)}>
            <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
            <span className="ranking-copy"><b>{item.id} <small>{item.source}</small></b><p>{item.text}</p><i><em style={{ width: `${Math.max(3, (item[scoreKey] / max) * 100)}%` }} /></i></span>
            <span className="score-number">{formatScore(item[scoreKey])}</span>
          </button>
        ))}
      </div>
      <div className="teaching-note"><b>观察</b><p>{scoreKey === "vector" ? "向量检索能找到措辞不同但语义相近的片段；本地演示使用哈希 n-gram 向量，生产系统可替换为 Embedding 模型。" : "BM25 对“付款、价格、客服”等明确关键词更敏感，但可能错过没有出现相同词语的语义相关内容。"}</p></div>
    </div>
  );
}

function FusionStage({ items, alpha, onSelect }: { items: RankedChunk[]; alpha: number; onSelect: (id: string) => void }) {
  return (
    <div className="stage-content fusion-stage">
      <div className="stage-intro">
        <div><span className="intro-icon">⌁</span><p><b>合并两路召回信号</b><br />先归一化分数，再按权重融合。既保留语义召回，也奖励准确关键词命中。</p></div>
        <span className="formula-chip">{alpha.toFixed(2)} × VECTOR + {(1 - alpha).toFixed(2)} × BM25</span>
      </div>
      <div className="fusion-legend"><span><i className="vector-key" /> VECTOR</span><span><i className="bm25-key" /> BM25</span><span><i className="hybrid-key" /> HYBRID</span></div>
      <div className="fusion-list">
        {items.slice(0, 7).map((item, index) => (
          <button key={item.id} className="fusion-row" onClick={() => onSelect(item.id)}>
            <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
            <span className="fusion-name"><b>{item.id}</b><small>{item.source}</small></span>
            <span className="score-track"><i className="vector-bar" style={{ width: `${item.vectorNorm * 100}%` }} /><em>{formatScore(item.vectorNorm)}</em></span>
            <span className="score-track"><i className="bm25-bar" style={{ width: `${item.bm25Norm * 100}%` }} /><em>{formatScore(item.bm25Norm)}</em></span>
            <strong>{formatScore(item.hybrid)}</strong>
          </button>
        ))}
      </div>
      <div className="balance-card">
        <div className="balance-visual"><span style={{ width: `${alpha * 100}%` }}>VECTOR {Math.round(alpha * 100)}%</span><span>BM25 {Math.round((1 - alpha) * 100)}%</span></div>
        <p>在左侧拖动 <b>VECTOR WEIGHT</b>，观察排名如何实时变化。</p>
      </div>
    </div>
  );
}

function RerankStage({
  before,
  after,
  topK,
  onSelect,
}: {
  before: RankedChunk[];
  after: RankedChunk[];
  topK: number;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="stage-content rerank-stage">
      <div className="stage-intro">
        <div><span className="intro-icon">⇅</span><p><b>逐对精排 Query × Chunk</b><br />对召回候选进行更细粒度的相关性判断，只把最有用的 {topK} 个片段送入答案上下文。</p></div>
        <span className="formula-chip">Cross-Encoder · 本地可解释模拟</span>
      </div>
      <div className="rerank-grid">
        <div className="before-list">
          <h3><span>BEFORE</span> 混合检索</h3>
          {before.slice(0, 6).map((item, index) => (
            <div key={item.id}><span>{index + 1}</span><b>{item.id}</b><em>{formatScore(item.hybrid)}</em></div>
          ))}
        </div>
        <div className="rerank-core">
          <span>QUERY</span><i>×</i><span>CHUNK</span><b>PAIRWISE<br />SCORING</b><small>覆盖率 · 短语命中 · 语义</small>
        </div>
        <div className="after-list">
          <h3><span>AFTER</span> 重排结果</h3>
          {after.slice(0, 6).map((item, index) => {
            const oldIndex = before.findIndex((candidate) => candidate.id === item.id);
            const delta = oldIndex - index;
            return (
              <button key={item.id} onClick={() => onSelect(item.id)} className={index < topK ? "kept" : ""}>
                <span>{index + 1}</span><b>{item.id}<small>{item.reasons[1]}</small></b>
                <i className={delta > 0 ? "up" : delta < 0 ? "down" : "flat"}>{delta > 0 ? `↑${delta}` : delta < 0 ? `↓${Math.abs(delta)}` : "—"}</i>
                <em>{formatScore(item.rerank)}</em>
              </button>
            );
          })}
        </div>
      </div>
      <div className="teaching-note"><b>生产提示</b><p>真实系统可接入 BGE Reranker、Cohere Rerank 等 Cross-Encoder。它比双塔向量检索慢，因此通常只重排召回后的少量候选。</p></div>
    </div>
  );
}

function AnswerStage({
  query,
  sentences,
  sources,
  onSelect,
}: {
  query: string;
  sentences: { id: string; sentence: string }[];
  sources: RankedChunk[];
  onSelect: (id: string) => void;
}) {
  return (
    <div className="stage-content answer-stage">
      <div className="stage-intro">
        <div><span className="intro-icon">✦</span><p><b>基于证据生成回答</b><br />将 Top-K 片段组成上下文，并要求答案引用来源，降低无依据生成。</p></div>
        <span className="formula-chip">QUERY + CONTEXT → ANSWER + CITATIONS</span>
      </div>
      <div className="answer-layout">
        <article className="answer-card">
          <div className="answer-card-head"><span>GENERATED ANSWER</span><i>● GROUNDED</i></div>
          <p className="answer-query">“{query}”</p>
          <div className="answer-copy">
            <span className="answer-spark">✦</span>
            <p>
              综合访谈证据，
              {sentences.length ? sentences.map((item, index) => (
                <span key={`${item.id}-${index}`}>{item.sentence.replace(/[。；]$/, "")}<button onClick={() => onSelect(item.id)}>[{item.id}]</button>{index < sentences.length - 1 ? "；" : "。"}</span>
              )) : "当前语料中没有足够信息支持回答，请尝试换一种提问。"}
            </p>
          </div>
          <div className="confidence"><span><i style={{ width: `${Math.round((sources[0]?.rerank ?? 0) * 100)}%` }} /></span><p><b>证据覆盖</b> {Math.round((sources[0]?.rerank ?? 0) * 100)}%</p></div>
        </article>
        <aside className="source-stack">
          <h3>CONTEXT WINDOW <span>{sources.length} CHUNKS</span></h3>
          {sources.map((source, index) => (
            <button key={source.id} onClick={() => onSelect(source.id)}>
              <span>{index + 1}</span>
              <b>{source.id}<small>{source.source}</small></b>
              <em>{Math.round(source.rerank * 100)}%</em>
            </button>
          ))}
          <div className="token-budget"><span><i style={{ width: `${Math.min(92, sources.reduce((sum, item) => sum + item.text.length, 0) / 6)}%` }} /></span><p>≈ {sources.reduce((sum, item) => sum + item.text.length, 0)} chars in context</p></div>
        </aside>
      </div>
      <div className="prompt-strip"><span>SYSTEM</span><code>仅根据给定上下文回答；信息不足时明确说明；每条结论附带 [chunk_id]。</code></div>
    </div>
  );
}
