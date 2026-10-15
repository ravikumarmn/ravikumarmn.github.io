/**
 * Architecture diagrams for the case studies. Coordinates are in SVG user units
 * (the figure scales responsively). Components named in the résumé are drawn as-is;
 * connective details that are inferred are called out in each figure caption.
 */
import type { DiagramSpec } from '~/lib/diagram';

export const multiAgentAssistant: DiagramSpec = {
  title: 'Governed multi-agent assistant',
  description:
    'Requests from business users pass through guardrails to a LangGraph orchestrator on Databricks Model Serving, which routes work to retrieval, Genie (natural language to SQL), reasoning, and task agents. Data access goes through Unity Catalog; prompts and responses are written to an audit log; models are versioned and deployed through Azure DevOps CI/CD.',
  width: 960,
  height: 550,
  groups: [{ x: 350, y: 24, w: 400, h: 420, label: 'DATABRICKS MODEL SERVING · LANGGRAPH' }],
  nodes: [
    { x: 20, y: 202, w: 130, label: 'Business users', sub: 'multi-tenant' },
    { x: 180, y: 202, w: 140, label: 'Guardrails', sub: 'input · output' },
    { x: 365, y: 202, w: 160, label: 'Orchestrator', sub: 'routing · shared state', tone: 'accent' },
    { x: 570, y: 58, w: 160, label: 'Retrieval agent', sub: 'unstructured context' },
    { x: 570, y: 150, w: 160, label: 'Genie agent', sub: 'NL → SQL on tables' },
    { x: 570, y: 242, w: 160, label: 'Reasoning agent', sub: 'multi-step reasoning' },
    { x: 570, y: 334, w: 160, label: 'Task agent', sub: 'workflow actions' },
    { x: 780, y: 64, w: 160, h: 140, label: 'Unity Catalog', sub: 'lineage · access control' },
    { x: 780, y: 320, w: 160, h: 84, label: 'Business systems', sub: 'tools · APIs', tone: 'quiet' },
    { x: 170, y: 476, w: 160, label: 'Audit log', sub: 'prompts · responses' },
    { x: 365, y: 476, w: 160, label: 'Model governance', sub: 'versions · policies' },
    { x: 570, y: 476, w: 160, label: 'Azure DevOps CI/CD', sub: 'build · test · deploy' },
  ],
  edges: [
    { points: [[150, 222], [180, 222]] },
    { points: [[180, 240], [150, 240]] },
    { points: [[320, 222], [365, 222]] },
    { points: [[365, 240], [320, 240]] },
    { points: [[525, 230], [548, 230], [548, 86], [570, 86]] },
    { points: [[548, 178], [570, 178]] },
    { points: [[548, 230], [548, 270], [570, 270]] },
    { points: [[548, 270], [548, 362], [570, 362]] },
    { points: [[730, 86], [780, 86]] },
    { points: [[730, 178], [780, 178]] },
    { points: [[730, 362], [780, 362]] },
    { points: [[250, 258], [250, 476]], dashed: true, label: { text: 'every turn', at: [258, 380], anchor: 'start' } },
    { points: [[445, 476], [445, 444]], dashed: true },
    { points: [[650, 476], [650, 444]], dashed: true },
  ],
};

export const agenticRag: DiagramSpec = {
  title: 'Agentic RAG copilot',
  description:
    'A query planner decomposes each question and routes it to hybrid retrieval (vector and keyword search, merged and ranked), a SQL agent for structured data, and tool calls. A synthesis step with memory and guardrails produces a grounded answer in about 1.8 seconds end to end. MLflow tracks, evaluates, and versions releases on Azure and Databricks.',
  width: 960,
  height: 460,
  groups: [{ x: 350, y: 20, w: 190, h: 174, label: 'HYBRID RETRIEVAL' }],
  nodes: [
    { x: 20, y: 190, w: 120, label: 'User question' },
    { x: 170, y: 190, w: 150, label: 'Query planner', sub: 'decompose · route', tone: 'accent' },
    { x: 370, y: 52, w: 150, label: 'Vector search', sub: 'dense · <120ms' },
    { x: 370, y: 126, w: 150, label: 'Keyword search', sub: 'lexical match' },
    { x: 370, y: 202, w: 150, label: 'SQL agent', sub: 'structured data' },
    { x: 370, y: 290, w: 150, label: 'Tool calls', sub: 'analytics · documents' },
    { x: 570, y: 88, w: 140, label: 'Merge & rank', sub: 'fused top-k' },
    { x: 750, y: 190, w: 190, label: 'Response synthesis', sub: 'memory · guardrails', tone: 'accent' },
    { x: 750, y: 300, w: 190, label: 'Grounded answer', sub: '1.8s end-to-end' },
    { x: 20, y: 392, w: 920, h: 48, label: 'MLflow: tracking · evaluation · versioned releases on Azure + Databricks', tone: 'quiet' },
  ],
  edges: [
    { points: [[140, 218], [170, 218]] },
    { points: [[320, 218], [340, 218], [340, 80], [370, 80]] },
    { points: [[340, 154], [370, 154]] },
    { points: [[340, 218], [340, 230], [370, 230]] },
    { points: [[340, 230], [340, 318], [370, 318]] },
    { points: [[520, 80], [545, 80], [545, 108], [570, 108]] },
    { points: [[520, 154], [545, 154], [545, 124], [570, 124]] },
    { points: [[710, 116], [730, 116], [730, 204], [750, 204]] },
    { points: [[520, 230], [750, 230]] },
    { points: [[520, 318], [730, 318], [730, 240], [750, 240]] },
    { points: [[845, 246], [845, 300]] },
  ],
};

export const documentIntelligence: DiagramSpec = {
  title: 'Agentic document intelligence',
  description:
    'Ingestion: documents (PDFs, forms, reports) are parsed by Docling, then an LLM extraction agent classifies and extracts them into structured JSON, a knowledge graph, and a vector index, in under 3 seconds per page. Query: a QA and summary agent retrieves context from the index and returns a grounded answer in about 1.5 seconds.',
  width: 960,
  height: 420,
  groups: [
    { x: 8, y: 8, w: 944, h: 244, label: 'INGESTION · <3s PER PAGE' },
    { x: 8, y: 276, w: 944, h: 136, label: 'QUERY · 1.5s GROUNDED QA' },
  ],
  nodes: [
    { x: 28, y: 110, w: 150, label: 'Documents', sub: 'PDFs · forms · reports' },
    { x: 208, y: 110, w: 150, label: 'Docling parser', sub: 'layout · tables · text', tone: 'accent' },
    { x: 388, y: 110, w: 170, label: 'LLM extraction agent', sub: 'classify · extract' },
    { x: 620, y: 40, w: 170, label: 'Structured JSON' },
    { x: 620, y: 110, w: 170, label: 'Knowledge graph' },
    { x: 620, y: 180, w: 170, label: 'Vector index', sub: 'chunks · embeddings' },
    { x: 28, y: 322, w: 150, label: 'User question' },
    { x: 388, y: 322, w: 170, label: 'QA & summary agent', sub: 'retrieve · ground', tone: 'accent' },
    { x: 780, y: 322, w: 160, label: 'Grounded answer', sub: '1.5s end-to-end' },
  ],
  edges: [
    { points: [[178, 138], [208, 138]] },
    { points: [[358, 138], [388, 138]] },
    { points: [[558, 138], [590, 138], [590, 68], [620, 68]] },
    { points: [[590, 138], [620, 138]] },
    { points: [[590, 138], [590, 208], [620, 208]] },
    { points: [[178, 350], [388, 350]] },
    { points: [[558, 350], [780, 350]] },
    {
      points: [[705, 236], [705, 300], [473, 300], [473, 322]],
      dashed: true,
      label: { text: 'top-k context', at: [713, 292], anchor: 'start' },
    },
  ],
};

export const modelAdaptation: DiagramSpec = {
  title: 'Model adaptation pipeline',
  description:
    'Curated, formatted domain data feeds three kinds of adaptation: supervised fine-tuning of an LLM with LoRA adapters, contrastive fine-tuning of embedding models on domain text pairs, and fine-tuning of transformer classifiers. Each passes an evaluation gate on task metrics before optimized serving; evaluation results feed back into data curation.',
  width: 960,
  height: 370,
  nodes: [
    { x: 20, y: 150, w: 150, label: 'Domain data', sub: 'curate · format' },
    { x: 230, y: 50, w: 230, label: 'LLM · SFT + LoRA', sub: 'frozen base, low-rank adapters' },
    { x: 230, y: 150, w: 230, label: 'Embeddings · contrastive', sub: 'domain text pairs' },
    { x: 230, y: 250, w: 230, label: 'Classifier · fine-tune', sub: '78% → 89% accuracy' },
    { x: 530, y: 150, w: 170, label: 'Evaluation gate', sub: 'task metrics', tone: 'accent' },
    { x: 770, y: 150, w: 170, label: 'Serving', sub: 'optimized inference' },
  ],
  edges: [
    { points: [[170, 178], [200, 178], [200, 78], [230, 78]] },
    { points: [[200, 178], [230, 178]] },
    { points: [[200, 178], [200, 278], [230, 278]] },
    { points: [[460, 78], [500, 78], [500, 170], [530, 170]] },
    { points: [[460, 178], [530, 178]] },
    { points: [[460, 278], [500, 278], [500, 186], [530, 186]] },
    { points: [[700, 178], [770, 178]] },
    {
      points: [[615, 206], [615, 345], [95, 345], [95, 206]],
      dashed: true,
      label: { text: 'iterate on data', at: [355, 337] },
    },
  ],
};

/** Each case study's diagram, keyed by its slug (used for the work-card previews). */
export const diagramsBySlug: Record<string, DiagramSpec> = {
  'governed-multi-agent-assistant': multiAgentAssistant,
  'agentic-rag-copilot': agenticRag,
  'document-intelligence': documentIntelligence,
  'model-adaptation': modelAdaptation,
};
