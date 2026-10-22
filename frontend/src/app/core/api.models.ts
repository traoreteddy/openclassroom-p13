/**
 * Types des réponses de l'API backend (`/api/v1`).
 * Miroir des modèles Pydantic : `app/tools/*.py` (sorties d'outils) et
 * `app/api/routes/agent.py` (réponse de l'agent).
 */

export interface Opening {
  eco: string;
  name: string;
}

export interface TheoryMove {
  san: string;
  uci: string;
  games: number;
  white_pct: number;
  draw_pct: number;
  black_pct: number;
  average_rating: number | null;
}

export interface ReferenceGame {
  id: string;
  white: string;
  black: string;
  white_rating: number | null;
  black_rating: number | null;
  year: number | null;
  winner: 'white' | 'black' | 'draw';
  url: string;
}

/** Sortie de `get_opening_theory` (Lichess Opening Explorer). */
export interface OpeningTheory {
  fen: string;
  database: 'masters' | 'lichess';
  opening: Opening | null;
  total_games: number;
  in_theory: boolean;
  moves: TheoryMove[];
  reference_games: ReferenceGame[];
}

export interface EngineLine {
  rank: number;
  score_cp: number | null;
  mate: number | null;
  move_san: string;
  move_uci: string;
  pv_san: string[];
}

/** Sortie de `evaluate_position` (Stockfish). */
export interface Evaluation {
  fen: string;
  depth: number;
  score_cp: number | null;
  mate: number | null;
  assessment: string;
  lines: EngineLine[];
}

export interface Passage {
  chunk_id: string;
  article_id: string;
  title: string;
  opening: string;
  opening_family: string;
  section: string;
  text: string;
  source_url: string;
  score: number;
}

/** Sortie de `search_opening_knowledge` (RAG). */
export interface KnowledgeContext {
  found: boolean;
  passages: Passage[];
}

export interface Video {
  video_id: string;
  title: string;
  channel: string;
  url: string;
  thumbnail_url: string | null;
  published_at: string | null;
}

/** Sortie de `find_opening_videos` (YouTube). */
export interface Videos {
  videos: Video[];
}

export type JobStatus = 'pending' | 'running' | 'done' | 'failed';

/** Réponse de `POST /analysis/{fen}` (202) : ticket de l'analyse lancée. */
export interface JobTicket {
  job_id: string;
  status: JobStatus;
  poll_after_s: number;
}

/** Réponse de `GET /analysis/{job_id}`. */
export interface JobState {
  job_id: string;
  status: JobStatus;
  elapsed_s: number;
  result: Evaluation | null;
  error: string | null;
  poll_after_s: number | null;
}

/** Un outil du parcours a échoué : le backend met son message à la place des données. */
export interface ToolFailure {
  error: string;
}

export type ToolData<T> = T | ToolFailure | null | undefined;

export function isToolFailure(data: unknown): data is ToolFailure {
  return typeof data === 'object' && data !== null && 'error' in data;
}

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  status: 'success' | 'error';
  outcome: string;
}

/** Réponse de `POST /agent/analyze` et `POST /agent/ask`. */
export interface AgentResponse {
  answer: string;
  in_theory: boolean | null;
  opening: string | null;
  data: {
    position?: ToolData<unknown>;
    theory?: ToolData<OpeningTheory>;
    evaluation?: ToolData<Evaluation>;
    context?: ToolData<KnowledgeContext>;
    videos?: ToolData<Videos>;
    [tool: string]: unknown;
  };
  tool_calls: ToolCall[];
  tokens_in: number;
  tokens_out: number;
  trace_id: string;
}

/** Corps d'erreur renvoyé par les routes outils (422 / 503). */
export interface ApiErrorBody {
  error: {
    kind: 'input' | 'transient' | 'unavailable';
    code: string;
    message: string;
    hint?: string;
    details?: Record<string, unknown>;
  };
}
