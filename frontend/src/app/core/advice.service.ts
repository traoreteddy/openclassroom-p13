import { DestroyRef, Injectable, InjectionToken, computed, inject, signal } from '@angular/core';
import { takeUntilDestroyed, toObservable } from '@angular/core/rxjs-interop';
import { Subject, debounceTime, distinctUntilChanged, of, switchMap, timer } from 'rxjs';
import { catchError, expand, filter, map, takeWhile } from 'rxjs/operators';

import { AgentResponse, Evaluation, JobState, OpeningTheory } from './api.models';
import { ApiService, describeError } from './api.service';
import { GameStateService } from './game-state.service';

/** Délai après le dernier changement de position avant d'interroger la théorie. */
export const THEORY_DEBOUNCE_MS = new InjectionToken<number>('THEORY_DEBOUNCE_MS', {
  providedIn: 'root',
  factory: () => 400,
});

/** Intervalle entre deux interrogations de l'état d'une analyse profonde. */
export const DEEP_POLL_MS = new InjectionToken<number>('DEEP_POLL_MS', {
  providedIn: 'root',
  factory: () => 2000,
});

/** Conseil de l'agent, lié à la position pour laquelle il a été demandé. */
export interface Analysis {
  fen: string;
  response: AgentResponse;
}

/** Analyse Stockfish profonde (tâche de fond côté serveur, interrogée périodiquement). */
export interface DeepAnalysis {
  fen: string;
  depth: number;
  status: 'pending' | 'running' | 'done' | 'failed';
  elapsedS: number;
  result: Evaluation | null;
  error: string | null;
}

export interface Exchange {
  question: string;
  response: AgentResponse | null;
  error: string | null;
}

type Outcome<T> = { ok: true; fen: string; value: T } | { ok: false; fen: string; error: string };

/**
 * Orchestration des appels à l'agent, synchronisés avec la position de l'échiquier.
 *
 * - **Théorie** (rapide) : rafraîchie automatiquement à chaque position, après un court
 *   délai, avec un cache par FEN. Une position changée pendant l'appel annule la requête
 *   en cours (`switchMap`) : la réponse affichée correspond toujours à la position.
 * - **Conseil complet** (lent, modèle local) : à la demande ; le résultat garde la FEN
 *   analysée et se signale « dépassé » si la position a bougé depuis.
 * - **Analyse profonde** (Stockfish, une à plusieurs minutes) : lancée en tâche de fond par
 *   l'API, interrogée toutes les 2 s jusqu'à `done` ; abandonnée si la position change.
 * - **Questions libres** : conversation dans la même session (regroupement des traces).
 */
@Injectable({ providedIn: 'root' })
export class AdviceService {
  private readonly api = inject(ApiService);
  private readonly game = inject(GameStateService);
  private readonly destroyRef = inject(DestroyRef);
  private readonly debounceMs = inject(THEORY_DEBOUNCE_MS);
  private readonly deepPollMs = inject(DEEP_POLL_MS);

  /** Identifiant de session : une par chargement de page. */
  readonly sessionId = newSessionId();

  private readonly theoryCache = new Map<string, OpeningTheory>();
  private readonly _theory = signal<OpeningTheory | null>(null);
  private readonly _theoryLoading = signal(false);
  private readonly _theoryError = signal<string | null>(null);
  readonly theory = this._theory.asReadonly();
  readonly theoryLoading = this._theoryLoading.asReadonly();
  readonly theoryError = this._theoryError.asReadonly();

  private readonly analyzeRequests = new Subject<{ fen: string; question: string }>();
  private pendingFen: string | null = null;
  private readonly _analysis = signal<Analysis | null>(null);
  private readonly _analysisLoading = signal(false);
  private readonly _analysisError = signal<string | null>(null);
  readonly analysis = this._analysis.asReadonly();
  readonly analysisLoading = this._analysisLoading.asReadonly();
  readonly analysisError = this._analysisError.asReadonly();
  /** Le conseil affiché porte sur une position antérieure. */
  readonly analysisStale = computed(() => {
    const analysis = this._analysis();
    return analysis !== null && analysis.fen !== this.game.fen();
  });

  private readonly deepRequests = new Subject<{ fen: string; depth: number } | null>();
  private readonly _deep = signal<DeepAnalysis | null>(null);
  readonly deep = this._deep.asReadonly();
  readonly deepRunning = computed(() => {
    const status = this._deep()?.status;
    return status === 'pending' || status === 'running';
  });

  private readonly _exchanges = signal<Exchange[]>([]);
  private readonly _asking = signal(false);
  readonly exchanges = this._exchanges.asReadonly();
  readonly asking = this._asking.asReadonly();

  constructor() {
    toObservable(this.game.fen)
      .pipe(
        debounceTime(this.debounceMs),
        distinctUntilChanged(),
        switchMap((fen) => this.loadTheory(fen)),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((outcome) => this.applyTheory(outcome));

    this.analyzeRequests
      .pipe(
        switchMap(({ fen, question }) =>
          this.api.analyze(fen, this.sessionId, question || undefined).pipe(
            map((value): Outcome<AgentResponse> => ({ ok: true, fen, value })),
            catchError((error) =>
              of<Outcome<AgentResponse>>({ ok: false, fen, error: describeError(error) }),
            ),
          ),
        ),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((outcome) => {
        this.pendingFen = null;
        this._analysisLoading.set(false);
        if (outcome.ok) {
          this._analysis.set({ fen: outcome.fen, response: outcome.value });
          this._analysisError.set(null);
          this.rememberTheory(outcome.fen, outcome.value);
        } else {
          this._analysisError.set(outcome.error);
        }
      });

    this.deepRequests
      .pipe(
        switchMap((request) => (request ? this.followDeepAnalysis(request) : of(null))),
        filter((state): state is DeepAnalysis => state !== null),
        takeUntilDestroyed(this.destroyRef),
      )
      .subscribe((state) => this._deep.set(state));

    // Une position qui change rend l'analyse profonde en cours sans objet.
    toObservable(this.game.fen)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((fen) => {
        const deep = this._deep();
        if (deep && deep.fen !== fen && this.deepRunning()) this.cancelDeepAnalysis();
      });
  }

  /** Démarre la tâche puis l'interroge jusqu'à son terme ; chaque état est émis. */
  private followDeepAnalysis({ fen, depth }: { fen: string; depth: number }) {
    const toState = (job: JobState): DeepAnalysis => ({
      fen,
      depth,
      status: job.status,
      elapsedS: job.elapsed_s,
      result: job.result,
      error: job.error,
    });
    const failed = (error: unknown): DeepAnalysis => ({
      fen,
      depth,
      status: 'failed',
      elapsedS: 0,
      result: null,
      error: describeError(error),
    });

    return this.api.startDeepAnalysis(fen, depth).pipe(
      switchMap((ticket) =>
        this.api.getDeepAnalysis(ticket.job_id).pipe(
          expand((job) =>
            job.status === 'done' || job.status === 'failed'
              ? of()
              : timer(this.deepPollMs).pipe(switchMap(() => this.api.getDeepAnalysis(job.job_id))),
          ),
          map(toState),
          takeWhile((state) => state.status !== 'done' && state.status !== 'failed', true),
        ),
      ),
      catchError((error) => of(failed(error))),
    );
  }

  /** Demande le conseil complet de l'agent sur la position courante (question facultative). */
  analyze(question = ''): void {
    const fen = this.game.fen();
    const text = question.trim();
    if (this.pendingFen === fen && !text) return; // déjà en cours pour cette position
    this.pendingFen = fen;
    this._analysisLoading.set(true);
    this._analysisError.set(null);
    this.analyzeRequests.next({ fen, question: text });
  }

  /** Lance (ou relance) l'analyse profonde de la position courante. */
  deepAnalyze(depth = 28): void {
    const fen = this.game.fen();
    const current = this._deep();
    if (current && current.fen === fen && current.depth === depth && this.deepRunning()) return;
    this._deep.set({ fen, depth, status: 'pending', elapsedS: 0, result: null, error: null });
    this.deepRequests.next({ fen, depth });
  }

  /** Abandonne le suivi de l'analyse profonde (le serveur finit son travail seul). */
  cancelDeepAnalysis(): void {
    this.deepRequests.next(null);
    this._deep.set(null);
  }

  /** Pose une question libre ; la réponse rejoint la conversation. */
  ask(question: string): void {
    const text = question.trim();
    if (!text || this._asking()) return;
    this._asking.set(true);
    this._exchanges.update((list) => [...list, { question: text, response: null, error: null }]);

    this.api
      .ask(text, this.sessionId)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe({
        next: (response) => this.completeExchange({ response, error: null }),
        error: (error) => this.completeExchange({ response: null, error: describeError(error) }),
      });
  }

  /** Relance la théorie pour la position courante (après une erreur). */
  retryTheory(): void {
    const fen = this.game.fen();
    this.theoryCache.delete(fen);
    this.loadTheory(fen)
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((outcome) => this.applyTheory(outcome));
  }

  private loadTheory(fen: string) {
    const cached = this.theoryCache.get(fen);
    if (cached) return of<Outcome<OpeningTheory>>({ ok: true, fen, value: cached });

    this._theoryLoading.set(true);
    this._theoryError.set(null);
    return this.api.theory(fen).pipe(
      map((value): Outcome<OpeningTheory> => ({ ok: true, fen, value })),
      catchError((error) =>
        of<Outcome<OpeningTheory>>({ ok: false, fen, error: describeError(error) }),
      ),
    );
  }

  private applyTheory(outcome: Outcome<OpeningTheory>): void {
    if (outcome.fen !== this.game.fen()) return; // position déjà changée : réponse ignorée
    this._theoryLoading.set(false);
    if (outcome.ok) {
      this.theoryCache.set(outcome.fen, outcome.value);
      this._theory.set(outcome.value);
      this._theoryError.set(null);
    } else {
      this._theory.set(null);
      this._theoryError.set(outcome.error);
    }
  }

  /** La théorie ramenée par l'agent alimente le cache : pas de second appel Lichess. */
  private rememberTheory(fen: string, response: AgentResponse): void {
    const theory = response.data.theory;
    if (theory && !('error' in theory) && !this.theoryCache.has(fen)) {
      this.theoryCache.set(fen, theory);
    }
  }

  private completeExchange(result: Pick<Exchange, 'response' | 'error'>): void {
    this._asking.set(false);
    this._exchanges.update((list) => {
      const last = list.at(-1);
      if (!last) return list;
      return [...list.slice(0, -1), { ...last, ...result }];
    });
  }
}

function newSessionId(): string {
  const random =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : Math.random().toString(16).slice(2);
  return `web-${random.replace(/-/g, '').slice(0, 12)}`;
}
