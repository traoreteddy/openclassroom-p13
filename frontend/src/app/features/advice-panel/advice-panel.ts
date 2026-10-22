import { Component, computed, inject, signal } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

import { AdviceService } from '../../core/advice.service';
import {
  Evaluation,
  KnowledgeContext,
  OpeningTheory,
  ToolData,
  Video,
  Videos,
  isToolFailure,
} from '../../core/api.models';
import { GameStateService } from '../../core/game-state.service';
import { RichTextPipe } from '../../core/rich-text.pipe';

interface MovePair {
  number: number;
  white: string;
  black?: string;
}

const STATUS_LABELS: Record<string, string> = {
  playing: '',
  check: 'Échec !',
  checkmate: 'Échec et mat',
  stalemate: 'Pat',
  draw: 'Partie nulle',
};

/**
 * Panneau d'information : position (FEN), coups joués et recommandations de l'agent
 * (coups théoriques, évaluation, contexte de l'ouverture, vidéos, questions libres).
 * Les données viennent d'`AdviceService`, synchronisé avec la position de l'échiquier.
 */
@Component({
  selector: 'app-advice-panel',
  imports: [RichTextPipe],
  templateUrl: './advice-panel.html',
  styleUrl: './advice-panel.scss',
})
export class AdvicePanel {
  protected readonly game = inject(GameStateService);
  protected readonly advice = inject(AdviceService);
  private readonly sanitizer = inject(DomSanitizer);

  protected readonly fenInput = signal('');
  protected readonly fenError = signal<string | null>(null);
  protected readonly copied = signal(false);
  protected readonly question = signal('');
  /** Question jointe à la demande de conseil (« pourquoi ce coup ? »). */
  protected readonly focus = signal('');
  /** Vidéo dont le lecteur est ouvert dans le panneau. */
  protected readonly playingVideo = signal<string | null>(null);

  protected readonly turnLabel = computed(() =>
    this.game.turn() === 'white' ? 'Trait aux Blancs' : 'Trait aux Noirs',
  );
  protected readonly statusLabel = computed(() => STATUS_LABELS[this.game.status()]);

  /** Coups regroupés par numéro : 1. e4 e5, 2. Cf3 Cc6… */
  protected readonly movePairs = computed<MovePair[]>(() => {
    const history = this.game.history();
    if (history.length === 0) return [];

    const firstNumber = Number(this.game.initialFen.split(' ')[5] ?? 1);
    const blackStarts = history[0].color === 'b';
    const sans = history.map((m) => m.san);
    if (blackStarts) sans.unshift('…');

    const pairs: MovePair[] = [];
    for (let i = 0; i < sans.length; i += 2) {
      pairs.push({ number: firstNumber + i / 2, white: sans[i], black: sans[i + 1] });
    }
    return pairs;
  });

  // --- Données de l'agent, extraites de la réponse ---------------------------------------

  protected readonly response = computed(() => this.advice.analysis()?.response ?? null);
  protected readonly evaluation = computed(() =>
    this.ok<Evaluation>(this.response()?.data.evaluation),
  );
  protected readonly context = computed(() =>
    this.ok<KnowledgeContext>(this.response()?.data.context),
  );
  protected readonly videos = computed(
    () => this.ok<Videos>(this.response()?.data.videos)?.videos ?? [],
  );
  protected readonly videosFailure = computed(() => {
    const data = this.response()?.data.videos;
    return isToolFailure(data) ? data.error : null;
  });
  protected readonly theoryFromAgent = computed(() =>
    this.ok<OpeningTheory>(this.response()?.data.theory),
  );
  /** Théorie affichée : celle de la position courante, sinon celle ramenée par l'agent. */
  protected readonly theory = computed(() => this.advice.theory() ?? this.theoryFromAgent());

  protected readonly scoreLabel = computed(() => formatScore(this.evaluation()));
  protected readonly deepScoreLabel = computed(() => formatScore(this.advice.deep()?.result));
  protected readonly deepStale = computed(() => {
    const deep = this.advice.deep();
    return deep !== null && deep.fen !== this.game.fen();
  });

  // --- Actions ----------------------------------------------------------------------------

  protected loadFen(): void {
    const error = this.game.load(this.fenInput());
    this.fenError.set(error);
    if (!error) this.fenInput.set('');
  }

  protected async copyFen(): Promise<void> {
    try {
      await navigator.clipboard.writeText(this.game.fen());
      this.copied.set(true);
      setTimeout(() => this.copied.set(false), 1500);
    } catch {
      this.copied.set(false);
    }
  }

  /** Joue sur l'échiquier un coup proposé par la théorie ou par le moteur. */
  protected playSuggested(san: string): void {
    this.game.play(san);
  }

  protected requestAdvice(): void {
    this.advice.analyze(this.focus());
    this.focus.set('');
  }

  protected askAgent(): void {
    const text = this.question().trim();
    if (!text) return;
    this.advice.ask(text);
    this.question.set('');
  }

  protected toggleVideo(video: Video): void {
    this.playingVideo.update((current) => (current === video.video_id ? null : video.video_id));
  }

  /** Lecteur intégré sans cookies ; l'URL est construite ici, jamais depuis l'API. */
  protected embedUrl(video: Video): SafeResourceUrl {
    const id = encodeURIComponent(video.video_id);
    return this.sanitizer.bypassSecurityTrustResourceUrl(
      `https://www.youtube-nocookie.com/embed/${id}`,
    );
  }

  protected percent(value: number): string {
    return `${Math.round(value)} %`;
  }

  private ok<T>(data: ToolData<T>): T | null {
    return data && !isToolFailure(data) ? data : null;
  }
}

/** « +0,35 », « −1,20 », « Mat en 3 » (du point de vue des Blancs). */
export function formatScore(evaluation: Evaluation | null | undefined): string {
  if (!evaluation) return '';
  if (evaluation.mate !== null) {
    return evaluation.mate > 0 ? `Mat en ${evaluation.mate}` : `Mat en ${-evaluation.mate} (Noirs)`;
  }
  if (evaluation.score_cp === null) return '';
  const pawns = evaluation.score_cp / 100;
  const sign = pawns > 0 ? '+' : pawns < 0 ? '−' : '';
  return `${sign}${Math.abs(pawns).toFixed(2).replace('.', ',')}`;
}
