import { Component, computed, inject, signal } from '@angular/core';

import { GameStateService } from '../../core/game-state.service';

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
 * Panneau d'information : position (FEN), coups joués et recommandations de l'agent.
 * Les recommandations seront branchées sur l'API à l'étape 3.
 */
@Component({
  selector: 'app-advice-panel',
  templateUrl: './advice-panel.html',
  styleUrl: './advice-panel.scss',
})
export class AdvicePanel {
  protected readonly game = inject(GameStateService);

  protected readonly fenInput = signal('');
  protected readonly fenError = signal<string | null>(null);
  protected readonly copied = signal(false);

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
}
