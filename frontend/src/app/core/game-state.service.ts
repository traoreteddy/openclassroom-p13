import { Injectable, computed, signal } from '@angular/core';
import { Chess, DEFAULT_POSITION, Move, validateFen } from 'chess.js';

export type GameStatus = 'playing' | 'check' | 'checkmate' | 'stalemate' | 'draw';

/**
 * Seule source de vérité de la partie.
 *
 * L'échiquier graphique ne fait qu'afficher et saisir les coups :
 * la position, la légalité et l'historique sont gérés ici par chess.js.
 */
@Injectable({ providedIn: 'root' })
export class GameStateService {
  private readonly chess = new Chess();

  private readonly _fen = signal(this.chess.fen());
  private readonly _history = signal<Move[]>([]);

  /** FEN complète de la position courante (celle envoyée au backend). */
  readonly fen = this._fen.asReadonly();
  /** Coups joués depuis la position de départ. */
  readonly history = this._history.asReadonly();
  readonly lastMove = computed(() => this._history().at(-1) ?? null);
  readonly turn = computed(() => (this.fenFields()[1] === 'w' ? 'white' : 'black'));
  readonly status = computed<GameStatus>(() => {
    this._fen(); // recalcule à chaque changement de position
    if (this.chess.isCheckmate()) return 'checkmate';
    if (this.chess.isStalemate()) return 'stalemate';
    if (this.chess.isDraw()) return 'draw';
    if (this.chess.inCheck()) return 'check';
    return 'playing';
  });

  private startFen = DEFAULT_POSITION;

  /**
   * Retrouve le coup légal qui mène au placement de pièces affiché par l'échiquier.
   * Comparer les placements gère roque, prise en passant et promotion sans dépendre
   * du format de coup de la librairie graphique.
   *
   * @returns le coup joué, ou `null` si aucun coup légal ne correspond.
   */
  applyBoardPosition(boardFen: string): Move | null {
    const target = placement(boardFen);
    const move = this.chess
      .moves({ verbose: true })
      .find((candidate) => placement(candidate.after) === target);

    if (!move) return null;
    this.chess.move(move);
    this.sync();
    return move;
  }

  /** Joue un coup en notation SAN (« Nf3 ») ou UCI (« g1f3 »). */
  play(move: string): Move | null {
    try {
      const played = this.chess.move(move);
      this.sync();
      return played;
    } catch {
      return null;
    }
  }

  undo(): Move | null {
    const undone = this.chess.undo();
    this.sync();
    return undone;
  }

  reset(): void {
    this.load(DEFAULT_POSITION);
  }

  /** Charge une position. Renvoie un message d'erreur si la FEN est invalide. */
  load(fen: string): string | null {
    const trimmed = fen.trim();
    const check = validateFen(trimmed);
    if (!check.ok) return check.error ?? 'FEN invalide';

    this.chess.load(trimmed);
    this.startFen = trimmed;
    this.sync();
    return null;
  }

  /** Position de départ de la partie courante (utile après un `load`). */
  get initialFen(): string {
    return this.startFen;
  }

  private fenFields(): string[] {
    return this._fen().split(' ');
  }

  private sync(): void {
    this._fen.set(this.chess.fen());
    this._history.set(this.chess.history({ verbose: true }));
  }
}

/** Premier champ de la FEN : uniquement le placement des pièces. */
export function placement(fen: string): string {
  return fen.trim().split(' ')[0];
}
