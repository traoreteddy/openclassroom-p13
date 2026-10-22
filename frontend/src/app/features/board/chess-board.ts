import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  effect,
  inject,
  input,
  output,
  signal,
  viewChild,
} from '@angular/core';
import { MoveChange, NgxChessBoardComponent } from 'ngx-chess-board';
import { Move } from 'chess.js';

import { GameStateService, placement } from '../../core/game-state.service';

const MIN_SIZE = 240;
const MAX_SIZE = 640;

/**
 * Enveloppe de ngx-chess-board.
 *
 * - transmet chaque coup saisi au GameStateService (chess.js valide) ;
 * - redessine l'échiquier quand la position change ailleurs (annuler, charger une FEN…).
 */
@Component({
  selector: 'app-chess-board',
  imports: [NgxChessBoardComponent],
  templateUrl: './chess-board.html',
  styleUrl: './chess-board.scss',
})
export class ChessBoard implements AfterViewInit {
  private readonly game = inject(GameStateService);
  private readonly host = inject<ElementRef<HTMLElement>>(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  private readonly board = viewChild.required<NgxChessBoardComponent>('board');

  /** Bloque la saisie (ex. pendant que l'agent réfléchit). */
  readonly locked = input(false);
  /** Coup accepté par chess.js. */
  readonly moved = output<Move>();
  /** Coup refusé : l'échiquier a été resynchronisé. */
  readonly rejected = output<string>();

  protected readonly size = signal(480);
  private flipped = false;

  constructor() {
    // Position modifiée hors de l'échiquier → on redessine.
    effect(() => {
      const fen = this.game.fen();
      const board = this.board();
      if (placement(board.getFEN()) !== placement(fen)) {
        this.redraw(fen);
      }
    });
  }

  ngAfterViewInit(): void {
    // Absent des environnements sans navigateur (tests) : on garde la taille par défaut.
    if (typeof ResizeObserver === 'undefined') return;

    const observer = new ResizeObserver(([entry]) => {
      const width = Math.floor(entry.contentRect.width);
      this.size.set(Math.max(MIN_SIZE, Math.min(MAX_SIZE, width)));
    });
    observer.observe(this.host.nativeElement);
    this.destroyRef.onDestroy(() => observer.disconnect());
  }

  flip(): void {
    this.flipped = !this.flipped;
    this.board().reverse();
  }

  protected onMoveChange(event: MoveChange): void {
    const move = this.game.applyBoardPosition(event.fen);
    if (move) {
      this.moved.emit(move);
      return;
    }
    this.redraw(this.game.fen());
    this.rejected.emit('Coup non reconnu : l’échiquier a été resynchronisé.');
  }

  private redraw(fen: string): void {
    const board = this.board();
    board.setFEN(fen);
    if (this.flipped) board.reverse();
  }
}
