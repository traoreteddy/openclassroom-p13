import { Component, inject, signal, viewChild } from '@angular/core';

import { GameStateService } from './core/game-state.service';
import { AdvicePanel } from './features/advice-panel/advice-panel';
import { ChessBoard } from './features/board/chess-board';

@Component({
  selector: 'app-root',
  imports: [ChessBoard, AdvicePanel],
  templateUrl: './app.html',
  styleUrl: './app.scss',
})
export class App {
  protected readonly game = inject(GameStateService);
  private readonly board = viewChild.required(ChessBoard);

  protected readonly notice = signal<string | null>(null);

  protected undo(): void {
    this.game.undo();
  }

  protected reset(): void {
    this.game.reset();
  }

  protected flip(): void {
    this.board().flip();
  }

  protected showNotice(message: string): void {
    this.notice.set(message);
    setTimeout(() => this.notice.set(null), 3000);
  }
}
