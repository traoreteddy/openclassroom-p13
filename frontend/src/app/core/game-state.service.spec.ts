import { TestBed } from '@angular/core/testing';
import { Chess, DEFAULT_POSITION } from 'chess.js';

import { GameStateService, placement } from './game-state.service';

/** FEN obtenue en jouant `moves` depuis `fen` (ce que l'échiquier afficherait). */
function boardFenAfter(moves: string[], fen = DEFAULT_POSITION): string {
  const chess = new Chess(fen);
  moves.forEach((m) => chess.move(m));
  return chess.fen();
}

describe('GameStateService', () => {
  let game: GameStateService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    game = TestBed.inject(GameStateService);
    game.reset();
  });

  it('démarre sur la position initiale, trait aux Blancs', () => {
    expect(game.fen()).toBe(DEFAULT_POSITION);
    expect(game.turn()).toBe('white');
    expect(game.history()).toEqual([]);
  });

  it('retrouve le coup joué à partir de la position affichée', () => {
    const move = game.applyBoardPosition(boardFenAfter(['e4']));

    expect(move?.san).toBe('e4');
    expect(game.fen()).toBe(boardFenAfter(['e4']));
    expect(game.turn()).toBe('black');
  });

  it('gère le roque (deux pièces bougent)', () => {
    const before = ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Bc5'];
    before.forEach((m) => game.play(m));

    const move = game.applyBoardPosition(boardFenAfter([...before, 'O-O']));

    expect(move?.san).toBe('O-O');
  });

  it('gère la prise en passant (pion capturé sur une autre case)', () => {
    const before = ['e4', 'a6', 'e5', 'd5'];
    before.forEach((m) => game.play(m));

    const move = game.applyBoardPosition(boardFenAfter([...before, 'exd6']));

    expect(move?.san).toBe('exd6');
  });

  it('distingue les promotions selon la pièce choisie', () => {
    const start = '8/P6k/8/8/8/8/8/K7 w - - 0 1';
    game.load(start);

    const move = game.applyBoardPosition(boardFenAfter(['a8=N'], start));

    expect(move?.san).toBe('a8=N');
  });

  it('refuse une position qui ne correspond à aucun coup légal', () => {
    const illegal = 'rnbqkbnr/pppppppp/8/8/8/4P3/PPP2PPP/RNBQKBNR b KQkq - 0 1';

    expect(game.applyBoardPosition(illegal)).toBeNull();
    expect(game.fen()).toBe(DEFAULT_POSITION);
  });

  it('annule le dernier coup', () => {
    game.play('e4');
    game.play('e5');

    game.undo();

    expect(game.fen()).toBe(boardFenAfter(['e4']));
    expect(game.history().map((m) => m.san)).toEqual(['e4']);
  });

  it('charge une FEN valide et rejette une FEN invalide', () => {
    const sicilian = boardFenAfter(['e4', 'c5']);

    expect(game.load(sicilian)).toBeNull();
    expect(game.fen()).toBe(sicilian);

    expect(game.load('pas une fen')).not.toBeNull();
    expect(game.fen()).toBe(sicilian);
  });

  it('détecte l’échec et mat', () => {
    ['f3', 'e5', 'g4', 'Qh4#'].forEach((m) => game.play(m));

    expect(game.status()).toBe('checkmate');
  });

  it('placement() ne garde que le premier champ de la FEN', () => {
    expect(placement(DEFAULT_POSITION)).toBe('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR');
  });
});
