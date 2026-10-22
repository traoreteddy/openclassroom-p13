import { HttpErrorResponse, provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { ApiService, describeError, encodeFen } from './api.service';

const FEN = 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3';

describe('ApiService', () => {
  let api: ApiService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    api = TestBed.inject(ApiService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('encode la FEN pour le chemin : espaces seulement, les « / » restent', () => {
    expect(encodeFen(' rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 ')).toBe(
      'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR%20w%20KQkq%20-%200%201',
    );
  });

  it('demande la théorie sans les parties de référence', () => {
    api.theory(FEN).subscribe();

    const request = http.expectOne((r) => r.url === `/api/v1/moves/${encodeFen(FEN)}`);
    expect(request.request.method).toBe('GET');
    expect(request.request.params.get('max_moves')).toBe('5');
    expect(request.request.params.get('include_games')).toBe('false');
    request.flush({ moves: [] });
  });

  it('envoie la position, la session et la question à l’agent', () => {
    api.analyze(FEN, 'web-abc', 'pourquoi Fc5 ?').subscribe();

    const request = http.expectOne('/api/v1/agent/analyze');
    expect(request.request.method).toBe('POST');
    expect(request.request.body).toEqual({
      fen: FEN,
      session_id: 'web-abc',
      question: 'pourquoi Fc5 ?',
    });
    request.flush({ answer: '' });
  });

  it('envoie une question nulle quand aucune n’est posée', () => {
    api.analyze(FEN, 'web-abc').subscribe();

    expect(http.expectOne('/api/v1/agent/analyze').request.body.question).toBeNull();
  });

  it('lance une analyse profonde puis lit son état', () => {
    api.startDeepAnalysis(FEN, 30).subscribe();
    const start = http.expectOne((r) => r.url === `/api/v1/analysis/${encodeFen(FEN)}`);
    expect(start.request.method).toBe('POST');
    expect(start.request.params.get('depth')).toBe('30');
    start.flush({ job_id: 'job1', status: 'pending', poll_after_s: 15 });

    api.getDeepAnalysis('job1').subscribe();
    expect(http.expectOne('/api/v1/analysis/job1').request.method).toBe('GET');
  });

  describe('describeError', () => {
    it('explique un serveur injoignable', () => {
      expect(describeError(new HttpErrorResponse({ status: 0 }))).toContain('injoignable');
    });

    it('reprend le message et le conseil de l’API', () => {
      const error = new HttpErrorResponse({
        status: 503,
        error: {
          error: {
            kind: 'unavailable',
            code: 'timeout',
            message: 'Lichess ne répond pas.',
            hint: 'Réessayer.',
          },
        },
      });
      expect(describeError(error)).toBe('Lichess ne répond pas. Réessayer.');
    });

    it('a un message générique pour une erreur serveur sans corps', () => {
      expect(describeError(new HttpErrorResponse({ status: 500 }))).toContain('serveur');
      expect(describeError('boom')).toBe('Erreur inattendue.');
    });
  });
});
