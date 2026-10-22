import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';
import { DEFAULT_POSITION } from 'chess.js';

import { AdviceService, DEEP_POLL_MS, THEORY_DEBOUNCE_MS } from './advice.service';
import { AgentResponse, OpeningTheory } from './api.models';
import { encodeFen } from './api.service';
import { GameStateService } from './game-state.service';

const theoryOf = (name: string): OpeningTheory => ({
  fen: '',
  database: 'masters',
  opening: { eco: 'C50', name },
  total_games: 10,
  in_theory: true,
  moves: [],
  reference_games: [],
});

const agentResponse = (answer: string): AgentResponse => ({
  answer,
  in_theory: true,
  opening: 'Italian Game',
  data: { theory: theoryOf('Italian Game') },
  tool_calls: [],
  tokens_in: 1,
  tokens_out: 1,
  trace_id: 'ab'.repeat(16),
});

/** Laisse passer les tâches asynchrones en attente (debounce à 0, timers à 0). */
const settle = async () => {
  TestBed.tick(); // exécute les effets (toObservable) avant d'attendre les timers
  await new Promise((resolve) => setTimeout(resolve, 10));
  TestBed.tick();
};

describe('AdviceService', () => {
  let advice: AdviceService;
  let game: GameStateService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: THEORY_DEBOUNCE_MS, useValue: 0 },
        { provide: DEEP_POLL_MS, useValue: 0 },
      ],
    });
    game = TestBed.inject(GameStateService);
    game.reset();
    advice = TestBed.inject(AdviceService);
    http = TestBed.inject(HttpTestingController);
  });

  const emptyVideos = { videos: [], queries: [], from_cache: false, quota_used_today: 0 };
  const videosUrl = (family: string) => (r: { url: string }) =>
    r.url === `/api/v1/videos/${encodeURIComponent(family)}`;

  afterEach(() => {
    // La théorie nomme une ouverture : la carte Vidéos interroge l'API une fois par famille.
    http
      .match((r) => r.url.startsWith('/api/v1/videos/'))
      .filter((r) => !r.cancelled)
      .forEach((r) => r.flush(emptyVideos));
    http.verify();
  });

  const theoryUrl = (fen: string) => (r: { url: string }) =>
    r.url === `/api/v1/moves/${encodeFen(fen)}`;

  it('charge la théorie de la position initiale puis la met en cache', async () => {
    await settle();
    expect(advice.theoryLoading()).toBe(true);
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    expect(advice.theory()?.opening?.name).toBe('Départ');
    expect(advice.theoryLoading()).toBe(false);

    game.play('e4');
    await settle();
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Après e4'));
    game.undo();
    await settle();

    http.expectNone(theoryUrl(DEFAULT_POSITION)); // servie par le cache
    expect(advice.theory()?.opening?.name).toBe('Départ');
  });

  it('annule la requête en cours quand la position change', async () => {
    await settle();
    const first = http.expectOne(theoryUrl(DEFAULT_POSITION));

    game.play('e4');
    await settle();

    expect(first.cancelled).toBe(true);
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Après e4'));
    expect(advice.theory()?.opening?.name).toBe('Après e4');
  });

  it('affiche une erreur réseau et permet de réessayer', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).error(new ProgressEvent('error'), { status: 0 });

    expect(advice.theory()).toBeNull();
    expect(advice.theoryError()).toContain('injoignable');

    advice.retryTheory();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    expect(advice.theoryError()).toBeNull();
    expect(advice.theory()?.opening?.name).toBe('Départ');
  });

  it('demande le conseil à l’agent et le marque dépassé si la position bouge', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    advice.analyze('pourquoi e4 ?');
    expect(advice.analysisLoading()).toBe(true);
    const request = http.expectOne('/api/v1/agent/analyze');
    expect(request.request.body.question).toBe('pourquoi e4 ?');
    expect(request.request.body.session_id).toBe(advice.sessionId);
    request.flush(agentResponse('Jouez e4.'));

    expect(advice.analysisLoading()).toBe(false);
    expect(advice.analysis()?.response.answer).toBe('Jouez e4.');
    expect(advice.analysisStale()).toBe(false);

    game.play('e4');
    expect(advice.analysisStale()).toBe(true);
    await settle();
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Après e4'));
  });

  it('ne relance pas un conseil déjà en cours pour la même position', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    advice.analyze();
    advice.analyze();

    http.expectOne('/api/v1/agent/analyze').flush(agentResponse('Une fois.'));
    expect(advice.analysis()?.response.answer).toBe('Une fois.');
  });

  it('suit une analyse profonde jusqu’à son terme', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    advice.deepAnalyze(30);
    expect(advice.deepRunning()).toBe(true);
    http
      .expectOne((r) => r.url === `/api/v1/analysis/${encodeFen(DEFAULT_POSITION)}`)
      .flush({ job_id: 'job1', status: 'pending', poll_after_s: 15 });
    http.expectOne('/api/v1/analysis/job1').flush({
      job_id: 'job1',
      status: 'running',
      elapsed_s: 1,
      result: null,
      error: null,
      poll_after_s: 15,
    });
    expect(advice.deep()?.status).toBe('running');

    await settle();
    http.expectOne('/api/v1/analysis/job1').flush({
      job_id: 'job1',
      status: 'done',
      elapsed_s: 42,
      result: {
        fen: DEFAULT_POSITION,
        depth: 30,
        score_cp: 20,
        mate: null,
        assessment: 'égal',
        lines: [],
      },
      error: null,
      poll_after_s: null,
    });

    expect(advice.deepRunning()).toBe(false);
    expect(advice.deep()?.result?.depth).toBe(30);
    await settle();
    http.expectNone('/api/v1/analysis/job1');
  });

  it('abandonne l’analyse profonde quand la position change', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    advice.deepAnalyze();
    const start = http.expectOne((r) => r.url.startsWith('/api/v1/analysis/'));

    game.play('e4');
    await settle();

    expect(start.cancelled).toBe(true);
    expect(advice.deep()).toBeNull();
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Après e4'));
  });

  it('enchaîne les questions libres dans la conversation', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Départ'));

    advice.ask('  Bonjour ?  ');
    expect(advice.asking()).toBe(true);
    expect(advice.exchanges()[0]).toEqual({ question: 'Bonjour ?', response: null, error: null });

    const request = http.expectOne('/api/v1/agent/ask');
    expect(request.request.body).toEqual({ question: 'Bonjour ?', session_id: advice.sessionId });
    request.flush(agentResponse('Bonjour !'));

    expect(advice.asking()).toBe(false);
    expect(advice.exchanges()[0].response?.answer).toBe('Bonjour !');
  });

  it('charge les vidéos de la famille d’ouverture une seule fois', async () => {
    await settle();
    http
      .expectOne(theoryUrl(DEFAULT_POSITION))
      .flush(theoryOf('Italian Game: Two Knights Defense'));
    await settle();

    expect(advice.openingFamily()).toBe('Italian Game');
    expect(advice.videosLoading()).toBe(true);
    const request = http.expectOne(videosUrl('Italian Game'));
    expect(request.request.params.get('max_results')).toBe('3');
    request.flush({
      ...emptyVideos,
      videos: [{ video_id: 'v1', title: 'Italienne', channel: 'c', url: 'u', embed_url: 'e' }],
    });

    expect(advice.videosLoading()).toBe(false);
    expect(advice.videos()?.videos[0].video_id).toBe('v1');

    game.play('e4');
    await settle();
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Italian Game: Evans Gambit'));
    await settle();

    http.expectNone(videosUrl('Italian Game')); // même famille : pas de nouvel appel
    expect(advice.videos()?.videos[0].video_id).toBe('v1');
  });

  it('explique l’absence de vidéos et les erreurs de la carte Vidéos', async () => {
    await settle();
    http.expectOne(theoryUrl(DEFAULT_POSITION)).flush(theoryOf('Bird Opening'));
    await settle();
    http.expectOne(videosUrl('Bird Opening')).flush({ ...emptyVideos, message: 'Aucune vidéo.' });

    expect(advice.videos()?.message).toBe('Aucune vidéo.');

    game.play('e4');
    await settle();
    http.expectOne(theoryUrl(game.fen())).flush(theoryOf('Sicilian Defense'));
    await settle();
    http
      .expectOne(videosUrl('Sicilian Defense'))
      .flush(
        { error: { kind: 'unavailable', code: 'quota_exceeded', message: 'Quota épuisé.' } },
        { status: 503, statusText: 'Service Unavailable' },
      );

    expect(advice.videos()).toBeNull();
    expect(advice.videosError()).toBe('Quota épuisé.');
  });
});
