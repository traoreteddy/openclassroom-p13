import { HttpClient, HttpErrorResponse, HttpParams } from '@angular/common/http';
import { Injectable, InjectionToken, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  AgentResponse,
  ApiErrorBody,
  Evaluation,
  JobState,
  JobTicket,
  OpeningTheory,
  Videos,
} from './api.models';

/**
 * Préfixe des appels à l'API. Relatif par défaut : en développement, `ng serve` relaie
 * `/api` vers le backend (proxy.conf.json) ; en production, le front et l'API partagent
 * la même origine derrière le même serveur.
 */
export const API_BASE_URL = new InjectionToken<string>('API_BASE_URL', {
  providedIn: 'root',
  factory: () => '/api/v1',
});

/** Accès HTTP à l'API backend : une méthode par route, aucune logique d'affichage. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);
  private readonly baseUrl = inject(API_BASE_URL);

  /** Coups théoriques (Lichess Opening Explorer) pour une position. */
  theory(fen: string, maxMoves = 5): Observable<OpeningTheory> {
    const params = new HttpParams().set('max_moves', maxMoves).set('include_games', false);
    return this.http.get<OpeningTheory>(`${this.baseUrl}/moves/${encodeFen(fen)}`, { params });
  }

  /** Évaluation Stockfish d'une position. */
  evaluate(fen: string, depth = 14): Observable<Evaluation> {
    const params = new HttpParams().set('depth', depth);
    return this.http.get<Evaluation>(`${this.baseUrl}/evaluate/${encodeFen(fen)}`, { params });
  }

  /** Vidéos explicatives YouTube sur une ouverture (nom Lichess), filtrées et mises en cache. */
  videos(opening: string, maxResults = 3): Observable<Videos> {
    const params = new HttpParams().set('max_results', maxResults);
    return this.http.get<Videos>(`${this.baseUrl}/videos/${encodeURIComponent(opening)}`, {
      params,
    });
  }

  /** Lance une analyse Stockfish profonde en tâche de fond (202 + ticket). */
  startDeepAnalysis(fen: string, depth = 28): Observable<JobTicket> {
    const params = new HttpParams().set('depth', depth);
    return this.http.post<JobTicket>(`${this.baseUrl}/analysis/${encodeFen(fen)}`, null, {
      params,
    });
  }

  /** État d'une analyse profonde ; `result` est rempli quand `status` vaut `done`. */
  getDeepAnalysis(jobId: string): Observable<JobState> {
    return this.http.get<JobState>(`${this.baseUrl}/analysis/${encodeURIComponent(jobId)}`);
  }

  /** Conseil complet de l'agent sur une position (Lichess, Stockfish, contexte, vidéos). */
  analyze(fen: string, sessionId: string, question?: string): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/agent/analyze`, {
      fen,
      session_id: sessionId,
      question: question || null,
    });
  }

  /** Question libre à l'agent (il décide lui-même des outils à appeler). */
  ask(question: string, sessionId: string): Observable<AgentResponse> {
    return this.http.post<AgentResponse>(`${this.baseUrl}/agent/ask`, {
      question,
      session_id: sessionId,
    });
  }
}

/**
 * La FEN va dans le chemin de l'URL : seuls les espaces sont encodés, les « / » entre
 * les rangées doivent rester tels quels (route FastAPI `{fen:path}`).
 */
export function encodeFen(fen: string): string {
  return fen.trim().replace(/ /g, '%20');
}

/** Message lisible pour l'utilisateur à partir d'une erreur HTTP. */
export function describeError(error: unknown): string {
  if (!(error instanceof HttpErrorResponse)) return 'Erreur inattendue.';
  if (error.status === 0) return 'Serveur injoignable : vérifiez que l’API est démarrée.';

  const body = error.error as Partial<ApiErrorBody> | { detail?: unknown } | null;
  if (body && typeof body === 'object' && 'error' in body && body.error) {
    const { message, hint } = body.error;
    return hint ? `${message} ${hint}` : message;
  }
  if (body && typeof body === 'object' && 'detail' in body && typeof body.detail === 'string') {
    return body.detail;
  }
  if (error.status === 422) return 'Requête refusée : position ou question invalide.';
  if (error.status >= 500) return 'Le serveur a rencontré une erreur. Réessayez.';
  return `Erreur ${error.status}.`;
}
