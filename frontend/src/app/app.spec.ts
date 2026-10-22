import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { TestBed } from '@angular/core/testing';

import { App } from './app';
import { THEORY_DEBOUNCE_MS } from './core/advice.service';

describe('App', () => {
  it('affiche l’échiquier et le panneau de conseils', async () => {
    await TestBed.configureTestingModule({
      imports: [App],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        { provide: THEORY_DEBOUNCE_MS, useValue: 0 },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('ngx-chess-board')).not.toBeNull();
    expect(el.querySelector('[data-testid="current-fen"]')?.textContent).toContain(
      'rnbqkbnr/pppppppp',
    );
    expect(el.querySelector('[data-testid="analyze"]')?.textContent).toContain('Demander conseil');
    expect(el.querySelector('[data-testid="deep-analyze"]')).not.toBeNull();
    expect(el.querySelector('[data-testid="ask-card"] input')).not.toBeNull();

    await new Promise((resolve) => setTimeout(resolve, 5));
    TestBed.inject(HttpTestingController)
      .match(() => true)
      .forEach((r) => r.flush({ moves: [] }));
  });
});
