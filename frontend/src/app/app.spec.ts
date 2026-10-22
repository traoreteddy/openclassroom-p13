import { TestBed } from '@angular/core/testing';

import { App } from './app';

describe('App', () => {
  it('affiche l’échiquier et le panneau de conseils', async () => {
    await TestBed.configureTestingModule({ imports: [App] }).compileComponents();
    const fixture = TestBed.createComponent(App);
    fixture.detectChanges();

    const el = fixture.nativeElement as HTMLElement;
    expect(el.querySelector('ngx-chess-board')).not.toBeNull();
    expect(el.querySelector('[data-testid="current-fen"]')?.textContent).toContain(
      'rnbqkbnr/pppppppp',
    );
  });
});
