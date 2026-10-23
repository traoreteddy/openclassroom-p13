import { RichTextPipe } from './rich-text.pipe';

describe('RichTextPipe', () => {
  const pipe = new RichTextPipe();

  it('met en gras et en italique sans interpréter le HTML du modèle', () => {
    expect(pipe.transform('Jouez **1...e5** (*solide*) <b>x</b>')).toBe(
      'Jouez <strong>1...e5</strong> (<em>solide</em>) &lt;b&gt;x&lt;/b&gt;',
    );
  });

  it('convertit les sauts de ligne et les éléments de liste', () => {
    expect(pipe.transform('Deux coups :\n1. e5\n- c5')).toBe(
      'Deux coups :<br><span class="item">e5</span><br><span class="item">c5</span>',
    );
  });

  it('transforme les liens Markdown http(s) en liens cliquables, et rien d’autre', () => {
    expect(
      pipe.transform('Voir [la vidéo](https://www.youtube.com/watch?v=abc) et [x](javascript:1)'),
    ).toBe(
      'Voir <a href="https://www.youtube.com/watch?v=abc" target="_blank" rel="noopener">la vidéo</a> et [x](javascript:1)',
    );
  });

  it('renvoie une chaîne vide sans texte', () => {
    expect(pipe.transform(null)).toBe('');
  });
});
