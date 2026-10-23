import { Pipe, PipeTransform } from '@angular/core';

/**
 * Met en forme le texte rédigé par le modèle : liens `[texte](https://…)`, gras `**…**`, italique `*…*`, listes
 * numérotées ou à puces en début de ligne, sauts de ligne. Tout le reste est échappé :
 * aucun HTML venant du modèle n'est interprété.
 */
@Pipe({ name: 'richText' })
export class RichTextPipe implements PipeTransform {
  transform(value: string | null | undefined): string {
    if (!value) return '';
    const escaped = value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return escaped
      .replace(
        /\[([^\]]+)\]\s*\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener">$1</a>',
      )
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>')
      .split(/\n/)
      .map((line) => line.replace(/^\s*(?:[-•]|\d+\.)\s+(.*)$/, '<span class="item">$1</span>'))
      .join('<br>');
  }
}
