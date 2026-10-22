# Frontend — Coach d'ouvertures (Angular 21)

Échiquier interactif (`ngx-chess-board`) et panneau de conseils de l'agent.

## Démarrage

```bash
cd frontend
npm install
npm start          # compile la librairie ngx-chess-board puis lance http://localhost:4200
```

Autres commandes :

```bash
npm test           # tests unitaires (Vitest)
npm run build      # build de production dans dist/app
```

> Si `npm install` échoue avec `Cannot read properties of null (reading 'edgesOut')`,
> c'est un bug de npm 10.9 : utiliser `npx npm@11 install`.

## Organisation

```
frontend/
├── projects/ngx-chess-board/   # librairie échiquier (source embarquée, voir plus bas)
└── src/app/
    ├── core/game-state.service.ts        # état de la partie (chess.js) — source de vérité
    ├── features/board/                   # enveloppe de ngx-chess-board
    └── features/advice-panel/            # position, coups joués, conseils de l'agent
```

## Synchronisation de la position (FEN)

`chess.js` est la **seule source de vérité** (`GameStateService`) :

1. le joueur bouge une pièce sur l'échiquier ;
2. `ChessBoard` reçoit la FEN affichée et demande au service quel coup légal mène à ce
   placement de pièces (gère roque, prise en passant, promotion) ;
3. si aucun coup ne correspond, l'échiquier est redessiné depuis `chess.js` ;
4. toute autre modification (annuler, charger une FEN, nouvelle partie) passe par le
   service, et l'échiquier se redessine automatiquement (`effect`).

La FEN envoyée au backend est donc toujours celle de `chess.js` : complète et valide.

## À propos de ngx-chess-board

Le paquet npm `ngx-chess-board` (3.0.0) n'est plus maintenu depuis 2023 et cible Angular 17.
On utilise la source fournie par OpenClassrooms
([material-chessboard](https://github.com/OpenClassrooms-Student-Center/material-chessboard),
licence MIT, auteur d'origine : grzegorz103), embarquée dans `projects/ngx-chess-board`
et compilée comme librairie Angular locale (`npm run build:board`, lancé automatiquement
avant `start`, `build` et `test`).

Modifications apportées à la source :

- import circulaire `from 'ngx-chess-board'` remplacé par un import relatif ;
- `peerDependencies` mises à jour pour Angular 21 ;
- compilation sans le mode TypeScript strict (code de 2020), l'application reste en mode strict.
