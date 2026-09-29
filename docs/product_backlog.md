## Context
la FFE est l'organisation officielle chargée de promouvoir et développer le jeu d'echec en France.
Elle veut un agent intelligent pour apprendre aux jeunes espoirs des **ouvertures** aux echecs.

## Fonctionnalité:
* leur proposant les meilleurs coups issus de la théorie,
* le contexte des ouvertures via des données enrichies par les parties historiques,
* des vidéos explicatives pertinentes,
* et une évaluation de la position par un moteur spécialisé (exemple Stockfish) si la partie s’écarte des sentiers battus.

## Objectifs:
* Propose une interface Angular avec échiquier interactif (ngx-chessboard) : 
  * https://github.com/OpenClassrooms-Student-Center/material-chessboard

* Guide l’utilisateur grâce à un agent IA connecté à :
  * une identification de la position en cours par identifiant [fen](https://www.chess.com/terms/fen-chess).
  * un [Stockfish](https://pypi.org/project/stockfish) (moteur d’échecs spécialisé),
  * des [APIs Lichess](https://lichess.org/api) (bibliothèque d’ouvertures et parties de références)
  * une intégration des données issues de [Wikichess](https://ficgs.com/wikichess_1.html) dans une base Milvus  pour la recherche vectorielle (toutes sources pertinentes sont acceptées)
  * un affichage de vidéos explicatives YouTube pertinentes (relatives à la position en cours) grâce à [youtube api](https://developers.google.com/youtube/v3?hl=fr). 
  * stocke les vidéos pertinentes,
  * analyse chaque vidéo pour en extraire les frames,
  * sur chaque Frame, détecte la présence d'un échiquier et convertit la position en notation FEN.


## Ressources 
* [classement des llms bons au echec](https://www.kaggle.com/game-arena)
* la position et du jeu d’échecs est transmise aux LLM (échiquier, derniers mouvements, coups possibles...)


## Questionnement:
1. Comprendre les règles du jeu d'échecs