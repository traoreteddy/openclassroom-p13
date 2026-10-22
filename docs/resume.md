Ce projet s'articule autour de la création d'un assistant intelligent pour la Fédération Française des Échecs (FFE), conçu pour aider les jeunes joueurs à maîtriser les ouvertures. La mission est divisée en deux volets distincts : le développement d'un prototype fonctionnel (POC) et la conception théorique d'un système de vision avancé.

**Partie 1 : Le Développement du POC (À coder en 2 semaines)**
L'objectif est de livrer une application complète orchestrant plusieurs outils d'intelligence artificielle et d'analyse d'échecs.

* **Interface Utilisateur :** Une application Angular intégrant un échiquier interactif (`ngx-chessboard`) pour permettre aux utilisateurs de jouer et visualiser les coups.
* **Orchestration de l'Agent IA :** Utilisation de LangGraph et FastAPI pour créer la logique de l'agent. L'agent devra analyser la position actuelle sur l'échiquier (identifiée par la notation FEN).
* **Outils de l'Agent :**
* **Stockfish :** Un moteur d'échecs pour évaluer la position, particulièrement utile si la partie quitte la théorie connue.
* **API Lichess :** Pour récupérer des statistiques sur les ouvertures et des parties de référence.
* **RAG (Retrieval-Augmented Generation) :** Intégration de données issues de Wikichess dans une base de données vectorielle Milvus, avec MongoDB pour la persistance des données.
* **API YouTube :** Pour proposer des vidéos explicatives basées sur la position.


* **Déploiement :** Le système doit être packagé via Docker Compose pour la démonstration au client.

**Partie 2 : L'Étude Stratégique d'un Serveur MCP de Vision (À concevoir uniquement)**
Le responsable, Alan, souligne que la simple recherche textuelle de vidéos YouTube montre vite ses limites, car elle risque de proposer une vidéo de 45 minutes alors que l'utilisateur a juste besoin de l'explication du coup en cours. Il demande de concevoir un système plus avancé.

* **Le Concept :** Le système devra stocker des vidéos pertinentes, extraire les frames de ces vidéos, et utiliser un modèle de vision (type board-to-FEN) pour détecter la présence d'un échiquier sur chaque frame afin de convertir la position en notation FEN.


* **L'Objectif :** Permettre de rechercher une position exacte dans le catalogue de vidéos et de renvoyer le lien avec le timestamp précis de l'explication.


* **L'Intégration :** Ce système d'indexation vidéo sera porté par un serveur MCP (Model Context Protocol) pour s'interfacer facilement avec l'application principale.


* **Les Livrables Attendus :** Vous ne devez pas coder cette partie, mais produire trois documents stratégiques :
1. Une note détaillant les bénéfices attendus et les limites de ce système.


2. Un schéma de l'architecture technique de la solution MCP.


3. Une étude de faisabilité estimant les coûts de mise en place et de fonctionnement.





Souhaitez-vous commencer par définir les différents nœuds et outils de l'agent LangGraph pour le POC, ou préférez-vous attaquer l'ébauche de l'architecture technique du serveur MCP ?