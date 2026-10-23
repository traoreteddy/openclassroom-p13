# Note de cadrage : retrouver la position exacte dans les vidéos d'ouvertures

**Système d'analyse vidéo par vision et serveur MCP « Vision » : bénéfices, limites,
architecture, faisabilité et coûts**

| | |
|---|---|
| Destinataire | Responsable produit FFE, direction technique |
| Auteur | Équipe agent « coach d'ouvertures » |
| Date | 23 octobre 2026 |
| Statut | Note de décision, système **non implémenté** |
| Décision demandée | Lancer ou non une preuve de faisabilité de 3 semaines (§ 9) |

---

## Résumé pour décision

Aujourd'hui, l'agent propose des vidéos par recherche textuelle sur le nom de l'ouverture.
Un jeune joueur qui veut comprendre le coup qu'il vient de jouer reçoit une vidéo de
20 minutes et doit la parcourir. Le système étudié ici indexe les vidéos **par position** :
il retrouve la seconde exacte où l'échiquier affiché montre la position courante et renvoie
un lien horodaté.

Ce que nous avons établi :

- **Techniquement faisable** avec des briques connues : les vidéos pédagogiques montrent
  des échiquiers 2D à l'écran, un cas plus simple que les plateaux filmés traités par la
  littérature (92 à 99 % de précision par case). Les coûts machine sont faibles : quelques
  dizaines d'euros pour indexer 500 vidéos, 30 à 80 € par mois pour faire tourner le service.
- **Le vrai coût est humain** : 45 à 75 jours de travail sur quatre phases, dont la
  constitution d'un jeu de test annoté et la négociation des droits.
- **Le risque principal est juridique, pas technique** : les conditions d'utilisation de
  YouTube interdisent de télécharger les vidéos. Le catalogue doit se limiter aux vidéos de
  la FFE, de partenaires qui fournissent leurs fichiers, et aux vidéos sous licence Creative
  Commons dont le fichier est obtenu auprès du créateur. Cela borne la couverture à quelques
  centaines, puis quelques milliers de vidéos.
- **Deux alternatives moins chères** existent (transcription audio ; chapitres et PGN
  fournis par les créateurs) et peuvent être combinées avec la vision.

**Recommandation** : lancer une preuve de faisabilité de 15 jours sur 20 vidéos annotées à
la main, avec un critère d'arrêt chiffré (90 % de positions exactes), avant tout engagement
sur le MVP. Ne pas investir dans l'industrialisation sans au moins 300 vidéos sous droits.

---

## 1. Contexte et problème

L'agent livré (FastAPI, LangGraph, Milvus, Stockfish, Lichess) appelle l'outil
`find_opening_videos` qui interroge l'API YouTube Data v3 avec le nom de l'ouverture
(« partie italienne échecs ouverture explication »), filtre les résultats (durée, vues,
langue) et renvoie des liens. La mesure sur 10 ouvertures du corpus donne 10 ouvertures
sur 10 avec des vidéos, 86 % des titres nommant l'ouverture, 100 % en français.

La limite est structurelle : la recherche texte travaille au niveau de **l'ouverture**, pas
de la **position**. Trois conséquences pour un jeune joueur :

| Situation | Ce qu'il reçoit | Ce qu'il voudrait |
|---|---|---|
| Il vient de jouer 4.Cg5 dans la défense des deux cavaliers | une vidéo de 25 minutes sur l'Italienne | l'instant où le présentateur montre 4.Cg5 |
| Sa position sort de la théorie (plus de nom d'ouverture) | rien, ou du hors-sujet | une vidéo où cette position, ou une très proche, apparaît |
| Il veut comparer deux plans dans une même variante | la même vidéo deux fois | deux horodatages différents |

Le backlog de la FFE décrit la réponse souhaitée : stocker les vidéos pertinentes, analyser
chaque vidéo pour en extraire les images, détecter la présence d'un échiquier sur chaque
image et convertir la position en notation FEN. La présente note instruit cette demande
sans l'implémenter.

## 2. Solution proposée et parcours utilisateur

### 2.1 Principe

Un pipeline hors ligne transforme chaque vidéo autorisée en une liste de **segments** :
« de la seconde 754 à la seconde 801, l'écran montre la position P ». Ces segments sont
rangés dans un index interrogeable par position. En ligne, l'agent demande à un serveur
**MCP « Vision »** les segments qui correspondent à la position courante et obtient des liens
du type `youtube.com/watch?v=XXX&t=754s`, affichés dans la carte « Vidéos pour réviser »
avec le lecteur intégré existant.

### 2.2 Parcours

1. Le joueur joue 1.e4 e5 2.Cf3 Cc6 3.Fc4 Cf6 4.Cg5 sur l'échiquier.
2. L'agent identifie la position (FEN) et l'ouverture (Lichess), comme aujourd'hui.
3. Il appelle l'outil MCP `find_video_by_fen` avec le **placement des pièces** (premier
   champ de la FEN).
4. Le serveur renvoie, par exemple : « Défense des deux cavaliers, partie pédagogique 2 »,
   de 12 min 34 s à 13 min 21 s, confiance 0,97 ; et deux autres segments.
5. La carte Vidéos affiche ces segments en tête, avec le lecteur démarrant à 12 min 34 s ; la
   recherche texte actuelle reste en repli si l'index ne connaît pas la position.

### 2.3 Ce qu'une image ne dit pas

Une image montre les pièces, pas le trait, les droits de roque ni la prise en passant.
L'index porte donc uniquement sur le placement des pièces (64 cases), et deux positions qui
ne diffèrent que par le trait sont confondues. C'est acceptable pour l'usage pédagogique :
le commentateur montre la position et en parle, quel que soit le trait. Il faut aussi
reconnaître l'échiquier **retourné** (vue côté Noirs), fréquent dans les vidéos sur les
défenses : chaque placement est indexé sous ses deux orientations.

## 3. État de l'art : détecter un échiquier et lire la position

### 3.1 Deux familles de problèmes

| Famille | Exemples | Difficulté | Notre cas |
|---|---|---|---|
| **Plateau physique filmé** (photos, caméra de tournoi) | LiveChess2FEN, chesscog, CVChess, ChessReD | perspective, éclairage, occultations, pièces 3D | rare dans les vidéos pédagogiques |
| **Échiquier 2D affiché à l'écran** (Lichess, Chess.com, logiciels d'analyse) | jeux de données d'échiquiers rendus (100 000 images, 28 styles de plateau, 32 styles de pièces) | grille parfaitement régulière ; variété des thèmes ; flèches, surlignages, curseur par-dessus | **cas majoritaire** |

Les vidéos d'ouvertures en français sont presque toutes des captures d'écran commentées :
le problème à résoudre est le second, nettement plus simple.

### 3.2 Résultats publiés

| Travail | Approche | Résultat rapporté | Enseignement |
|---|---|---|---|
| **LiveChess2FEN** (2020, code ouvert) | détection du plateau, puis classification de chaque case par réseau convolutif ; tourne sur Jetson Nano | 95 % de détection du plateau, 92 % de classification des pièces ; 96 % « en conditions optimales » dans une réimplémentation | une case à la fois, en moins d'une seconde par position |
| **chesscog** (2021) | localisation du plateau par géométrie, occupation puis classe de pièce par réseaux | précision élevée sur son jeu de données ; dégradation quand l'angle de vue s'éloigne de la vue de dessus | les méthodes sont évaluées en vue de dessus : notre cas 2D est une vue de dessus parfaite |
| **CVChess** (2025) | transformée de Hough, redressement, 64 cases, réseau résiduel à 13 classes | 98,9 % par case sur le domaine d'entraînement, moins bien hors domaine | le **décalage de domaine** (thèmes inconnus) est le risque à mesurer |
| Projets universitaires 2024-2025 | rendus 3D synthétiques, puis photos réelles | environ 1,7 erreur par échiquier sur le synthétique ; performance insuffisante sur photos réelles | ne pas s'entraîner seulement sur du synthétique |
| Jeux de données d'échiquiers **2D rendus** | 100 000 images 400 × 400 avec FEN, 28 plateaux × 32 jeux de pièces | des classifieurs simples dépassent 99 % par case sur ce domaine | bon point de départ pour notre cas, à compléter par des captures réelles |

### 3.3 Les briques que nous retiendrions

| Étape | Technique | Pourquoi |
|---|---|---|
| Échantillonnage des images | **ffmpeg** avec détection de changement de scène (`select='gt(scene,0.3)'`) ou PySceneDetect, plutôt qu'une image toutes les 2 secondes | une vidéo de 20 minutes contient 60 à 150 positions distinctes, pas 600 : on divise le travail par 5 à 10 |
| Détection de l'échiquier | d'abord **OpenCV** (détection de lignes de Hough, recherche d'une grille 8 × 8 régulière, homographie) ; un petit détecteur appris (type YOLO nano) si les thèmes « sans bordure » mettent la géométrie en échec | la grille 2D est régulière et axée : la géométrie classique suffit souvent, et elle ne coûte rien |
| Lecture de la position | découpage en 64 cases, **classifieur à 13 classes** (6 pièces × 2 couleurs + vide), réseau convolutif léger entraîné sur échiquiers rendus **et** sur des captures réelles annotées (thèmes Lichess, Chess.com, Chessbase, flèches, surlignages) | c'est l'approche de LiveChess2FEN et CVChess ; sur du 2D elle dépasse 99 % par case en domaine |
| Validation | **python-chess** : rejet des placements illégaux (deux rois, pions sur la première rangée, plus de 16 pièces par camp) ; cohérence temporelle : une position doit tenir au moins deux images successives | élimine la majorité des erreurs isolées sans modèle supplémentaire |
| Orientation | tester le placement et son miroir ; retenir celui qui est légal, ou les deux | gère l'échiquier vu côté Noirs |

Ordre de grandeur de calcul : 30 à 50 ms par image sur un GPU d'entrée de gamme (L4),
10 à 20 fois plus sur CPU. Le décodage vidéo par ffmpeg pèse autant que la vision.

## 4. Architecture technique avec MCP

### 4.1 Pourquoi MCP

Le Model Context Protocol expose des **outils** et des **ressources** à n'importe quel
agent, dans un contrat standard (schéma JSON des paramètres, description lisible par le
modèle). Notre agent LangGraph consomme déjà sept outils par un registre interne ; le
serveur Vision devient une **source d'outils supplémentaire** branchée par un client MCP,
sans toucher au graphe. Le même serveur pourra servir d'autres usages FFE (site, bot,
application mobile) et être remplacé ou déployé indépendamment.

Nous retiendrions **FastMCP** (version 3.x) en Python : un décorateur `@mcp.tool` sur une
fonction typée suffit à publier un outil, les annotations génèrent le schéma, les fonctions
asynchrones sont supportées, et le transport **Streamable HTTP** est recommandé pour un
déploiement serveur (STDIO sert aux outils lancés localement ; SSE est déprécié).

### 4.2 Schéma d'architecture

```
┌───────────────────────── Application existante ─────────────────────────┐
│  Angular (échiquier, panneau)  ──►  FastAPI  ──►  Agent LangGraph        │
│                                                     │  registre d'outils │
│                                                     │  (7 outils) +      │
│                                                     │  client MCP        │
└─────────────────────────────────────────────────────┼────────────────────┘
                                                      │ Streamable HTTP (/mcp)
                                                      ▼
┌────────────────────── Serveur MCP « Vision » (FastMCP) ──────────────────┐
│  outils : find_video_by_fen(placement, limit)   → segments + liens t=    │
│           list_segments(video_id)               → positions d'une vidéo │
│           index_video(url, source, licence)     → ticket (administration)│
│  ressource : catalog://videos (vidéos indexées, licences, dates)         │
└──────────────┬──────────────────────────────────────┬────────────────────┘
               │ lecture                              │ écriture (ticket)
               ▼                                      ▼
   ┌──────────────────────┐              ┌─────────────────────────────┐
   │ Index des positions  │              │ File de tâches (Redis / RQ) │
   │ MongoDB (ou Postgres)│              └──────────────┬──────────────┘
   │ {placement, video_id,│                             │
   │  t_début, t_fin,     │                             ▼
   │  confiance, vignette}│   ┌──────────────── Workers d'indexation ───────────────┐
   │ index sur placement  │   │ 1. récupération du fichier (FFE, partenaire, CC)     │
   └──────────▲───────────┘   │    → stockage objet temporaire, supprimé après       │
              │               │ 2. ffmpeg : images aux changements de scène          │
              │               │ 3. détecteur d'échiquier (OpenCV, puis YOLO si besoin)│
              │               │ 4. board-to-FEN : 64 cases, classifieur 13 classes   │
              │               │ 5. validation python-chess, cohérence temporelle     │
              │               │ 6. regroupement en segments, vignettes → stockage    │
              └───────────────┤ 7. écriture dans l'index                              │
                              └──────────────────────────────────────────────────────┘
                                   GPU à la demande (lot), pas de GPU permanent
```

### 4.3 Rôle de chaque brique

| Brique | Rôle | Choix et justification |
|---|---|---|
| Client MCP dans l'agent | découvre les outils du serveur, les présente au modèle comme les autres | adaptateur MCP → `ToolSpec` existant : même politique de délai, de reprise et de disjoncteur que les sept outils ; repli sur la recherche texte si le serveur ne répond pas |
| Serveur MCP Vision | expose la recherche par position ; seul point d'entrée en ligne | FastMCP, HTTP, authentification par jeton pour `index_video` ; sans état, déployable en conteneur dans le `docker compose` actuel |
| Index des positions | répondre en quelques millisecondes à « quelles vidéos montrent ce placement ? » | MongoDB déjà présent dans la pile ; clé = placement normalisé ; 160 000 segments pèsent moins de 100 Mo |
| File de tâches et workers | indexer en lot, reprendre après panne, limiter le GPU aux créneaux utiles | Redis + RQ (simple) ou Celery ; les workers sont des conteneurs éphémères avec GPU |
| Stockage objet | fichiers vidéo le temps du traitement, vignettes des segments de façon durable | compatible S3 (Scaleway, OVHcloud) ; les vidéos sont **supprimées** après indexation, pour le droit et le coût |
| Modules de vision | détection, lecture, validation | chaque module a une interface stable (image → coordonnées ; vignette de case → classe) et se remplace sans toucher au serveur ni à l'agent |

### 4.4 Modularité et évolutivité

- Changer de modèle de vision, c'est remplacer un worker. L'agent ne voit que des segments.
- Monter en charge, c'est ajouter des workers pendant la nuit d'indexation ; le service en
  ligne ne fait que des lectures d'index.
- Ajouter la transcription audio (§ 8) est un worker de plus qui enrichit les segments
  (« ici le présentateur dit : le cavalier attaque f7 »).
- Le serveur MCP peut porter demain d'autres outils (« positions similaires », « vidéos
  par thème ») sans changer de protocole.

## 5. Bénéfices attendus

| Bénéfice | Pour qui | Comment le mesurer |
|---|---|---|
| **Vidéo calée sur le coup** : plus de parcours de 20 minutes | le jeune joueur | taux de clics sur les segments, temps avant lecture, enquête de satisfaction en club |
| **Fonctionne hors théorie** : si une vidéo a montré la position, l'index la connaît, nom d'ouverture ou pas | le joueur sorti des sentiers battus, cas où la recherche texte échoue | part des positions hors théorie avec au moins un segment |
| **Un actif propre à la FFE** : un index de positions sur un catalogue maîtrisé, réutilisable | la fédération | nombre de vidéos et de segments indexés, nombre d'applications consommatrices |
| **Qualité pédagogique mesurable** : chaque segment a une confiance et peut être validé par un entraîneur | la direction pédagogique | taux de segments validés, taux de signalements |
| **Indépendance vis-à-vis de la recherche YouTube** : moins de quota, résultats stables | l'équipe technique | unités d'API consommées par jour (aujourd'hui 201 par ouverture non cachée) |

Hypothèse de valeur à vérifier en preuve de faisabilité : pour une position de la théorie
courante (10 premiers coups des 12 ouvertures du corpus), **au moins un segment est trouvé
dans 80 % des cas** avec un catalogue de 300 vidéos. En dessous, l'intérêt par rapport à la
recherche texte devient marginal.

## 6. Limites et risques

### 6.1 Risques juridiques et de conformité (bloquants)

| Risque | Analyse | Parade |
|---|---|---|
| **Téléchargement interdit** par les conditions d'utilisation de YouTube (accès au contenu uniquement par le lecteur, pas de téléchargement sans autorisation écrite de YouTube et des ayants droit) | s'applique à toute vidéo YouTube, y compris celles sous licence Creative Commons : la licence CC autorise la **réutilisation** du contenu, pas l'**accès au fichier** par un outil tiers | ne traiter que des fichiers **remis** par leurs auteurs : vidéos FFE, partenaires sous accord écrit, créateurs de vidéos CC sollicités pour fournir le fichier (la licence rend l'accord simple) |
| Droit d'auteur sur les images extraites | les vignettes de segments sont des extraits ; l'index lui-même (positions et horodatages) n'est pas protégé | ne conserver que des vignettes réduites, mention de la source et de la licence, suppression sur demande |
| Données personnelles | aucune donnée de joueur n'entre dans le pipeline ; les sessions de l'agent restent celles d'aujourd'hui | registre des traitements inchangé |
| Dépendance à la plateforme | les liens `t=` et le lecteur intégré dépendent de YouTube | les segments stockent aussi l'identifiant de la vidéo source chez le partenaire : un hébergement FFE reste possible |

### 6.2 Risques techniques

| Risque | Probabilité | Impact | Parade |
|---|---|---|---|
| Décalage de domaine : thèmes de pièces inconnus, flèches, surlignages, curseur | élevée | positions fausses ou manquées | jeu d'entraînement mêlant rendus synthétiques (28 plateaux × 32 jeux de pièces) et 1 000 à 2 000 captures réelles annotées ; validation de légalité ; seuil de confiance ; cohérence temporelle |
| Échiquiers partiels, animés ou très petits (incrustation en coin) | moyenne | segments manqués | détecteur appris en complément de la géométrie ; accepter de manquer les incrustations |
| Plateaux physiques filmés | faible dans notre corpus | hors périmètre | ne pas traiter ; signaler la vidéo comme non indexable |
| Positions ambiguës (trait, roques) | certaine | deux positions confondues | assumé : indexation du placement seul (§ 2.3) |
| Dérive des outils externes (format des liens, lecteur) | faible | liens cassés | tests de bout en bout périodiques |

### 6.3 Risques business

| Risque | Analyse | Parade |
|---|---|---|
| **Couverture insuffisante** : sans droits, le catalogue reste petit et l'index ne connaît pas la position du joueur | c'est le risque qui décide de la valeur | fixer un seuil (300 vidéos sous droits) avant d'industrialiser ; repli automatique sur la recherche texte |
| Investissement pour un gain marginal | si les créateurs partenaires fournissent déjà un PGN horodaté, la vision est superflue | évaluer l'alternative 2 (§ 8) avec les premiers partenaires |
| Coût humain sous-estimé | l'annotation et les accords prennent plus de temps que le code | provisionner 20 % de marge ; jalons avec critères d'arrêt |
| Relations avec les créateurs | indexer sans accord abîmerait la relation | accord écrit, attribution visible, possibilité de retrait |

## 7. Faisabilité et coûts

### 7.1 Hypothèses

| Paramètre | Hypothèse basse | Hypothèse haute |
|---|---|---|
| Catalogue | 500 vidéos | 2 000 vidéos |
| Durée moyenne | 20 min | 25 min |
| Images analysées (changements de scène) | 100 par vidéo → 50 000 | 150 par vidéo → 300 000 |
| Temps de vision par image (GPU L4) | 50 ms | 50 ms |
| Décodage ffmpeg | 0,5 × durée réelle sur 4 cœurs | idem |
| Nouvelles vidéos par mois en régime de croisière | 20 | 100 |
| Prix GPU L4 à la demande | 0,9 €/h (Scaleway, OVHcloud) ; 0,4 à 0,5 €/h chez des fournisseurs spécialisés | idem |
| VM 2 vCPU / 8 Go | 30 à 40 €/mois | idem |
| Stockage objet | 0,015 €/Go/mois ; sortie 0,01 €/Go au-delà de 75 Go | idem |

Les prix sont ceux relevés en octobre 2026 chez des fournisseurs européens ; ils varient
de 10 à 20 % selon la région et le mode de facturation.

### 7.2 Coûts de construction (build)

**Machine, ponctuel**

| Poste | Basse | Haute | Calcul |
|---|---|---|---|
| Vision GPU, indexation initiale | 3 € | 15 € | 50 000 × 50 ms = 0,7 h ; 300 000 × 50 ms = 4,2 h ; × 0,9 €/h, arrondi avec les relances |
| Décodage et détection de scènes (CPU) | 10 € | 60 € | 500 × 20 min × 0,5 = 83 h de CPU 4 cœurs ; 2 000 × 25 min × 0,5 = 417 h ; VM à 0,1 à 0,15 €/h |
| Stockage temporaire des fichiers | 3 € | 15 € | 500 × 300 Mo = 150 Go ; 2 000 × 400 Mo = 800 Go ; quelques jours seulement |
| Entraînement et évaluation du classifieur | 20 € | 50 € | 20 à 50 h de GPU, plusieurs itérations |
| **Total machine** | **≈ 40 €** | **≈ 140 €** | négligeable devant l'humain |

**Humain, en jours de travail** (un développeur confirmé, appui ponctuel d'un entraîneur
pour l'annotation et la validation)

| Phase | Contenu | Basse | Haute |
|---|---|---|---|
| 0. Cadrage et droits | recensement des vidéos FFE, premiers accords partenaires, modèle d'accord, sollicitation des créateurs CC | 5 | 10 |
| 1. Preuve de faisabilité | 20 vidéos, 2 000 images annotées, entraînement du classifieur, mesure de précision, décision | 10 | 15 |
| 2. MVP | pipeline complet, serveur MCP (3 outils, 1 ressource), client MCP dans l'agent, 200 vidéos indexées, carte Vidéos adaptée | 15 | 25 |
| 3. Industrialisation | file de tâches, workers GPU à la demande, reprise sur panne, supervision, validation par les entraîneurs, retrait sur demande | 10 | 15 |
| 4. Enrichissement | transcription audio des segments (§ 8), recherche de positions proches | 5 | 10 |
| **Total** | | **45 jours** | **75 jours** |

En coût complet, à 500 à 700 € par jour (prestation) ou 350 à 450 € (interne), le build
représente **16 000 à 52 000 €**. C'est là que se décide le projet, pas sur la machine.

### 7.3 Coûts de fonctionnement (opex, par mois)

| Poste | Basse | Haute | Remarque |
|---|---|---|---|
| Serveur MCP + index | 0 à 40 € | 40 € | 0 € si hébergé sur la VM backend existante ; 1 VM 2 vCPU / 8 Go sinon |
| Indexation des nouvelles vidéos (GPU + CPU à la demande) | 2 € | 15 € | 20 vidéos × 5 min de GPU ; 100 vidéos × 5 min |
| Stockage durable (vignettes, index) | 1 € | 5 € | 160 000 vignettes × 20 Ko = 3,2 Go ; index < 1 Go |
| File de tâches (Redis) | 0 € | 10 € | sur la même VM, ou service géré |
| API YouTube | 0 € | 0 € | métadonnées seulement, dans le quota gratuit |
| Supervision, sauvegardes | 5 € | 20 € | |
| Maintenance humaine | 1 jour | 3 jours | reprise des vidéos en échec, mises à jour, validations |
| **Total machine** | **≈ 10 à 50 €** | **≈ 90 €** | |

À comparer : un GPU réservé en permanence coûterait 650 €/mois pour rien. Le choix du
« GPU à la demande, en lot » est ce qui rend l'opex marginal.

### 7.4 Verdict de faisabilité

| Critère | Verdict |
|---|---|
| Technique | **faisable** avec des briques éprouvées ; incertitude sur la précision hors domaine, à lever en phase 1 |
| Coût machine | **négligeable** (dizaines d'euros en build, dizaines d'euros par mois) |
| Coût humain | **significatif** (45 à 75 jours), concentré sur l'annotation, l'intégration et l'industrialisation |
| Juridique | **conditionne tout** : sans fichiers remis par les ayants droit, pas de projet |
| Valeur | **à prouver** : dépend de la taille du catalogue ; seuil proposé de 300 vidéos sous droits |

## 8. Alternatives

### 8.1 Transcription audio et reconstruction des coups

Transcrire chaque vidéo (API de transcription à environ 0,006 $ la minute, soit 60 $ pour
500 vidéos de 20 min, 300 $ pour l'hypothèse haute ; ou un modèle Whisper ouvert sur GPU),
puis demander à un modèle de langage de repérer les coups annoncés (« cavalier f3 »,
« on prend en d5 ») et de rejouer la partie avec python-chess pour obtenir des positions
horodatées.

| Pour | Contre |
|---|---|
| pas de vision, pas de GPU dédié, s'appuie sur ce que nous savons faire (LLM, python-chess) | dépend de la qualité du commentaire : coups non annoncés, « on joue ici », variantes sautées |
| donne en plus le **contexte** parlé, que l'image n'a pas | les horodatages sont ceux de la parole, moins précis que l'image |
| coût machine très bas | les mêmes contraintes de droits sur le fichier audio |

Verdict : excellente **complément** de la vision (elle enrichit les segments de ce qui est
dit), **insuffisante seule** sur les vidéos où le présentateur ne dicte pas les coups.

### 8.2 Chapitres YouTube, descriptions et PGN fournis par les créateurs

Beaucoup de créateurs mettent dans la description la liste des coups, un PGN ou des
chapitres horodatés (« 03:12 variante 4.Cg5 »). L'API YouTube donne descriptions et
chapitres **sans télécharger la vidéo**. Un partenariat demande au créateur un PGN
horodaté (export d'un logiciel d'analyse, quelques minutes de travail par vidéo).

| Pour | Contre |
|---|---|
| **aucun risque juridique**, aucun GPU, quelques jours de développement | couverture limitée aux vidéos bien documentées ou aux partenaires |
| précision parfaite quand le PGN existe | demande un effort aux créateurs ; variable |
| peut démarrer en deux semaines dans l'outil existant | n'adresse pas le stock de vidéos existantes non documentées |

Verdict : **à faire d'abord**, quoi qu'il arrive. C'est le chemin le plus court vers des
liens horodatés, et il mesure l'appétence des créateurs. La vision ne se justifie que pour
le stock de vidéos non documentées.

### 8.3 Comparaison

| Critère | Vision (cette note) | Transcription audio | Chapitres et PGN |
|---|---|---|---|
| Précision des horodatages | image près (± 2 s) | phrase près (± 10 s) | parfaite si fournie |
| Couverture possible | toute vidéo avec échiquier à l'écran **et** fichier disponible | toute vidéo commentée **et** fichier disponible | vidéos documentées ou partenaires |
| Risque juridique | fort sans accord | fort sans accord | nul |
| Build | 45 à 75 jours | 15 à 25 jours | 5 à 10 jours |
| Opex | 10 à 90 €/mois | 10 à 60 €/mois | ≈ 0 |
| Dépendance à un modèle de vision | oui | non | non |

## 9. Étapes de développement et recommandation

```
  Phase 0            Phase 1                 Phase 2              Phase 3           Phase 4
  Droits et PGN      Preuve de faisabilité   MVP vision + MCP     Industrialisation  Enrichissement
  5 à 10 j           10 à 15 j               15 à 25 j            10 à 15 j          5 à 10 j
  ───────────────►   ───────────────────►    ────────────────►    ──────────────►    ───────────►
  ▲ critère : 50     ▲ critère : 90 % de     ▲ critère : 80 %     ▲ critère : 95 %   ▲ transcription
    vidéos sous        positions exactes       des positions de     de vidéos          audio sur les
    droits, 20 PGN     sur 500 images          théorie courante     indexées sans      segments
    horodatés          réelles, < 2 % de       ont un segment       intervention
                       faux positifs           (catalogue 200)
```

| Phase | Livrables | Critère d'arrêt (go / no go) |
|---|---|---|
| **0. Droits et chemin court** | modèle d'accord, 50 vidéos sous droits (FFE, partenaires, CC sollicités), **alternative 8.2 livrée** dans l'outil existant (liens horodatés depuis les descriptions et PGN) | moins de 50 vidéos obtenues → arrêt : la vision n'aura pas de matière |
| **1. Preuve de faisabilité** | 20 vidéos, 2 000 images annotées par un entraîneur, classifieur entraîné (rendus + réel), mesure : exactitude par case, par échiquier, taux de faux positifs après validation | **≥ 90 % d'échiquiers exacts et < 2 % de faux positifs** ; sinon, arrêt ou bascule sur 8.1 |
| **2. MVP** | pipeline hors ligne, serveur FastMCP (3 outils, 1 ressource), client MCP dans l'agent avec repli texte, 200 vidéos indexées, carte Vidéos avec horodatage | ≥ 80 % des positions de la théorie courante ont un segment ; retours de 2 clubs |
| **3. Industrialisation** | file de tâches, GPU à la demande, supervision, validation et retrait, documentation d'exploitation | 95 % des nouvelles vidéos indexées sans intervention ; opex mesuré ≤ 100 €/mois |
| **4. Enrichissement** | transcription des segments (8.1), positions proches, statistiques d'usage | gain mesuré sur le temps avant lecture |

**Recommandation.** Faire la phase 0 immédiatement, parce qu'elle livre de la valeur sans
risque (liens horodatés depuis les PGN et chapitres) et parce qu'elle répond à la seule
question qui compte : la FFE obtiendra-t-elle des fichiers vidéo sous droits ? Lancer la
phase 1 dès 50 vidéos obtenues, avec son critère chiffré. Ne décider du MVP qu'avec le
résultat de la phase 1 en main. Dans tous les cas, garder la recherche texte actuelle en
repli : elle fonctionne, elle est mesurée, et elle couvre les cas que l'index ignorera.

---

## Annexe A. Ce que le système existant apporte déjà

| Brique livrée | Réutilisation dans le projet vision |
|---|---|
| Registre d'outils avec contrat, délai, reprise, disjoncteur | le client MCP s'y branche comme un adaptateur : mêmes garanties |
| Outil `find_opening_videos` (requêtes combinées, filtre qualité, cache, quota) | repli et source de candidates ; les métadonnées (titre, chaîne, licence) alimentent le catalogue |
| MongoDB, stockage des caches, compteur de quota | héberge l'index des positions |
| Carte « Vidéos pour réviser » avec lecteur intégré | affiche les segments avec `start` au bon horodatage |
| Identification de la position et de l'ouverture (FEN, Lichess) | fournit le placement à chercher |
| Journal de décision et traces OpenTelemetry | mesurent l'usage des segments et le taux de repli |

## Annexe B. Sources consultées

- LiveChess2FEN : article (arXiv 2012.06858) et page NVIDIA Jetson (95 % de détection,
  92 % de classification) ; réimplémentation universitaire (96 % en conditions optimales).
- chesscog (Wölflein et Arandjelović, 2021), cité par VICE : View-Invariant Chess
  Estimation (2023) pour la dégradation hors vue de dessus.
- CVChess : A Deep Learning Framework for Converting Chessboard Images to FEN (arXiv
  2511.11522) : 98,93 % par case en domaine, moins hors domaine ; jeu de données ChessReD.
- Projets CS231N (Stanford, 2024-2025) : rendus synthétiques et photos réelles.
- Jeu de données d'échiquiers 2D rendus : 100 000 images, 28 plateaux, 32 jeux de pièces.
- FastMCP : documentation officielle (gofastmcp.com), outils, transports (Streamable HTTP
  recommandé, SSE déprécié), version 3.2.x.
- YouTube Data API v3 : paramètre `videoLicense=creativeCommon` du point d'entrée
  `search.list` ; conditions d'utilisation YouTube (accès au contenu par le lecteur,
  téléchargement interdit sans autorisation écrite).
- Prix relevés en octobre 2026 : GPU L4 à la demande 0,88 à 1,00 €/h (Scaleway, OVHcloud),
  0,39 à 0,49 $/h (RunPod) ; VM OVHcloud b3-8 (2 vCPU, 8 Go) 28 à 44 €/mois selon la
  source ; stockage objet Scaleway 0,0146 €/Go/mois, sortie 0,01 €/Go au-delà de 75 Go ;
  transcription OpenAI gpt-4o-transcribe 0,006 $/min, gpt-4o-mini-transcribe 0,003 $/min.
