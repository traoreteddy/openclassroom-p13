"""Embedder : Qwen3-Embedding-0.6B via sentence-transformers.

- **Asymétrique** : les questions sont encodées avec la consigne de recherche du modèle
  (`prompt_name="query"`), les documents sans consigne (recommandation du modèle).
- **Normalisé** : vecteurs de norme 1, la similarité cosinus devient un produit scalaire.
- **Préchauffé en arrière-plan** : l'API lance le chargement (≈ 8 s, 1,2 Go) au démarrage
  sans l'attendre (`start_warm_up`).
- **Jamais bloquant pour l'agent** : une question reçue pendant le chargement obtient
  aussitôt l'erreur `model_loading` (avec un délai de nouvel essai), au lieu d'attendre
  au-delà du timeout de l'outil. Les scripts appellent `warm_up()` et attendent.
- **Calcul dans un fil séparé** (`asyncio.to_thread`) : la boucle d'événements reste libre.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.tools.errors import ToolUnavailableError

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self._device = device
        self._dtype = dtype
        self._batch_size = batch_size
        self._model: Any = None
        self._lock = threading.Lock()
        self._warm_up_task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def warm_up(self) -> None:
        """Charge le modèle, puis encode une phrase : la première vraie requête est rapide."""
        await asyncio.to_thread(self._load)
        await asyncio.to_thread(self._encode, ["préchauffage de l'encodeur"], prompt_name="query")

    def start_warm_up(self) -> asyncio.Task[None]:
        """Lance le chargement en arrière-plan, sans l'attendre (démarrage de l'API)."""
        if self._warm_up_task is None:
            self._warm_up_task = asyncio.get_running_loop().create_task(self.warm_up())
        return self._warm_up_task

    @property
    def dimension(self) -> int:
        return int(self._load().get_embedding_dimension())

    async def embed(self, text: str) -> list[float]:
        """Vecteur d'une question (avec la consigne de recherche).

        Si le modèle n'est pas encore chargé, lance le chargement et répond aussitôt
        `model_loading` : l'agent continue sans attendre.
        """
        if not self.ready:
            task = self.start_warm_up()
            if task.done() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
            if not self.ready:
                raise ToolUnavailableError(
                    code="model_loading",
                    message="Le modèle d'embedding est en cours de chargement.",
                    hint="Réessayer dans une dizaine de secondes ; en attendant, répondre "
                    "sans la base de connaissances.",
                    retry_after_s=10,
                )
        vectors = await asyncio.to_thread(self._encode, [text], prompt_name="query")
        return vectors[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Vecteurs des chunks à indexer (sans consigne)."""
        return await asyncio.to_thread(self._encode, texts, prompt_name=None)

    def _encode(self, texts: list[str], *, prompt_name: str | None) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def _torch_dtype(self) -> Any:
        import torch

        try:
            return getattr(torch, self._dtype)
        except AttributeError as exc:
            raise ValueError(
                f"EMBEDDING_DTYPE inconnu : {self._dtype!r} (auto, float32, bfloat16)"
            ) from exc

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    # « auto » = bfloat16 (défaut du modèle), rapide sur GPU et Mac M mais
                    # très lent sur CPU sans bf16 natif : le compose impose float32.
                    explicit = self._dtype not in ("", "auto")  # vide = variable non renseignée
                    kwargs = {"model_kwargs": {"dtype": self._torch_dtype()}} if explicit else {}
                    self._model = SentenceTransformer(
                        self.model_name, device=self._device, **kwargs
                    )
                except OSError as exc:  # modèle absent du cache et pas de réseau
                    raise ToolUnavailableError(
                        code="embedding_unavailable",
                        message=f"Le modèle d'embedding {self.model_name} est indisponible.",
                        hint="Vérifier le cache du modèle (volume hf_models) ou l'accès réseau.",
                    ) from exc
            return self._model
