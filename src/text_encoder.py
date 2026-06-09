import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, text: str | list[str]) -> torch.Tensor:
        """
        Returns normalized embedding(s).
        """
        embeddings = self.model.encode(
            text,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=self.device,
        )
        return embeddings
