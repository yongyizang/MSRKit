# scoring_program/metrics.py
import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

try:
    import laion_clap  # type: ignore
except Exception:
    laion_clap = None
    logging.warning(
        "The `laion_clap` package is not available. "
        "FAD-CLAP will use random embeddings (diagnostic only)."
    )


class Metric(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.register_buffer("dummy_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_buffer.device

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any): ...

    @abstractmethod
    def compute(self) -> Dict[str, float]: ...

class SI_SNR(Metric):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.reset()

    def reset(self):
        self.register_buffer("sum_scores", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("sum_sq_scores", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0, dtype=torch.int64))

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # pred/target: shape (B, T)
        score = self._compute_si_snr(pred, target).detach()
        self.sum_scores += torch.sum(score)
        self.sum_sq_scores += torch.sum(score.pow(2))
        self.count += score.numel()

    def compute(self) -> Dict[str, float]:
        n = int(self.count.item())
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mean_val = (self.sum_scores / self.count).item()
        var = (self.sum_sq_scores / self.count) - (self.sum_scores / self.count).pow(2)
        std_val = torch.sqrt(var).item() if var > 0 and n > 1 else 0.0
        return {"mean": mean_val, "std": std_val, "count": n}

    def _compute_si_snr(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1, pred.shape[-1])
        target = target.view(-1, target.shape[-1])
        pred_zm = pred - pred.mean(dim=-1, keepdim=True)
        target_zm = target - target.mean(dim=-1, keepdim=True)
        alpha = (pred_zm * target_zm).sum(dim=-1, keepdim=True) / (
            target_zm.pow(2).sum(dim=-1, keepdim=True) + self.eps
        )
        target_scaled = alpha * target_zm
        noise = pred_zm - target_scaled
        si_snr_val = (target_scaled.pow(2).sum(dim=-1)) / (noise.pow(2).sum(dim=-1) + self.eps)
        return 10 * torch.log10(si_snr_val + self.eps)

class FAD_CLAP(Metric):
    def __init__(self, embedding_dim: int = 512, model_name: str = "HTSAT-tiny", ckpt_path: Optional[str] = None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.clap_model = self._load_clap_model(model_name, ckpt_path)
        self.pred_embeddings: List[torch.Tensor] = []
        self.target_embeddings: List[torch.Tensor] = []
        self.reset()

    def _load_clap_model(self, model_name: str, ckpt_path: Optional[str]) -> Optional[nn.Module]:
        if laion_clap is None:
            logging.warning("`laion_clap` not installed. FAD will use random embeddings.")
            return None
        try:
            logging.info(f"Loading CLAP model '{model_name}' for FAD metric...")
            model = laion_clap.CLAP_Module(enable_fusion=False, amodel=model_name)
            model.load_ckpt(ckpt_path)
            model.eval()
            logging.info("CLAP model loaded.")
            return model
        except Exception as e:
            logging.warning(f"Failed to load CLAP model: {e}. FAD will use random embeddings.")
            return None

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def reset(self):
        self.pred_embeddings.clear()
        self.target_embeddings.clear()

    def update(self, pred_audio_48k: torch.Tensor, target_audio_48k: torch.Tensor):
        self.pred_embeddings.append(self._extract_embedding(pred_audio_48k).cpu())
        self.target_embeddings.append(self._extract_embedding(target_audio_48k).cpu())

    def compute(self) -> Dict[str, float]:
        if not self.pred_embeddings or not self.target_embeddings:
            return {"fad": float("inf"), "count": 0}
        pred_emb_all = torch.cat(self.pred_embeddings, dim=0).to(self.device)
        target_emb_all = torch.cat(self.target_embeddings, dim=0).to(self.device)
        if pred_emb_all.shape[0] < 2 or target_emb_all.shape[0] < 2:
            logging.warning(
                f"FAD requires at least 2 samples per set, got {pred_emb_all.shape[0]} and {target_emb_all.shape[0]}."
            )
            return {"fad": float("inf"), "count": int(pred_emb_all.shape[0])}
        mu_pred, sigma_pred = self._get_mu_and_sigma(pred_emb_all)
        mu_target, sigma_target = self._get_mu_and_sigma(target_emb_all)
        fad_score = self._frechet_distance(mu_pred, sigma_pred, mu_target, sigma_target)
        return {"fad": fad_score.item(), "count": int(pred_emb_all.shape[0])}

    @torch.no_grad()
    def _extract_embedding(self, audio_48k: torch.Tensor) -> torch.Tensor:
        if self.clap_model is None:
            B = audio_48k.shape[0]
            return torch.randn(B, self.embedding_dim, device=audio_48k.device)
        self.clap_model.to(audio_48k.device)
#         audio_dict = {"waveform": audio_48k, "sample_rate": 48000}
        return self.clap_model.get_audio_embedding_from_data(x=audio_48k, use_tensor=True)

    def _get_mu_and_sigma(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = embeddings.mean(dim=0)
        sigma = torch.cov(embeddings.T)
        return mu, sigma

    def _frechet_distance(self, mu1, sigma1, mu2, sigma2) -> torch.Tensor:
        diff = mu1 - mu2
        mean_dist_sq = diff.dot(diff)
        try:
            offset = torch.eye(sigma1.shape[0], device=self.device, dtype=sigma1.dtype) * 1e-6
            cov_sqrt = torch.linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset)).real
        except Exception:
            logging.warning("Matrix sqrt failed; using diagonal approximation for FAD.")
            cov_sqrt = torch.sqrt(torch.diag(sigma1) * torch.diag(sigma2))
#         trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(cov_sqrt)
        trace_term = torch.trace(sigma1) + torch.trace(sigma2) - 2 * cov_sqrt.sum()
        return mean_dist_sq + trace_term
