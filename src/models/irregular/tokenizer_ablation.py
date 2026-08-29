"""Controlled point-token-point model for tokenizer ablations."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTokenOperator(nn.Module):
    """Shared operator with hard quadtree or learned soft token assignment."""

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 4,
        d_model: int = 56,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_tokens: int = 256,
        tokenizer: str = "hard",
        dropout: float = 0.1,
        temperature: float = 0.5,
    ):
        super().__init__()
        if tokenizer not in {"hard", "learned"}:
            raise ValueError("tokenizer must be 'hard' or 'learned'")
        if max_tokens < 2:
            raise ValueError("max_tokens must be at least 2")

        self.tokenizer = tokenizer
        self.d_model = d_model
        self.max_tokens = max_tokens

        self.point_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.assignment_key = nn.Linear(d_model, d_model, bias=False)
        self.token_embed = nn.Parameter(torch.empty(max_tokens, d_model))
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.position_encoder = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.point_decoder = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.token_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _hard_aggregate(
        features: torch.Tensor,
        positions: torch.Tensor,
        token_ids: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n_points, width = features.shape
        index = token_ids.view(1, n_points, 1).expand(batch_size, -1, width)
        tokens = features.new_zeros(batch_size, num_tokens, width)
        tokens.scatter_add_(1, index, features)

        counts = features.new_zeros(batch_size, num_tokens)
        counts.scatter_add_(
            1,
            token_ids.view(1, n_points).expand(batch_size, -1),
            features.new_ones(batch_size, n_points),
        )
        tokens = tokens / counts.clamp_min(1.0).unsqueeze(-1)

        position_index = token_ids.view(1, n_points, 1).expand(batch_size, -1, 2)
        centers = positions.new_zeros(batch_size, num_tokens, 2)
        centers.scatter_add_(1, position_index, positions)
        centers = centers / counts.clamp_min(1.0).unsqueeze(-1)
        return tokens, centers, counts

    def _learned_aggregate(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.token_embed[:num_tokens]
        keys = self.assignment_key(features)
        temperature = self.temperature.clamp_min(0.05)
        logits = torch.einsum("bnd,pd->bnp", keys, queries)
        logits = logits / (math.sqrt(self.d_model) * temperature)
        assignment = F.softmax(logits, dim=-1)
        mass = assignment.sum(dim=1)
        tokens = torch.einsum("bnp,bnd->bpd", assignment, features)
        tokens = tokens / mass.clamp_min(1e-6).unsqueeze(-1)
        centers = torch.einsum("bnp,bnc->bpc", assignment, positions)
        centers = centers / mass.clamp_min(1e-6).unsqueeze(-1)
        return tokens, centers, mass, assignment

    def forward(
        self,
        positions: torch.Tensor,
        flow: torch.Tensor,
        num_tokens: int,
        token_ids: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ):
        if num_tokens < 2 or num_tokens > self.max_tokens:
            raise ValueError(
                f"num_tokens={num_tokens} outside [2, {self.max_tokens}]"
            )
        features = self.point_encoder(torch.cat([positions, flow], dim=-1))

        assignment = None
        if self.tokenizer == "hard":
            if token_ids is None:
                raise ValueError("hard tokenizer requires token_ids")
            token_ids = token_ids.to(device=features.device, dtype=torch.long)
            if token_ids.numel() != positions.shape[1]:
                raise ValueError("token_ids length must equal the point count")
            if token_ids.min().item() < 0 or token_ids.max().item() >= num_tokens:
                raise ValueError("token_ids contains an inactive token index")
            tokens, centers, mass = self._hard_aggregate(
                features, positions, token_ids, num_tokens
            )
        else:
            tokens, centers, mass, assignment = self._learned_aggregate(
                features, positions, num_tokens
            )

        tokens = (
            tokens
            + self.position_encoder(centers)
            + self.token_embed[:num_tokens].unsqueeze(0)
        )
        encoded = self.backbone(tokens)

        if self.tokenizer == "hard":
            context = encoded[:, token_ids, :]
            entropy = features.new_zeros(())
            confidence = features.new_ones(())
        else:
            assert assignment is not None
            context = torch.einsum("bnp,bpd->bnd", assignment, encoded)
            entropy = -(
                assignment.clamp_min(1e-12).log() * assignment
            ).sum(dim=-1).mean() / math.log(num_tokens)
            confidence = assignment.max(dim=-1).values.mean()

        point_position = self.position_encoder(positions)
        prediction = self.point_decoder(torch.cat([point_position, context], dim=-1))
        if not return_diagnostics:
            return prediction

        diagnostics: dict[str, torch.Tensor] = {
            "assignment_entropy": entropy.detach(),
            "assignment_confidence": confidence.detach(),
            "token_mass_min": mass.min().detach(),
            "token_mass_mean": mass.mean().detach(),
            "token_mass_max": mass.max().detach(),
        }
        return prediction, diagnostics


class PointTokenLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(prediction, target)
