"""Configuration for the OSM directed-edge Emit/Trans training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slhmm.configs.runtime import ARTIFACT_ROOT, CHECKPOINT_ROOT, DATA_ROOT

DATASET_PROFILES = {
    "hangzhou": dict(snapshot="hangzhou_abl_v2", data_dir="osm_filtered_10098", metric_epsg="EPSG:32651", candidate_k=200),
    "xian": dict(snapshot="xian_val112_abl_v2", data_dir="xian_val200", metric_epsg="EPSG:32649", candidate_k=1200),
}


@dataclass
class TrainingConfig:
    """Resolved defaults shared by every training stage."""

    experiment_id: str = "osm_retrain_v1"
    dataset: str = "hangzhou"
    snapshot: str | None = None
    variant: str = "full"
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    num_workers: int = 0
    limit_trajectories: int | None = None

    road_hidden_dim: int = 128
    road_layers: int = 2
    road_dropout: float = 0.1
    highway_embedding_dim: int = 16
    coordinate_frequencies: int = 8
    cell_id_embedding_dim: int = 32
    local_road_k: int = 30
    trajectory_hidden_dim: int = 64
    trajectory_layers: int = 2
    candidate_k: int = 200
    emit_top_k: int = 30
    trans_negative_count: int = 64

    road_pretrain_epochs: int = 20
    emit_epochs: int = 20
    trans_epochs: int = 15
    joint_epochs: int = 10
    batch_candidate_budget: int = 20_000
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0

    road_pretrain_lr: float = 1e-3
    task_road_lr: float = 1e-4
    task_head_lr: float = 1e-3
    joint_road_lr: float = 5e-5
    joint_head_lr: float = 1e-4
    weight_decay: float = 1e-4
    masked_attribute_rate: float = 0.15
    early_stopping_patience: int = 5

    viterbi_trans_weight: float = 1.0
    geometry_tolerance_m: float = 15.0
    minimum_topology_continuity: float = 0.99

    use_road_pretrain: bool = True
    use_trajectory_context: bool = True
    use_h3: bool = True
    emit_mode: str = "decoupled"
    use_reliability_gate: bool = True
    use_path_costs: bool = True
    use_lattice_loss: bool = True
    use_route_cutoff: bool = True
    skip_joint: bool = False
    h3_resolutions: tuple[int, ...] = (7, 8, 9)
    h3_embedding_dim: int = 32
    h3_unk_dropout: tuple[float, ...] = (0.05, 0.10, 0.15)
    eta_turn: float = 30.0
    eta_uturn: float = 300.0
    eta_tunnel: float = 200.0
    route_gamma_cut: float = 4.0
    route_abs_slack_m: float = 3000.0
    d_min_m: float = 200.0
    detour_gamma: float = 1.5
    lambda_len: float = 0.1
    mu_back: float = 2.0
    mu_uturn: float = 2.0
    beta_tunnel: float = 1.0
    rank_levels: int = 7
    cost_clamp: float = 50.0
    lambda_emit: float = 0.5
    lambda_trans: float = 0.5
    lambda_gate: float = 0.1
    p_perturb: float = 0.08
    perturb_ring_m: tuple[float, float] = (1000.0, 3000.0)
    lattice_pair_budget: int = 600_000
    trans_chunk_size: int = 8192
    route_source_batch: int = 8
    route_cache_pairs: int = 300_000

    def __post_init__(self) -> None:
        if self.dataset not in DATASET_PROFILES:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        profile = DATASET_PROFILES[self.dataset]
        self.snapshot = self.snapshot or profile["snapshot"]
        self.candidate_k = profile["candidate_k"]
        if self.variant == "baseline":
            self.use_h3 = self.use_reliability_gate = self.use_path_costs = False
            self.use_lattice_loss = self.use_route_cutoff = False
            self.emit_mode = "concat"
        elif self.variant == "no_road_pretrain":
            self.use_road_pretrain = False
        elif self.variant == "no_context_gate":
            self.use_trajectory_context = self.use_reliability_gate = False
            self.emit_mode = "concat"
        elif self.variant == "no_h3":
            self.use_h3 = False
        elif self.variant == "no_path_costs":
            self.use_path_costs = False
        elif self.variant == "no_lattice":
            self.use_lattice_loss = False
        elif self.variant == "no_joint":
            self.skip_joint = True
        elif self.variant != "full":
            raise ValueError(f"Unknown variant: {self.variant}")
        if not self.experiment_id or Path(self.experiment_id).name != self.experiment_id or self.experiment_id in {".", ".."}:
            raise ValueError("experiment_id must be a single directory name")

    @property
    def graph_path(self) -> Path:
        return DATA_ROOT / DATASET_PROFILES[self.dataset]["data_dir"] / "graph" / "graph.graphml"

    @property
    def training_data_root(self) -> Path:
        return DATA_ROOT / DATASET_PROFILES[self.dataset]["data_dir"] / "training"

    @property
    def metric_epsg(self) -> str:
        return DATASET_PROFILES[self.dataset]["metric_epsg"]

    @property
    def feature_root(self) -> Path:
        return ARTIFACT_ROOT / "features" / self.snapshot

    @property
    def run_root(self) -> Path:
        return ARTIFACT_ROOT / "runs" / self.experiment_id

    @property
    def checkpoint_root(self) -> Path:
        return CHECKPOINT_ROOT / self.experiment_id

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "graph_path": str(self.graph_path),
                "training_data_root": str(self.training_data_root),
                "feature_root": str(self.feature_root),
                "run_root": str(self.run_root),
                "checkpoint_root": str(self.checkpoint_root),
            }
        )
        return value

