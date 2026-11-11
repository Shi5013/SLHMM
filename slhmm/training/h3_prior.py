"""Training-only hierarchical H3 vocabulary and learned spatial prior."""
from __future__ import annotations

import h3
import torch
from torch import nn
from slhmm.training.common import load_pickle, write_json, atomic_torch_save, sha256_file


def prepare_h3_artifacts(config):
    root = config.training_data_root
    mappings = load_pickle(root / "mappings.obj")
    alignments = {s: load_pickle(root / f"alignment_{s}.obj") for s in ("train", "val")}
    cells = {s: sorted({int(c) for a in records.values() for c in a["cell_ids"]}) for s, records in alignments.items()}
    finest = max(config.h3_resolutions)
    fine = [h3.latlng_to_cell(float(p[1]), float(p[0]), finest) for p in mappings["cell_id_to_position"]]
    # Derive coarse logical parents explicitly: direct geocoding at adjacent
    # resolutions need not yield nested cells near hexagon boundaries.
    grids = [[h3.cell_to_parent(f, res) for f in fine] for res in config.h3_resolutions]
    vocabularies = [{g: i + 1 for i, g in enumerate(sorted({grid[c] for c in cells["train"]}))} for grid in grids]
    ids = torch.tensor([[vocab.get(grid[c], 0) for vocab, grid in zip(vocabularies, grids)] for c in range(len(fine))])
    metadata = {
        "schema_version": 2, "resolutions": config.h3_resolutions,
        "construction": "geocode finest resolution, derive logical parents",
        "fit_split": "train", "vocab_sizes": [len(v) for v in vocabularies],
        "vocabularies": vocabularies,
        "alignment_train_sha256": sha256_file(root / "alignment_train.obj"),
        "mappings_sha256": sha256_file(root / "mappings.obj"),
        "val_distinct_cell_oov": (ids[cells["val"]] == 0).float().mean(0).tolist(),
        "val_observation_oov": (ids[[c for a in alignments["val"].values() for c in a["cell_ids"]]] == 0).float().mean(0).tolist(),
    }
    dest = config.feature_root / "h3"
    atomic_torch_save(dest / "cell_h3_ids.pt", ids)
    write_json(dest / "metadata.json", metadata)
    return {k: v for k, v in metadata.items() if k != "vocabularies"}


class H3Prior(nn.Module):
    def __init__(self, ids, vocab_sizes, config):
        super().__init__()
        self.register_buffer("cell_h3_ids", ids.long(), persistent=False)
        self.register_buffer("dropout_probabilities", torch.tensor(config.h3_unk_dropout))
        self.embeddings = nn.ModuleList(nn.Embedding(n + 1, config.h3_embedding_dim) for n in vocab_sizes)
        for emb in self.embeddings:
            nn.init.zeros_(emb.weight)
        self.projection = nn.Linear(config.h3_embedding_dim, config.road_hidden_dim, bias=False)
        self.norm = nn.LayerNorm(config.road_hidden_dim)

    def forward(self, coordinate, cell_ids):
        ids = self.cell_h3_ids[cell_ids]
        if self.training:
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.dropout_probabilities, 0)
        embedding = sum(emb(ids[..., m]) for m, emb in enumerate(self.embeddings))
        return self.norm(coordinate + self.projection(embedding))
