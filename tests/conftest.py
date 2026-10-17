from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pyBigWig
import pytest
import torch

from rna_pred.common.config import load_config


@pytest.fixture
def synthetic_project(tmp_path):
    config = load_config(Path(__file__).parents[1] / "code/rna_pred/baseline/configs/default.yaml")
    config.update(data_dir=str(tmp_path / "raw"), prepared_dir=str(tmp_path / "prepared"),
                  cache_dir=str(tmp_path / "cache"), reference_fasta=str(tmp_path / "hg38.fa"),
                  run_dir=str(tmp_path / "run"))
    config["data_provenance"] = {"kind": "synthetic", "description": "Synthetic test dataset with artificial expression labels and mocked sequence embeddings"}
    config["train"].update(epochs=2, patience=8, batch_size=2, cpu_threads=1, device="cpu")
    config["extraction"].update(shard_windows=3, batch_size=2, mixed_precision=False)
    root = Path(config["data_dir"])
    genes = root / "CAGE-train" / "CAGE-train"
    genes.mkdir(parents=True)
    sizes = {"chr1": 300000, "chr2": 300000, "chr14": 300000}
    anchors = [100000, 110000, 120000]
    for cell, split, chrom in [("X1", "train", "chr2"), ("X2", "train", "chr2"),
                              ("X1", "val", "chr14"), ("X2", "val", "chr14"), ("X3", "test", "chr1")]:
        rows = []
        for i, anchor in enumerate(anchors):
            if cell == "X2" and i == 2:
                anchor += 50
            rows.append(dict(gene_name=f"gene{i}", chr=chrom, gene_start=anchor - 100,
                             gene_end=anchor + 3000, TSS_start=anchor - 25, TSS_end=anchor + 25,
                             strand="-" if i == 1 else "+"))
        pd.DataFrame(rows).to_csv(genes / f"{cell}_{split}_info.tsv", sep="\t", index=False)
        if split != "test":
            pd.DataFrame({"gene_name": [f"gene{i}" for i in range(3)], "gex": [0.0, 5.0, 30.0]}).to_csv(
                genes / f"{cell}_{split}_y.tsv", sep="\t", index=False)
    for cell_idx, cell in enumerate(("X1", "X2", "X3")):
        peak_dir = root / "DNase-bed"
        peak_dir.mkdir(exist_ok=True)
        with (peak_dir / f"{cell}.bed").open("w") as handle:
            for chrom in sizes:
                for summit in [90000, 99000, 100000, 103000, 107000, 113000, 123000, 220000]:
                    handle.write(f"{chrom}\t{summit-75}\t{summit+75}\t.\t0\t.\t2\t-1\t-1\t75\n")
        for track_idx, track in enumerate(config["tracks"]):
            folder = root / f"{track}-bigwig"
            folder.mkdir(exist_ok=True)
            bw = pyBigWig.open(str(folder / f"{cell}.bw"), "w")
            bw.addHeader(list(sizes.items()))
            for chrom, size in sizes.items():
                starts = list(range(0, size, 1000))
                values = [float((i % 20) + track_idx + cell_idx) for i in range(len(starts))]
                bw.addEntries([chrom] * len(starts), starts, ends=[s + 1000 for s in starts], values=values)
            bw.close()
    with Path(config["reference_fasta"]).open("w") as handle:
        for chrom, size in sizes.items():
            handle.write(f">{chrom}\n")
            sequence = "ACGT" * (size // 4)
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i:i + 80] + "\n")
    return config


@pytest.fixture
def fake_alphagenome(monkeypatch):
    from rna_pred.common.data import cache

    class FakeModel(torch.nn.Module):
        calls = 0

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def encode(self, dna, organism_index, resolutions):
            assert not self.training
            assert not torch.is_grad_enabled()
            assert organism_index == 0 and resolutions == (128,)
            type(self).calls += len(dna)
            channels = torch.sin(torch.arange(3072, device=dna.device).float())
            hidden = channels[None, None, :].expand(len(dna), dna.shape[1] // 128, -1)
            return {"embeddings_128bp": hidden}

    class Policy:
        compute_dtype = torch.float32

        @classmethod
        def full_float32(cls):
            return cls()

        mixed_precision = full_float32

    module = types.ModuleType("alphagenome_pytorch")
    module.AlphaGenome = FakeModel
    module_config = types.ModuleType("alphagenome_pytorch.config")
    module_config.DtypePolicy = Policy
    monkeypatch.setitem(sys.modules, "alphagenome_pytorch", module)
    monkeypatch.setitem(sys.modules, "alphagenome_pytorch.config", module_config)
    monkeypatch.setattr(cache, "installed_port_provenance", lambda revision: {"revision": revision, "version": "test"})
    monkeypatch.setattr(cache, "resolve_checkpoint", lambda config, folder: (Path("unused"), {"sha256": "test-only"}))
    return FakeModel

