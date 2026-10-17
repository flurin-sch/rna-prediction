"""Resumable frozen-feature extraction and lazy, indexed feature access."""
from collections import OrderedDict
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta

from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.common.data.prepare import verify_prepared
from rna_pred.common.data.regions import overlap_weights


def fetch_window(reference, chrom, start, end):
    if chrom not in reference:
        raise ValueError(f"Reference lacks chromosome {chrom}")
    size = len(reference[chrom])
    lo, hi = max(0, start), min(size, end)
    if hi <= lo:
        raise ValueError(f"Window does not overlap chromosome {chrom}")
    native = str(reference[chrom][lo:hi]).upper()
    sequence = "N" * (lo - start) + native + "N" * (end - hi)
    if len(sequence) != end - start:
        raise ValueError("Fetched window length does not match coordinate frame")
    return sequence, (hi - lo) / (end - start)


def one_hot(sequence):
    lookup = np.zeros((256, 4), dtype=np.float32)
    for index, base in enumerate("ACGT"):
        lookup[ord(base), index] = lookup[ord(base.lower()), index] = 1
    return lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]


def pool_regions(hidden, window_start, window_bp, regions):
    if hidden.shape != (window_bp // 128, 3072):
        raise ValueError(f"Unexpected contextual feature shape: {tuple(hidden.shape)}")
    vectors = []
    for region in regions.itertuples(index=False):
        weights = overlap_weights(window_start, window_bp, region.start, region.end)
        indices = np.flatnonzero(weights)
        selected = hidden[torch.as_tensor(indices, device=hidden.device)].float()
        w = torch.as_tensor(weights[indices], device=hidden.device)
        vector = (selected * w[:, None]).sum(0) / w.sum()
        if not torch.isfinite(vector).all():
            raise ValueError(f"Nonfinite embedding for {region.region_id}")
        vectors.append(vector)
    return torch.stack(vectors).cpu().numpy().astype(np.float32)


def resolve_checkpoint(config, cache_dir):
    settings = config["extraction"]
    path = settings.get("checkpoint_path")
    lock_path = cache_dir / "checkpoint.json"
    if path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        provenance = {"source": "local", "path": str(path.resolve()),
                      "declared_repo": settings["checkpoint_repo"],
                      "declared_revision": settings.get("checkpoint_revision")}
    else:
        from huggingface_hub import HfApi, hf_hub_download
        revision = settings.get("checkpoint_revision")
        if revision is None and lock_path.exists():
            lock = read_json(lock_path)
            if lock.get("repo") == settings["checkpoint_repo"] and lock.get("file") == settings["checkpoint_file"]:
                revision = lock["revision"]
        if revision is None:
            revision = HfApi().model_info(settings["checkpoint_repo"]).sha
        path = Path(hf_hub_download(settings["checkpoint_repo"], settings["checkpoint_file"],
                                   revision=revision, cache_dir=cache_dir / "hf"))
        provenance = {"source": "huggingface", "repo": settings["checkpoint_repo"],
                      "file": settings["checkpoint_file"], "revision": revision}
    provenance["sha256"] = file_sha256(path)
    atomic_json(lock_path, provenance)
    return path, provenance


def installed_port_provenance(required_revision):
    dist = importlib.metadata.distribution("alphagenome-pytorch")
    direct = dist.read_text("direct_url.json")
    if not direct:
        raise ValueError("Install the pinned AlphaGenome git dependency; version-only installs are unsupported")
    metadata = json.loads(direct)
    actual = metadata.get("vcs_info", {}).get("commit_id")
    if actual != required_revision:
        raise ValueError(f"AlphaGenome port must be pinned to {required_revision}; found {actual}")
    return {"revision": actual, "version": dist.version}


def build_cache(config, device="cuda", limit_windows=None, verify_repeat=False):
    if limit_windows is not None and limit_windows < 1:
        raise ValueError("--limit-windows must be positive")
    prepared = Path(config["prepared_dir"])
    prepared_manifest = verify_prepared(prepared)
    if prepared_manifest["spec"]["regions"] != config["regions"] or prepared_manifest["spec"]["tracks"] != config["tracks"]:
        raise ValueError("Prepared data settings do not match configuration; run prepare first")
    regions = pd.read_parquet(prepared / "regions.parquet").sort_values(["window_id", "region_id"])
    out = Path(config["cache_dir"])
    out.mkdir(parents=True, exist_ok=True)
    reference_path = Path(config["reference_fasta"])
    if not reference_path.is_file():
        raise FileNotFoundError(f"Missing hg38 reference: {reference_path}; see README for setup")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run extraction in a GPU allocation")
    from alphagenome_pytorch import AlphaGenome
    from alphagenome_pytorch.config import DtypePolicy
    port = installed_port_provenance(config["extraction"]["port_revision"])
    checkpoint_path, checkpoint = resolve_checkpoint(config, out)
    policy = (DtypePolicy.mixed_precision() if config["extraction"]["mixed_precision"]
              and device.startswith("cuda") else DtypePolicy.full_float32())
    reference_identity = {"assembly": "hg38", "sha256": file_sha256(reference_path)}
    spec = {"schema": 1, "reference": reference_identity, "checkpoint": checkpoint,
            "port": port, "window_bp": config["regions"]["window_bp"], "resolution": 128,
            "embedding_dim": 3072, "organism_index": 0, "pooling": "overlap-weighted-mean", "orientation": "reference",
            "boundary": "N-padding", "compute": "bf16" if policy.compute_dtype == torch.bfloat16 else "float32",
            "storage_dtype": "float32", "torch": torch.__version__}
    # Location/source metadata is provenance, not a different representation.
    effective_spec = dict(spec, checkpoint={"sha256": checkpoint["sha256"]},
                          port={"revision": port["revision"]})
    identity = digest(effective_spec)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["identity"] != identity:
            raise ValueError("Incompatible cache: choose a new cache_dir for changed extraction settings")
    else:
        manifest = {"identity": identity, "spec": spec, "shards": [], "prepared_identities": [], "complete": False}
        atomic_json(manifest_path, manifest)
    # Shard metadata is the transaction commit marker. Recover committed shards
    # if a process stopped before updating the global manifest/index.
    committed = []
    for path in sorted(out.glob("shard-*.json")):
        meta = read_json(path)
        if meta["identity"] != identity:
            raise ValueError(f"Incompatible shard: {path}")
        for kind in ("vectors", "metadata"):
            target = out / meta[kind]["file"]
            if not target.is_file() or file_sha256(target) != meta[kind]["sha256"]:
                raise ValueError(f"Missing or corrupted shard file: {target}")
        committed.append(meta)
    frames = [pd.read_parquet(out / s["metadata"]["file"]) for s in committed]
    existing = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["region_id"])
    if existing.region_id.duplicated().any():
        raise ValueError("Duplicate region IDs in feature cache")
    pending = regions[~regions.region_id.isin(existing.region_id)]
    groups = list(pending.groupby("window_id", sort=True))
    if limit_windows is not None:
        groups = groups[:limit_windows]
    new_index = frames.copy()
    model = None
    started = time.perf_counter()
    repeat_max_difference = None
    torch.set_num_threads(config["train"]["cpu_threads"])
    if groups:
        model = AlphaGenome.from_pretrained(checkpoint_path, dtype_policy=policy, device=device).eval()
        model.requires_grad_(False)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
    reference = Fasta(str(reference_path), as_raw=True, sequence_always_upper=True)
    try:
        for offset in range(0, len(groups), config["extraction"]["shard_windows"]):
            shard_groups = groups[offset:offset + config["extraction"]["shard_windows"]]
            pooled, metadata = [], []
            for batch_start in range(0, len(shard_groups), config["extraction"]["batch_size"]):
                batch = shard_groups[batch_start:batch_start + config["extraction"]["batch_size"]]
                sequences, fractions = [], []
                for _, group in batch:
                    row = group.iloc[0]
                    if row.chrom not in reference or len(reference[row.chrom]) != row.chrom_size:
                        raise ValueError(f"Reference/track chromosome length mismatch for {row.chrom}")
                    sequence, fraction = fetch_window(reference, row.chrom, int(row.window_start), int(row.window_end))
                    sequences.append(one_hot(sequence))
                    fractions.append(fraction)
                dna = torch.from_numpy(np.stack(sequences)).to(device)
                with torch.inference_mode():
                    output = model.encode(dna, organism_index=0, resolutions=(128,))
                    hidden = output["embeddings_128bp"]
                    if verify_repeat and repeat_max_difference is None:
                        repeated = model.encode(dna, organism_index=0, resolutions=(128,))["embeddings_128bp"]
                        torch.testing.assert_close(hidden, repeated, rtol=1e-4, atol=1e-4)
                        repeat_max_difference = float((hidden.float() - repeated.float()).abs().max().item())
                        del repeated
                    for j, (_, group) in enumerate(batch):
                        row = group.iloc[0]
                        vectors = pool_regions(hidden[j], int(row.window_start), config["regions"]["window_bp"], group)
                        pooled.append(vectors)
                        frame = group.copy()
                        frame["native_base_fraction"] = fractions[j]
                        frame["valid_base_fraction"] = float(sequences[j].any(axis=1).mean())
                        metadata.append(frame)
                del output, hidden, dna
            number = len(committed)
            name = f"shard-{number:06d}"
            values = np.concatenate(pooled)
            frame = pd.concat(metadata, ignore_index=True)
            frame["shard"] = f"{name}.npy"
            frame["vector_row"] = np.arange(len(values))
            temp = out / f"{name}.tmp.npy"
            np.save(temp, values, allow_pickle=False)
            os.replace(temp, out / f"{name}.npy")
            atomic_parquet(frame, out / f"{name}.parquet")
            meta = {"identity": identity, "rows": len(values),
                    "vectors": {"file": f"{name}.npy", "sha256": file_sha256(out / f"{name}.npy")},
                    "metadata": {"file": f"{name}.parquet", "sha256": file_sha256(out / f"{name}.parquet")}}
            atomic_json(out / f"{name}.json", meta)
            committed.append(meta)
            new_index.append(frame)
            print(f"Cached windows {min(offset + len(shard_groups), len(groups))}/{len(groups)}", flush=True)
    finally:
        reference.close()
    index = pd.concat(new_index, ignore_index=True) if new_index else existing
    if index.empty:
        raise ValueError("No feature vectors were extracted")
    atomic_parquet(index, out / "index.parquet")
    complete = set(regions.region_id).issubset(set(index.region_id))
    manifest.update(shards=committed, complete=complete, index_sha256=file_sha256(out / "index.parquet"))
    manifest["prepared_identities"] = sorted(set(manifest["prepared_identities"] + [prepared_manifest["identity"]]))
    atomic_json(manifest_path, manifest)
    report = {"windows_extracted": len(groups), "regions_available": len(index), "complete": complete,
              "elapsed_seconds": time.perf_counter() - started, "device": device,
              "python": platform.python_version(), "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None}
    report["repeat_max_abs_difference"] = repeat_max_difference
    atomic_json(out / "last_extraction.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return manifest


class CacheStore:
    def __init__(self, folder, expected_dim=3072):
        self.folder = Path(folder)
        self.manifest = read_json(self.folder / "manifest.json")
        if "index_sha256" not in self.manifest or not (self.folder / "index.parquet").is_file():
            raise ValueError("Feature cache has no committed index; resume extraction first")
        if file_sha256(self.folder / "index.parquet") != self.manifest["index_sha256"]:
            raise ValueError("Feature index is corrupted")
        self.index = pd.read_parquet(self.folder / "index.parquet")
        if self.index.region_id.duplicated().any():
            raise ValueError("Duplicate cache region IDs")
        self.lookup = {row.region_id: (row.shard, int(row.vector_row)) for row in self.index.itertuples()}
        self.expected_dim = expected_dim
        self.arrays = OrderedDict()
        for shard in self.manifest["shards"]:
            path = self.folder / shard["vectors"]["file"]
            if file_sha256(path) != shard["vectors"]["sha256"]:
                raise ValueError(f"Feature shard is corrupted: {path}")

    def get_many(self, region_ids):
        result = np.empty((len(region_ids), self.expected_dim), dtype=np.float32)
        for i, region_id in enumerate(region_ids):
            if region_id not in self.lookup:
                raise ValueError(f"Missing sequence cache entry {region_id}; finish extraction first")
            name, row = self.lookup[region_id]
            if name not in self.arrays:
                self.arrays[name] = np.load(self.folder / name, mmap_mode="r", allow_pickle=False)
                if (self.arrays[name].ndim != 2 or self.arrays[name].shape[1] != self.expected_dim
                        or self.arrays[name].dtype != np.float32):
                    raise ValueError(f"Invalid cached vector shape: {name}")
            self.arrays.move_to_end(name)
            result[i] = self.arrays[name][row]
            if len(self.arrays) > 8:
                self.arrays.popitem(last=False)
        if not np.isfinite(result).all():
            raise ValueError("Cached sequence vectors contain nonfinite values")
        return result


def validate_cache(config, require_full=True):
    """Check requested-region coverage, shard integrity, and every stored value."""
    prepared = Path(config["prepared_dir"])
    manifest = verify_prepared(prepared)
    if require_full and manifest["spec"]["limit_per_table"] is not None:
        raise ValueError("Full cache validation requires unlimited prepared data")
    store = CacheStore(config["cache_dir"], config["model"]["embedding_dim"])
    if not store.manifest["complete"] or manifest["identity"] not in store.manifest["prepared_identities"]:
        raise ValueError("Cache is not marked complete for these prepared data")
    regions = pd.read_parquet(prepared / "regions.parquet")
    examples = pd.read_parquet(prepared / "examples.parquet")
    if not set(regions.region_id).issubset(store.lookup):
        raise ValueError("Cache does not cover all requested regions")
    actual_rows, vector_bytes = 0, 0
    for shard in store.manifest["shards"]:
        metadata_path = store.folder / shard["metadata"]["file"]
        if file_sha256(metadata_path) != shard["metadata"]["sha256"]:
            raise ValueError(f"Corrupted shard metadata: {metadata_path}")
        metadata = pd.read_parquet(metadata_path)
        vectors = np.load(store.folder / shard["vectors"]["file"], mmap_mode="r", allow_pickle=False)
        if vectors.shape != (shard["rows"], store.expected_dim) or vectors.dtype != np.float32:
            raise ValueError("Invalid shard shape or dtype")
        if metadata.vector_row.tolist() != list(range(shard["rows"])):
            raise ValueError("Invalid shard row mapping")
        if metadata.shard.nunique() != 1 or metadata.shard.iloc[0] != shard["vectors"]["file"]:
            raise ValueError("Invalid shard filename mapping")
        for offset in range(0, len(vectors), 8192):
            if not np.isfinite(vectors[offset:offset + 8192]).all():
                raise ValueError("Nonfinite cached sequence features")
        for row in metadata.itertuples():
            if store.lookup.get(row.region_id) != (row.shard, int(row.vector_row)):
                raise ValueError("Global index does not match shard metadata")
        actual_rows += len(vectors)
        vector_bytes += vectors.nbytes
    if actual_rows != len(store.index):
        raise ValueError("Shard row counts do not match cache index")
    report = {"complete": True, "examples": len(examples), "windows": int(regions.window_id.nunique()),
              "requested_regions": len(regions), "cached_regions": actual_rows,
              "embedding_dim": store.expected_dim, "vector_bytes": vector_bytes,
              "shards": len(store.manifest["shards"]), "all_finite": True,
              "checksums_verified": True, "prepared_identity": manifest["identity"],
              "cache_identity": store.manifest["identity"]}
    atomic_json(store.folder / "validation.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report
