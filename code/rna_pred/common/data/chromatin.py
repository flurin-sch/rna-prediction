"""Strand-oriented, multi-scale measured chromatin features; no sequence extraction."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.common.data.prepare import find_track, verify_prepared
from rna_pred.common.data.regions import tss_anchor


DEFAULT_FEATURES = {
    "tracks": ["DNase", "H3K27ac", "H3K4me3", "H3K9me3", "H3K27me3", "H3K4me1", "H3K36me3"],
    "segments": [[-100000, -10240, 6], [-10240, -2048, 4], [-2048, 2048, 32],
                 [2048, 10240, 4], [10240, 100000, 6]],
    "promoter_widths": [250, 500, 2048], "body_bins": 10,
}


def binned_signal(bw, chrom, start, end, bins):
    """Exact full-width means and covered fractions, including chromosome padding."""
    size = bw.chroms().get(chrom)
    if size is None:
        raise ValueError(f"Missing chromosome {chrom}")
    if end <= start or bins < 1:
        raise ValueError("Invalid bin interval")
    if start >= 0 and end <= size and end - start >= bins:
        means = bw.stats(chrom, int(start), int(end), nBins=bins, type="mean", exact=True)
        coverage = bw.stats(chrom, int(start), int(end), nBins=bins, type="coverage", exact=True)
        covered = np.array([0. if c is None else c for c in coverage])
        values = np.array([0. if m is None else m for m in means]) * covered
    else:
        edges = np.linspace(start, end, bins + 1).astype(np.int64)
        values, covered = np.zeros(bins), np.zeros(bins)
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            lo, hi = max(0, int(a)), min(size, int(b))
            if hi <= lo or b <= a:
                continue
            mean = bw.stats(chrom, lo, hi, type="mean", exact=True)[0]
            fraction = bw.stats(chrom, lo, hi, type="coverage", exact=True)[0]
            covered[i] = (fraction or 0.) * (hi - lo) / (b - a)
            values[i] = (mean or 0.) * covered[i]
    if not np.isfinite(values).all() or not np.isfinite(covered).all():
        raise ValueError(f"Nonfinite track values at {chrom}:{start}-{end}")
    return values, covered


def gene_features(row, handles, settings):
    anchor = tss_anchor(row.TSS_start, row.TSS_end)
    reverse = row.strand == "-"
    result = {"gene_length": float(row.gene_end - row.gene_start)}
    if result["gene_length"] <= 0:
        raise ValueError("Invalid gene body")
    for track, bw in handles.items():
        def add(name, start, end, bins, orient=False):
            signal, coverage = binned_signal(bw, row.chr, start, end, bins)
            if orient and reverse:
                signal, coverage = signal[::-1], coverage[::-1]
            for i, (value, fraction) in enumerate(zip(signal, coverage)):
                result[f"{track}/{name}/{i}/signal"] = float(value)
                result[f"{track}/{name}/{i}/coverage"] = float(fraction)
        for a, b, bins in settings["segments"]:
            lo, hi = (anchor - b, anchor - a) if reverse else (anchor + a, anchor + b)
            add(f"tss_{a}_{b}", lo, hi, bins, orient=True)
        for width in settings["promoter_widths"]:
            add(f"promoter_{width}", anchor - width // 2, anchor + width // 2, 1)
        add("body", int(row.gene_start), int(row.gene_end), settings["body_bins"], orient=True)
    return result


def prepare_features(config, folder):
    """Write a separate checksummed sidecar, preserving prepared/cache identities."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    settings = config.get("chromatin", DEFAULT_FEATURES)
    prepared = verify_prepared(config["prepared_dir"])
    inputs = []
    for cell in ("X1", "X2", "X3"):
        for track in settings["tracks"]:
            p = find_track(config["data_dir"], track, cell, "bigwig").resolve()
            st = p.stat()
            inputs.append({"path": str(p), "size": st.st_size, "mtime_ns": st.st_mtime_ns})
    spec = {"schema": 1, "settings": settings, "inputs": inputs,
            "prepared_identity": prepared["identity"], "summary": "exact-mean-zero-uncovered"}
    identity = digest(spec)
    if (folder / "manifest.json").exists():
        manifest = read_json(folder / "manifest.json")
        if manifest["identity"] != identity:
            raise ValueError("Chromatin feature inputs/settings changed; choose a new feature directory")
        load_features(folder, prepared["identity"])
        return manifest
    examples = pd.read_parquet(Path(config["prepared_dir"]) / "examples.parquet")
    # Per-cell commits allow preprocessing to resume after interruption.
    for cell, frame in examples.groupby("cell", sort=True):
        target = folder / f"{cell}.parquet"
        marker = folder / f"{cell}.json"
        if marker.exists():
            record = read_json(marker)
            if record["identity"] == identity and target.exists() and file_sha256(target) == record["sha256"]:
                continue
            raise ValueError(f"Incompatible/corrupted partial feature file: {target}")
        rows = []
        with ExitStack() as stack:
            handles = {t: stack.enter_context(pyBigWig.open(str(find_track(config["data_dir"], t, cell, "bigwig"))))
                       for t in settings["tracks"]}
            for chrom in frame.chr.unique():
                sizes = {handle.chroms().get(chrom) for handle in handles.values()}
                if None in sizes or len(sizes) != 1:
                    raise ValueError(f"Track chromosome lengths disagree for {cell}/{chrom}")
            for i, row in enumerate(frame.itertuples(index=False)):
                rows.append({"example_id": row.example_id, **gene_features(row, handles, settings)})
                if (i + 1) % 1000 == 0:
                    print(f"Chromatin features {cell}: {i + 1}/{len(frame)}", flush=True)
        atomic_parquet(pd.DataFrame(rows), target)
        atomic_json(marker, {"identity": identity, "sha256": file_sha256(target)})
    files = {c: read_json(folder / f"{c}.json")["sha256"] for c in sorted(examples.cell.unique())}
    columns = pd.read_parquet(folder / "X1.parquet").columns.tolist()[1:]
    manifest = {"identity": identity, "spec": spec, "files": files, "columns": columns}
    atomic_json(folder / "manifest.json", manifest)
    return manifest


def load_features(folder, prepared_identity):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest["spec"]["prepared_identity"] != prepared_identity:
        raise ValueError("Chromatin features do not match prepared examples")
    frames = []
    for cell, checksum in manifest["files"].items():
        path = folder / f"{cell}.parquet"
        if file_sha256(path) != checksum:
            raise ValueError(f"Corrupted chromatin features: {path}")
        frame = pd.read_parquet(path)
        if frame.columns.tolist() != ["example_id", *manifest["columns"]]:
            raise ValueError("Chromatin feature columns disagree")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).set_index("example_id", verify_integrity=True)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Nonfinite chromatin sidecar")
    return result, manifest


def feature_matrix(table, examples):
    values = table.loc[examples.example_id].to_numpy(dtype=np.float32, copy=True)
    # Coverage fractions retain their original interpretation.
    signal = np.array([not c.endswith("/coverage") for c in table.columns])
    values[:, signal] = np.sign(values[:, signal]) * np.log1p(np.abs(values[:, signal]))
    return values
