from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig

from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.common.data.regions import centered_interval, load_peak_index, select_regions, tss_anchor

TABLES = (("X1", "train"), ("X1", "val"), ("X2", "train"), ("X2", "val"), ("X3", "test"))
TRAIN_CHROMS = {f"chr{i}" for i in range(2, 23)} - {"chr14", "chr19"}
SPLIT_CHROMS = {"train": TRAIN_CHROMS, "val": {"chr14", "chr19"}, "test": {"chr1"}}


def find_track(root, assay, cell, kind):
    suffixes = (".bw", ".bigwig") if kind == "bigwig" else (".bed",)
    matches = [Path(root) / f"{assay}-{kind}" / f"{cell}{suffix}" for suffix in suffixes]
    matches = [p for p in matches if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {kind} track for {assay}/{cell}: {matches}")
    return matches[0]


def signal_summary(bw, chrom, start, end):
    sizes = bw.chroms()
    if chrom not in sizes:
        raise ValueError(f"Chromosome {chrom} is absent from BigWig")
    lo, hi = max(0, start), min(sizes[chrom], end)
    if hi <= lo:
        return 0.0, 0.0
    values = np.asarray(bw.values(chrom, lo, hi), dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError(f"Infinite BigWig signal at {chrom}:{lo}-{hi}")
    valid = np.isfinite(values)
    width = end - start
    return float(values[valid].sum() / width), float(valid.sum() / width)


def load_examples(root, cell, split, limit):
    folder = Path(root) / "CAGE-train" / "CAGE-train"
    info = pd.read_csv(folder / f"{cell}_{split}_info.tsv", sep="\t")
    expected = ["gene_name", "chr", "gene_start", "gene_end", "TSS_start", "TSS_end", "strand"]
    if list(info.columns) != expected or not info.gene_name.is_unique:
        raise ValueError(f"Invalid gene table schema or duplicate genes: {cell}/{split}")
    if not set(info.chr).issubset(SPLIT_CHROMS[split]):
        raise ValueError(f"Unexpected chromosomes in {cell}/{split}")
    if not set(info.strand).issubset({"+", "-"}):
        raise ValueError("Gene strands must be + or -")
    if split != "test":
        labels = pd.read_csv(folder / f"{cell}_{split}_y.tsv", sep="\t")
        if list(labels.columns) != ["gene_name", "gex"] or not labels.gene_name.is_unique:
            raise ValueError("Invalid labels or duplicate label genes")
        if set(labels.gene_name) != set(info.gene_name):
            raise ValueError("Gene and label tables do not contain the same genes")
        info = info.merge(labels, on="gene_name", how="left", validate="one_to_one", sort=False)
        if not np.isfinite(info.gex).all() or (info.gex < 0).any():
            raise ValueError("CAGE labels must be finite and nonnegative for log1p supervision")
    else:
        info["gex"] = np.nan
    info["row_order"] = np.arange(len(info))
    if limit is not None:
        info = info.iloc[:limit].copy()
    info["cell"] = cell
    info["split"] = split
    info["example_id"] = [f"{cell}/{split}/{i}" for i in info.row_order]
    return info


def prepare(config, limit=None, force=False):
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    root, out = Path(config["data_dir"]), Path(config["prepared_dir"])
    tracks = config["tracks"]
    settings = config["regions"]
    input_paths = list((root / "CAGE-train" / "CAGE-train").glob("*.tsv"))
    input_paths += [find_track(root, "DNase", cell, "bed") for cell in ("X1", "X2", "X3")]
    input_paths += [find_track(root, track, cell, "bigwig") for cell in ("X1", "X2", "X3") for track in tracks]
    inputs = [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
               "sha256": file_sha256(p) if p.suffix == ".tsv" or p.suffix == ".bed" else None}
              for p in sorted(input_paths)]
    spec = {"schema": 1, "regions": settings, "tracks": tracks, "inputs": inputs, "limit_per_table": limit,
            "coordinates": "0-based-half-open", "orientation": "reference", "signal": "mean-zero-uncovered"}
    identity = digest(spec)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and not force:
        old = read_json(manifest_path)
        if old["identity"] != identity:
            # A raw-data directory may move to scratch behind its original symlink.
            # Resolve recorded paths again, retaining all size/time/hash checks and
            # the original prepared identity used by caches and trained checkpoints.
            relocated_spec = {**old["spec"], "inputs": [
                {**record, "path": str(Path(record["path"]).resolve())}
                for record in old["spec"]["inputs"]
            ]}
            if digest(relocated_spec) != identity:
                raise ValueError("Prepared data configuration/inputs changed; use another prepared_dir or --force")
        verify_prepared(out, old)
        print(f"Reusing prepared data: {out}", flush=True)
        return old
    out.mkdir(parents=True, exist_ok=True)
    examples, mappings, regions = [], [], {}
    for cell in ("X1", "X2", "X3"):
        index = load_peak_index(find_track(root, "DNase", cell, "bed"))
        handles = {}
        summaries = {}
        try:
            for track in tracks:
                handles[track] = pyBigWig.open(str(find_track(root, track, cell, "bigwig")))
            for table_cell, split in TABLES:
                if table_cell != cell:
                    continue
                table = load_examples(root, cell, split, limit)
                examples.append(table)
                for row in table.itertuples(index=False):
                    anchor = tss_anchor(row.TSS_start, row.TSS_end)
                    sizes = [handles[track].chroms().get(row.chr) for track in tracks]
                    if None in sizes or len(set(sizes)) != 1:
                        raise ValueError(f"Track chromosome lengths disagree for {cell}/{row.chr}")
                    if row.TSS_end > sizes[0]:
                        raise ValueError(f"TSS falls outside {row.chr}")
                    window_start, window_end = centered_interval(anchor, settings["window_bp"])
                    window_id = digest([row.chr, window_start, window_end])
                    for slot, (start, end, center) in enumerate(select_regions(row.chr, anchor, index, settings)):
                        region_id = digest([window_id, start, end])
                        regions[region_id] = dict(region_id=region_id, window_id=window_id, chrom=row.chr,
                                                   window_start=window_start, window_end=window_end,
                                                   start=start, end=end, chrom_size=sizes[0])
                        key = (row.chr, start, end)
                        if key not in summaries:
                            summaries[key] = {track: signal_summary(handles[track], row.chr, start, end)
                                              for track in tracks}
                        record = dict(example_id=row.example_id, region_id=region_id, slot=slot,
                                      distance=(center - anchor) * (1 if row.strand == "+" else -1) / settings["radius"],
                                      is_promoter=float(slot == 0))
                        for track, (mean, coverage) in summaries[key].items():
                            record[track] = mean
                            record[f"{track}_coverage"] = coverage
                        mappings.append(record)
                print(f"Prepared {cell}/{split}: {len(table)} genes", flush=True)
        finally:
            for handle in handles.values():
                handle.close()
    frames = {"examples": pd.concat(examples, ignore_index=True),
              "mapping": pd.DataFrame(mappings), "regions": pd.DataFrame(regions.values())}
    files = {}
    for name, frame in frames.items():
        path = out / f"{name}.parquet"
        atomic_parquet(frame, path)
        files[name] = {"sha256": file_sha256(path), "rows": len(frame)}
    manifest = {"identity": identity, "spec": spec, "files": files,
                "feature_columns": tracks + ["distance", "is_promoter"],
                "max_regions": settings["max_candidates"] + 1}
    atomic_json(manifest_path, manifest)
    print(f"Prepared {len(regions)} sequence regions in {frames['regions'].window_id.nunique()} windows", flush=True)
    return manifest


def verify_prepared(folder, manifest=None):
    folder = Path(folder)
    manifest = manifest or read_json(folder / "manifest.json")
    for name, record in manifest["files"].items():
        path = folder / f"{name}.parquet"
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise ValueError(f"Prepared table is missing or changed: {path}")
    return manifest

