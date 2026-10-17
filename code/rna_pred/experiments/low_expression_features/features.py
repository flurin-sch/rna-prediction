"""Label-free feature sidecars for the low-expression investigation."""
from bisect import bisect_left, bisect_right
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from pyfaidx import Fasta

from rna_pred.common.data.chromatin import binned_signal, load_features
from rna_pred.common.config import atomic_json, atomic_parquet, digest, file_sha256, read_json
from rna_pred.common.data.prepare import find_track, verify_prepared
from rna_pred.common.data.regions import load_peak_index, tss_anchor

TRACKS = ("DNase", "H3K27ac", "H3K4me3", "H3K9me3", "H3K27me3", "H3K4me1", "H3K36me3")
FAMILIES = ("enhancer", "body", "promoter")


class Intervals:
    """Union intervals, with logarithmic-time covered-length queries."""
    def __init__(self, starts=(), ends=()):
        merged = []
        for a, b in sorted(zip(starts, ends)):
            if b <= a:
                raise ValueError("Invalid interval")
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], int(b))
            else:
                merged.append([int(a), int(b)])
        self.starts = np.array([a for a, _ in merged], dtype=np.int64)
        self.ends = np.array([b for _, b in merged], dtype=np.int64)
        self.prefix = np.r_[0, np.cumsum(self.ends - self.starts)]

    def integral(self, position):
        i = np.searchsorted(self.ends, position, side="right")
        value = int(self.prefix[i])
        if i < len(self.starts):
            value += max(0, int(position) - int(self.starts[i]))
        return value

    def overlap(self, start, end):
        return max(0, self.integral(end) - self.integral(start)) if end > start else 0


def peak_intervals(path):
    peaks = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2],
                        names=["chrom", "start", "end"])
    return {c: Intervals(g.start, g.end) for c, g in peaks.groupby("chrom")}


def candidate_indices(summits, anchor, settings):
    lo = bisect_left(summits, anchor - settings["radius"])
    hi = bisect_right(summits, anchor + settings["radius"])
    indices = np.arange(lo, hi)
    half_p, half_c = settings["promoter_bp"] // 2, settings["candidate_bp"] // 2
    keep = ~((summits[indices] - half_c < anchor + half_p) &
             (summits[indices] + half_c > anchor - half_p))
    indices = indices[keep]
    order = np.lexsort((summits[indices], np.abs(summits[indices] - anchor)))
    return indices, indices[order[:settings["max_candidates"]]]


def body_pieces(start, end, anchor, promoter_bp=2048):
    """Gene-body pieces outside the target promoter, with no invented TSS repair."""
    lo, hi = anchor - promoter_bp // 2, anchor + promoter_bp // 2
    return [(a, b) for a, b in [(start, min(end, lo)), (max(start, hi), end)] if b > a]


def sequence_features(sequence):
    seq = sequence.upper()
    valid = sum(seq.count(b) for b in "ACGT")
    c, g = seq.count("C"), seq.count("G")
    return {"gc": (c + g) / valid if valid else 0.,
            "cpg_density": seq.count("CG") / max(1, len(seq) - 1),
            "cpg_observed_expected": seq.count("CG") * valid / max(1, c * g),
            "ambiguous_fraction": 1 - valid / max(1, len(seq))}


def summarize_candidates(indices, anchor, summits, signals, coverages, flags, prefix):
    n = len(indices)
    result = {f"enhancer/{prefix}/count": float(n), f"enhancer/{prefix}/none": float(n == 0)}
    distance = np.abs(summits[indices] - anchor)
    result[f"enhancer/{prefix}/nearest_bp"] = float(distance.min()) if n else 100001.
    result[f"enhancer/{prefix}/mean_distance_bp"] = float(distance.mean()) if n else 100001.
    weight = np.exp(-distance / 25000.)
    for name, values in flags.items():
        count = int(values[indices].sum())
        result[f"enhancer/{prefix}/{name}_count"] = float(count)
        result[f"enhancer/{prefix}/{name}_none"] = float(count == 0)
        result[f"enhancer/{prefix}/{name}_fraction"] = count / n if n else 0.
    for k, track in enumerate(TRACKS):
        values = signals[indices, k]
        for name, value in [("sum", values.sum()), ("mean", values.mean() if n else 0.),
                            ("max", values.max() if n else 0.), ("distance_sum", (values * weight).sum())]:
            result[f"enhancer/{prefix}/{track}/{name}"] = float(value)
        result[f"enhancer/{prefix}/{track}/coverage"] = float(coverages[indices, k].mean()) if n else 0.
    return result


def prepare_extra(config, root, base_dir):
    root = Path(root)
    folder = root / "features-extra-v1"
    folder.mkdir(parents=True, exist_ok=True)
    prepared = verify_prepared(config["prepared_dir"])
    base, base_manifest = load_features(base_dir, prepared["identity"])
    examples = pd.read_parquet(Path(config["prepared_dir"]) / "examples.parquet")
    paths = [Path(config["reference_fasta"]), Path(config["reference_fasta"] + ".fai")]
    paths += [find_track(config["data_dir"], t, c, kind) for c in examples.cell.unique()
              for t in TRACKS for kind in ("bed", "bigwig")]
    inputs = [{"path": str(p.resolve()), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
              for p in paths]
    spec = {"schema": 1, "prepared_identity": prepared["identity"], "base_identity": base_manifest["identity"],
            "inputs": inputs, "source_sha256": file_sha256(__file__), "regions": config["regions"],
            "activity": "DNase candidate overlapping H3K27ac AND H3K4me1 BED, excluding annotated promoters",
            "summary": "exact-mean-zero-uncovered; BED union breadth; no expression-dependent thresholds"}
    identity = digest(spec)
    if (folder / "manifest.json").exists():
        old = read_json(folder / "manifest.json")
        if old["identity"] != identity:
            raise ValueError("Feature extraction specification changed; choose a new investigation root")
        load_features(folder, prepared["identity"])
        return folder
    with Fasta(config["reference_fasta"], as_raw=True, sequence_always_upper=True, rebuild=False) as fasta:
        for cell, frame in examples.groupby("cell", sort=True):
            target, marker = folder / f"{cell}.parquet", folder / f"{cell}.json"
            if marker.exists():
                old = read_json(marker)
                if old["identity"] != identity or file_sha256(target) != old["sha256"]:
                    raise ValueError("Corrupt or incompatible partial features")
                continue
            peaks = {t: peak_intervals(find_track(config["data_dir"], t, cell, "bed")) for t in TRACKS}
            summits_by_chrom = load_peak_index(find_track(config["data_dir"], "DNase", cell, "bed"))
            rows = []
            with ExitStack() as stack:
                handles = {t: stack.enter_context(pyBigWig.open(str(find_track(config["data_dir"], t, cell, "bigwig")))) for t in TRACKS}
                for chrom, genes in frame.groupby("chr", sort=True):
                    anchors = np.array([(r.TSS_start + r.TSS_end) // 2 for r in genes.itertuples()])
                    annotated = Intervals(anchors - 1024, anchors + 1024)
                    starts, ends = np.sort(genes.gene_start), np.sort(genes.gene_end)
                    summits = summits_by_chrom.get(chrom, np.empty(0, dtype=np.int64))
                    # Only summarize raw candidates that lie in at least one supplied gene window.
                    needed = np.zeros(len(summits), dtype=bool)
                    for a in anchors:
                        all_i, _ = candidate_indices(summits, a, config["regions"])
                        needed[all_i] = True
                    signals, coverage = np.zeros((len(summits), len(TRACKS))), np.zeros((len(summits), len(TRACKS)))
                    overlaps = {t: np.zeros(len(summits), dtype=bool) for t in TRACKS}
                    promoters = np.zeros(len(summits), dtype=bool)
                    for i in np.flatnonzero(needed):
                        a, b = int(summits[i]) - 256, int(summits[i]) + 256
                        promoters[i] = annotated.overlap(a, b) > 0
                        for k, t in enumerate(TRACKS):
                            signal, covered = binned_signal(handles[t], chrom, a, b, 1)
                            signals[i, k], coverage[i, k] = signal[0], covered[0]
                            overlaps[t][i] = peaks[t].get(chrom, Intervals()).overlap(a, b) > 0
                    flags = {"acetylated": overlaps["H3K27ac"], "annotated_promoter": promoters,
                             "promoter_marked": overlaps["H3K4me3"],
                             "activity_supported": overlaps["H3K27ac"] & overlaps["H3K4me1"] & ~promoters,
                             "repression_supported": overlaps["H3K27me3"] | overlaps["H3K9me3"]}
                    for row in genes.itertuples(index=False):
                        anchor = tss_anchor(row.TSS_start, row.TSS_end)
                        result = {"example_id": row.example_id}
                        all_i, retained = candidate_indices(summits, anchor, config["regions"])
                        for name, idx in [("all", all_i), ("retained", retained)]:
                            result.update(summarize_candidates(idx, anchor, summits, signals, coverage, flags, name))
                        result["enhancer/truncated_count"] = float(len(all_i) - len(retained))
                        pieces = body_pieces(int(row.gene_start), int(row.gene_end), anchor)
                        width = sum(b - a for a, b in pieces)
                        result["body/excluded_width"] = float(width)
                        result["body/empty_after_exclusion"] = float(width == 0)
                        result["body/tss_outside"] = float(not row.gene_start <= anchor < row.gene_end)
                        result["body/tss_edge_distance"] = float(min(abs(anchor-row.gene_start), abs(anchor-row.gene_end)))
                        result["body/overlapping_gene_count"] = float(np.searchsorted(starts, row.gene_end, side="left") -
                                                                     np.searchsorted(ends, row.gene_start, side="right") - 1)
                        for t in TRACKS:
                            total, covered = 0., 0.
                            for a, b in pieces:
                                s, c = binned_signal(handles[t], chrom, a, b, 1)
                                total += s[0] * (b-a)
                                covered += c[0] * (b-a)
                            result[f"body/{t}/excluded_signal"] = total / width if width else 0.
                            result[f"body/{t}/coverage"] = covered / width if width else 0.
                            result[f"body/{t}/peak_breadth"] = sum(peaks[t].get(chrom, Intervals()).overlap(a,b) for a,b in pieces) / width if width else 0.
                            bins = np.array([base.at[row.example_id, f"{t}/body/{i}/signal"] for i in range(10)])
                            result[f"body/{t}/end_start_contrast"] = float(np.log1p(bins[-3:].mean()) - np.log1p(bins[:3].mean()))
                            promoter = base.at[row.example_id, f"{t}/promoter_2048/0/signal"]
                            result[f"body/{t}/body_promoter_contrast"] = float(np.log1p(max(0,total/max(1,width))) - np.log1p(max(0,promoter)))
                        for active in ("H3K36me3", "H3K27ac"):
                            for repressive in ("H3K27me3", "H3K9me3"):
                                result[f"body/{active}_{repressive}_contrast"] = float(
                                    np.log1p(max(0,result[f"body/{active}/excluded_signal"])) -
                                    np.log1p(max(0,result[f"body/{repressive}/excluded_signal"])))
                        seq_start, seq_end = max(0,anchor-1024), min(len(fasta[chrom]),anchor+1024)
                        sequence = "N" * max(0,1024-anchor) + str(fasta[chrom][seq_start:seq_end]) + "N" * max(0,anchor+1024-len(fasta[chrom]))
                        result.update({f"promoter/{k}": v for k,v in sequence_features(sequence).items()})
                        for radius in (10000, 100000):
                            result[f"promoter/neighbors_{radius}"] = float(((anchors>=anchor-radius)&(anchors<=anchor+radius)).sum()-1)
                        for a,b in [("H3K4me3","H3K27me3"),("H3K27ac","H3K27me3"),("H3K4me1","H3K4me3")]:
                            result[f"promoter/{a}_{b}_contrast"] = float(np.log1p(max(0,base.at[row.example_id,f"{a}/promoter_2048/0/signal"]))-
                                                                          np.log1p(max(0,base.at[row.example_id,f"{b}/promoter_2048/0/signal"])))
                        rows.append(result)
                    print(f"Features {cell}/{chrom}: {len(rows)}/{len(frame)} genes; {needed.sum()} distinct candidate regions", flush=True)
            output = pd.DataFrame(rows)
            if not np.isfinite(output.drop(columns="example_id").to_numpy()).all():
                raise ValueError("Nonfinite extra features")
            atomic_parquet(output, target)
            atomic_json(marker, {"identity": identity, "sha256": file_sha256(target)})
    columns = pd.read_parquet(folder / "X1.parquet").columns.tolist()[1:]
    atomic_json(folder / "manifest.json", {"identity": identity, "spec": spec,
                "columns": columns, "files": {c: read_json(folder/f"{c}.json")["sha256"] for c in examples.cell.unique()},
                "families": {f:[c for c in columns if c.startswith(f+"/")] for f in FAMILIES}})
    return folder

