"""Coordinate-only region construction; all intervals are 0-based, half-open."""
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd


def tss_anchor(start, end):
    if start < 0 or end <= start:
        raise ValueError(f"Invalid TSS interval [{start}, {end})")
    return (int(start) + int(end)) // 2


def centered_interval(anchor, width):
    return int(anchor) - width // 2, int(anchor) + width // 2


def load_peak_index(path):
    peaks = pd.read_csv(path, sep="\t", header=None, comment="#", usecols=[0, 1, 2, 9],
                        names=["chrom", "start", "end", "summit_offset"])
    if ((peaks.start < 0) | (peaks.end <= peaks.start)).any():
        raise ValueError(f"Invalid DNase coordinates in {path}")
    usable = (peaks.summit_offset >= 0) & (peaks.summit_offset < peaks.end - peaks.start)
    peaks["summit"] = np.where(usable, peaks.start + peaks.summit_offset,
                               (peaks.start + peaks.end) // 2).astype(np.int64)
    return {chrom: np.unique(rows.summit.to_numpy()) for chrom, rows in peaks.groupby("chrom")}


def select_regions(chrom, anchor, summit_index, settings):
    promoter = centered_interval(anchor, settings["promoter_bp"])
    summits = summit_index.get(chrom, np.empty(0, dtype=np.int64))
    left = bisect_left(summits, anchor - settings["radius"])
    right = bisect_right(summits, anchor + settings["radius"])
    candidates = []
    for summit in summits[left:right]:
        start, end = centered_interval(int(summit), settings["candidate_bp"])
        if start < promoter[1] and end > promoter[0]:
            continue
        candidates.append((abs(int(summit) - anchor), start, end, int(summit)))
    candidates.sort()
    return [(*promoter, anchor)] + [(start, end, summit) for _, start, end, summit
                                    in candidates[:settings["max_candidates"]]]


def overlap_weights(window_start, window_bp, region_start, region_end, resolution=128):
    if region_start < window_start or region_end > window_start + window_bp or region_end <= region_start:
        raise ValueError("Pooling region must lie within the extraction window")
    bins = window_start + np.arange(window_bp // resolution) * resolution
    weights = np.maximum(0, np.minimum(bins + resolution, region_end) - np.maximum(bins, region_start))
    if weights.sum() != region_end - region_start:
        raise ValueError("Pooling weights do not cover the region")
    return weights.astype(np.float32)

