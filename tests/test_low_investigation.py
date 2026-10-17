import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from rna_pred.experiments.low_expression_features.features import (Intervals, body_pieces, candidate_indices,
                                   sequence_features, summarize_candidates, TRACKS)


def test_interval_union_half_open_and_gaps():
    index = Intervals([10, 15, 30], [20, 25, 40])
    assert index.overlap(0, 100) == 25
    assert index.overlap(25, 30) == 0
    assert index.overlap(18, 35) == 12
    assert Intervals().overlap(-10, 100) == 0


def test_candidates_match_nearest_selection_and_exclude_promoter():
    settings = dict(radius=10000, promoter_bp=2048, candidate_bp=512, max_candidates=2)
    summits = np.array([0, 3000, 5000, 7000, 10000])
    eligible, kept = candidate_indices(summits, 5000, settings)
    np.testing.assert_array_equal(eligible, [0, 1, 3, 4])
    np.testing.assert_array_equal(kept, [1, 3])
    assert len(candidate_indices(np.array([], dtype=int), 5000, settings)[0]) == 0


def test_promoter_exclusion_empty_and_displaced_tss():
    assert body_pieces(0, 10000, 0) == [(1024, 10000)]
    assert body_pieces(0, 10000, 10000) == [(0, 8976)]
    assert body_pieces(4000, 6000, 5000) == []
    assert body_pieces(0, 10000, 20000) == [(0, 10000)]


def test_empty_is_distinct_from_inactive_and_uncovered():
    summits = np.array([5000])
    values = np.zeros((1, len(TRACKS)))
    flags = {"activity_supported": np.zeros(1, dtype=bool)}
    empty = summarize_candidates(np.array([], dtype=int), 0, summits, values, values, flags, "all")
    inactive = summarize_candidates(np.array([0]), 0, summits, values, np.ones_like(values), flags, "all")
    assert empty["enhancer/all/none"] == 1
    assert inactive["enhancer/all/none"] == 0
    assert inactive["enhancer/all/activity_supported_none"] == 1
    assert inactive["enhancer/all/DNase/coverage"] == 1
    assert empty["enhancer/all/DNase/coverage"] == 0


def test_sequence_ambiguity_and_cpg_denominators():
    values = sequence_features("ACGTCGNN")
    assert values["gc"] == 4/6
    assert values["ambiguous_fraction"] == .25
    assert values["cpg_density"] == 2/7
    assert sequence_features("NNNN")["gc"] == 0


def test_screen_folds_keep_cells_and_chromosomes_together():
    from rna_pred.experiments.low_expression_features.screens import chromosome_folds, screen_conditions
    frame=pd.DataFrame({"chr":["chr2","chr2","chr3","chr3","chr4","chr4"],"cell":["X1","X2"]*3})
    folds,_=chromosome_folds(frame)
    np.testing.assert_array_equal(folds,[0,0,1,1,2,2])
    conditions=screen_conditions(["gene_length","H3K36me3/body/0/signal"],["enhancer/all/count","body/width","promoter/gc"])
    assert conditions["remove_existing_body"]==["gene_length"]
    assert "body/width" not in conditions["enhancer_counts"]


def test_undefined_ranking_and_source_threshold_stay_explicit():
    from rna_pred.experiments.low_expression_features.exploration import flat_metrics
    from rna_pred.common.metrics import fit_thresholds
    source=pd.DataFrame({"cell":["X2"]*5,"gex":[0,.1,.3,1,100]})
    target=pd.DataFrame({"cell":["X1"]*4,"gex":[0,.1,.1,100]})
    result=flat_metrics(target,np.arange(4.),fit_thresholds(source,.25)).iloc[0]
    assert result.threshold < 1
    assert result.distinct_low == 1
    assert np.isnan(result.low_spearman)
    assert result.zero_low_auc == 1


def test_genomic_resampling_keeps_matched_cells_paired():
    from rna_pred.experiments.low_expression_features.results import bootstrap_indices
    blocks=np.array(["a","a","b","b","c","c"])
    indices=bootstrap_indices(blocks,np.random.default_rng(42))
    for k in range(0,len(indices),2):
        assert indices[k]//2 == indices[k+1]//2
        assert indices[k]%2 == 0 and indices[k+1]%2 == 1


def test_paired_intervals_same_scores_and_discordant_tss(monkeypatch):
    import rna_pred.experiments.low_expression_features.results as module
    genes=np.tile([f"gene{i}" for i in range(6)],2)
    frame=pd.DataFrame({"gene_name":genes,"cell":["X1"]*6+["X2"]*6,"chr":["chr2"]*12,
                        "TSS_start":np.r_[np.arange(6)*5000000+4999990,np.arange(6)*5000000+5000010],
                        "gex":np.tile(np.arange(6),2)})
    frame["TSS_end"]=frame.TSS_start+2
    original=module.bootstrap_indices
    def checked(blocks,rng):
        assert np.array_equal(blocks[:6],blocks[6:])
        return original(blocks,rng)
    monkeypatch.setattr(module,"bootstrap_indices",checked)
    scores=np.array([np.tile(np.arange(6),2)],dtype=float)
    ci,_=module.paired_intervals(frame,scores,scores,{"pooled":3.,"per_cell":{}},repeats=3)
    assert np.allclose(ci.ci_low.dropna(),0)
    assert np.allclose(ci.ci_high.dropna(),0)


@pytest.mark.parametrize("command",["screen","activity-screen"])
def test_frozen_training_selection_cannot_be_overwritten(tmp_path,monkeypatch,command):
    from rna_pred.experiments.low_expression_features.run import main
    record=tmp_path/"training_submission.json"
    record.write_text('{"frozen": true}\n')
    config=Path(__file__).parents[1]/"code/rna_pred/experiments/chromatin_representation/configs/default.yaml"
    monkeypatch.setattr("sys.argv",["investigation",command,"--config",str(config),"--root",str(tmp_path)])
    with pytest.raises(ValueError,match="selection is frozen"):
        main()
    assert record.read_text()=='{"frozen": true}\n'
