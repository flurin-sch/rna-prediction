import numpy as np
import pandas as pd
import pytest
import torch

from rna_pred.common.data.cache import fetch_window, one_hot, pool_regions
from rna_pred.common.data.dataset import fit_normalizer
from rna_pred.baseline.model import ExpressionModel, promoter_attention_mask
from rna_pred.common.data.regions import load_peak_index, overlap_weights, select_regions, tss_anchor


def test_candidate_limits_overlap_ties_and_boundaries(tmp_path):
    settings = dict(promoter_bp=2048, candidate_bp=512, radius=100000, max_candidates=2)
    index = {"chr2": np.array([0, 98000, 99000, 100000, 102000, 200000, 200001])}
    result = select_regions("chr2", 100000, index, settings)
    assert result[0] == (98976, 101024, 100000)
    assert [r[2] for r in result[1:]] == [98000, 102000]
    settings["max_candidates"] = 60
    assert [r[2] for r in select_regions("chr2", 100000, index, settings)[1:]] == [98000, 102000, 0, 200000]
    assert len(select_regions("chr1", 100000, index, settings)) == 1
    path = tmp_path / "peaks.bed"
    path.write_text("chr2\t100\t200\t.\t0\t.\t1\t-1\t-1\t-1\nchr2\t100\t200\t.\t0\t.\t1\t-1\t-1\t50\n")
    assert load_peak_index(path)["chr2"].tolist() == [150]
    assert tss_anchor(100, 151) == 125


def test_partial_bin_pooling_and_coordinate_frame():
    weights = overlap_weights(-128, 2048, -64, 144)
    assert weights[:3].tolist() == [64, 128, 16]
    hidden = torch.arange(16).float()[:, None].expand(16, 3072)
    regions = pd.DataFrame([dict(region_id="test", start=-64, end=144)])
    pooled = pool_regions(hidden, -128, 2048, regions)
    np.testing.assert_allclose(pooled, (128 + 32) / 208)
    with pytest.raises(ValueError, match="within"):
        overlap_weights(0, 2048, -1, 128)


def test_reference_padding_and_ambiguous_base_encoding():
    sequence, fraction = fetch_window({"chr1": "ACGTN"}, "chr1", -3, 8)
    assert sequence == "NNNACGTNNNN"
    assert fraction == 5 / 11
    encoded = one_hot(sequence)
    assert encoded.sum() == 4
    np.testing.assert_array_equal(encoded[3:7], np.eye(4))
    with pytest.raises(ValueError, match="lacks"):
        fetch_window({"chr1": "ACGT"}, "chr2", 0, 4)


def test_normalization_excludes_validation_and_padding():
    mapping = pd.DataFrame({"example_id": ["train", "train", "val"], "DNase": [0., 3., 1000000.]})
    normalizer = fit_normalizer(mapping, ["train"], ["DNase"])
    assert normalizer["mean"] == pytest.approx([np.log(4) / 2])
    assert normalizer["scale"] == pytest.approx([np.log(4) / 2])


def test_mask_padding_permutation_and_promoter_only():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = ExpressionModel(7, embedding_dim=12).eval()
    sequence, features = torch.randn(2, 5, 12), torch.randn(2, 5, 7)
    valid = torch.tensor([[True, True, True, False, False], [True, False, False, False, False]])
    mask = promoter_attention_mask(5)
    assert not mask[0].any() and not mask[:, 0].any() and not mask.diag().any()
    assert mask[1, 2] and mask[2, 1]
    with torch.no_grad():
        original = model(sequence, features, valid)
        altered_sequence, altered_features = sequence.clone(), features.clone()
        altered_sequence[~valid], altered_features[~valid] = float("nan"), float("inf")
        torch.testing.assert_close(original, model(altered_sequence, altered_features, valid), rtol=0, atol=0)
        permutation = [0, 3, 2, 4, 1]
        torch.testing.assert_close(original, model(sequence[:, permutation], features[:, permutation], valid[:, permutation]))
        assert torch.isfinite(original).all()
        valid[:, 0] = False
        with pytest.raises(ValueError, match="promoter"):
            model(sequence, features, valid)

