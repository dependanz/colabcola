import importlib.util
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import tune_quality as tune


def test_split_overlap_rejected_and_fixed_split_valid():
    tune.validate_splits(tune.SPLITS)
    with pytest.raises(ValueError, match='overlapping'):
        tune.validate_splits({'train': ('a', 0, 10), 'validation': ('a', 9, 3)})


def test_selection_uses_validation_only_and_deterministic_tie():
    rows = [{'name': 'b', 'validation_mse': .2, 'test_mse': 0.0},
            {'name': 'z', 'validation_mse': .1, 'test_mse': 1.0},
            {'name': 'a', 'validation_mse': .1, 'test_mse': 2.0}]
    assert tune.select(rows)[0]['name'] == 'a'
    with pytest.raises(ValueError, match='nonfinite'):
        tune.select([{'name': 'bad', 'validation_mse': float('nan')}])


def test_edge_metric_separates_brightness_from_lost_edge():
    truth = torch.zeros(1, 3, 4, 4)
    truth[:, :, :, 2:] = 1
    mse, edge = tune.image_errors(truth + .1, truth)
    assert mse.item() == pytest.approx(.01)
    assert edge.item() == pytest.approx(0, abs=1e-12)
    mse, edge = tune.image_errors(torch.full_like(truth, .5), truth)
    assert mse.item() == pytest.approx(.25)
    assert edge.item() == pytest.approx(1 / 6)

def test_assessment_separates_generated_and_trainable_parameters(tmp_path):
    cfg = {'width': 32, 'latent': 64, 'activation': 'hosc', 'hosc_beta': 1., 'hosc_omega': 3.}
    net = tune.workload.model(**cfg)
    checkpoint = tune.workload.save_checkpoint(tmp_path, {
        'schema': 1, 'step': 1, 'identity': {'config': cfg}, 'model': net.state_dict()})
    measured, predicted = tune.assess(checkpoint, torch.zeros(2, 3, 8, 8))
    assert measured['generated_parameters'] == 1763
    assert measured['trainable_parameters'] == 152515
    assert predicted.shape == (2, 3, 8, 8)
