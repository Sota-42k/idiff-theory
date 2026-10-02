"""The contextual large-basis probe must never change the frozen-grid selection."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'selection', Path(__file__).resolve().parents[1]/'scripts/select_potential.py')
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


def rows(values):
    return [dict(case='mixing', n_features=width, ridge=1e-8, oracle_aligned_mse=value)
            for width, value in values]


def test_probe_better_than_every_candidate_is_only_context():
    result = selection.choose_setting(rows([(1024, 2.), (2048, 1.), (8192, .001)]),
                                      'mixing', [(1024, 1e-8), (2048, 1e-8)], [(8192, 1e-8)])
    assert result['n_features'] == 2048
    assert '8192:1e-08' not in result['grid']
    assert result['reference_only_not_candidates']['8192:1e-08'] == pytest.approx(.001)


def test_declared_five_percent_tie_chooses_smaller_basis():
    result = selection.choose_setting(rows([(1024, 1.04), (2048, 1.)]),
                                      'mixing', [(1024, 1e-8), (2048, 1e-8)], [])
    assert result['n_features'] == 1024


def test_missing_or_nonfinite_cell_is_not_silently_selected():
    for values in ([], [(1024, float('nan'))], [(1024, 0.)]):
        with pytest.raises(ValueError, match='finite positive'):
            selection.choose_setting(rows(values), 'mixing', [(1024, 1e-8)], [])
