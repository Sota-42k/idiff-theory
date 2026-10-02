"""Small deterministic fixtures for the classifier-checkpoint supplement (CLASSIFIER-CHECKPOINT-PLAN.md)."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from recovery_estimators import experiment, logodds
from recovery_estimators.checkpoint_experiment import classifier_checkpoint_rows
from recovery_estimators.checkpoint_records import (CHECKPOINT_SCHEMA_VERSION,
                                                     load_checkpoint_results,
                                                     validate_checkpoint_supplement)
from recovery_estimators.logodds_checkpoints import train_with_checkpoints
from recovery_estimators.networks import NetworkConfig, state_hash
from recovery_estimators.records import digest, identifier, write

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    c = yaml.safe_load((ROOT/'configs/estimators.yaml').read_text())
    c.update(seeds=[0], widths=[8], case='artanh', sizes=dict(fitting=12, calibration=12, test=12))
    c['network'].update(hidden_layers=1, embedding_dim=4)
    c['training'].update(updates=4, batch_size=8)
    c['dsm']['evaluate_at'] = [1, 2]
    c['potential'].update(n_features=4, ridge=1e-4, chunk_size=64)
    c['convex'].update(n_features=[4], max_iterations=50)
    return c


def logodds_config(config, width):
    return logodds.LogOddsConfig(**config['training'],
                                 network=NetworkConfig(**{**config['network'], 'width': width}))


def test_train_with_checkpoints_matches_plain_training(config):
    """Snapshotting must not perturb the trajectory: the final state hash is unchanged."""
    evaluation = experiment.Evaluation.generate(config, 0)
    width = config['widths'][0]
    cfg = logodds_config(config, width)
    conditions = evaluation.case.params.q+1
    plain_model, plain_training = logodds.train(evaluation.observations, conditions, cfg, 0)
    checkpointed_model, snapshots, training = train_with_checkpoints(
        evaluation.observations, conditions, cfg, 0, (1, 2))
    assert state_hash(checkpointed_model) == state_hash(plain_model)
    assert training['final_state_sha256'] == plain_training['final_state_sha256']
    assert sorted(snapshots) == [1, 2]
    with pytest.raises(ValueError, match='evaluate_at'):
        train_with_checkpoints(evaluation.observations, conditions, cfg, 0, (2, 1))
    with pytest.raises(ValueError, match='evaluate_at'):
        train_with_checkpoints(evaluation.observations, conditions, cfg, 0, (0, 2))


def test_classifier_checkpoint_rows_cover_grid_and_verify_final_hash(config):
    evaluation = experiment.Evaluation.generate(config, 0)
    width = config['widths'][0]
    cfg = logodds_config(config, width)
    conditions = evaluation.case.params.q+1
    _, plain_training = logodds.train(evaluation.observations, conditions, cfg, 0)
    final_hash = plain_training['final_state_sha256']
    evaluate_at = tuple(config['dsm']['evaluate_at'])
    rows = classifier_checkpoint_rows(evaluation, config, 0, width, evaluate_at, final_hash)
    final = config['training']['updates']
    expected_updates = sorted(set(evaluate_at)|{final})
    assert sorted({r['update'] for r in rows}) == expected_updates
    assert {r['route_evaluated'] for r in rows} == {'direct', 'through_the_potential'}
    assert len(rows) == len(expected_updates)*2
    by_update: dict[int, set[str]] = {}
    for r in rows:
        assert r['estimator'] == 'logodds' and r['width'] == width
        assert r['evaluated_update'] == r['update']
        by_update.setdefault(r['update'], set()).add(r['evaluated_state_sha256'])
    assert all(len(v) == 1 for v in by_update.values())
    assert len({next(iter(v)) for v in by_update.values()}) == len(by_update)
    final_rows = [r for r in rows if r['update'] == final]
    assert all(r['evaluated_state_sha256'] == final_hash for r in final_rows)
    with pytest.raises(ValueError, match='replay'):
        classifier_checkpoint_rows(evaluation, config, 0, width, evaluate_at, '0'*64)


def fake_base_record(config, seed=0, logodds_hash='a'*64):
    width = config['widths'][0]
    rows = [
        dict(estimator='dsm', width=width, update=config['training']['updates'],
             evaluated_update=config['training']['updates'], route_evaluated='through_the_potential',
             evaluated_state_sha256='d'*64, final_state_sha256='d'*64, affine_nmse=.3,
             field_error=1., raw_readout_mse=.2, oracle_aligned_mse=.4),
        dict(estimator='logodds', width=width, update=config['training']['updates'],
             evaluated_update=config['training']['updates'], route_evaluated='through_the_potential',
             evaluated_state_sha256=logodds_hash, affine_nmse=.1, field_error=.05,
             raw_readout_mse=.2, oracle_aligned_mse=.3),
        dict(estimator='logodds', width=width, update=config['training']['updates'],
             evaluated_update=config['training']['updates'], route_evaluated='direct',
             evaluated_state_sha256=logodds_hash, affine_nmse=.1, raw_readout_mse=.2,
             oracle_aligned_mse=.3),
    ]
    return dict(config=config, config_id=identifier(config), status='completed', seed=seed, rows=rows)


def fake_checkpoint_record(config, base, base_file, seed=0):
    width = config['widths'][0]
    final = config['training']['updates']
    final_hash = base['rows'][1]['evaluated_state_sha256']

    def row(update, route, hash_value):
        common = dict(estimator='logodds', width=width, update=update, evaluated_update=update,
                      route_evaluated=route, evaluated_state_sha256=hash_value, affine_nmse=.1,
                      raw_readout_mse=.2, oracle_aligned_mse=.3)
        if route == 'through_the_potential':
            common['field_error'] = .05
        return common

    rows = []
    for update, h in [(1, 'b'*64), (2, 'c'*64), (final, final_hash)]:
        rows.append(row(update, 'through_the_potential', h))
        rows.append(row(update, 'direct', h))
    return dict(config=config, config_id=identifier(config), schema_version=CHECKPOINT_SCHEMA_VERSION,
               mode='logodds-checkpoints', seed=seed, status='completed',
               base_seed_sha256=digest(base_file), rows=rows)


def checkpoint_files(tmp_path, config):
    base = tmp_path/'base'
    base.mkdir()
    record = fake_base_record(config)
    write(base/'seed0.json', record)
    write(base/'manifest.json', dict(config=config, config_id=identifier(config),
                                     runs={'0': dict(status='completed')}))
    supplement = tmp_path/'supplement'
    supplement.mkdir()
    fixed = fake_checkpoint_record(config, record, base/'seed0.json')
    write(supplement/'seed0.json', fixed)
    write(supplement/'manifest.json', dict(config=config, config_id=identifier(config),
                                          schema_version=CHECKPOINT_SCHEMA_VERSION,
                                          mode='logodds-checkpoints',
                                          base_manifest_sha256=digest(base/'manifest.json'),
                                          runs={'0': dict(status='completed')}))
    return base, supplement, fixed


def test_checkpoint_supplement_is_purely_additive(tmp_path, config):
    base, supplement, fixed = checkpoint_files(tmp_path, config)
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    records_out, manifest, notes = load_checkpoint_results(base, supplement, (1, 2))
    assert not notes
    rows = records_out[0]['rows']
    logodds_rows = [r for r in rows if r['estimator'] == 'logodds']
    assert sorted({r['update'] for r in logodds_rows}) == [1, 2, 4]
    finals = [r for r in logodds_rows if r['update'] == 4]
    assert len(finals) == 2  # from the base record, not duplicated from the supplement
    assert {p.name: p.read_bytes() for p in base.iterdir()} == before


def test_partial_checkpoint_supplement_notes_missing_seeds_without_raising(tmp_path, config):
    base, supplement, fixed = checkpoint_files(tmp_path, config)
    manifest = json.loads((supplement/'manifest.json').read_text())
    manifest['runs']['0']['status'] = 'failed'
    write(supplement/'manifest.json', manifest)
    records_out, _, notes = load_checkpoint_results(base, supplement, (1, 2))
    assert any('seed' in note.lower() for note in notes)
    assert sorted({r['update'] for r in records_out[0]['rows'] if r['estimator'] == 'logodds'}) == [4]


@pytest.mark.parametrize('damage', ['hash', 'seed', 'config', 'width', 'duplicate', 'extra',
                                    'state', 'nan', 'mismatch'])
def test_bad_checkpoint_supplement_is_rejected(tmp_path, config, damage):
    base, supplement, fixed = checkpoint_files(tmp_path, config)
    if damage == 'hash': fixed['base_seed_sha256'] = 'b'*64
    if damage == 'seed': fixed['seed'] = 1
    if damage == 'config': fixed['config'] = dict(config, n=3)
    if damage == 'width': fixed['rows'][0]['width'] = 512
    if damage == 'duplicate': fixed['rows'] = fixed['rows']+[fixed['rows'][0]]
    if damage == 'extra':
        fixed['rows'].append(dict(estimator='logodds', width=config['widths'][0], update=999,
                                  evaluated_update=999, route_evaluated='direct',
                                  evaluated_state_sha256='f'*64, affine_nmse=.1,
                                  raw_readout_mse=.2, oracle_aligned_mse=.3))
    if damage == 'state': fixed['rows'][0]['evaluated_state_sha256'] = 'e'*64
    if damage == 'nan': fixed['rows'][0]['affine_nmse'] = float('nan')
    if damage == 'mismatch': fixed['rows'][-1]['affine_nmse'] = 999.0
    (supplement/'seed0.json').write_text(json.dumps(fixed))
    with pytest.raises(ValueError):
        load_checkpoint_results(base, supplement, (1, 2))


def test_validate_checkpoint_supplement_direct(tmp_path, config):
    base, supplement, fixed = checkpoint_files(tmp_path, config)
    record = json.loads((base/'seed0.json').read_text())
    validate_checkpoint_supplement(fixed, record, config, base/'seed0.json', (1, 2))


def test_cli_dry_run_and_checkpoint_completion(tmp_path, config):
    config_path = tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    env = {**os.environ, 'PYTHONPATH': str(ROOT/'src')}
    base_output = tmp_path/'base'
    base_command = [sys.executable, str(ROOT/'scripts/run_estimators.py'), '--config', str(config_path),
                    '--output', str(base_output)]
    full = subprocess.run(base_command, capture_output=True, text=True, env=env)
    assert full.returncode == 0, full.stdout+full.stderr
    checkpoint_output = tmp_path/'checkpoints'
    command = [sys.executable, str(ROOT/'scripts/run_classifier_checkpoints.py'), '--config',
              str(config_path), '--base', str(base_output), '--output', str(checkpoint_output)]
    dry = subprocess.run(command+['--dry-run'], capture_output=True, text=True, env=env)
    assert dry.returncode == 0, dry.stderr
    assert not checkpoint_output.exists()
    before = {p.name: p.read_bytes() for p in base_output.iterdir()}
    run = subprocess.run(command, capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stdout+run.stderr
    assert {p.name: p.read_bytes() for p in base_output.iterdir()} == before
    records_out, _, notes = load_checkpoint_results(base_output, checkpoint_output,
                                                     tuple(config['dsm']['evaluate_at']))
    assert not notes
    logodds_rows = [r for r in records_out[0]['rows'] if r['estimator'] == 'logodds']
    expected = sorted(set(config['dsm']['evaluate_at'])|{config['training']['updates']})
    assert sorted({r['update'] for r in logodds_rows}) == expected
    repeat = subprocess.run(command, capture_output=True, text=True, env=env)
    assert repeat.returncode != 0 and 'already exists' in repeat.stderr
