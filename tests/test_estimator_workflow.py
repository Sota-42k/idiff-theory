"""Small deterministic fixtures for the final-iterate bug and supplement lifecycle."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from recovery_estimators import experiment
from recovery_estimators.networks import state_hash
from recovery_estimators.records import (base_records, digest, identifier, load_results,
                                         provenance, run_directory, validate_supplement, write)
from recovery_estimators.report import render

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


def test_final_iterate_survives_loading_earlier_snapshots(config, monkeypatch):
    """Final-only and budget-curve paths must evaluate the independently trained final state."""
    evaluation = experiment.Evaluation.generate(config, 0)
    real_train = experiment.dsm.train
    final_scores = []

    def capture(*args):
        model, snapshots, training = real_train(*args)
        final_scores.append(experiment.dsm.scores(model, evaluation.test.x, args[2].t_min,
                                                  args[3], args[2]))
        assert state_hash(model) == training['final_state_sha256']
        return model, snapshots, training

    monkeypatch.setattr(experiment.dsm, 'train', capture)
    rows, _ = experiment.dsm_rows(evaluation, config, 0, 8)
    single, _ = experiment.dsm_rows(evaluation, config, 0, 8, final_only=True,
                                    expected_final_hash=rows[-1]['final_state_sha256'])
    assert [r['update'] for r in rows] == [1, 2, 4]
    assert rows[-1]['evaluated_state_sha256'] == single[0]['evaluated_state_sha256']
    assert rows[1]['evaluated_state_sha256'] != rows[-1]['evaluated_state_sha256']
    assert rows[-1]['field_error'] == pytest.approx(
        experiment.field_error(evaluation.case, evaluation.test.x, final_scores[0]))
    for key in ('affine_nmse', 'raw_readout_mse', 'oracle_aligned_mse', 'field_error'):
        assert rows[-1][key] == pytest.approx(single[0][key], abs=1e-12)
    config['dsm']['evaluate_at'] = [1, 4]
    deduped, _ = experiment.dsm_rows(evaluation, config, 0, 8)
    assert [r['update'] for r in deduped] == [1, 4]
    with pytest.raises(ValueError, match='replay'):
        experiment.dsm_rows(evaluation, config, 0, 8, final_only=True, expected_final_hash='0'*64)


def fake_record(config, seed=0):
    common = dict(estimator='dsm', width=8, route_evaluated='through_the_potential',
                  affine_nmse=.3, field_error=1., raw_readout_mse=.2, oracle_aligned_mse=.4,
                  final_state_sha256='a'*64)
    return dict(config=config, config_id=identifier(config), status='completed', seed=seed,
                analytic_reference=dict(affine_nmse=.001), rows=[
                    dict(common, update=2), dict(common, update=4),
                    dict(estimator='logodds', width=8, update=4, route_evaluated='direct', affine_nmse=.1)])


def files(tmp_path, config):
    base = tmp_path/'base'
    base.mkdir()
    record = fake_record(config)
    write(base/'seed0.json', record)
    write(base/'manifest.json', dict(config=config, config_id=identifier(config),
                                     runs={'0': dict(status='completed')}))
    supplement = tmp_path/'supplement'
    supplement.mkdir()
    metadata = dict(config=config, config_id=identifier(config), schema_version=2, mode='dsm-final',
                    base_manifest_sha256=digest(base/'manifest.json'))
    fixed = dict(**metadata, seed=0, status='completed', base_seed_sha256=digest(base/'seed0.json'),
                 rows=[dict(record['rows'][1], evaluated_update=4, evaluated_state_sha256='a'*64)])
    write(supplement/'seed0.json', fixed)
    write(supplement/'manifest.json', dict(**metadata, runs={'0': dict(status='completed')}))
    return base, supplement, fixed


def test_report_excludes_legacy_final_and_overlays_only_valid_supplement(tmp_path, config):
    base, supplement, fixed = files(tmp_path, config)
    before = {p.name: p.read_bytes() for p in base.iterdir()}
    rows, manifest, notes = load_results(base)
    assert [r['update'] for r in rows[0]['rows'] if r['estimator']=='dsm'] == [2]
    text = render(rows, manifest, notes)
    assert 'Excluded 1 invalid legacy' in text and 'not measured (n=0)' in text
    corrected, manifest, notes = load_results(base, supplement)
    assert len(corrected[0]['rows']) == 3
    assert corrected[0]['rows'][-1]['evaluated_state_sha256'] == 'a'*64
    assert {p.name:p.read_bytes() for p in base.iterdir()} == before


@pytest.mark.parametrize('damage', ['hash','seed','config','width','duplicate','extra','state','nan','partial'])
def test_bad_supplement_is_rejected(tmp_path, config, damage):
    base, supplement, fixed = files(tmp_path, config)
    if damage=='hash': fixed['base_seed_sha256'] = 'b'*64
    if damage=='seed': fixed['seed'] = 1
    if damage=='config': fixed['config'] = dict(config, n=3)
    if damage=='width': fixed['rows'][0]['width'] = 512
    if damage=='duplicate': fixed['rows'] *= 2
    if damage=='extra': fixed['rows'].append(dict(estimator='logodds'))
    if damage=='state': fixed['rows'][0]['evaluated_state_sha256'] = 'b'*64
    if damage=='nan': fixed['rows'][0]['affine_nmse'] = float('nan')
    if damage=='partial':
        m=json.loads((supplement/'manifest.json').read_text())
        m['runs']['0']['status']='failed'
        write(supplement/'manifest.json',m)
    (supplement/'seed0.json').write_text(json.dumps(fixed))
    with pytest.raises(ValueError): load_results(base, supplement)


def test_supplement_trains_only_dsm(config, monkeypatch):
    base = fake_record(config)
    def forbidden(*args, **kwargs):
        pytest.fail('Final-only path invoked another experiment')
    monkeypatch.setattr(experiment, 'classifier_rows', forbidden)
    monkeypatch.setattr(experiment, 'convex_rows', forbidden)
    monkeypatch.setattr(experiment.Evaluation, 'scores', forbidden)
    calls = []
    def dsm_only(*args, **kwargs):
        calls.append(kwargs)
        return [dict(estimator='dsm', update=4)], []
    monkeypatch.setattr(experiment, 'dsm_rows', dsm_only)
    record = experiment.run_seed(config,0,base=base)
    assert 'analytic_reference' not in record
    assert calls == [dict(final_only=True, expected_final_hash='a'*64)]


def test_run_directory_preserves_other_seeds_and_checks_provenance(tmp_path, config):
    directory=tmp_path/'run'
    metadata=provenance(config)
    assert 'src/recovery_estimators/experiment.py' in metadata['source_hashes']
    assert metadata['environment']['training_dtype']=='float32'
    with run_directory(directory, metadata) as manifest:
        manifest['runs']['0']=dict(status='completed')
        write(directory/'manifest.json',manifest)
        with pytest.raises(FileExistsError):
            with run_directory(directory,metadata): pass
    with run_directory(directory,metadata) as manifest:
        assert manifest['runs']['0']['status']=='completed'
    with pytest.raises(ValueError,match='source_hashes'):
        with run_directory(directory,dict(metadata,source_hashes={})): pass
    assert not (directory/'.run.lock').exists()


def test_cli_dry_run_and_small_final_completion(tmp_path, config):
    config_path=tmp_path/'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    output=tmp_path/'full'
    command=[sys.executable,str(ROOT/'scripts/run_estimators.py'),'--config',str(config_path)]
    import os
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    dry=subprocess.run(command+['--output',str(output),'--dry-run'],capture_output=True,text=True,env=env)
    assert dry.returncode==0,dry.stderr
    assert not output.exists()
    full=subprocess.run(command+['--output',str(output)],capture_output=True,text=True,env=env)
    assert full.returncode==0,full.stdout+full.stderr
    report=tmp_path/'report.md'
    analysis=[sys.executable,str(ROOT/'scripts/analyse_estimators.py'),
              '--results',str(output),'--out',str(report)]
    first=subprocess.run(analysis,capture_output=True,text=True,env=env)
    assert first.returncode==0,first.stderr
    content=report.read_bytes()
    second=subprocess.run(analysis,capture_output=True,text=True,env=env)
    assert second.returncode==0,second.stderr
    assert report.read_bytes()==content
    assert report.read_text().count('# Which estimator reaches the construction')==1
    old={p.name:p.read_bytes() for p in output.iterdir()}
    supplement=tmp_path/'final'
    repair=subprocess.run(command+['--complete-final-from',str(output),'--output',str(supplement)],
                          capture_output=True,text=True,env=env)
    assert repair.returncode==0,repair.stdout+repair.stderr
    rows,_,_=load_results(output,supplement)
    assert {p.name:p.read_bytes() for p in output.iterdir()}==old
    stored=json.loads((supplement/'seed0.json').read_text())
    assert len(stored['rows'])==1 and stored['rows'][0]['update']==4
    repeated=subprocess.run(command+['--complete-final-from',str(output),'--output',str(supplement)],
                            capture_output=True,text=True,env=env)
    assert repeated.returncode!=0 and 'already exists' in repeated.stderr


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'bad_hash'])
def test_preflight_rejects_unusable_final_reference(tmp_path, config, damage):
    base, _, _ = files(tmp_path, config)
    record=json.loads((base/'seed0.json').read_text())
    if damage=='missing': record['rows'].pop(1)
    if damage=='duplicate': record['rows'].append(record['rows'][1])
    if damage=='bad_hash': record['rows'][1]['final_state_sha256']='bad'
    write(base/'seed0.json',record)
    with pytest.raises(ValueError, match='Base'):
        base_records(base,config)


def test_failed_seed_retry_keeps_completed_seed_and_attempt_history(tmp_path, config, monkeypatch):
    import importlib.util
    config['seeds'] = [0, 1]
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    output = tmp_path/'run'
    spec = importlib.util.spec_from_file_location('estimator_cli', ROOT/'scripts/run_estimators.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    calls = []

    def run(config, seed, **kwargs):
        calls.append(seed)
        if seed == 1 and calls.count(1) == 1:
            raise ValueError('intentional test failure')
        record = fake_record(config, seed)
        record['rows'][1].update(evaluated_update=4, evaluated_state_sha256='a'*64)
        return record

    monkeypatch.setattr(cli, 'run_seed', run)
    command = ['run_estimators', '--config', str(path), '--output', str(output)]
    monkeypatch.setattr(sys, 'argv', command)
    with pytest.raises(SystemExit, match='Failed seeds'):
        cli.main()
    before = (output/'seed0.json').read_bytes()
    assert not (output/'seed1.json').exists()
    monkeypatch.setattr(sys, 'argv', command+['--seeds', '1'])
    cli.main()
    manifest = json.loads((output/'manifest.json').read_text())
    assert calls == [0, 1, 1]
    assert (output/'seed0.json').read_bytes() == before
    assert all(r['status']=='completed' for r in manifest['runs'].values())
    assert manifest['previous_attempts']['1'][0]['status']=='failed'
    assert 'intentional test failure' in manifest['previous_attempts']['1'][0]['failure']
