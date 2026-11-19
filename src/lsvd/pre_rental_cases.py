"""Measurements called only by the bounded pre_rental_probe entry."""

import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import time
import urllib.request


def empty_record():
    fields = json.loads(Path(__file__).with_name('probe_fields.json').read_text())
    return {section: {name: None for name in fields[section]}
              for section in ['protocol', 'throughput', 'timings_seconds',
                              'resources', 'correctness_and_retrieval']}


def record(args, model):
    from .pre_rental_deploy import code_manifest
    from .two_tower_prepare import sha256_file
    from .two_tower_runtime import process_metrics, available_resources

    result = empty_record()
    result['protocol'].update(source_sha=code_manifest(Path(__file__).resolve().parents[2])['source_sha256'],
                               data_sha=(hashlib.sha256(b'synthetic-round30-seed42').hexdigest()
                                         if args.synthetic else sha256_file(args.dataset / 'manifest.json')),
                               seed=42, model=model)
    result['resources'].update(process_metrics(args.device))
    result['resources'].update(available_resources())
    result.update(status='complete', execution='local_cpu_smoke' if args.local_smoke else 'cuda_probe')
    return result


def load_case_data(args, config, *, placement='cpu'):
    from .two_tower_runtime import configure_threads
    configure_threads(args.threads)
    if args.synthetic:
        from .two_tower_data import synthetic
        from .two_tower_tensors import TensorCatalog, TensorEventStore
        catalog, store, data, _ = synthetic(config, user_count=64, item_count=1500)
        catalog = TensorCatalog.from_catalog(catalog, placement,
                                             args.device == 'cuda' and placement == 'cpu')
        store = TensorEventStore.from_store(store, catalog, placement,
                                             args.device == 'cuda' and placement == 'cpu')
        return catalog, store, data
    from .two_tower_prepare import load_tensor_dataset
    _, catalog, store, data = load_tensor_dataset(
        args.dataset, config, storage_device=placement,
        pinned=args.device == 'cuda' and placement == 'cpu')
    return catalog, store, data


def prepare_shared_order(args, output):
    import numpy as np
    from .two_tower_tensors import disk_shuffle

    count = 64 * 40 if args.synthetic else json.loads(
        (args.dataset / 'manifest.json').read_text())['counts']['training_targets']
    rng = np.random.default_rng(42)
    before = copy.deepcopy(rng.bit_generator.state)
    started = time.perf_counter()
    path = output / 'shared-order.npy'
    order = disk_shuffle(path, count, rng)
    del order
    state = dict(seed=42, before=before, after=rng.bit_generator.state,
                  seconds=time.perf_counter()-started, targets=count)
    with (output / 'shared-order-state.json').open('x') as handle:
        json.dump(state, handle, indent=2)
    return path


def reuse_order(session, path):
    if path is None:
        return
    import numpy as np
    state = json.loads(path.with_name('shared-order-state.json').read_text())
    session.order = np.load(path, mmap_mode='r')
    session.shuffle_before = state['before']
    session.order_rng.bit_generator.state = state['after']


def machine(args):
    import torch
    from .two_tower_runtime import configure_threads, process_metrics

    limits = configure_threads(args.threads)
    result = record(args, 'machine_inventory')
    result['resources'].update(limits)
    result['resources'].update(process_metrics(args.device))
    result['resources']['gpu_vram_bytes'] = (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None)
    result['resources']['gpu_name'] = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    result['cuda_runtime'] = torch.version.cuda
    result['torch_version'] = torch.__version__
    result['cuda_arch_list'] = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    result['cuda_available'] = torch.cuda.is_available()
    result['gpu_count'] = torch.cuda.device_count()
    if torch.cuda.is_available():
        completed = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,power.limit,driver_version,clocks.sm,clocks.mem',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
        rows = [line.split(',') for line in completed.stdout.strip().splitlines()]
        result['nvidia_smi_query_succeeded'] = completed.returncode == 0
        if len(rows) == 1:
            values = [value.strip() for value in rows[0]]
            result['driver_version'] = values[3]
            for key, index in [('gpu_power_limit_watts', 2), ('sm_clock_mhz', 4),
                               ('memory_clock_mhz', 5)]:
                try:
                    result['resources'][key] = float(values[index])
                except ValueError:
                    pass
        else:
            result['gpu_mapping_note'] = 'multiple nvidia-smi rows; per-device power/clock UNKNOWN'
    path = args.output / 'disk-read.bin'
    size = 16 * 1024**2 if args.local_smoke else 256 * 1024**2
    with path.open('xb') as handle:
        block = bytes(1024**2)
        for _ in range(size // len(block)):
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    fio = shutil.which('fio') if not args.local_smoke else None
    if fio:
        disk = {}
        for mode, block_size in [('read', '1M'), ('randread', '4k')]:
            completed = subprocess.run([fio, '--name=lsvd-read', '--filename=' + str(path),
                '--rw=' + mode, '--bs=' + block_size, '--direct=1', '--ioengine=sync',
                '--runtime=10', '--time_based=1', '--output-format=json'],
                capture_output=True, text=True, timeout=30, check=True)
            disk[mode] = json.loads(completed.stdout)['jobs'][0]['read']
        result['resources']['disk_seq_mb_per_second'] = disk['read']['bw_bytes'] / 1e6
        result['resources']['disk_random_iops'] = disk['randread']['iops']
        percentile = disk['randread']['clat_ns']['percentile']
        result['resources']['disk_p50_p95_ms'] = [percentile['50.000000'] / 1e6,
                                                  percentile['95.000000'] / 1e6]
        result['disk_method'] = 'fio direct=1 sequential 1MiB and random 4KiB QD1'
    else:
        descriptor = os.open(path, os.O_RDONLY)
        started = time.perf_counter()
        for offset in range(0, size, 1024**2):
            os.pread(descriptor, 1024**2, offset)
        result['resources']['disk_seq_mb_per_second'] = size / (time.perf_counter()-started) / 1e6
        rng = random.Random(42)
        latencies = []
        started = time.perf_counter()
        for _ in range(256):
            before = time.perf_counter()
            os.pread(descriptor, 4096, rng.randrange(size // 4096) * 4096)
            latencies.append((time.perf_counter()-before)*1000)
        result['resources']['disk_random_iops'] = 256 / (time.perf_counter()-started)
        os.close(descriptor)
        latencies.sort()
        result['resources']['disk_p50_p95_ms'] = [latencies[128], latencies[243]]
        result['disk_method'] = 'Python pread, warm/page-cache measurement'
    if args.synthetic:
        result['network_download_status'] = 'skipped_synthetic_test'
    elif not args.download_url:
        result['network_download_status'] = 'skipped_disabled'
    else:
        try:
            started = time.perf_counter()
            count = 0
            with urllib.request.urlopen(args.download_url, timeout=20) as response:
                for block in iter(lambda: response.read(1024**2), b''):
                    count += len(block)
                    if count >= 32 * 1024**2:
                        break
            result['network_download_mbps'] = count * 8 / (time.perf_counter()-started) / 1e6
            result['network_download_bytes'] = count
        except Exception as error:
            result['network_download_status'] = type(error).__name__
    return result


def _fixed_loss(model, catalog, store, config, selected, device):
    import torch
    from .two_tower import corrected_loss
    from .two_tower_data import encode_batch

    batch, ids, positives, q, weights = store.train_batch(selected, config, device)
    users = encode_batch(model, catalog, batch, device)
    items = model.item(batch['target_features'])
    loss, _ = corrected_loss(users, items, ids.long(), positives, q, weights,
                             config.temperature)
    return loss, users, items


def fp32_snapshot(model, catalog, store, config, selected, device):
    import torch
    from .two_tower import corrected_loss
    from .two_tower_data import encode_batch
    from .two_tower_runtime import precision_mode

    with torch.no_grad(), precision_mode(device, 'fp32'):
        batch, ids, positives, q, weights = store.train_batch(selected, config, device)
        mask = positives.allowed(ids)
        users = encode_batch(model, catalog, batch, device)
        items = model.item(batch['target_features'])
        loss, skipped = corrected_loss(users, items, ids.long(), mask, q, weights,
                                       config.temperature)
    inputs = {f'history.{name}': value for name, value in batch.items()
              if name not in ('item_features', 'target_features')}
    for section in ('item_features', 'target_features'):
        inputs.update({f'{section}.{name}': value for name, value in batch[section].items()})
    inputs.update(target_item_ids=ids, positive_user_rows=positives.users, mask=mask,
                  logq_probability=q, target_weight=weights,
                  skipped_rows=torch.tensor(skipped))
    outputs = dict(loss=loss, users=users, items=items)
    return ({name: value.cpu() for name, value in inputs.items()},
            {name: value.cpu() for name, value in outputs.items()})


def compare_fp32(cpu, cuda):
    import torch

    cpu_inputs, cpu_outputs = cpu
    cuda_inputs, cuda_outputs = cuda
    mismatches = sorted(set(cpu_inputs) ^ set(cuda_inputs))
    for name in sorted(cpu_inputs.keys() & cuda_inputs.keys()):
        before, after = cpu_inputs[name], cuda_inputs[name]
        if before.dtype != after.dtype or not torch.equal(before, after):
            mismatches.append(name)
    differences = {}
    tolerance = 1e-4
    for name, before in cpu_outputs.items():
        after = cuda_outputs[name]
        comparable = (before.shape == after.shape and before.dtype == after.dtype
                      and bool(torch.isfinite(before).all() and torch.isfinite(after).all()))
        differences[name] = float((before - after).abs().max()) if comparable else None
    passed = not mismatches and all(
        difference is not None and difference <= tolerance for difference in differences.values()
    )
    return dict(status='pass' if passed else 'fail', severity='ok' if passed else 'red',
                tolerance=tolerance, checked_fields=sorted(cpu_inputs.keys() & cuda_inputs.keys()),
                mismatched_fields=mismatches, max_abs_diff=differences)


def fp32_cpu_cuda(args):
    import numpy as np
    import torch
    from .two_tower import TowerConfig, TwoTower

    if args.device != 'cuda':
        return dict(status='skipped_cpu')
    if not torch.cuda.is_available():
        return dict(status='cuda_unavailable')
    config = TowerConfig(batch_size=1024, id_min_count=20, seed=42)
    cpu_args = copy.copy(args)
    cpu_args.device = 'cpu'
    cpu_catalog, cpu_store, data = load_case_data(cpu_args, config)
    order = np.load(args.shared_order, mmap_mode='r')
    selected = order[:min(config.batch_size, len(cpu_store.samples))]
    torch.manual_seed(config.seed)
    model = TwoTower(cpu_catalog, config, 'causal').float().eval()
    # A nonzero gate makes the diagnostic sensitive to the attention residual.
    with torch.no_grad():
        model.user.gate.fill_(1.)
    cpu = fp32_snapshot(model, cpu_catalog, cpu_store, config, selected, 'cpu')
    del cpu_catalog, cpu_store, data

    cuda_catalog, cuda_store, _ = load_case_data(args, config, placement='cuda')
    cuda = fp32_snapshot(model.to('cuda'), cuda_catalog, cuda_store, config, selected, 'cuda')
    result = record(args, 'mlp_causal')
    result['protocol'].update(precision='fp32', batch_size=config.batch_size,
                              training_target_positions=len(selected),
                              id_min_count=config.id_min_count)
    result.update(compare_fp32(cpu, cuda), synthetic=args.synthetic,
                  batch_sha256=hashlib.sha256(selected.astype('<i8', copy=False)).hexdigest(),
                  weights='same seeded eval model with diagnostic causal gate set to 1',
                  data_placement='cuda')
    return result


def two_tower_case(args):
    import numpy as np
    import torch
    from .two_tower import TowerConfig
    from .two_tower_training import TensorTrainingSession
    from .two_tower_runtime import precision_mode, process_metrics
    from .two_tower_measure import phase_clock
    from .pre_rental_deploy import sha256_file

    spec = json.loads(args.case_file.read_text())
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    config = TowerConfig(batch_size=spec['batch_size'], id_min_count=spec['id_min_count'], seed=42)
    catalog, store, _ = load_case_data(args, config, placement=spec['data_placement'])
    session = TensorTrainingSession(catalog, store, config, spec['structure'],
                                     args.device, args.output / 'session',
                                     precision=spec['precision'])
    reuse_order(session, args.shared_order)
    result = record(args, spec['structure'])
    result['protocol'].update(precision=spec['precision'], batch_size=config.batch_size,
                               history_events=config.max_history,
                               training_target_positions=config.batch_size,
                               catalog_items=len(catalog.ids), id_min_count=config.id_min_count)
    result['data_placement'] = spec['data_placement']
    result['tower_config'] = config.__dict__
    result['warmup_steps'] = args.warmup
    result['measurement_steps'] = args.steps
    result['timings_seconds']['load'] = time.perf_counter()-started
    result['id_table_rows'] = catalog.id_count
    if args.shared_order is not None:
        result['timings_seconds']['epoch_shuffle'] = json.loads(
            args.shared_order.with_name('shared-order-state.json').read_text())['seconds']
    session._ensure_order()
    selected = session.order[:min(config.batch_size, len(store.samples))]
    session.model.eval()
    with torch.no_grad(), precision_mode(args.device, 'fp32'):
        loss32, users32, items32 = _fixed_loss(session.model, catalog, store, config,
                                               selected, args.device)
    if args.device == 'cuda' and torch.cuda.is_bf16_supported():
        with torch.no_grad(), precision_mode(args.device, 'bf16'):
            loss16, users16, items16 = _fixed_loss(session.model, catalog, store, config,
                                                   selected, args.device)
        result['fp32_bf16_loss_difference'] = abs(float(loss32)-float(loss16))
        result['correctness_and_retrieval']['fixed_batch_forward_max_difference'] = max(
            float((users32-users16).abs().max()), float((items32-items16).abs().max()))
        result['precision_comparison_mode'] = 'eval; same weights and batch, dropout inactive'
    else:
        result['fp32_bf16_loss_difference'] = None
        result['precision_comparison_mode'] = 'skipped_cuda_unavailable'
    del users32, items32, loss32
    for _ in range(args.warmup):
        session.step()
    cpu_before = time.process_time()
    started = time.perf_counter()
    rows = [session.step(profile=True) for _ in range(args.steps)]
    wall = time.perf_counter()-started
    result['throughput'].update(
        effective_targets_per_second=sum(row['targets']-row['skipped'] for row in rows)/wall,
        steps_per_second=len(rows)/wall,
        distinct_users_per_batch=float(np.mean([row['distinct_users'] for row in rows])))
    result['timings_seconds']['steady_window_wall'] = wall
    result['resources']['process_cpu_time_over_wall'] = (time.process_time()-cpu_before)/wall
    phases = {'batch_host': 'take_batch', 'history_features_host': 'history',
              'h2d_wait': 'h2d', 'forward_cuda': 'forward',
              'loss_mask_cuda_or_host': 'mask', 'backward_cuda': 'backward',
              'optimizer_cuda': 'optimizer'}
    result['phase_clocks'] = {}
    for field, phase in phases.items():
        clock = phase_clock(phase, args.device, spec['data_placement'])
        result['phase_clocks'][field] = clock
        values = [row['timings'][clock]
                  .get(phase, 0.) for row in rows]
        result['timings_seconds'][field] = float(np.median(values))
    result['feature_lookup_host_seconds'] = float(np.median([
        row['timings']['host_seconds']['feature_lookup'] for row in rows]))
    result['cross_entropy_and_logq_seconds'] = float(np.median([
        row['timings']['host_seconds']['loss'] for row in rows]))
    result['correctness_and_retrieval']['finite_loss'] = all(math.isfinite(row['loss']) for row in rows)
    if not result['correctness_and_retrieval']['finite_loss']:
        result['status'] = 'failed_nonfinite'
    result['resources'].update(process_metrics(args.device))
    result['steps'] = rows
    if (result['status'] == 'complete' and args.device == 'cuda'
            and spec == dict(structure='mlp_causal', precision='bf16', id_min_count=20,
                             data_placement='cuda', batch_size=1024)):
        path = args.output / 'reference-model.pt'
        with path.open('xb') as handle:
            torch.save(dict(model=session.model.state_dict(), config=config.__dict__,
                             probe_updates=session.global_step), handle)
        result.update(reference_sha256=sha256_file(path),
                      reference_probe_updates=session.global_step)
    return result


def generator_case(args):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from vkgr import tiger_arch
    from .two_tower_measure import StepTimer
    from .two_tower_runtime import configure_threads, precision_mode, process_metrics

    configure_threads(args.threads)
    spec = json.loads(args.case_file.read_text())
    if args.device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)
    previous_positions = tiger_arch.POSITIONS
    tiger_arch.POSITIONS = spec['positions']
    try:
        model = tiger_arch.TigerArchModel(args.generator_vocab, d_model=128,
                    heads=6, head_dim=64, encoder_layers=4, decoder_layers=4,
                    ff=1024, dropout=.1).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        source = torch.randint(1, args.generator_vocab,
                                (spec['batch_size'], spec['positions']), device=args.device)
        decoder = torch.randint(1, args.generator_vocab,
                                 (spec['batch_size'], args.decoder_positions), device=args.device)
        labels = torch.randint(1, args.generator_vocab, decoder.shape, device=args.device)
        rows = []
        started = None
        for step in range(args.warmup + args.steps):
            if step == args.warmup:
                started = time.perf_counter()
            timer = StepTimer(args.device)
            with precision_mode(args.device, spec['precision']):
                with timer.region('forward'):
                    logits = model(source, decoder)
                with timer.region('loss'):
                    loss = F.cross_entropy(logits.float().reshape(-1, args.generator_vocab),
                                            labels.reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                with timer.region('backward'):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                with timer.region('optimizer'):
                    optimizer.step()
            timing = timer.finish()
            if step >= args.warmup:
                rows.append(dict(loss=float(loss.detach()), timing=timing))
        wall = time.perf_counter()-started
        result = record(args, 'legacy_A1_synthetic')
        result['protocol'].update(precision=spec['precision'], batch_size=spec['batch_size'],
                                   history_events=spec['positions'],
                                   training_target_positions=spec['batch_size'],
                                   sid_supervised_tokens=spec['batch_size']*args.decoder_positions)
        result['architecture'] = dict(model.spec)
        result['synthetic_vocabulary'] = args.generator_vocab
        result['warmup_steps'] = args.warmup
        result['measurement_steps'] = args.steps
        result['decoder_positions'] = args.decoder_positions
        result['parameters'] = sum(value.numel() for value in model.parameters())
        result['throughput'].update(effective_targets_per_second=len(rows)*spec['batch_size']/wall,
            sid_supervised_tokens_per_second=len(rows)*spec['batch_size']*args.decoder_positions/wall,
            steps_per_second=len(rows)/wall)
        result['timings_seconds']['steady_window_wall'] = wall
        result['resources'].update(process_metrics(args.device))
        result['correctness_and_retrieval']['finite_loss'] = all(math.isfinite(row['loss']) for row in rows)
        if not result['correctness_and_retrieval']['finite_loss']:
            result['status'] = 'failed_nonfinite'
        result['steps'] = rows
        return result
    finally:
        tiger_arch.POSITIONS = previous_positions


def checkpoint(args):
    import torch
    from .two_tower import TowerConfig
    from .two_tower_training import TensorTrainingSession
    from .two_tower_runtime import process_metrics
    from .two_tower_prepare import sha256_file

    config = TowerConfig(batch_size=8 if args.local_smoke else 1024,
                          id_min_count=100, seed=42)
    catalog, store, _ = load_case_data(args, config)
    shared = args.output.parent / 'two_tower/shared-order.npy'
    if not shared.exists():
        shared = prepare_shared_order(args, args.output)
    continuous = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                        args.device, args.output / 'continuous')
    reuse_order(continuous, shared)
    before, after = (2, 2) if args.local_smoke else (3, 3)
    expected = [continuous.step()['loss'] for _ in range(before + after)]
    continuous_parameters = {name: value.detach().cpu().clone()
                              for name, value in continuous.model.state_dict().items()}
    del continuous
    gc.collect()
    interrupted = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                         args.device, args.output / 'interrupted')
    reuse_order(interrupted, shared)
    actual = [interrupted.step()['loss'] for _ in range(before)]
    started = time.perf_counter()
    path = interrupted.save()
    write_seconds = time.perf_counter()-started
    del interrupted
    gc.collect()
    resumed = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                     args.device, args.output / 'resumed', resume=path)
    reuse_order(resumed, shared)
    actual.extend(resumed.step()['loss'] for _ in range(after))
    exact_parameters = all(torch.equal(value, resumed.model.state_dict()[name].cpu())
                           for name, value in continuous_parameters.items())
    result = record(args, 'checkpoint_resume')
    result['resume_losses_exact'] = expected == actual
    result['resume_parameters_exact'] = exact_parameters
    result['continuous_losses'] = expected
    result['resumed_losses'] = actual
    result['steps_before_after'] = [before, after]
    result['timings_seconds'].update(checkpoint_write=write_seconds,
        checkpoint_read_resume=sum(resumed.restore_timings.values()))
    result['checkpoint_read_seconds'] = resumed.restore_timings['read_seconds']
    result['checkpoint_restore_seconds'] = resumed.restore_timings['restore_seconds']
    result['checkpoint_bytes'] = path.stat().st_size
    # An artificial score exercises transport of a previous full best checkpoint.
    # It is confined to this recovery probe and is not a validation measurement.
    best = resumed.register_epoch(0., dict(epoch=1, diagnostic_only=True))
    best_sha = sha256_file(best)
    started = time.perf_counter()
    with_best = resumed.save()
    result['checkpoint_with_best_write_seconds'] = time.perf_counter()-started
    result['checkpoint_with_best_bytes'] = with_best.stat().st_size
    del resumed
    gc.collect()
    started = time.perf_counter()
    imported = TensorTrainingSession(catalog, store, config, 'mlp_causal',
                                      args.device, args.output / 'best-import', resume=with_best)
    result['checkpoint_with_best_session_restore_seconds'] = time.perf_counter()-started
    result['best_checkpoint_restored_exact'] = sha256_file(imported.manager.best) == best_sha
    result['best_payload_probe_score'] = 'artificial recovery fixture; no validation score'
    result['resources'].update(process_metrics(args.device))
    if (not result['resume_losses_exact'] or not exact_parameters
            or not result['best_checkpoint_restored_exact']):
        result['status'] = 'failed_equivalence'
    return result


def evaluation_reference(two_tower_output, identity):
    from .pre_rental_deploy import sha256_file

    case = two_tower_output / 'mlp_causal-bf16-20-cuda'
    record_path = case / 'result.json'
    if not record_path.is_file():
        return None
    receipt = json.loads(record_path.read_text())
    if receipt['status'] != 'complete':
        return None
    expected = dict(source_sha=identity['source_sha'], data_sha=identity['data_sha'],
                    model='mlp_causal', precision='bf16', id_min_count=20, batch_size=1024)
    if (any(receipt['protocol'].get(key) != value for key, value in expected.items())
            or receipt.get('data_placement') != 'cuda'
            or not receipt.get('reference_probe_updates', 0)):
        raise ValueError('reference model identity differs from primary causal probe')
    path = case / 'reference-model.pt'
    if sha256_file(path) != receipt.get('reference_sha256'):
        raise ValueError('reference model sha256 mismatch')
    return path, receipt['reference_probe_updates']


def evaluate(args):
    import numpy as np
    import torch
    from .two_tower import TowerConfig, TwoTower, user_hash
    from .two_tower_data import subset_evaluation
    from .two_tower_retrieval import export_item_embeddings, retrieve
    from .two_tower_runtime import precision_mode, process_metrics

    config = TowerConfig(batch_size=1024, id_min_count=20, seed=42)
    catalog, store, data = load_case_data(args, config)
    result = record(args, 'export_and_early_retrieval')
    torch.manual_seed(42)
    model = TwoTower(catalog, config, 'causal').to(args.device)
    reference = evaluation_reference(args.output.parent / 'two_tower', result['protocol'])
    model_source = 'seeded_causal_initialization'
    probe_updates = 0
    if reference is not None:
        path, probe_updates = reference
        saved = torch.load(path, weights_only=True, map_location='cpu')
        if saved['config'] != config.__dict__ or saved['probe_updates'] != probe_updates:
            raise ValueError('reference model configuration or update count differs from primary causal probe')
        model.load_state_dict(saved['model'])
        model_source = 'trained_primary_causal_bf16_min20_cuda'
        del saved
    early = np.array([user for user in data.user_ids
                       if user_hash(user, config.user_hash_seed) % 10 == 0])
    if args.local_smoke:
        early = early[:16]
    if not len(early):
        raise ValueError('no users in the fixed early-stopping hash subset')
    data = subset_evaluation(data, early)
    with precision_mode(args.device, 'fp32'):
        exported = export_item_embeddings(model, catalog, args.device,
                                           args.output / 'catalog-vectors')
        predictions, details = retrieve(model, catalog, store, data, args.device,
                                          args.output / 'retrieval', exact_users=8,
                                          save_index=True,
                                          vector_file=args.output / 'catalog-vectors/vectors.npy')
    result['model_source'] = model_source
    result['user_structure'] = 'causal'
    result['reference_probe_updates'] = probe_updates
    result['protocol'].update(catalog_items=len(catalog.ids), precision='fp32',
                               id_min_count=config.id_min_count,
                               history_events=config.max_history)
    result['query_users'] = len(early)
    result['timings_seconds'].update(catalog_export=exported['seconds'],
        index_fit=details['index_fit_seconds'], index_add=details['index_add_seconds'],
        early_validation=details['query_encode_seconds'] + details['search_seconds'])
    result['query_encode_seconds'] = details['query_encode_seconds']
    result['search_seconds'] = details['search_seconds']
    result['correctness_and_retrieval'].update(
        ann_exact_topk_overlap=details['mean_overlap'],
        filtered_unique_items=sum(len(values) for values in predictions.values()))
    result['resources'].update(process_metrics(args.device))
    return result
