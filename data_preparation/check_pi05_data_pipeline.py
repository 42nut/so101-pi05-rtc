#!/usr/bin/env python3
"""Attach train-only stats to a converted copy and check native LeRobot loading.

This checks data loading and numeric (un)normalization, not Pi05 model forward,
tokenization, image resize, or robot execution. No image/video/data rows change.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--prep', required=True, type=Path)
    args = parser.parse_args()
    root, prep = args.root.resolve(), args.prep.resolve()

    import numpy as np
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.datasets import LeRobotDataset
    from lerobot.processor import TransitionKey, batch_to_transition
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    info = json.loads((root/'meta/info.json').read_text())
    if info.get('codebase_version') != 'v3.0':
        raise ValueError('--root must point to the converted v3 dataset, not meta/')
    if (info['total_episodes'], info['total_frames'], info['fps']) != (50, 30969, 30):
        raise ValueError('Dataset counts or FPS differ from the expected recording')
    split = json.loads((prep/'split_manifest.json').read_text())
    fitted = json.loads((prep/'train_numeric_statistics.json').read_text())
    train_ids, val_ids = split['train_episodes'], split['validation_episodes']
    if len(train_ids) != 45 or len(val_ids) != 5 or set(train_ids) & set(val_ids) or set(train_ids+val_ids) != set(range(50)):
        raise ValueError('Expected a disjoint 45/5 episode split covering all 50 episodes')
    if fitted['training_episode_ids'] != train_ids:
        raise ValueError('Normalization statistics do not belong to this training split')

    stats_path = root/'meta/stats.json'
    old_bytes = stats_path.read_bytes()
    merged = json.loads(old_bytes)
    numeric_keys = ('action', 'observation.state')
    for key in numeric_keys:
        f = fitted['stats'][key]
        if f['names'] != info['features'][key]['names']:
            raise ValueError(f'{key}: joint order differs from fitted statistics')
        if info['features'][key]['shape'] != [6] or f['count'] != [split['train_frames']]:
            raise ValueError(f'{key}: invalid shape or fitted sample count')
        replacement = {}
        for name in ('min', 'max', 'mean', 'std', 'q01', 'q99', 'count'):
            x = np.asarray(f[name])
            if not np.isfinite(x).all():
                raise ValueError(f'Nonfinite statistics in {key}/{name}')
            replacement[name] = f[name]
        if not np.all(np.asarray(f['q99']) > np.asarray(f['q01'])):
            raise ValueError(f'{key}: constant or invalid quantile interval')
        merged[key] = replacement

    # Only this working copy's numeric normalization metadata is updated.
    # Retain the original metadata and image statistics for inspection/recovery.
    backup = root/'meta/stats_before_train_only.json'
    if not backup.exists():
        backup.write_bytes(old_bytes)
    pending = stats_path.with_name('stats.json.pending')
    pending.write_text(json.dumps(merged, ensure_ascii=False, indent=2)+'\n')
    pending.replace(stats_path)
    print('Attached numeric statistics fitted on 45 training episodes.', flush=True)

    common = dict(repo_id='local/so101_yellow_cube_to_tray', root=root,
                  video_backend='torchcodec', delta_timestamps={'action':[i/30 for i in range(50)]})
    train = LeRobotDataset(episodes=train_ids, **common)
    val = LeRobotDataset(episodes=val_ids, **common)
    assert len(train) == split['train_frames'] == 28083, len(train)
    assert len(val) == split['validation_frames'] == 2886, len(val)
    for label, dataset, expected_ids in [('train', train, train_ids), ('validation', val, val_ids)]:
        actual_ids = set(int(x) for x in dataset.hf_dataset['episode_index'])
        if actual_ids != set(expected_ids):
            raise ValueError(f'{label}: native loader selected the wrong episodes')
        for key in numeric_keys:
            for name in ('q01', 'q99'):
                if name not in dataset.meta.stats[key]:
                    raise ValueError(f'{label}: native loader has lost {key}/{name}; stop before training')
                np.testing.assert_allclose(np.asarray(dataset.meta.stats[key][name]),
                                           fitted['stats'][key][name], rtol=1e-6, atol=1e-6)
    print('Native split check: train=45 / 28083 frames; validation=5 / 2886 frames.', flush=True)

    features = {
        'observation.state': PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        'action': PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    norm_map = {FeatureType.STATE:NormalizationMode.QUANTILES,
                FeatureType.ACTION:NormalizationMode.QUANTILES}
    tensor_stats = {key: {name:torch.tensor(value, dtype=torch.float32) for name,value in merged[key].items()}
                    for key in numeric_keys}
    normalizer = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=tensor_stats)
    unnormalizer = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=tensor_stats)
    results = []
    for label, dataset in [('train',train), ('validation',val)]:
        # Include the final row to exercise native action-chunk boundary padding.
        for index in [0, len(dataset)//2, len(dataset)-1]:
            sample = dataset[index]
            assert tuple(sample['action'].shape) == (50,6), sample['action'].shape
            assert tuple(sample['observation.state'].shape) == (6,), sample['observation.state'].shape
            images = {}
            for key in ['observation.images.front','observation.images.wrist']:
                image = sample[key]
                assert tuple(image.shape) == (3,480,640), (key,image.shape)
                assert torch.isfinite(image).all()
                assert float(image.min()) >= -1e-5 and float(image.max()) <= 1.00001
                images[key] = list(image.shape)
            batch = {key:sample[key].float().unsqueeze(0) for key in numeric_keys}
            transition = batch_to_transition(copy.deepcopy(batch))
            normalized = normalizer(transition)
            recovered = unnormalizer(normalized)
            record = {'split':label, 'dataset_row':index, 'episode_index':int(sample['episode_index']),
                      'action_shape':list(batch['action'].shape), 'images':images}
            for key, value in [('action',normalized[TransitionKey.ACTION]),
                               ('observation.state',normalized[TransitionKey.OBSERVATION]['observation.state'])]:
                assert torch.isfinite(value).all(), f'{label}: {key} normalization produced NaN/Inf'
                q01 = tensor_stats[key]['q01']; q99 = tensor_stats[key]['q99']
                torch.testing.assert_close(value, 2*(batch[key]-q01)/(q99-q01)-1, atol=1e-5, rtol=1e-5)
                restored = (recovered[TransitionKey.ACTION] if key=='action' else
                            recovered[TransitionKey.OBSERVATION]['observation.state'])
                torch.testing.assert_close(restored, batch[key], atol=1e-4, rtol=1e-5)
                record[key] = {'normalized_min':float(value.min()), 'normalized_max':float(value.max()),
                               'roundtrip_max_error':float((restored-batch[key]).abs().max())}
            results.append(record)
    report = {'status':'passed', 'dataset_root':str(root), 'repo_id':common['repo_id'],
              'normalization':'QUANTILES', 'action_representation':'absolute recorded motor positions',
              'normalization_fit_episodes':train_ids, 'validation_episodes':val_ids,
              'train_frames':len(train), 'validation_frames':len(val),
              'action_chunk_size':50, 'fps':30, 'sample_checks':results,
              'normalization_stats_sha256':hashlib.sha256(stats_path.read_bytes()).hexdigest(),
              'scope':'Checks native decoding, exact episode selection and numeric normalization; model/tokenizer/resize not checked'}
    (prep/'normalization_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (prep/'normalization_scope.json').write_text(json.dumps({
        'numeric_stats_in_dataset_meta':'fitted on training split only',
        'image_stats':'preserved; Pi05 visual normalization uses IDENTITY here',
        'data_rows':'all 50 episodes remain in storage; train loader MUST select train_episodes',
        'train_episodes':train_ids,'validation_episodes':val_ids,
        'absolute_actions':True,'raw_dataset_modified':False,
    },ensure_ascii=False,indent=2)+'\n')
    print('Native QUANTILES normalization + inverse: PASS', flush=True)
    print('Decoded both cameras; checked 50-step chunks in train and validation.', flush=True)
    print('Report:', prep/'normalization_check.json')


if __name__ == '__main__':
    main()
