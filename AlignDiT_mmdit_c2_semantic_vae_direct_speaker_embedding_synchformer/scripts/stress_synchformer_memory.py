#!/usr/bin/env python3
"""Read-only real-video RSS stress test for the frozen Synchformer extractor."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import torch
from aligndit.model.synchformer_features import (
    FrozenSynchformerExtractor, atomic_json, load_synchformer_payload, read_inventory,
)


def rss_mib():
    return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 1024**2


def main():
    root = Path(os.environ.get('ROOT_PREFIX', '') + '/zjw524/projects/data')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--max-rss-mib', type=int, default=12000)
    parser.add_argument('--inventory', type=Path, default=root/'CelebVDub_svae1000k_sample_seed666_fp32/manifests/inventory.jsonl')
    parser.add_argument('--video-root', type=Path, default=root/'CelebVDub/video')
    parser.add_argument('--cache-dir', type=Path, default=root/'CelebVDub/synchformer_25fps_16f_stride8')
    parser.add_argument('--report', type=Path, default=Path('logs/synchformer_rss_stress.json'))
    parser.add_argument('--cleanup-interval', type=int, default=16)
    parser.add_argument('--compare-cached', type=int, default=8)
    args = parser.parse_args()
    if args.count < 1 or args.max_rss_mib < 1024:
        raise ValueError('Invalid count or RSS ceiling')
    torch.set_num_threads(2)
    stop = threading.Event()
    peak = [rss_mib()]
    def monitor():
        while not stop.wait(.1):
            current = rss_mib()
            peak[0] = max(peak[0], current)
            if current > args.max_rss_mib:
                print(json.dumps({'error': 'RSS ceiling exceeded', 'rss_mib': current}), flush=True)
                os._exit(2)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    records = read_inventory(args.inventory)
    model = FrozenSynchformerExtractor(device=args.device, cleanup_interval=args.cleanup_interval)
    comparisons = []
    start = time.monotonic()
    try:
        for record in records[:args.compare_cached]:
            key = record['audio_relative_path'].removesuffix('.wav')
            expected = load_synchformer_payload(args.cache_dir, key)['features']
            actual = model.extract(args.video_root/(key+'.mp4'), key)['features']
            delta = (actual.float()-expected.float()).abs().max().item()
            if not torch.equal(actual, expected):
                raise AssertionError(f'Feature regression for {key}: max difference {delta}')
            comparisons.append({'clip_key':key,'max_abs_difference':delta,'exact_equal':True})
            del expected, actual
        print(json.dumps({'stage':'comparison','count':len(comparisons),'all_exact_equal':True}), flush=True)
        samples = []
        for index in range(args.count):
            record = records[(index*263)%len(records)]
            key = record['audio_relative_path'].removesuffix('.wav')
            payload = model.extract(args.video_root/(key+'.mp4'), key)
            if not torch.isfinite(payload['features']).all():
                raise AssertionError(f'Nonfinite features: {key}')
            del payload
            entry = {'count':index+1,'rss_mib':rss_mib(),'peak_rss_mib':peak[0],
                     'elapsed_seconds':time.monotonic()-start}
            samples.append(entry)
            if (index+1)%25==0 or index==0:
                print(json.dumps(entry), flush=True)
        result = {'complete':True,'count':args.count,'cleanup_interval':args.cleanup_interval,
            'peak_rss_mib':peak[0],'final_rss_mib':rss_mib(),
            'elapsed_seconds':time.monotonic()-start,'cached_feature_comparisons':comparisons,'samples':samples}
        atomic_json(args.report,result)
        print(json.dumps({key:value for key,value in result.items() if key not in {'samples','cached_feature_comparisons'}}),flush=True)
    finally:
        stop.set()
        monitor_thread.join(timeout=1)

if __name__=='__main__':
    main()
