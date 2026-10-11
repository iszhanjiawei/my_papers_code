"""Persistent two-GPU pipeline. Launch this controller using setsid and -u."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from protocol import DEFAULT_RUN, PROTOCOL, rows, save_json, sha

HERE = Path(__file__).resolve().parent
METHODS = ('repa', 'baseline', 'styledubber', 'produbber')
TASKS = ('sim', 'wer', 'emosim', 'emoembed', 'avsync')


class Pipeline:
    def __init__(self, run, gpus):
        self.run, self.gpus = run, gpus
        self.state = {'protocol': PROTOCOL, 'controller_pid': os.getpid(),
                      'gpus': gpus, 'jobs': {}, 'status': 'running'}
        (run / 'logs').mkdir(parents=True, exist_ok=True)
        self.lock = (run / 'pipeline.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def update(self):
        self.state['updated_unix'] = time.time()
        save_json(self.run / 'pipeline_state.json', self.state)

    def batch(self, jobs):
        active = []
        for name, script, argv, gpu in jobs:
            logfile = self.run / 'logs' / (name + '.log')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            env['PYTHONPATH'] = str(self.run / 'python_deps') + os.pathsep + env.get('PYTHONPATH', '')
            command = [sys.executable, '-u', str(HERE / script), '--run', str(self.run), *argv]
            log = logfile.open('a')
            child = subprocess.Popen(command, cwd=HERE.parents[1], env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            self.state['jobs'][name] = {'pid': child.pid, 'gpu': gpu, 'command': command,
                                       'log': str(logfile), 'started_unix': time.time(), 'status': 'running'}
            active.append((name, child, log))
            print(f'Started {name}: PID {child.pid}, GPU {gpu}', flush=True)
        self.update()
        while active:
            remaining = []
            for name, child, log in active:
                code = child.poll()
                if code is None:
                    remaining.append((name, child, log))
                else:
                    log.close()
                    self.state['jobs'][name].update(exit_code=code, finished_unix=time.time(),
                                                    status='complete' if code == 0 else 'failed')
                    print(f'Finished {name}: exit {code}', flush=True)
            active = remaining
            self.update()
            if active:
                time.sleep(5)
        return {name: self.state['jobs'][name]['exit_code'] for name, *_ in jobs}

    def execute(self):
        rows(self.run)
        # References already prepared are immutable and can be reused.
        if not (self.run / 'references/features.json').exists():
            result = self.batch([('references', 'features.py', ['--stage', 'references'], self.gpus[0])])
            if any(result.values()):
                raise RuntimeError('Reference feature extraction failed')
        visual = [('visual_gpu' + str(gpu), 'features.py',
                   ['--stage', 'visual', '--shard', str(i), '--nshard', '2'], gpu)
                  for i, gpu in enumerate(self.gpus)]
        if any(self.batch(visual).values()):
            raise RuntimeError('Visual preprocessing failed; valid records remain resumable')
        # One clip from each GRID speaker before formal generation. Separate
        # output directories ensure smoke samples never count toward the 660.
        smoke = [(m + '_smoke33', 'infer.py', ['--method', m, '--limit', '33'],
                  self.gpus[i // 2]) for i, m in enumerate(METHODS)]
        tested = self.batch(smoke)
        ready = [m for m in METHODS if tested[m + '_smoke33'] == 0]
        generated = self.batch([(m + '_infer660', 'infer.py', ['--method', m],
                                self.gpus[METHODS.index(m) // 2]) for m in ready])
        complete = [m for m in ready if generated[m + '_infer660'] == 0]
        avjobs = [('gt_avfeatures', 'avfeatures.py', ['--method', 'gt'], self.gpus[0])]
        avjobs += [(m + '_avfeatures', 'avfeatures.py', ['--method', m],
                    self.gpus[(i + 1) % 2]) for i, m in enumerate(complete)]
        # At most two simultaneous AV-HuBERT workers per card.
        avresults = {}
        for i in range(0, len(avjobs), 4):
            avresults.update(self.batch(avjobs[i:i + 4]))
        self.batch([('gt_wer', 'metrics.py', ['--task', 'wer', '--gt'], self.gpus[0])])
        # Two workers per GPU; preserve all failures and continue other models.
        for task in TASKS:
            jobs = [(m + '_' + task, 'metrics.py', ['--method', m, '--task', task], self.gpus[i // 2])
                    for i, m in enumerate(complete)
                    if task != 'avsync' or (avresults['gt_avfeatures'] == 0 and avresults[m + '_avfeatures'] == 0)]
            self.batch(jobs)
        reports = {}
        for m in complete:
            values = {}
            for task in TASKS:
                path = self.run / 'outputs' / m / ('_' + task + '_summary.json')
                if path.exists():
                    summary = json.loads(path.read_text())
                    if summary['count'] != 660 or summary['manifest_sha256'] != sha(self.run / 'pairs.jsonl'):
                        raise ValueError('Incorrect metric coverage')
                    values[task] = summary[task]
            reports[m] = values
        self.state['status'] = 'complete' if len(reports) == 4 and all(len(v) == 5 for v in reports.values()) else 'incomplete'
        save_json(self.run / 'comparison.json', {'protocol': PROTOCOL, 'count_per_model': 660,
                 'reference_count': 66, 'complete': self.state['status'] == 'complete', 'models': reports})
        self.update()
        print(json.dumps(self.state['status']), flush=True)
        return 0 if self.state['status'] == 'complete' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--gpus', type=int, nargs=2, default=[0, 1])
    args = parser.parse_args()
    pipeline = Pipeline(args.run, args.gpus)
    try:
        sys.exit(pipeline.execute())
    except Exception as exc:
        pipeline.state.update(status='incomplete', error=repr(exc))
        pipeline.update()
        raise
