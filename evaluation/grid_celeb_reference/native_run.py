"""Persistent, isolated Baseline/REPA native rerun on GPUs 0/1.

Launch this controller with setsid and the environment Python -u. Each job uses
the child run belonging to its own model. No cross-model visual cache reuse.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from grid_wer import report
from native_frontend import NATIVE_RUN, verify_audit
from protocol import DEFAULT_RUN, rows, save_json, sha
from run import Pipeline, TASKS


class NativePipeline(Pipeline):
    existing_repa_pid = 0

    def job(self, name, script, method, argv, gpu):
        return name, script, ['--run', str(self.run/method), *argv], gpu

    def required_batch(self, jobs):
        result = self.batch(jobs)
        if any(result.values()):
            raise RuntimeError('Failed native jobs: '+str({k:v for k,v in result.items() if v}))

    def execute(self):
        for method in ('baseline','repa'):
            rows(self.run/method)
            verify_audit(self.run/method, method)
        if self.existing_repa_pid:
            proc = Path(f'/proc/{self.existing_repa_pid}/cmdline')
            if proc.exists():
                command = proc.read_bytes().replace(b'\0',b' ').decode()
                if str(self.run/'repa') not in command or '--method repa' not in command or '--limit 0' not in command:
                    raise ValueError('Cannot adopt a different inference process')
            self.state['jobs']['repa_infer660_adopted'] = {'pid':self.existing_repa_pid,'gpu':self.gpus[1],
                'status':'running','log':str(self.run/'logs/repa_infer660.log'),
                'note':'Existing verified native REPA inference retained during Baseline cached-Mel correction'}
            self.update()
        visual = []
        for m, gpu in zip(('baseline','repa'), self.gpus):
            child = self.run/m
            records = [child/'assets/records'/(r['target_id']+'.json') for r in rows(child)]
            if all(p.exists() for p in records):
                for path in records:
                    record = json.loads(path.read_text())
                    if record['frontend_method'] != m:
                        raise ValueError('Cannot reuse a different model frontend')
                    for key, value in record['artifact_sha256'].items():
                        if sha(record['artifact_paths'][key]) != value:
                            raise ValueError('Changed completed native visual cache')
                print(f'Verified and retained all 660 own native visual records: {m}',flush=True)
                continue
            for shard in range(self.visual_workers):
                visual.append(self.job(f'{m}_native_visual{shard}', 'native_frontend.py', m,
                    ['--stage','visual','--method',m,'--shard',str(shard),'--nshard',str(self.visual_workers)], gpu))
        if visual:
            self.required_batch(visual)
        for limit in (33, 0):
            self.required_batch([self.job(f'{m}_infer{limit or 660}', 'native_frontend.py', m,
                ['--stage','infer','--method',m,'--limit',str(limit)], gpu)
                for m,gpu in zip(('baseline','repa'),self.gpus)
                if not (m == 'repa' and self.existing_repa_pid)])
        if self.existing_repa_pid:
            while Path(f'/proc/{self.existing_repa_pid}/cmdline').exists():
                if not Path(f'/proc/{self.existing_repa_pid}/cmdline').read_bytes():
                    break  # completed orphan may briefly remain as a zombie
                self.update()
                time.sleep(5)
            from metrics import validate
            validate(self.run/'repa','repa',rows(self.run/'repa',evaluation=True))
            self.state['jobs']['repa_infer660_adopted'].update(status='complete',exit_code=0)
            self.update()
        avjobs = []
        for m,gpu in zip(('baseline','repa'), self.gpus):
            for kind in ('gt', m):
                avjobs.append(self.job(f'{m}_{kind}_joint_av', 'avfeatures.py', m, ['--method',kind], gpu))
        self.required_batch(avjobs)
        self.required_batch([self.job('grid_gt_wer','metrics.py','baseline',['--task','wer','--gt'],self.gpus[0])])
        for task in TASKS:
            self.required_batch([self.job(f'{m}_{task}', 'metrics.py', m, ['--method',m,'--task',task], gpu)
                for m,gpu in zip(('baseline','repa'),self.gpus)])
        gt = report(self.run/'baseline', 'baseline', gt=True)
        models = {}
        for m in ('baseline','repa'):
            child = self.run/m
            normalized = report(child,m)
            values = {}
            for task in TASKS:
                summary = json.loads((child/'outputs'/m/f'_{task}_summary.json').read_text())
                if summary['count'] != 660 or summary['manifest_sha256'] != sha(child/'pairs.jsonl'):
                    raise ValueError('Metric coverage differs from native run')
                values[task] = summary[task]
            values['wer_grid_normalized'] = normalized['wer']
            models[m] = values
        save_json(self.run/'comparison.json', {'complete':True,'count_per_model':660,'reference_count':66,
            'frontend':'model_isolated_aligndit_native_v1','models':models,'grid_gt_wer_normalized':gt['wer'],
            'frozen_source_run':str(DEFAULT_RUN), 'native_audit':{m:str(self.run/m/'native_frontend.json') for m in models},
            'caveat':'Canonical mean-face restored from upstream; absent historical template hash cannot be independently verified. No human ASR edits.'})
        self.state['status'] = 'complete'
        self.update()
        print(json.dumps(models,indent=2),flush=True)
        return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, default=NATIVE_RUN)
    p.add_argument('--gpus', type=int, nargs=2, default=[0,1])
    p.add_argument('--visual-workers-per-gpu', type=int, choices=[1,2], default=2)
    p.add_argument('--existing-repa-pid',type=int,default=0)
    args = p.parse_args()
    pipeline = NativePipeline(args.run,args.gpus,args.visual_workers_per_gpu)
    pipeline.existing_repa_pid = args.existing_repa_pid
    try:
        sys.exit(pipeline.execute())
    except Exception as exc:
        pipeline.state.update(status='incomplete',error=repr(exc))
        pipeline.update()
        raise
