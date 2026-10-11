"""Score a completed native model while the other GPU continues inference."""
import argparse
from pathlib import Path

from metrics import validate
from native_frontend import NATIVE_RUN, verify_audit
from protocol import rows
from run import Pipeline


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=NATIVE_RUN/'repa')
    p.add_argument('--method',choices=['baseline','repa'],default='repa')
    p.add_argument('--gpu',type=int,default=1)
    a = p.parse_args()
    verify_audit(a.run,a.method)
    validate(a.run,a.method,rows(a.run,evaluation=True))
    pipeline = Pipeline(a.run,[a.gpu,a.gpu],2)
    try:
        jobs = [('generated_joint_av','avfeatures.py',['--method',a.method],a.gpu)]
        jobs += [(task,'metrics.py',['--method',a.method,'--task',task],a.gpu) for task in ('wer','sim')]
        if any(pipeline.batch(jobs).values()):
            raise RuntimeError('Early AV/WER/SPK scoring failed')
        jobs = [(task,'metrics.py',['--method',a.method,'--task',task],a.gpu)
                for task in ('emosim','emoembed','avsync')]
        if any(pipeline.batch(jobs).values()):
            raise RuntimeError('Early emotion/AV scoring failed')
        pipeline.state['status'] = 'complete'
        pipeline.update()
    except Exception as exc:
        pipeline.state.update(status='incomplete',error=repr(exc))
        pipeline.update()
        raise
