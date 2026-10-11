"""Official LRS3 EMA400k on frozen GRID660/Celeb66, with process sharding.

Model/preprocessing imports come only from the user-specified official tree.
Metric imports run in separate processes and reuse historical Setting 1 workers.
All assets, shard records and resumable outputs live outside the source repo.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from protocol import DATA, PROJECT, DEFAULT_RUN, PROTOCOL, REPA, rows, sha, save_json
from run import Pipeline, TASKS

TREE = PROJECT / 'AlignDiT_original_LRS3_trained/AlignDiT'
CKPT = PROJECT / 'AlignDiT_LRS3_Checkpoint/AlignDiT_finetune_LRS3_400000.pt'
RESOURCES = DATA / 'evaluations/official_lrs3_resources'
RUN = DATA / 'evaluations/grid660_celeb66_seed0_official_lrs3_ema400k'
METHOD = 'official_lrs3'
AV_CKPT = PROJECT.parent.parent / 'alignDiT_pretrain_models/large_vox_iter5.pt'


def official_imports():
    sys.path.insert(0, str(TREE / 'src'))
    import aligndit.model
    if Path(aligndit.model.__file__).resolve().parent != (TREE/'src/aligndit/model').resolve():
        raise ValueError('Unexpected model snapshot')


def fairseq_imports():
    sys.path.insert(0, str(DATA/'av_hubert/fairseq/fairseq'))
    sys.path.insert(0, str(DATA/'av_hubert/avhubert/avhubert'))
    import fairseq
    fairseq.utils.import_user_module(argparse.Namespace(user_dir=str(DATA/'av_hubert/avhubert/avhubert')))


def copy_verified(source, dest):
    source, dest = Path(source), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha(dest) != sha(source):
            raise ValueError('Changed immutable asset: '+str(dest))
    else:
        shutil.copyfile(source, dest)


def initialize(run):
    import numpy as np
    import cv2
    official_imports()
    from aligndit.script.misc.extract_mel import get_tacotron_mel_spectrogram_processor, single_job
    if (run/'selection.json').exists():
        rows(run)
        protocol=json.loads((run/'protocol.json').read_text())
        protocol.update(source_container_adapter='Lossless MPG-to-Matroska stream copy, timestamps start at zero; every decoded RGB frame verified exactly equal to direct PyAV decoding of original MPG',
            adapter_sha256=sha(__file__))
        save_json(run/'protocol.json',protocol)
        return
    old = rows(DEFAULT_RUN, evaluation=True)
    run.mkdir(parents=True, exist_ok=True)
    processor = get_tacotron_mel_spectrogram_processor()
    references = {}
    inference, evaluation = [], []
    for original in old:
        row = {k:v for k,v in original.items() if k not in (
            'target_gt_audio','source_lip_feature','source_face_feature','lip_feature',
            'face_feature','ref_latent_path','ref_speaker_path','target_video_path')}
        ident = Path(row['target_id'])
        row['mouth_video'] = str(run/'assets/mouth'/ident.with_suffix('.mp4'))
        row['video_25hz_path'] = str(run/'assets/avhubert_video_feat'/ident.with_suffix('.npy'))
        ref = Path(row['ref_id']).relative_to('celebvdub')
        row['ref_audio'] = str(run/'references/audio'/ref.with_suffix('.wav'))
        copy_verified(original['ref_audio'], row['ref_audio'])
        if row['ref_id'] not in references:
            if sha(row['ref_audio']) != row['ref_audio_sha256']:
                raise ValueError('Frozen reference changed')
            mel = single_job(processor, row['ref_audio'],16000).cpu().numpy()
            melpath = run/'references/mel_tacotron'/ref.with_suffix('.npy')
            melpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(melpath, mel, allow_pickle=False)
            video_source = DATA/'CelebVDub/video/test'/ref.with_suffix('.mp4')
            cap = cv2.VideoCapture(str(video_source))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if frame_count <= 0 or abs(fps-25)>1e-4:
                raise ValueError('Reference video length unavailable')
            # The native builder replaces all prompt video with zeros. Only
            # its original frame count controls prompt-Mel cut_or_pad length.
            vp = run/'references/avhubert_video_feat'/ref.with_suffix('.npy')
            vp.parent.mkdir(parents=True, exist_ok=True)
            np.save(vp, np.zeros((frame_count,1024),np.float32), allow_pickle=False)
            references[row['ref_id']] = {'audio_sha256':sha(row['ref_audio']),
                'mel_path':str(melpath),'mel_sha256':sha(melpath),
                'reference_video_length_source':str(video_source),
                'reference_video_frames':frame_count,'reference_video_condition_used':False}
        inference.append(row)
        evaluation.append({**row,'target_gt_audio':original['target_gt_audio']})
    for name, items in [('inference',inference),('pairs',evaluation)]:
        (run/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
    selection=json.loads((DEFAULT_RUN/'selection.json').read_text())
    selection.update(pairs_sha256=sha(run/'pairs.jsonl'),inference_sha256=sha(run/'inference.jsonl'),
        frozen_source_run=str(DEFAULT_RUN),frozen_source_pairs_sha256=sha(DEFAULT_RUN/'pairs.jsonl'))
    save_json(run/'selection.json',selection)
    code=[TREE/'src/aligndit'/p for p in ('config/finetune.yaml','model/cfm_vt.py',
        'model/backbone/dit_vt_c.py','script/eval/utils.py','script/misc/extract_mel.py',
        'script/misc/extract_avhubert_from_only_video.py','script/misc/avhubert_utils.py')]
    code += list((RESOURCES/'auto_avsr/preparation/detectors/retinaface').glob('*'))
    save_json(run/'protocol.json',{'protocol':PROTOCOL,'count':660,'reference_count':66,
        'model_tree':str(TREE),'checkpoint':str(CKPT),'checkpoint_sha256':sha(CKPT),
        'source_sha256':{str(p):sha(p) for p in code if p.is_file()},
        'avhubert_checkpoint':str(AV_CKPT),'avhubert_sha256':sha(AV_CKPT),
        'vocab_path':str(TREE/'data/LRS3_char/vocab.txt'),'vocab_sha256':sha(TREE/'data/LRS3_char/vocab.txt'),
        'weights':'EMA','finetune_update':400000,'seed':0,'nfe':32,'ode':'euler',
        'epss':True,'cfg_text':5.,'cfg_video':2.,'sway':-1.,'target_seconds':3.,
        'target_acoustic_inputs_used':False,'mouth_frontend':'Auto-AVSR RetinaFace/FAN; official color preprocessing, 96x96, H264 via torchvision.write_video; AV-HuBERT grayscale/center crop per checkpoint',
        'reference_frontend':'Official extract_mel.py raw cached TacotronSTFT; original get_inference_prompt_vt and output RMS rule',
        'references':references,'metric_protocol':'Existing CelebV-Dub Setting 1 worker implementations; WER corpus edits/words; other scores sample means'})
    rows(run)
    print('Initialized unchanged 660 targets, 66 references',flush=True)


def audit(run):
    protocol=json.loads((run/'protocol.json').read_text())
    for path,expected in protocol['source_sha256'].items():
        if sha(path)!=expected:
            raise ValueError('Changed native source: '+path)
    return protocol


def visual(args):
    import numpy as np
    import torch
    import av
    import torchvision
    official_imports()
    fairseq_imports()
    for folder in ('face_detection','face_alignment'):
        sys.path.insert(0,str(RESOURCES/folder))
    sys.path.insert(0,str(RESOURCES/'auto_avsr/preparation'))
    from data.data_module import AVSRDataLoader
    from utils import save2vid
    from aligndit.script.misc.extract_avhubert_from_only_video import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    audit(args.run)
    loader=AVSRDataLoader(modality='video',detector='retinaface',convert_gray=False,gpu_type='cuda')
    reader=AVHubertFeatureReader(str(AV_CKPT),custom_utils=avhubert_utils)
    items=rows(args.run)[args.shard::args.nshard]
    for i,row in enumerate(items,1):
        recordpath=args.run/'assets/records'/(row['target_id']+'.json')
        if recordpath.exists():
            record=json.loads(recordpath.read_text())
            if sha(row['mouth_video'])!=record['mouth_sha256'] or sha(row['video_25hz_path'])!=record['feature_sha256']:
                raise ValueError('Changed resumed native visual artifact')
            continue
        if sha(row['source_video'])!=row['source_video_sha256']:
            raise ValueError('Changed original target')
        # torchvision.read_video seeks incorrectly on some GRID MPEG streams
        # (e.g. returns 63/75 frames). Remux without re-encoding so the native
        # Auto-AVSR reader receives the complete, pixel-identical RGB video.
        container=args.run/'assets/container'/(row['target_id']+'.mkv')
        container.parent.mkdir(parents=True,exist_ok=True)
        if not container.exists():
            subprocess.run([str(Path(sys.executable).with_name('ffmpeg')),
                '-hide_banner','-loglevel','error','-n','-i',row['source_video'],
                '-map','0:v:0','-an','-c:v','copy','-avoid_negative_ts','make_zero',str(container)],check=True)
        with av.open(row['source_video']) as original:
            original_rgb=np.stack([f.to_ndarray(format='rgb24') for f in original.decode(video=0)])
        remux_rgb=torchvision.io.read_video(str(container),pts_unit='sec')[0].numpy()
        if original_rgb.shape!=(75,288,360,3) or not np.array_equal(original_rgb,remux_rgb):
            raise ValueError('Lossless container adapter changed frames')
        with torch.inference_mode():
            mouth=loader.load_data(str(container))
            if tuple(mouth.shape)!=(75,96,96,3):
                raise ValueError('Native mouth frames changed')
            save2vid(row['mouth_video'],mouth,25)
            feat=reader.get_feats((row['mouth_video'],None)).cpu().float().numpy()
        if feat.shape!=(75,1024) or not np.isfinite(feat).all():
            raise ValueError('Invalid native features')
        dest=Path(row['video_25hz_path']);dest.parent.mkdir(parents=True,exist_ok=True)
        np.save(dest,feat,allow_pickle=False)
        save_json(recordpath,{'target_id':row['target_id'],'mouth_sha256':sha(row['mouth_video']),
            'feature_sha256':sha(dest),'frontend':'official Auto-AVSR RetinaFace/FAN',
            'source_container':str(container),'source_container_sha256':sha(container),
            'original_rgb_frame_count':len(original_rgb),'remux_rgb_pixel_equal':True})
        print(f'Visual {i}/{len(items)} {row["target_id"]}',flush=True)


def infer(args):
    import numpy as np
    import torch
    import torchaudio
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    official_imports()
    from aligndit.model import CFM_VT
    from aligndit.model.modules import MelSpec_tacotron
    from aligndit.script.eval.utils import get_inference_prompt_vt,load_vocoder
    from f5_tts.infer.utils_infer import load_checkpoint
    from f5_tts.model.utils import get_tokenizer
    protocol=audit(args.run)
    if sha(CKPT)!=protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(0)
    np.random.seed(0)
    cfg=OmegaConf.load(TREE/'src/aligndit/config/finetune.yaml')
    # Explicitly use the official 38-char vocabulary; no CelebV-Dub vocab.
    vocab,size=get_tokenizer('LRS3','char')
    cls=get_class('aligndit.model.'+cfg.model.backbone)
    model=CFM_VT(transformer=cls(**cfg.model.arch,text_num_embeds=size,mel_dim=80),
        mel_spec_module=MelSpec_tacotron(**cfg.model.mel_spec),
        mel_spec_kwargs={k:v for k,v in cfg.model.mel_spec.items() if k!='mel_spec_type'},
        odeint_kwargs={'method':'euler'},vocab_char_map=vocab).cuda()
    model=load_checkpoint(model,str(CKPT),'cuda:0',dtype=torch.float32,use_ema=True)
    decoder=load_vocoder('hifigan_16k',True,str(TREE/'ckpts/hifigan_16k_LRS3/g_01000000'),device='cuda:0')
    allitems=rows(args.run)
    # Pilot the longest input to establish a conservative per-process peak.
    items=[max(allitems,key=lambda r:r['ref_num_samples'])] if args.stage=='pilot' else allitems[args.shard::args.nshard]
    out=args.run/('pilot' if args.stage=='pilot' else 'outputs/official_lrs3')
    metainfo=[]
    for row in items:
        dummy=args.run/'assets/audio'/(row['target_id']+'.wav')
        if dummy.exists():
            raise ValueError('Target GT must not be accessible to input builder')
        metainfo.append((row['target_id'],row['ref_text'],row['ref_audio'],' '+row['target_text'],str(dummy)))
    prompts=get_inference_prompt_vt(metainfo,tokenizer='char',target_sample_rate=16000,
        n_fft=640,win_length=640,n_mel_channels=80,hop_length=160,mel_spec_type='hifigan_16k',
        target_rms=.1,use_truth_duration=False,infer_batch_size=1,min_secs=1,audio_video_ratio=4)
    byid={r['target_id']:r for r in items}
    torch.cuda.reset_peak_memory_stats()
    for i,prompt in enumerate(prompts,1):
        utts,rms,cond,lens,total,text,video=prompt
        row=byid[utts[0]]
        dest=out/(row['target_id']+'.wav')
        recordpath=out/'records'/(row['target_id']+'.json')
        if recordpath.exists():
            record=json.loads(recordpath.read_text())
            if sha(dest)!=record['sha256']:
                raise ValueError('Changed resumed audio')
            continue
        if total[0]-lens[0]!=300 or len(text[0])+1>total[0]:
            raise ValueError('Native duration inconsistent')
        start=time.time()
        with torch.inference_mode():
            generated,_=model.sample(cond=cond.cuda(),text=text,duration=torch.tensor(total,device='cuda'),
                video=video.cuda(),lens=torch.tensor(lens,device='cuda'),steps=32,cfg_strength=5.,
                cfg_strength_v=2.,sway_sampling_coef=-1.,seed=0,use_epss=True)
            wave=decoder(generated[:,lens[0]:total[0]].float().transpose(1,2)).squeeze(1).cpu()
            if rms[0]<.1:
                wave=wave*rms[0]/.1
        if tuple(wave.shape)!=(1,48000) or not torch.isfinite(wave).all() or not torch.any(wave):
            raise ValueError('Invalid unadjusted native output')
        dest.parent.mkdir(parents=True,exist_ok=True)
        torchaudio.save(str(dest),wave,16000,encoding='PCM_S',bits_per_sample=16)
        record={'target_id':row['target_id'],'pair_id':row['pair_id'],'ref_id':row['ref_id'],
            'ref_speaker_id':row['ref_speaker_id'],'relative_path':row['target_id']+'.wav',
            'sha256':sha(dest),'samples':48000,'tail_adjustment_samples_16k':0,
            'prompt_frames':lens[0],'target_frames':300,'seconds':time.time()-start,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'target_acoustic_inputs_used':False}
        save_json(recordpath,record)
        print(f'Infer {i}/{len(prompts)} {row["target_id"]} {record["seconds"]:.2f}s peak={record["peak_reserved_bytes"]/2**30:.2f}GiB',flush=True)
    if args.stage=='pilot':
        save_json(args.run/'pilot_summary.json',{'max_reserved_bytes':torch.cuda.max_memory_reserved(),
            'longest_reference_id':items[0]['ref_id'],'count':1})


def gather(run):
    import numpy as np
    import soundfile as sf
    out=run/'outputs/official_lrs3'
    records=[]
    for row in rows(run):
        record=json.loads((out/'records'/(row['target_id']+'.json')).read_text())
        path=out/(row['target_id']+'.wav')
        wave,sr=sf.read(path)
        if record['ref_id']!=row['ref_id'] or sha(path)!=record['sha256'] or sr!=16000 or wave.shape!=(48000,) or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError('Invalid inference record')
        records.append(record)
    actual={str(p.relative_to(out)) for p in (out/'test').rglob('*.wav')}
    if actual!={r['target_id']+'.wav' for r in rows(run)}:
        raise ValueError('Unexpected WAV coverage')
    save_json(out/'inference_summary.json',{'protocol':PROTOCOL,'count':660,'partial_smoke_test':False,
        'pair_manifest_sha256':sha(run/'pairs.jsonl'),'checkpoint':str(CKPT),'outputs':records})


def avfeatures(args):
    import numpy as np
    official_imports()
    fairseq_imports()
    from aligndit.script.misc.extract_avhubert import AVHubertFeatureReader
    from aligndit.script.misc import avhubert_utils
    reader=AVHubertFeatureReader(str(AV_CKPT),custom_utils=avhubert_utils)
    root=args.run/'assets/gt_avhubert' if args.kind=='gt' else args.run/'outputs/official_lrs3/avhubert_feat'
    for i,row in enumerate(rows(args.run,evaluation=True)[args.shard::args.nshard],1):
        audio=row['target_gt_audio'] if args.kind=='gt' else str(args.run/'outputs/official_lrs3'/(row['target_id']+'.wav'))
        dest=root/(row['target_id']+'.npy')
        recordpath=dest.with_suffix('.json')
        if recordpath.exists():
            record=json.loads(recordpath.read_text())
            if record['audio_sha256']!=sha(audio) or record['mouth_sha256']!=sha(row['mouth_video']) or record['feature_sha256']!=sha(dest):
                raise ValueError('Changed joint AV feature')
            continue
        feat=reader.get_feats((row['mouth_video'],audio)).cpu().float().numpy()
        if feat.shape!=(75,1024) or not np.isfinite(feat).all():
            raise ValueError('Invalid joint AV features')
        dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,feat,allow_pickle=False)
        save_json(recordpath,{'target_id':row['target_id'],'audio_sha256':sha(audio),
            'mouth_sha256':sha(row['mouth_video']),'feature_sha256':sha(dest)})
        print(f'AV {args.kind} {i} {row["target_id"]}',flush=True)


def metric(args):
    # Deliberately separate from official model process: identical historical
    # metric implementations, with generated WAVs as their only model input.
    sys.path.insert(0,str(REPA/'src'))
    from aligndit.script.eval import eval_celebvdub_grid_reference as native
    from metrics import validate
    items=rows(args.run,evaluation=True)
    out,_=validate(args.run,METHOD,items)
    subset=items[args.shard::args.nshard]
    test=[(str(out/(r['target_id']+'.wav')),r['ref_audio'] if args.task=='sim' else r['target_gt_audio'],r['target_text']) for r in subset]
    worker=argparse.Namespace(eval_task=args.task,gen_wav_dir=out,
        wavlm_ckpt=PROJECT.parent.parent/'alignDiT_pretrain_models/wavlm_large_finetune.pth',
        wavlm_base_ckpt=PROJECT.parent.parent/'alignDiT_pretrain_models/wavlm_large_s3prl.pt',
        asr_ckpt=DATA/'faster-whisper-large-v3',emo_ckpt=DATA/'emotion2vec_plus_large',
        gt_av_feat=args.run/'assets/gt_avhubert')
    if args.task=='avsync':
        native.validate_av_features(subset,out,worker.gt_av_feat)
    result=native.attach_identities(subset,native.run_worker(worker,test),args.task)
    path=out/'metric_shards'/f'{args.task}_{args.shard}.json'
    save_json(path,{'manifest_sha256':sha(args.run/'pairs.jsonl'),'results':result})


def summarize(run,nshard):
    sys.path.insert(0,str(REPA/'src'))
    from aligndit.script.eval import eval_celebvdub_grid_reference as native
    from grid_wer import report
    expected=[r['target_id'] for r in rows(run)]
    out=run/'outputs/official_lrs3'; values={}
    for task in TASKS:
        records=[]
        for i in range(nshard):
            shard=json.loads((out/'metric_shards'/f'{task}_{i}.json').read_text())
            if shard['manifest_sha256']!=sha(run/'pairs.jsonl'):
                raise ValueError('Metric shard manifest changed')
            records.extend(shard['results'])
        byid={r['target_id']:r for r in records}
        if len(records)!=660 or set(byid)!=set(expected):
            raise ValueError('Incomplete metric coverage')
        records=[byid[k] for k in expected]
        summary=native.summarize(records,task)
        summary.update(task=task,method=METHOD,checkpoint=str(CKPT),manifest_sha256=sha(run/'pairs.jsonl'))
        save_json(out/f'_{task}_summary.json',summary)
        (out/f'_{task}_results.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records))
        values[task]=summary[task]
    values['wer_grid_normalized']=report(run,METHOD)['wer']
    save_json(run/'comparison.json',{'complete':True,'count_per_model':660,'reference_count':66,
        'models':{METHOD:values},'protocol_file':str(run/'protocol.json')})
    print(json.dumps(values,indent=2),flush=True)


class OfficialPipeline(Pipeline):
    def required(self,jobs):
        results=self.batch(jobs)
        if any(results.values()):
            raise RuntimeError('Failed jobs: '+str(results))

    def jobs(self,stage,nshard,**extra):
        argv=[str(x) for k,v in extra.items() for x in ('--'+k.replace('_','-'),v)]
        return [(f'{stage}_{extra.get("kind",extra.get("task",""))}_{i}','official_lrs3.py',
            ['--stage',stage,'--shard',str(i),'--nshard',str(nshard),*argv],self.gpus[i%2]) for i in range(nshard)]

    def execute(self):
        self.required([('initialize','official_lrs3.py',['--stage','initialize'],0)])
        self.required(self.jobs('visual',4))
        self.required([('pilot_longest','official_lrs3.py',['--stage','pilot'],0)])
        peak=json.loads((self.run/'pilot_summary.json').read_text())['max_reserved_bytes']
        # Reserve 3 GiB for context/fragmentation per card, and 1 GiB overhead
        # per worker beyond the measured longest-utterance allocator peak.
        total=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total','--format=csv,noheader,nounits'],text=True)
        budget=(min(int(v) for v in total.split())*2**20)-3*2**30
        workers=max(1,min(4,int(budget/(peak+2**30))))
        self.state['inference_workers_per_gpu']=workers
        self.state['pilot_peak_reserved_bytes']=peak;self.update()
        print(f'Inference workers/card: {workers}; longest pilot peak {peak/2**30:.2f}GiB',flush=True)
        self.required(self.jobs('infer',workers*2))
        gather(self.run)
        self.required(self.jobs('av',4,kind='gt')+self.jobs('av',4,kind='generated'))
        # Two workers/card for heavy ASR/WavLM/emotion models. Corpus summaries
        # are reconstructed from full per-utterance records across all shards.
        for task in TASKS:
            self.required(self.jobs('metric',4,task=task))
        summarize(self.run,4)
        self.state['status']='complete';self.update()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--stage',choices=['controller','initialize','visual','pilot','infer','av','metric','summarize'],required=True)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--nshard',type=int,default=1)
    p.add_argument('--kind',choices=['gt','generated'],default='generated')
    p.add_argument('--task',choices=TASKS)
    args=p.parse_args()
    os.chdir(TREE)
    if args.stage=='controller':
        pipeline=OfficialPipeline(args.run,[0,1])
        try:
            pipeline.execute()
        except Exception as exc:
            pipeline.state.update(status='incomplete',error=repr(exc));pipeline.update()
            raise
    elif args.stage=='initialize':initialize(args.run)
    elif args.stage=='visual':visual(args)
    elif args.stage in ('pilot','infer'):infer(args)
    elif args.stage=='av':avfeatures(args)
    elif args.stage=='metric':metric(args)
    else:summarize(args.run,args.nshard)
