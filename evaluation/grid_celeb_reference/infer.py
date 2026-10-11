"""Native frozen model inference on the same 660 GRID/Celeb reference pairs."""
import argparse
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from scipy.signal import resample_poly

from protocol import (DEFAULT_RUN, PROJECT, DATA, REPA, BASELINE, STYLE, PRO,
                      TOOLS, WEIGHTS, PROTOCOL, rows, checkpoint, sha, save_json, load_module)


def finalize(wave, sr, n):
    wave = np.asarray(wave, dtype=np.float32).reshape(-1)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Invalid generated waveform')
    if sr != 16000:
        g = math.gcd(sr, 16000)
        wave = resample_poly(wave, 16000 // g, sr // g).astype(np.float32)
    tail = n - len(wave)
    if abs(tail) > 1600:
        raise ValueError(f'Native output / video discrepancy exceeds 100ms: {tail}')
    wave = np.pad(wave, (0, max(0, tail)))[:n]
    if float(np.sqrt(np.mean(wave ** 2))) < 1e-5:
        raise ValueError('Near-silent generated waveform')
    return wave, tail


def publish(row, wave, sr, out, details):
    wave, tail = finalize(wave, sr, row['target_num_samples'])
    path = out / (row['target_id'] + '.wav')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    sf.write(path, wave, 16000, subtype='PCM_16')
    return {'pair_id': row['pair_id'], 'target_id': row['target_id'], 'ref_id': row['ref_id'],
            'ref_speaker_id': row['ref_speaker_id'], 'relative_path': str(path.relative_to(out)),
            'sha256': sha(path), 'samples': len(wave), 'native_sampling_rate': sr,
            'tail_adjustment_samples_16k': tail, **details}


def aligndit(args, items):
    if args.method == 'baseline':
        raise RuntimeError('Legacy online-RMS Mel conditioning is not the Baseline cached-Mel training path. '
                           'Use native_frontend.py --stage infer --method baseline instead.')
    if not (args.run/'native_frontend.json').is_file():
        raise RuntimeError('REPA requires its own audited native visual/reference caches. '
                           'Use native_frontend.py and native_run.py; comparison-model crops are forbidden.')
    from native_frontend import verify_audit
    verify_audit(args.run,args.method)
    tree = BASELINE if args.method == 'baseline' else REPA
    sys.path.insert(0, str(tree / 'src'))
    native = load_module('native_grid_inference', tree / 'src/aligndit/script/eval/infer_celebvdub_grid_reference.py')
    device = torch.device('cuda:0')
    if args.method == 'baseline':
        model, cfg, ema_step = native.build_model(tree / 'src/aligndit/config/finetune_celebvdub.yaml', checkpoint(args.method), DATA/'CelebVDub_char/vocab.txt', 100000, device)
        from aligndit.model.modules import MelSpec_tacotron
        from aligndit.script.eval.utils import load_vocoder
        extractor = MelSpec_tacotron(**cfg.model.mel_spec)
        decoder = load_vocoder(vocoder_name='hifigan_16k', is_local=True, local_path=str(tree/'ckpts/hifigan_16k_LRS3/g_01000000'), device=device)
        mean = std = None
    else:
        from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder
        config = tree/'src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml'
        model = native.build_model(config, checkpoint(args.method), 150000, device)
        mean, std, _ = native.load_normalization(DATA/'LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json')
        spec = json.loads((DATA/'CelebVDub_svae1000k_sample_seed666_fp32/state/latents/spec.json').read_text())
        decoder, decoder_meta = load_semantic_vae_decoder(repo=PROJECT/'papers_codes/Semantic-VAE', checkpoint_root=PROJECT/'Semantic-VAE/semantic_vae_1000k',cache_spec=spec,device=device)
        mean_tensor, std_tensor = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
    def generate(row):
        if args.method == 'baseline':
            audio, rate = __import__('torchaudio').load(row['ref_audio'])
            if rate != 16000:
                raise ValueError('Wrong reference sample rate')
            rms = float(audio.square().mean().sqrt())
            scale = min(.1/rms, (1-1e-6)/float(audio.abs().max())) if rms < .1 else 1.
            prompt = extractor(audio * scale).T
            pvideo = math.ceil(row['ref_num_samples']/640)
            p = pvideo * 4
            prompt = F.pad(prompt, (0,max(0,p-prompt.shape[-1])),mode='replicate')[:,:p].T[None].cuda()
            video = torch.from_numpy(np.load(row['video_25hz_path'],allow_pickle=False)).cuda()
            t = len(video)*4
            combined_video = torch.cat([torch.zeros(pvideo,1024,device=device),video])[None]
            text = native.combined_text(row['ref_text'],row['target_text'])
            extra = {}
        else:
            ref = np.load(row['ref_latent_path'],allow_pickle=False)
            video = np.load(row['target_video_path'],allow_pickle=False)
            text = native.reference_target_text(row['ref_text'],row['target_text'])
            ref, combined_video, p, t, total = native.make_sampling_inputs(ref,video,text,mean,std)
            prompt = torch.from_numpy(ref)[None].cuda()
            combined_video = torch.from_numpy(combined_video)[None].cuda()
            extra = {'speaker_embedding':torch.from_numpy(np.load(row['ref_speaker_path'],allow_pickle=False))[None].cuda()}
        total = p+t
        if total > 4096 or len(text)+1 > total:
            raise ValueError('Sampler text/duration limit exceeded')
        with torch.inference_mode():
            generated,_ = model.sample(cond=prompt,text=[text],duration=torch.tensor([total],device=device),video=combined_video,
                lens=torch.tensor([p],device=device),steps=32,cfg_strength=5.,cfg_strength_v=2.,sway_sampling_coef=-1.,seed=0,use_epss=True,**extra)
            target = generated[:,p:p+t].float()
            if args.method == 'repa':
                wave = decoder((target*std_tensor+mean_tensor).transpose(1,2)).squeeze().cpu().numpy()
            else:
                wave = decoder(target.transpose(1,2)).squeeze().cpu().numpy()/scale
        return wave,16000,{'prompt_frames':p,'target_frames':t,'nfe':32,'cfg_text':5.,'cfg_video':2.,'sway':-1.,'weights':'EMA'}
    return generate


def styledubber(args, items):
    import yaml
    import librosa
    from scipy.io import wavfile
    sys.path.insert(0,str(STYLE))
    sys.path.insert(0,str(TOOLS))
    from celebvdub_runtime import build_models, get_vocoder, load_torch, predict
    from stage6_spk_embedding import wav_to_mel
    from audio_stft import TacotronSTFT
    from text import text_to_sequence
    configs=[yaml.safe_load((STYLE/f'config/CelebVDub/{name}.yaml').read_text()) for name in ['preprocess','model','train']]
    configs[0]['path']['preprocessed_path']=str(STYLE.parent/'data/CelebVDub_StyleDubber')
    vocdir=PROJECT/'papers_codes/StyleDubber/vocoder/UNIVERSAL_V1'
    configs[1]['vocoder'].update(config_path=str(vocdir/'config.json'),checkpoint_file=str(vocdir/'g_02500000'))
    device=torch.device('cuda:0')
    model,fusion=build_models(configs,device)
    state=load_torch(checkpoint('styledubber'),'cpu')
    if state.get('step') != 100000:
        raise ValueError('Unexpected StyleDubber step')
    model.load_state_dict(state['model'],strict=True)
    fusion.load_state_dict(state['fusion_model'],strict=True)
    del state
    model.eval().requires_grad_(False)
    fusion.eval().requires_grad_(False)
    vocoder=get_vocoder(configs[1],device)
    encoder=torch.jit.load(str(WEIGHTS/'dvector.pt'),map_location=device).eval()
    stft=TacotronSTFT(1024,256,1024,80,22050,0,8000).cuda().eval()
    program='import json,sys; from g2p_en import G2p; g=G2p(); print(json.dumps([[p for p in g(t) if p.strip() and any(c.isalpha() for c in p)] for t in json.load(sys.stdin)]))'
    # Do not expose Python-3.10-only runtime packages to the original 3.8 G2P
    # interpreter. It already has the author's g2p_en/NLTK environment.
    g2p_env=os.environ.copy()
    g2p_env.pop('PYTHONPATH',None)
    result=subprocess.run([str(PROJECT.parent.parent/'ENTER/envs/hpmdubbing/bin/python'),'-c',program],
                          input=json.dumps([r['target_text'] for r in items]),env=g2p_env,
                          text=True,capture_output=True)
    if result.returncode:
        raise RuntimeError('Native G2P failed: '+result.stderr)
    phones=dict(zip([r['target_id'] for r in items],json.loads(result.stdout)))
    references={}
    for row in items:
        ident=row['ref_id']
        if ident in references:
            continue
        dest=args.run/'references/styledubber'/ (ident+'.wav')
        dest.parent.mkdir(parents=True,exist_ok=True)
        # Exactly the uploaded Stage 1 resampling/peak-normalization convention.
        wave,_=librosa.load(row['ref_audio'],sr=22050)
        pcm=(wave/np.max(np.abs(wave))*.95*32768).astype(np.int16)
        wavfile.write(dest,22050,pcm)
        with torch.inference_mode():
            emb=encoder.embed_utterance(torch.from_numpy(wav_to_mel(str(dest)))[None].cuda()).cpu().numpy().reshape(256)
            mel,_=stft.mel_spectrogram(torch.from_numpy(pcm.astype(np.float32)/32768)[None].cuda())
        references[ident]=(mel[0].T.cpu().numpy(),emb)
    del stft,encoder
    torch.cuda.empty_cache()
    def generate(row):
        ref,emb=references[row['ref_id']]
        phone=np.asarray(text_to_sequence('{'+ ' '.join(phones[row['target_id']])+'}', ['english_cleaners']),dtype=np.int64)
        nt,nr,nv=len(phone),len(ref),row['native_frames']
        batch={'id':[row['target_id']],'sid':np.zeros(1,np.int64),'text':phone[None], 'src_len':np.asarray([nt]),
            'ref_mels':ref[None],'ref_mel_lens':np.asarray([nr]),'mel_target':np.zeros((1,1,80),np.float32),
            'mel_len':np.ones(1,np.int64),'D':np.zeros((1,nt),np.int64),'f0':np.zeros((1,nt),np.float32),
            'energy':np.zeros((1,nt),np.float32),'ref_linguistics':np.zeros((1,nr),np.int64),
            'face_lens':np.asarray([nv]),'Lipmotion':np.load(row['lip_feature'])[None],
            'face_embedding':np.load(row['face_feature'])[None],'spk_embedding':emb[None]}
        with torch.inference_mode():
            mel,lengths=predict(model,fusion,batch)
            frames=int(lengths[0])
            wave=vocoder(mel[:,:frames].transpose(1,2)).squeeze().cpu().numpy()
        return wave,22050,{'mel_frames':frames,'reference_frontend':'uploaded librosa Stage1 + Stage6 GE2E + audio_stft; full reference','target_acoustic_inputs':False}
    return generate


def loop(args, items, generate):
    out=args.run/'outputs'/(args.method+(f'_smoke{args.limit}' if args.limit else ''))
    out.mkdir(parents=True,exist_ok=True)
    record_path=out/'generation_records.jsonl'
    completed={}
    if record_path.exists():
        completed={r['target_id']:r for r in [json.loads(x) for x in record_path.read_text().splitlines()]}
    records=[]
    started=time.time()
    for i,row in enumerate(items,1):
        if row['target_id'] in completed:
            record=completed[row['target_id']]
            if record['ref_id']!=row['ref_id'] or sha(out/record['relative_path'])!=record['sha256']:
                raise ValueError('Changed resumed output')
        else:
            wave,sr,details=generate(row)
            record=publish(row,wave,sr,out,details)
            with record_path.open('a') as f:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
        records.append(record)
        save_json(out/'progress.json',{'completed':len(records),'requested':len(items),'last_target':row['target_id']})
        print(f'{args.method} {i}/{len(items)}: {row["target_id"]}',flush=True)
    save_json(out/'inference_summary.json',{'protocol':PROTOCOL,'method':args.method,'count':len(records),
        'partial_smoke_test':len(records)!=660,'pair_manifest_sha256':sha(args.run/'pairs.jsonl'),
        'inference_manifest_sha256':sha(args.run/'inference.jsonl'),'checkpoint':{'path':str(checkpoint(args.method)),
        'sha256':sha(checkpoint(args.method)),'weights':'EMA' if args.method in {'repa','baseline'} else 'native trained parameters'},
        'target_acoustic_inputs_used':False,'duration_source':'complete GRID video frames at 25fps',
        'reference_dataset':'CelebV-Dub test pool','source_sha256':sha(__file__),'seed':0,'elapsed_seconds':time.time()-started,'outputs':records})


def produbber(args, items):
    import yaml
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    if hasattr(EspeakWrapper,'set_data_path'):
        EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
    else:
        os.environ['ESPEAK_DATA_PATH']=espeakng_loader.get_data_path()
    sys.path.insert(0,str(PRO))
    # Reuse the native model construction, strict checkpoint loading, style
    # frontend and sampler without changing any architecture or tensor weights.
    runner=types.ModuleType('setting2_dubbing_runner')
    def callback(native_args, path, data_params, compute_style, inference, device):
        references={}
        def generate(row):
            if row['ref_id'] not in references:
                references[row['ref_id']]=compute_style(row['ref_audio'])[0]
            with torch.inference_mode():
                wave=inference(row['target_text'],references[row['ref_id']],mel_len=round(row['native_frames']*3.2),
                     emotion_feature=torch.from_numpy(np.load(row['face_feature'])).to(device),
                     lip_feature=torch.from_numpy(np.load(row['lip_feature'])).to(device))
            return wave,24000,{'alpha':0.,'beta':.3,'diffusion_steps':5,'embedding_scale':1.,'reference_frontend':'native ProDubber 24k style/predictor encoders with 5000-sample padding'}
        loop(args,items,generate)
    runner.run_setting2=callback
    sys.modules['setting2_dubbing_runner']=runner
    config=yaml.safe_load((PRO/'Configs/config_celebvdub.yml').read_text())
    for key in ['F0_path','ASR_config','ASR_path','PLBERT_dir']:
        config[key]=str(PRO/config[key])
    dest=args.run/'produbber_config.yml'
    dest.write_text(yaml.safe_dump(config))
    sys.argv=[str(PRO/'inference_grid.py'),'-n','celebvdub_stage2','--epoch','12','--setting','2',
              '--config_path',str(dest),'--checkpoint_path',str(checkpoint('produbber')),
              '--setting2_bundle',str(args.run),'--output_dir',str(args.run/'outputs/produbber')]
    runpy.run_path(str(PRO/'inference_grid.py'),run_name='__main__')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=DEFAULT_RUN)
    p.add_argument('--method',choices=['repa','baseline','styledubber','produbber'],required=True)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('high')
    items=rows(args.run)
    if args.limit:
        items=items[:args.limit]
    for row in items:
        required=['lip_feature','face_feature'] if args.method in {'styledubber','produbber'} else ['video_25hz_path'] if args.method=='baseline' else ['target_video_path','ref_latent_path','ref_speaker_path']
        for key in required:
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])
        if sha(row['ref_audio'])!=row['ref_audio_sha256']:
            raise ValueError('Reference changed')
    if args.method=='produbber':
        produbber(args,items)
    else:
        generate=styledubber(args,items) if args.method=='styledubber' else aligndit(args,items)
        loop(args,items,generate)
