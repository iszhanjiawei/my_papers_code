"""Baseline's actual cached-Mel CelebV-Dub inference path, not online RMS Mel."""
import json
from pathlib import Path
import sys

import torch

from protocol import BASELINE, DATA, checkpoint, load_module, sha, save_json


def cached_baseline(args, items):
    from native_frontend import verified_copy
    sys.path.insert(0, str(BASELINE/'src'))
    from aligndit.script.eval.utils import get_inference_prompt_vt, load_vocoder
    native = load_module('baseline_ema_loader', BASELINE/'src/aligndit/script/eval/infer_celebvdub_grid_reference.py')
    device = torch.device('cuda:0')
    config_path = BASELINE/'src/aligndit/config/finetune_celebvdub.yaml'
    model, config, ema_step = native.build_model(config_path, checkpoint('baseline'),
        DATA/'CelebVDub_char/vocab.txt',100000,device)
    decoder = load_vocoder(vocoder_name='hifigan_16k',is_local=True,
        local_path=str(BASELINE/'ckpts/hifigan_16k_LRS3/g_01000000'),device=device)
    metainfo, reference_records = [], {}
    for row in items:
        ident = Path(row['ref_id']).relative_to('celebvdub')
        mel_source = DATA/'CelebVDub/mel_tacotron/test'/ident.with_suffix('.npy')
        video_source = DATA/'CelebVDub/avhubert_video_feat/test'/ident.with_suffix('.npy')
        # Preserve the exact path-replacement convention in native utils.py.
        mel_dest = Path(row['ref_audio'].replace('/audio/','/mel_tacotron/')).with_suffix('.npy')
        video_dest = Path(row['ref_audio'].replace('/audio/','/avhubert_video_feat/')).with_suffix('.npy')
        verified_copy(mel_source,mel_dest)
        verified_copy(video_source,video_dest)
        reference_records[row['ref_id']] = {'ref_id':row['ref_id'],
            'mel_source':str(mel_source),'mel_path':str(mel_dest),'mel_sha256':sha(mel_dest),
            'reference_video_length_source':str(video_source),
            'reference_video_condition_used':False}
        dummy_target_audio = args.run/'assets/audio'/(row['target_id']+'.wav')
        if dummy_target_audio.exists():
            raise ValueError('Dummy path must not expose target GT audio')
        target_video = Path(str(dummy_target_audio).replace('/audio/','/avhubert_video_feat/')).with_suffix('.npy')
        verified_copy(row['video_25hz_path'],target_video)
        metainfo.append((row['target_id'],row['ref_text'],row['ref_audio'],
            ' '+row['target_text'],str(dummy_target_audio)))
    # Calls the unmodified original CelebV-Dub input builder: raw cached Mel,
    # native text spaces, zero prompt video, native cut_or_pad and frame ratios.
    # use_truth_duration=False means dummy_target_audio is never opened.
    prompts = get_inference_prompt_vt(metainfo,tokenizer='char',
        target_sample_rate=16000,n_fft=640,win_length=640,n_mel_channels=80,
        hop_length=160,mel_spec_type='hifigan_16k',target_rms=.1,
        use_truth_duration=False,infer_batch_size=1,audio_video_ratio=4)
    by_id = {}
    for prompt in prompts:
        if len(prompt[0]) != 1:
            raise ValueError('Expected native batch size one')
        by_id[prompt[0][0]] = prompt
    if set(by_id) != {r['target_id'] for r in items}:
        raise ValueError('Native prompt coverage mismatch')
    audit = json.loads((args.run/'native_frontend.json').read_text())
    audit.update(reference_processing='exact cached unscaled CelebVDub Mel; original get_inference_prompt_vt; original output RMS scaling',
        baseline_native_reference_cache=list(reference_records.values()),
        baseline_native_inference_adapter_sha256=sha(__file__))
    for name in ('script/eval/utils.py','model/modules.py','model/dataset.py','model/cfm_vt.py'):
        path = BASELINE/'src/aligndit'/name
        audit['source_sha256'][str(path)] = sha(path)
    save_json(args.run/'native_frontend.json',audit)
    def generate(row):
        utts, rms, cond, prompt_lens, total_lens, text, video = by_id[row['target_id']]
        p, total = prompt_lens[0], total_lens[0]
        if total-p != 300 or total > 4096:
            raise ValueError('Native Baseline target duration changed')
        with torch.inference_mode():
            generated,_ = model.sample(cond=cond.to(device),text=text,
                duration=torch.tensor(total_lens,device=device),video=video.to(device),
                lens=torch.tensor(prompt_lens,device=device),steps=32,cfg_strength=5.,
                cfg_strength_v=2.,sway_sampling_coef=-1.,seed=0,use_epss=True)
            wave = decoder(generated[:,p:total].float().transpose(1,2)).squeeze(1).cpu()
            # Exactly the original infer.py output rule, including the cached-
            # Mel path: do not invent a peak-safe online input normalization.
            if rms[0] < .1:
                wave = wave * rms[0] / .1
        return wave.squeeze().numpy(),16000,{'prompt_frames':p,'target_frames':300,
            'nfe':32,'cfg_text':5.,'cfg_video':2.,'sway':-1.,'weights':'EMA',
            'reference_frontend':'native cached unscaled Mel',
            'reference_mel_sha256':reference_records[row['ref_id']]['mel_sha256'],
            'target_acoustic_inputs':False,'native_adapter_sha256':sha(__file__)}
    return generate
