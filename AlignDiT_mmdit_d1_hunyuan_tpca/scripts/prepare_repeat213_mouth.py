"""Use existing FAN/canonical ROI preprocessing, rejecting whole-face fallbacks."""
import argparse
import json
from pathlib import Path
import cv2
import face_alignment
import numpy as np
import torch
from aligndit.script.misc import crop_mouth_celebvdub as crop

p=argparse.ArgumentParser()
p.add_argument('--dataset',type=Path,required=True)
p.add_argument('--mean-face',type=Path,required=True)
p.add_argument('--rank',type=int,default=0)
p.add_argument('--nshard',type=int,default=1)
a=p.parse_args()
cv2.setNumThreads(1)
torch.set_num_threads(1)
rows=[json.loads(s) for s in (a.dataset/'manifest.jsonl').read_text().splitlines()]
rows=rows[a.rank::a.nshard]
mean=np.load(a.mean_face)
assert mean.shape==(68,2)
fa=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cuda')
failures=[]
for index,row in enumerate(rows):
    key=row['utterance_key'].removeprefix('celebvdub/')
    dest=a.dataset/'CelebVDub/video_mouth'/(key+'.mp4')
    meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        print('SKIP',key,flush=True); continue
    try:
        source=a.dataset/'CelebVDub/video'/(key+'.mp4')
        frames=crop.read_video_frames(str(source))
        assert len(frames)==row['video_frames_25hz'], (key,len(frames),row['video_frames_25hz'])
        landmarks=crop.detect_landmarks_video(fa,[cv2.cvtColor(f,cv2.COLOR_BGR2RGB) for f in frames],'cuda')
        detected=sum(x is not None for x in landmarks)
        landmarks=crop.landmarks_interpolate(landmarks)
        if landmarks is None: raise RuntimeError('No face detected; refusing whole-frame resize fallback')
        patches=crop.crop_patch(frames,landmarks,mean,crop.STABLE_PNTS_IDS,crop.STD_SIZE,crop.WINDOW_MARGIN,crop.START_IDX,crop.STOP_IDX,crop.CROP_HEIGHT,crop.CROP_WIDTH)
        gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) if f.ndim==3 else f for f in patches]
        dest.parent.mkdir(parents=True,exist_ok=True)
        crop.write_video_mp4(gray,str(dest))
        cap=cv2.VideoCapture(str(dest))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(frames)
        assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(96,96)
        assert cap.get(cv2.CAP_PROP_FPS)==25
        cap.release()
        meta.write_text(json.dumps(dict(utterance_id=key,frames=len(frames),detected_landmark_frames=detected,whole_frame_fallback=False,source=str(source.resolve())))+'\n')
        print(f'{index+1}/{len(rows)} OK {key} frames={len(frames)}',flush=True)
        del frames,landmarks,patches,gray
        torch.cuda.empty_cache()
    except Exception as e:
        failures.append((key,str(e)))
        print('FAILED',key,repr(e),flush=True)
print('CROP_COMPLETE',a.rank,'failures',failures,flush=True)
if failures: raise RuntimeError(failures)
