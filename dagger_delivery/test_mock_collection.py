import sys, types, tempfile, importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
JOINTS=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
KEYS=[j+'.pos' for j in JOINTS]
TASK='test task'
health=dict(protocol='pi05-rtc-v1',ready=True,rtc=True,chunk_size=50,prefix_space='normalized',action_dim=6,task=TASK,joint_order=JOINTS,fps=30,model_token='test')
class Config:
    def __init__(self,**kw):self.__dict__.update(kw)
class Device:
    def __init__(self,cfg):
        self.calibration={'saved':True};self.is_connected=False;self.is_calibrated=True
        self.bus=self;self.motors=dict.fromkeys(JOINTS);self.q=np.zeros(6);self.writes=[]
    def disable_torque(self):self.torque=False
    def enable_torque(self):self.torque=True
    def write(self,name,j,value):pass
    def sync_write(self,name,values):self.q=np.array([values[j] for j in JOINTS])
    def connect(self,calibrate=True):
        assert calibrate is False;self.is_connected=True
    def disconnect(self):self.is_connected=False
    def sync_read(self,name):return dict(zip(JOINTS,self.q))
    def get_action(self):return dict(zip(KEYS,np.zeros(6)))
    def get_observation(self):return dict(zip(KEYS,self.q),front=np.zeros((480,640,3),dtype=np.uint8),wrist=np.zeros((480,640,3),dtype=np.uint8))
    def send_action(self,a):self.writes.append(a.copy());self.q=np.array([a[k] for k in KEYS]);return a
class DS:
    instance=None
    @classmethod
    def create(cls,**kw):
        cls.instance=cls();cls.instance.frames=[];cls.instance.saved=False;return cls.instance
    def add_frame(self,f,task):self.frames.append(f)
    def save_episode(self):self.saved=True
    def stop_image_writer(self):pass
m=types.ModuleType('pi05_rtc_client')
for name,value in dict(JOINTS=JOINTS,KEYS=KEYS,TASK=TASK,encode_rgb=lambda a:'image',SO101Follower=Device,SO101FollowerConfig=Config,OpenCVCameraConfig=Config).items():setattr(m,name,value)
def req(url,payload=None,timeout=5):
    if payload is None:return health
    return dict(request_id=payload['request_id'],model_token='test',rtc_guided=payload['prefix_normalized'] is not None,normalized_actions=np.zeros((50,6)).tolist(),actions=np.zeros((50,6)).tolist(),inference_ms=1)
m.request_json=req;sys.modules['pi05_rtc_client']=m
for name,attrs in [('lerobot.datasets.lerobot_dataset',{'LeRobotDataset':DS}),('lerobot.teleoperators.so101_leader',{'SO101Leader':Device,'SO101LeaderConfig':Config})]:
    z=types.ModuleType(name);z.__dict__.update(attrs);sys.modules[name]=z
spec=importlib.util.spec_from_file_location('collector',str(Path(__file__).resolve().parent/'pi05_dagger_collect.py'));mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class KB:
    def __enter__(self):self.n=0;return self
    def __exit__(self,*a):pass
    def read(self):
        self.n+=1
        if self.n==1:return 'a'
        if self.n==3:return 't'
        if self.n > 1000:raise AssertionError('State transition stalled')
        if self.n>=4 and len(DS.instance.frames)==0:return 'r'
        return 's' if len(DS.instance.frames)>=32 else ''
class Future:
    def __init__(self,fn,*a):self.result_value=fn(*a)
    def done(self):return True
    def result(self):return self.result_value
    def cancel(self):pass
class Pool:
    def __init__(self,**kw):pass
    def submit(self,fn,*a):return Future(fn,*a)
    def shutdown(self,**kw):pass
clock=[0]
def mono():clock[0]+=.005;return clock[0]
with tempfile.TemporaryDirectory() as root,patch.object(mod,'Keyboard',KB),patch.object(mod,'ThreadPoolExecutor',Pool),patch.object(mod.time,'monotonic',mono),patch.object(mod.time,'sleep',lambda x:None),patch('builtins.input',side_effect=['upper left','YES','corrected']),patch.object(sys,'argv',['collect','--episode','16','--root',root]):
    mod.main()
    assert DS.instance.saved and len(DS.instance.frames)==32
    import json
    meta=json.loads((Path(root)/'episode_000016/metadata.json').read_text())
    assert meta['split']=='validation'
    logs=[json.loads(x) for x in (Path(root)/'episode_000016/rollout.jsonl').read_text().splitlines()]
    t=next(i for i,x in enumerate(logs) if x['type']=='takeover')
    assert not any(x['type']=='policy_action' for x in logs[t:])
    assert sum(x['type']=='human_action' for x in logs)==32
    assert sum(x['type']=='policy_action' for x in logs)>0
    print('PASS: full mocked auto -> takeover -> alignment -> human-only frames -> save; validation split and no policy actions after takeover')
