import importlib.util
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import numpy as np
from rtc_queue import Queue, StaleReply


def chunk(start):
    return np.repeat(np.arange(start, start+50, dtype=np.float32)[:,None], 6, axis=1)

class QueueTests(unittest.TestCase):
    def test_actual_delay_alignment(self):
        for delay in (0, 3, 9, 17, 29):
            q = Queue(); q.bootstrap(chunk(0), chunk(0))
            for _ in range(20): q.pop()
            s = q.snapshot()
            for _ in range(delay): q.pop()
            q.merge(s, chunk(20), chunk(20))
            self.assertEqual(q.pop()[0], 20+delay)

    def test_repeated_merge_preserves_timeline(self):
        q = Queue(); q.bootstrap(chunk(0), chunk(0))
        for delay in (4, 12, 7, 16)*10:
            while q.cursor < 20:
                self.assertEqual(q.pop()[0], q.consumed-1)
            s = q.snapshot(); anchor = q.consumed
            for _ in range(delay): self.assertEqual(q.pop()[0], q.consumed-1)
            q.merge(s, chunk(anchor), chunk(anchor))
        self.assertGreater(q.consumed, 700)

    def test_generation_rejects_duplicate(self):
        q = Queue(); q.bootstrap(chunk(0),chunk(0)); s=q.snapshot()
        q.merge(s,chunk(0),chunk(0))
        with self.assertRaises(StaleReply): q.merge(s,chunk(0),chunk(0))

    def test_exhaustion_and_reset(self):
        q = Queue(); q.bootstrap(chunk(0),chunk(0)); s=q.snapshot()
        for _ in range(50): q.pop()
        with self.assertRaises(StaleReply): q.merge(s,chunk(0),chunk(0))
        with self.assertRaises(StaleReply): q.pop()
        q.bootstrap(chunk(0),chunk(0))
        with self.assertRaises(StaleReply): q.merge(s,chunk(0),chunk(0))

    def test_invalid_reply_does_not_mutate_queue(self):
        q = Queue(); q.bootstrap(chunk(0),chunk(0)); s=q.snapshot()
        bad=chunk(0);bad[2,0]=np.nan
        with self.assertRaises(ValueError): q.merge(s,bad,chunk(0))
        self.assertEqual(q.remaining(),50)
        self.assertEqual(q.generation,1)


class IntegrationTests(unittest.TestCase):
    def run_client(self, dry, bad_reply=False):
        owner = threading.get_ident()
        instances=[]; requests=[]
        joints=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        class Config:
            def __init__(self, **kwargs): self.__dict__.update(kwargs)
        class Robot:
            def __init__(self,cfg):
                self.is_connected=False;self.bus=self;self.motors=dict.fromkeys(joints)
                self.writes=[];instances.append(self)
            def check(self): assert threading.get_ident()==owner
            def connect(self): self.check();self.is_connected=True
            def disconnect(self): self.check();self.is_connected=False
            def sync_read(self,name): self.check();return dict.fromkeys(joints,0.)
            def get_observation(self):
                self.check()
                return {**{j+'.pos':0. for j in joints},
                        'front':np.zeros((480,640,3),dtype=np.uint8),
                        'wrist':np.zeros((480,640,3),dtype=np.uint8)}
            def send_action(self,a): self.check();self.writes.append(a);return a
        robot_module=types.ModuleType('lerobot.robots.so101_follower')
        robot_module.SO101Follower=Robot;robot_module.SO101FollowerConfig=Config
        camera_module=types.ModuleType('lerobot.cameras.opencv.configuration_opencv')
        camera_module.OpenCVCameraConfig=Config
        with patch.dict(sys.modules, {'lerobot.robots.so101_follower':robot_module,
                         'lerobot.cameras.opencv.configuration_opencv':camera_module}):
            spec=importlib.util.spec_from_file_location('rtc_client_test',Path(__file__).with_name('pi05_rtc_client.py'))
            client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
        def http(url,payload=None,timeout=5):
            if payload is None:
                return dict(protocol='pi05-rtc-v1',ready=True,rtc=True,chunk_size=50,
                    prefix_space='normalized',action_dim=6,task=client.TASK,joint_order=joints,
                    fps=30,model_token='test',checkpoint='pi05_recovery_cosine_tc_v1')
            requests.append(payload)
            prefix=payload['prefix_normalized']; guided=prefix is not None
            if guided:
                assert threading.get_ident()!=owner
                assert 0<=payload['inference_delay']<len(prefix)
                time.sleep(.1)
            actions=np.zeros((50,6)).tolist()
            return dict(request_id='wrong' if bad_reply and guided else payload['request_id'],
                model_token='test',rtc_guided=guided,actions=actions,normalized_actions=actions,inference_ms=50)
        argv=['test','--seconds','1.2']+(['--dry-run'] if dry else [])
        with tempfile.TemporaryDirectory() as tmp, patch.object(client,'__file__',str(Path(tmp)/'client.py')), \
             patch.object(client,'request_json',http),patch.object(sys,'argv',argv):
            if bad_reply:
                with self.assertRaisesRegex(RuntimeError,'Stale or foreign'):
                    client.main()
            else: client.main()
        self.assertGreaterEqual(len(requests),2)
        self.assertFalse(instances[0].is_connected)
        if dry: self.assertEqual(instances[0].writes,[])
        else: self.assertGreater(len(instances[0].writes),10)
    def test_dry_run_never_writes(self): self.run_client(True)
    def test_live_mock_single_hardware_owner(self): self.run_client(False)
    def test_wrong_response_stops_and_disconnects(self): self.run_client(False,True)

if __name__=='__main__': unittest.main(verbosity=2)
