import unittest
import numpy as np
from leader_alignment import LeaderAlignment

J=['a','b','c','d','e','f']
class Bus:
    def __init__(self):self.q=np.zeros(6);self.calls=[];self.fail=False
    def disable_torque(self):self.calls.append('off')
    def enable_torque(self):
        self.calls.append('on')
        if self.fail:raise RuntimeError('enable failed')
    def write(self,n,j,v):self.calls.append(('seed',j,v))
    def sync_read(self,n):return dict(zip(J,self.q))
    def sync_write(self,n,v):self.calls.append(('move',v.copy()));self.q=np.array(list(v.values()))
class Tests(unittest.TestCase):
    def setup_aligner(self):
        b=Bus();t=[0.];a=LeaderAlignment(b,J,clock=lambda:t[0]);return b,t,a
    def test_preseed_rate_and_release(self):
        b,t,a=self.setup_aligner();target=np.ones(6)*10;a.start(target)
        self.assertEqual(b.calls[0],'off');self.assertEqual(b.calls[7],'on')
        t[0]=1;a.tick(target)
        self.assertLessEqual(max(b.q),5/30+1e-9)
        for _ in range(400):
            t[0]+=.034
            if a.tick(target):break
        self.assertFalse(a.active);self.assertEqual(b.calls[-1],'off')
    def test_enable_error_releases(self):
        b,t,a=self.setup_aligner();b.fail=True
        with self.assertRaises(RuntimeError):a.start(np.ones(6))
        self.assertEqual(b.calls[-1],'off')
    def test_follower_drift_releases(self):
        b,t,a=self.setup_aligner();a.start(np.ones(6)*10)
        with self.assertRaises(RuntimeError):a.tick(np.ones(6)*15)
        self.assertEqual(b.calls[-1],'off')
    def test_cancel_releases(self):
        b,t,a=self.setup_aligner();a.start(np.ones(6)*10);a.stop()
        self.assertEqual(b.calls[-1],'off');self.assertFalse(a.active)
    def test_timeout_releases(self):
        b,t,a=self.setup_aligner();a.start(np.ones(6)*10);t[0]=61
        with self.assertRaises(TimeoutError):a.tick(np.ones(6)*10)
        self.assertEqual(b.calls[-1],'off')
if __name__=='__main__':unittest.main()
