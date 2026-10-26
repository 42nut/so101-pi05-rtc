import unittest
import numpy as np
from dagger_control import Ownership, aligned, bounded_command, plan
from rtc_queue import Queue

class Tests(unittest.TestCase):
    def test_no_reply_after_takeover(self):
        o=Ownership();g=o.generation
        self.assertTrue(o.accepts(g))
        o.takeover()
        self.assertFalse(o.accepts(g))
        o.record(True)
        self.assertFalse(o.accepts(g))
    def test_no_record_without_alignment(self):
        o=Ownership()
        with self.assertRaises(RuntimeError):o.record(True)
        o.takeover()
        with self.assertRaises(RuntimeError):o.record(False)
        o.record(True);self.assertEqual(o.phase,'record')
    def test_joint_alignment(self):
        self.assertTrue(aligned([1]*6,[0]*6,5))
        self.assertFalse(aligned([0]*5+[6],[0]*6,5))
        with self.assertRaises(ValueError):aligned([float('nan')]*6,[0]*6,5)
    def test_actual_command_bounds(self):
        np.testing.assert_array_equal(bounded_command([200]*6,[0]*6,10),[10]*6)
        np.testing.assert_array_equal(bounded_command([-200]*6,[0]*6,10),[-10]*5+[0])
    def test_fixed_split(self):
        self.assertEqual(sum(plan(i)['split']=='train' for i in range(20)),16)
        self.assertEqual(sum(plan(i)['split']=='validation' for i in range(20)),4)
    def test_queue_reset_is_empty(self):
        q=Queue();q.bootstrap(np.zeros((50,6)),np.zeros((50,6)))
        q=Queue();self.assertEqual(q.remaining(),0)

if __name__=='__main__':unittest.main()
