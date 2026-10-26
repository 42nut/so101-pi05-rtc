"""Pure control helpers: no hardware or network access."""
import numpy as np


def vector(values):
    x = np.asarray(values, dtype=np.float64)
    if x.shape != (6,) or not np.isfinite(x).all():
        raise ValueError('Expected six finite joint values')
    return x


def aligned(leader, follower, tolerance):
    return bool(np.all(np.abs(vector(leader)-vector(follower)) <= tolerance))


def bounded_command(target, last, slew):
    target = np.clip(vector(target), [-100]*5+[0], [100]*6)
    return vector(last) + np.clip(target-vector(last), -slew, slew)


def plan(index):
    if not 0 <= index < 20:
        raise ValueError('Episode must be 0..19')
    category = ('upper_left' if index < 6 or index in (16,17)
                else 'empty_grasp' if index < 11 or index == 18 else 'fixed_finger_contact')
    return {'planned_episode': index, 'split': 'train' if index < 16 else 'validation',
            'category': category}


class Ownership:
    def __init__(self):
        self.phase = 'auto'
        self.generation = 0

    def takeover(self):
        self.generation += 1
        self.phase = 'align'

    def record(self, is_aligned):
        if self.phase != 'align' or not is_aligned:
            raise RuntimeError('Recording requires aligned leader and follower')
        self.phase = 'record'

    def finish(self):
        self.generation += 1
        self.phase = 'review'

    def accepts(self, generation):
        return self.phase == 'auto' and self.generation == generation
