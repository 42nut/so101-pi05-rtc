"""Non-blocking leader-only coarse alignment; all units are calibrated positions."""
import time
import numpy as np
from dagger_control import vector


class LeaderAlignment:
    def __init__(self, bus, joints, rate=5.0, timeout=60.0, clock=None):
        self.bus, self.joints = bus, joints
        self.rate, self.timeout, self.clock = rate, timeout, clock or time.monotonic
        self.active = False
        self.torque_attempted = False

    def read(self):
        q = self.bus.sync_read('Present_Position')
        return vector([q[j] for j in self.joints])

    def stop(self):
        # Always attempt release if enabling was attempted, even if that call failed.
        if self.torque_attempted:
            self.bus.disable_torque()
            self.torque_attempted = False
        self.active = False

    def start(self, target):
        self.target = vector(target).copy()
        if np.any(self.target < np.array([-100]*5+[0])) or np.any(self.target > 100):
            raise ValueError('Follower pose outside calibrated leader command range')
        try:
            self.bus.disable_torque()
            self.goal = self.read()
            # Seed present position BEFORE enabling torque; do not revive old goals.
            for j, value in zip(self.joints, self.goal):
                self.bus.write('Goal_Position', j, float(value))
            self.torque_attempted = True
            self.bus.enable_torque()
            self.started = self.last_tick = self.clock()
            self.active = True
        except BaseException:
            self.stop()
            raise

    def tick(self, follower):
        if not self.active:
            return True
        now = self.clock()
        if np.max(np.abs(vector(follower)-self.target)) > 2:
            self.stop()
            raise RuntimeError('Follower moved during alignment; finish manually')
        if now-self.started >= self.timeout:
            self.stop()
            raise TimeoutError('Coarse alignment timed out; finish manually')
        if now-self.last_tick < 1/30:
            return False
        # Never catch up a blocked loop with a large position jump.
        dt = min(now-self.last_tick, 1/30)
        self.last_tick = now
        current = self.read()
        tolerance = np.array([2,2,4,2,2,2])
        if np.all(np.abs(current-self.target) <= tolerance):
            self.stop()
            return True
        self.goal += np.clip(self.target-self.goal, -self.rate*dt, self.rate*dt)
        self.goal = np.clip(self.goal, current-2, current+2)
        self.bus.sync_write('Goal_Position', dict(zip(self.joints,self.goal.tolist())))
        return False
