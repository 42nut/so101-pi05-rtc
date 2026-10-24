"""Single-owner queue. Indices count actions actually consumed, never HTTP ticks."""
import numpy as np


class StaleReply(RuntimeError):
    pass


def checked_chunk(value):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (50, 6) or not np.isfinite(result).all():
        raise ValueError(f"Invalid action chunk: {result.shape}")
    return result.copy()


class Queue:
    def __init__(self):
        self.normalized = None
        self.actions = None
        self.cursor = 0
        self.consumed = 0
        self.generation = 0

    def remaining(self):
        return 0 if self.actions is None else len(self.actions) - self.cursor

    def bootstrap(self, normalized, actions):
        self.normalized = checked_chunk(normalized)
        self.actions = checked_chunk(actions)
        self.cursor = 0
        self.generation += 1

    def snapshot(self):
        if self.remaining() < 1:
            raise StaleReply("Empty queue: cannot build an RTC prefix")
        return dict(generation=self.generation, consumed=self.consumed,
                    remaining=self.remaining(),
                    prefix=self.normalized[self.cursor:].copy().tolist())

    def merge(self, snapshot, normalized, actions):
        delay = self.consumed - snapshot['consumed']
        if snapshot['generation'] != self.generation or not 0 <= delay < snapshot['remaining']:
            raise StaleReply("Reply belongs to an old/exhausted action timeline")
        norm, physical = checked_chunk(normalized), checked_chunk(actions)
        if delay >= len(physical):
            raise StaleReply("Entire reply has expired")
        self.normalized, self.actions = norm, physical
        self.cursor = delay  # Do not replay actions elapsed during inference.
        self.generation += 1
        return delay

    def pop(self):
        if self.remaining() < 1:
            raise StaleReply("Queue underrun: stop instead of replaying old actions")
        action = self.actions[self.cursor].copy()
        self.cursor += 1
        self.consumed += 1
        return action
