"""CPU-only regression checks; no model or LeRobot dependency."""
import ast
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('dagger', HERE / 'train_pi05_dagger.py')
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

class Checks(unittest.TestCase):
    def test_ratios_and_resume(self):
        pools = [list(range(17)), list(range(100, 129)), list(range(200, 211))]
        sampler = m.GroupSampler(pools)
        totals = [0, 0, 0]
        membership = {v: g for g, p in enumerate(pools) for v in p}
        for u in range(m.TOTAL):
            draw = sampler.draw_update(u)
            counts = [sum(membership[x] == g for x in draw) for g in range(3)]
            self.assertEqual(tuple(counts), m.update_counts(u))
            totals = [a+b for a,b in zip(totals, counts)]
        self.assertEqual(totals, [307200,307200,409600])
        for step in [1, 499, 500, 1000, 3999]:
            resumed = m.GroupSampler(pools, start_step=step)
            self.assertEqual(list(itertools.islice(resumed, 256)), sampler.draw_update(step))
            self.assertEqual(m.accumulated_sampler_state(step, 59403, 8, 1)['start_index'], step*256)
        random.seed(4)
        a = sampler.draw_update(20)
        random.seed(10000)
        self.assertEqual(a, sampler.draw_update(20))

    def test_native_transform(self):
        source=(HERE/'reference_native_train.py').read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(), m.NATIVE_SHA256)
        tree=m.transform_formal(source)
        compile(tree, '<test>', 'exec')
        text=ast.unparse(tree)
        self.assertNotIn('EpisodeAwareSampler(', text)
        self.assertNotIn('compute_sampler_state(', text)
        self.assertEqual(text.count('_dagger_sampler(dataset, cfg, step, active_cfg)'),1)
        self.assertEqual(text.count('_dagger_count_batch(batch)'),1)
        self.assertIn('drop_last=True',text)
        self.assertIn('range(step, min(cfg.steps, _formal_stop_step))',text)

    def test_filtered_index_space(self):
        from types import SimpleNamespace
        episodes=[]; frames=[]
        for group, total in zip(m.GROUP_EPISODES,[28083,17760,13560]):
            q,r=divmod(total,len(group))
            for k,ep in enumerate(group):
                n=q+int(k<r)
                episodes.extend([ep]*n)
                frames.extend(range(n))
        class Columns(dict):
            def select_columns(self, names):
                return self
        class Dataset:
            hf_dataset=Columns(episode_index=episodes,frame_index=frames)
            def __len__(self): return len(episodes)
        sampler=m.make_group_sampler(Dataset(),SimpleNamespace(seed=42),500,SimpleNamespace(drop_n_last_frames=2))
        for group,pool in zip(m.GROUP_EPISODES,sampler.pools):
            self.assertTrue(all(episodes[i] in group for i in pool))
        self.assertEqual([len(p) for p in sampler.pools],[28083-90,17760-76,13560-36])

    def test_schedule(self):
        self.assertAlmostEqual(m.LR*m.wsd_factor(0),1e-7)
        self.assertAlmostEqual(m.LR*m.wsd_factor(99),1e-5)
        self.assertAlmostEqual(m.LR*m.wsd_factor(3999),1e-6)
        self.assertGreater(m.wsd_factor(999),m.wsd_factor(1999))

    def test_split(self):
        path=HERE/'reference_split.json'
        s=json.loads(path.read_text())
        m.validate_split(path,Path(s['dataset_root']))
        self.assertEqual(len(m.TRAIN_EPISODES),101)
        self.assertFalse(set(m.TRAIN_EPISODES)&set(m.VALIDATION_EPISODES))

if __name__ == '__main__':
    unittest.main()
