import json
from pathlib import Path
import argparse
ap=argparse.ArgumentParser()
ap.add_argument('--root',default='/run/media/k/新加卷/lerobot_datasets/so101_dagger_v1')
a=ap.parse_args()
root=Path(a.root)
counts={'train':0,'validation':0}
for p in sorted(root.glob('episode_*/metadata.json')):
    d=json.loads(p.read_text())
    info_path=p.parent/'dataset/meta/info.json'
    info=json.loads(info_path.read_text()) if info_path.exists() else {}
    print(p.parent.name,d.get('split'),d.get('category'),d.get('status'),
          'frames=',d.get('frames',0),'fps=',round(d.get('effective_fps',0),2),
          'saved_episodes=',info.get('total_episodes',0))
    if d.get('status')=='accepted' and info.get('total_episodes')==1:
        counts[d['split']]+=1
print('可合并数量：',counts,'目标 train=16 validation=4')
print('只合并 accepted；其他状态需要检查或重录。每个子数据集内部 episode_index=0。')
