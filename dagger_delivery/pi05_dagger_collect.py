"""One independently saved human correction per run, with explicit train/val identity."""
import argparse
import inspect
import json
import math
import select
import sys
import termios
import time
import tty
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from pi05_rtc_client import (JOINTS, KEYS, TASK, request_json, encode_rgb,
                            SO101Follower, SO101FollowerConfig, OpenCVCameraConfig)
from rtc_queue import Queue, StaleReply
from dagger_control import Ownership, aligned, bounded_command, vector, plan
from leader_alignment import LeaderAlignment
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


class Keyboard:
    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError('Run in an interactive foreground terminal, not nohup')
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        return sys.stdin.read(1).lower() if select.select([sys.stdin], [], [], 0)[0] else ''

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def connect_existing(device):
    # Refuse calibration prompts or writing replacement calibration files.
    if not getattr(device, 'calibration', None):
        raise RuntimeError(f'No existing calibration loaded for {device}; stop and check ID')
    if 'calibrate' not in inspect.signature(device.connect).parameters:
        raise RuntimeError('connect(calibrate=False) unavailable; no connection attempted')
    device.connect(calibrate=False)
    if not device.is_calibrated:
        raise RuntimeError('Hardware calibration does not match saved calibration; no actions allowed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--episode', required=True, type=int)
    ap.add_argument('--retry', action='store_true', help='Archive a non-accepted attempt before re-recording')
    ap.add_argument('--root', default='/run/media/k/新加卷/lerobot_datasets/so101_dagger_v1')
    ap.add_argument('--url', default='http://127.0.0.1:18088')
    ap.add_argument('--port', default='/dev/ttyACM1')
    ap.add_argument('--leader-port', default='/dev/ttyACM0')
    ap.add_argument('--request-after', type=int, default=10)
    ap.add_argument('--initial-delay', type=int, default=12)
    ap.add_argument('--slew', type=float, default=10.0)
    ap.add_argument('--max-relative-target', type=float, default=20.0)
    ap.add_argument('--timeout', type=float, default=3.0)
    ap.add_argument('--align-tolerance', type=float, default=2.0)
    ap.add_argument('--seconds', type=float, default=120, help='Maximum autonomous duration')
    ap.add_argument('--record-seconds', type=float, default=120)
    args = ap.parse_args()
    planned = plan(args.episode)
    if not (1 <= args.request_after <= 35 and 1 <= args.initial_delay < 50-args.request_after
            and all(math.isfinite(v) and v > 0 for v in
                    (args.slew,args.max_relative_target,args.timeout,args.seconds,args.record_seconds))
            and 0 < args.align_tolerance <= 2):
        ap.error('Invalid timing/limit settings')
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    slot = root / f'episode_{args.episode:06d}'
    if slot.exists() and args.retry:
        old = json.loads((slot/'metadata.json').read_text())
        if old.get('status') == 'accepted':
            raise RuntimeError('Accepted episode already exists; refusing to replace it')
        archive = root/'rejected_attempts'
        archive.mkdir(exist_ok=True)
        slot.rename(archive/(slot.name+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    slot.mkdir()  # Never overwrite an existing attempt.
    info = dict(planned, status='incomplete', task=TASK, args=vars(args),
                recorded_policy='human_only', dataset_episode=0)
    def metadata():
        tmp = slot/'metadata.tmp'
        tmp.write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
        tmp.replace(slot/'metadata.json')
    metadata()
    print('本条计划:', planned, flush=True)
    info['scene_note'] = input('记录方块位置/朝向及计划起始姿态：').strip()
    health = request_json(args.url.rstrip('/')+'/health', timeout=args.timeout)
    for k,v in dict(protocol='pi05-rtc-v1',ready=True,rtc=True,chunk_size=50,
                    prefix_space='normalized',action_dim=6,task=TASK,joint_order=JOINTS,fps=30).items():
        if health.get(k) != v: raise RuntimeError(f'Server mismatch: {k}')
    info['server'] = health
    metadata()
    robot = SO101Follower(SO101FollowerConfig(
        port=args.port,
        id="my_awesome_follower_arm",
        use_degrees=False,
        disable_torque_on_disconnect=False,
        max_relative_target=args.max_relative_target,
        cameras={
            "front": OpenCVCameraConfig(
                index_or_path=Path(
                    "/dev/v4l/by-id/"
                    "usb-icSpring_icspring_camera_202404160005-video-index0"
                ),
                width=640, height=480, fps=30,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=Path(
                    "/dev/v4l/by-id/"
                    "usb-icSpring_icspring_camera-video-index0"
                ),
                width=640, height=480, fps=30,
            ),
        },
    ))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port,
                         id='my_awesome_leader_arm', use_degrees=False))
    features = {
        'observation.state': {'dtype':'float32','shape':(6,), 'names':KEYS},
        'action': {'dtype':'float32','shape':(6,), 'names':KEYS},
        **{f'observation.images.{cam}': {'dtype':'video','shape':(480,640,3),
            'names':['height','width','channels']} for cam in ('front','wrist')},
    }
    ds = LeRobotDataset.create(repo_id=f'local/so101_dagger_{args.episode:06d}',
        root=slot/'dataset', fps=30, features=features, robot_type='so101_follower',
        use_videos=True, image_writer_processes=0, image_writer_threads=4)
    aligner = LeaderAlignment(leader.bus, JOINTS)
    fine_since = None
    owner = Ownership()
    queue = Queue()
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    latencies = deque(maxlen=20)
    last = None
    frames = 0
    frame_times = []
    log = (slot/'rollout.jsonl').open('x', buffering=1)
    start = time.monotonic()
    def emit(kind, **kw):
        log.write(json.dumps(dict(type=kind,t=time.monotonic()-start,**kw),allow_nan=False)+'\n')
    def state():
        x = robot.bus.sync_read('Present_Position')
        return vector([x[j] for j in JOINTS])
    def hold():
        nonlocal last
        last = state()
        robot.send_action(dict(zip(KEYS,last.tolist())))
    def takeover(reason):
        nonlocal pending, queue, fine_since
        owner.takeover()  # Invalidates every outstanding autonomous reply first.
        if pending is not None: pending[0].cancel()
        pending = None
        queue = Queue()
        hold()
        info['takeover_reason'] = reason
        info['takeover_state'] = last.tolist()
        metadata()
        emit('takeover',reason=reason,state=last.tolist())
        fine_since = None
        print('\n从臂已保持。主臂即将以5单位/秒自动对齐，请松开主臂；T取消自动对齐，Q退出。',flush=True)
        try:
            aligner.start(last)
            owner.phase = 'auto_align'
        except Exception as exc:
            aligner.stop()
            emit('alignment_error', error=repr(exc))
            print('自动对齐未启动，已释放主臂，改为人工微调：',exc,flush=True)
    def observation():
        obs = robot.get_observation()
        x = vector([obs[k] for k in KEYS])
        return obs, (x, np.array(obs['front'],copy=True), np.array(obs['wrist'],copy=True))
    def fetch(obs,prefix,delay,rid):
        x,front,wrist=obs
        response=request_json(args.url.rstrip('/')+'/infer',dict(protocol='pi05-rtc-v1',
            model_token=health['model_token'],request_id=rid,state=x.tolist(),
            front=encode_rgb(front),wrist=encode_rgb(wrist),prefix_normalized=prefix,
            inference_delay=delay),timeout=args.timeout)
        if response.get('model_token') != health['model_token'] or response.get('request_id') != rid:
            raise RuntimeError('Foreign/stale reply')
        if response.get('rtc_guided') != (prefix is not None): raise RuntimeError('RTC mismatch')
        return response
    try:
        connect_existing(robot)
        connect_existing(leader)
        if set(robot.bus.motors) != set(JOINTS): raise RuntimeError('Unexpected motors')
        hold()
        print('连接完成；先保持。A开始模型控制；T接管；R对齐后录制；S结束录制；Q退出。',flush=True)
        owner.phase='idle'
        auto_start=None
        last_display=0
        next_tick=time.monotonic()
        with Keyboard() as keyboard:
            while owner.phase != 'review':
                key=keyboard.read()
                if key=='q':
                    aligner.stop()
                    hold(); owner.finish(); break
                if key=='a' and owner.phase=='idle':
                    owner.phase='auto'
                    auto_start=time.monotonic()
                    _,obs=observation()
                    rid=uuid.uuid4().hex
                    pending=(pool.submit(fetch,obs,None,0,rid),None,time.monotonic(),0,rid,owner.generation)
                    print('模型控制启动；T随时接管。',flush=True)
                if key=='t' and owner.phase in ('auto','idle'):
                    takeover('manual')
                    key = ''  # Do not treat the same T as an alignment cancellation.
                if owner.phase=='auto':
                    try:
                        now=time.monotonic()
                        if now-auto_start > args.seconds:
                            takeover('autonomous_time_limit'); continue
                        if pending is not None:
                            future,snap,sent_at,predicted,rid,generation=pending
                            age=now-sent_at
                            budget=args.timeout if snap is None else snap['remaining']/30
                            if age >= budget: raise StaleReply('Request expired')
                            if future.done():
                                result=future.result()
                                if not owner.accepts(generation): raise StaleReply('Control owner changed')
                                if snap is None:
                                    queue.bootstrap(result['normalized_actions'],result['actions'])
                                    # Bootstrap may include cold-start overhead; do not use it as steady RTC latency.
                                    emit('bootstrap',response=result,request_ms=age*1000)
                                    next_tick=time.monotonic()
                                else:
                                    delay=queue.merge(snap,result['normalized_actions'],result['actions'])
                                    latencies.append(age)
                                    emit('merge',response=result,actual_delay=delay,predicted=predicted,request_ms=age*1000)
                                    print(f'RTC actual={delay} predicted={predicted} request={age*1000:.0f}ms queue={queue.remaining()}',flush=True)
                                pending=None
                        if queue.actions is not None and pending is None and queue.cursor>=args.request_after:
                            predicted=max(args.initial_delay,math.ceil(max(latencies,default=0)*30)+2)
                            if predicted>=queue.remaining(): raise StaleReply('Latency exceeds queue budget')
                            _,obs=observation()
                            snap=queue.snapshot(); rid=uuid.uuid4().hex
                            pending=(pool.submit(fetch,obs,snap['prefix'],predicted,rid),snap,
                                     time.monotonic(),predicted,rid,owner.generation)
                        if queue.actions is not None and time.monotonic()>=next_tick:
                            target=queue.pop()
                            cmd=bounded_command(target,last,args.slew)
                            sent=robot.send_action(dict(zip(KEYS,cmd.tolist())))
                            last=vector([sent[k] for k in KEYS])
                            emit('policy_action',target=target.tolist(),sent=last.tolist())
                            next_tick=max(next_tick+1/30,time.monotonic())
                    except Exception as exc:
                        emit('auto_error',error=repr(exc))
                        takeover('auto_error: '+repr(exc))
                elif owner.phase=='auto_align':
                    try:
                        done = False
                        if key == 't':
                            aligner.stop()
                            done = True
                        else:
                            done = aligner.tick(state())
                        if done:
                            owner.phase = 'align'
                            fine_since = None
                            print('主臂力矩已释放。手扶主臂微调，全部关节差值<=2稳定0.5秒后按R。',flush=True)
                    except Exception as exc:
                        aligner.stop()
                        owner.phase = 'align'
                        fine_since = None
                        emit('alignment_error', error=repr(exc))
                        print('自动对齐停止，已释放主臂，改为人工微调：',exc,flush=True)
                elif owner.phase=='align':
                    follower=state()
                    lead=leader.get_action()
                    lead=vector([lead[k] for k in KEYS])
                    delta=lead-follower
                    if aligned(lead,follower,args.align_tolerance):
                        if fine_since is None: fine_since = time.monotonic()
                    else:
                        fine_since = None
                    if time.monotonic()-last_display>1:
                        print('主臂-从臂差值:',dict(zip(JOINTS,np.round(delta,1).tolist())),flush=True)
                        last_display=time.monotonic()
                    if key=='r':
                        if fine_since is None or time.monotonic()-fine_since < .5:
                            print('需全部关节差值 <=',args.align_tolerance,'并稳定0.5秒；稍后再按R。',flush=True)
                        else:
                            owner.record(True)
                            last=follower
                            record_start=time.monotonic()
                            next_tick=record_start
                            emit('human_record_start',state=follower.tolist(),leader=lead.tolist())
                            print('正在录制人工纠正，完成放置后按S。',flush=True)
                elif owner.phase=='record':
                    if key=='s' or time.monotonic()-record_start>=args.record_seconds:
                        hold();owner.finish();break
                    if time.monotonic()>=next_tick:
                        obs,_=observation()  # Observation BEFORE corresponding expert action.
                        lead=leader.get_action()
                        target=vector([lead[k] for k in KEYS])
                        cmd=bounded_command(target,last,args.slew)
                        sent=robot.send_action(dict(zip(KEYS,cmd.tolist())))
                        last=vector([sent[k] for k in KEYS])
                        stamp=time.monotonic()
                        frame_times.append(stamp)
                        ds.add_frame({'observation.state':np.asarray([obs[k] for k in KEYS],dtype=np.float32),
                                      'action':last.astype(np.float32),
                                      'observation.images.front':np.array(obs['front'],copy=True),
                                      'observation.images.wrist':np.array(obs['wrist'],copy=True)},task=TASK)
                        frames+=1
                        emit('human_action',frame=frames-1,requested=target.tolist(),sent=last.tolist(),
                             state=[float(obs[k]) for k in KEYS])
                        next_tick=max(next_tick+1/30,time.monotonic())
                time.sleep(.001 if owner.phase!='align' else .03)
        if frames:
            intervals=np.diff(frame_times)
            effective_fps=(frames-1)/(frame_times[-1]-frame_times[0]) if frames>1 else 0
            info.update(frames=frames,effective_fps=effective_fps,
                        max_frame_interval_s=float(max(intervals,default=0)))
            print(f'纠正帧数={frames}, 实际平均频率={effective_fps:.2f}Hz',flush=True)
            print('YES=成功纠正并保存；其他输入=标记废弃，本次不纳入训练。')
            accept=input('是否保存为有效示范：').strip()=='YES'
            info['failure_note']=input('记录实际失败、偏移方向及接管操作：').strip()
            if accept:
                # Preserve low-rate takes for review rather than silently training on wrong time spacing.
                ds.save_episode()
                timing_ok=frames>=30 and effective_fps>=27 and info['max_frame_interval_s']<=.2
                info['status']='accepted' if timing_ok else 'timing_review'
                print('已保存：',slot/'dataset','状态：',info['status'],flush=True)
            else:
                info['status']='rejected'
        else:
            info.update(status='rejected',frames=0)
        metadata()
    except KeyboardInterrupt:
        info.update(status='interrupted',frames=frames)
        metadata()
        print('中断；未确认的数据不纳入训练。',flush=True)
    except Exception as exc:
        info.update(status='error',frames=frames,error=repr(exc));metadata()
        raise
    finally:
        try:
            aligner.stop()
        except Exception as exc:
            print('主臂力矩释放失败，请硬件停止：',exc,flush=True)
        owner.finish()
        if pending is not None: pending[0].cancel()
        pool.shutdown(wait=False,cancel_futures=True)
        if robot.is_connected:
            try: hold()
            except Exception as exc: print('保持姿态失败，请硬件停止：',exc,flush=True)
        for device in (leader,robot):
            if device.is_connected:
                try: device.disconnect()
                except Exception as exc: print('断开失败：',exc,flush=True)
        stop_writer=getattr(ds,'stop_image_writer',None)
        if callable(stop_writer): stop_writer()
        log.close()


if __name__=='__main__': main()
