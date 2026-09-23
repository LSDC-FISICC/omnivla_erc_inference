"""In-place turn dynamics of every side-step in one or more bags: onset lag, rate,
overshoot after the command is cut, final angle. Source of the numbers behind
sidestep.py's lead_s / settle_s.

    source /opt/ros/jazzy/setup.bash
    python3 bag_turns.py <bag> [<bag> ...]
"""
import sys, re, numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
rows=[]
for bag in sys.argv[1:]:
    r=rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag,storage_id=''),rosbag2_py.ConverterOptions('',''))
    types={t.name:t.type for t in r.get_all_topics_and_types()}
    H=[];C=[];E=[]
    while r.has_next():
        tp,data,t=r.read_next(); t*=1e-9
        if tp=='/erc/heading_deg': H.append((t,deserialize_message(data,get_message(types[tp])).data))
        elif tp=='/cmd_vel':
            m=deserialize_message(data,get_message(types[tp])); C.append((t,m.linear.x,m.angular.z))
        elif tp=='/omnivla_debug':
            s=deserialize_message(data,get_message(types[tp])).data
            m=re.search(r'\[side-step\] (turn (left|right)|settle turned)',s)
            if m: E.append((t,m.group(1)))
    H=np.array(H);C=np.array(C)
    hu=np.degrees(np.unwrap(np.radians(H[:,1])))
    # turn start events: 'turn left/right'; settle events follow
    starts=[(t,k) for t,k in E if k.startswith('turn')]
    settles=[t for t,k in E if k.startswith('settle')]
    for t0,k in starts:
        ts=min([s for s in settles if s>t0],default=None)
        if ts is None: continue
        sgn= 1 if k=='turn right' else -1   # compass CW: right = increasing
        # command on/off as seen on /cmd_vel (after shaper)
        cm=(C[:,0]>t0-0.5)&(C[:,0]<ts+3)
        cw=C[cm]; on=cw[np.abs(cw[:,2])>0.05]
        if len(on)==0: continue
        t_on=on[0,0]; t_off=on[-1,0]
        h0=np.interp(t_on,H[:,0],hu)
        rel=lambda t: sgn*(np.interp(t,H[:,0],hu)-h0)
        tt=np.arange(t_on,t_off+8,0.05); rr=np.array([rel(x) for x in tt])
        mv=tt[np.argmax(rr>3)] if (rr>3).any() else np.nan
        final=rr.max()
        # heading stops: last time rr increases by >1 deg within next 0.5s
        inc=np.array([rel(x+0.5)-rel(x) for x in tt])
        stop_t=tt[np.where(inc>1)[0][-1]]+0.5 if (inc>1).any() else np.nan
        at_off=rel(t_off)
        # mid rate between 20% and 80% of final
        a=tt[np.argmax(rr>0.2*final)]; b=tt[np.argmax(rr>0.8*final)]
        rate=0.6*final/(b-a) if b>a else np.nan
        rows.append((mv-t_on, t_off-t_on, at_off, final, stop_t-t_off, rate, final-at_off))
R=np.array(rows)
names=['onset lag s','cmd duration s','turn read at cmd-off','final turn','keeps turning after off s','rate deg/s','overshoot after off deg']
for i,n in enumerate(names):
    v=R[:,i]; v=v[np.isfinite(v)]
    print(f'{n:28s} p10 {np.percentile(v,10):6.1f} p50 {np.median(v):6.1f} p90 {np.percentile(v,90):6.1f}  n={len(v)}')
