"""The local costmap built from a field bag's /erc/free_space exactly as the node builds it,
plus (optional) a replay of the replanner's decisions on the leg's real route.

Works on bags recorded WITHOUT local_replan (it only needs /erc/free_space,
/erc/gps/filtered, /erc/heading_deg, /goal_gps). Open loop: the rover in the bag
did not follow these plans. Runs under /usr/bin/python3 (rosbag2_py):

    source /opt/ros/jazzy/setup.bash
    python3 bag_local_map.py <bag> map.png                       # the map only
    python3 bag_local_map.py <bag> map.png <leg start, unix s> 3 # + first 3 replans -> map_replan.png
"""
import os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'erc_inference')))
import re, math, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from erc_inference import local_planner as lp
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag,out=sys.argv[1],sys.argv[2]
r=rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag,storage_id=''),rosbag2_py.ConverterOptions('',''))
types={t.name:t.type for t in r.get_all_topics_and_types()}
G=[];H=[];SC=[];B=[];GOAL=[]
while r.has_next():
    tp,data,t=r.read_next(); t*=1e-9
    if tp=='/erc/gps/filtered':
        m=deserialize_message(data,get_message(types[tp])); G.append((t,m.latitude,m.longitude))
    elif tp=='/erc/heading_deg': H.append((t,deserialize_message(data,get_message(types[tp])).data))
    elif tp=='/erc/free_space':
        m=deserialize_message(data,get_message(types[tp])); SC.append((t,np.array(m.ranges),m.angle_min,m.angle_increment,m.range_max))
    elif tp=='/goal_gps':
        m=deserialize_message(data,get_message(types[tp])); GOAL.append((t,m.latitude,m.longitude))
    elif tp=='/omnivla_debug':
        s=deserialize_message(data,get_message(types[tp])).data
        if '[side-step] brake ahead' in s: B.append(t)
G=np.array(G);H=np.array(H)
lat0,lon0=G[0,1],G[0,2]; kN=math.radians(1)*6378137.0; kE=kN*math.cos(math.radians(lat0))
E=(G[:,2]-lon0)*kE; N=(G[:,1]-lat0)*kN
hu=np.degrees(np.unwrap(np.radians(H[:,1])))
pl=lp.LocalReplanner(np.array([[E.min(),N.min()],[E.max(),N.max()]]))
t0=G[0,0]
for t,rng,amin,inc,rmax in SC:
    if t<G[0,0] or t>G[-1,0]: continue
    x=np.interp(t,G[:,0],E); y=np.interp(t,G[:,0],N)
    yaw=math.radians(90-np.interp(t,H[:,0],hu))
    b=np.degrees(amin+np.arange(len(rng))*inc)
    pl.observe(t,x,y,yaw,b,rng,rng<rmax-0.05)
m=pl.map; print(f'scans {pl.scans} skipped (turning) {pl.skipped_scans}; occupied cells {m.occupied().sum()}')
fig,ax=plt.subplots(figsize=(8,10))
ext=[m.ox,m.ox+m.w*m.p['resolution_m'],m.oy,m.oy+m.h*m.p['resolution_m']]
ax.imshow(np.where(m.seen,m.L,np.nan),origin='lower',extent=ext,cmap='RdBu_r',vmin=-2,vmax=3.5)
ax.plot(E,N,'g-',lw=1,label='GPS track (filtered)')
bi=[np.searchsorted(G[:,0],t) for t in B]
ax.plot(E[bi],N[bi],'kx',ms=9,mew=2,label='side-step brakes')
if GOAL:
    Gg=np.array(GOAL); ax.plot((Gg[:,2]-lon0)*kE,(Gg[:,1]-lat0)*kN,'.',color='orange',ms=2,label='carrot')
ax.set_xlim(E.min()-6,E.max()+6); ax.set_ylim(N.min()-4,N.max()+6); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7)
ax.set_title(f'{bag}: local costmap from /erc/free_space (red = occupied)',fontsize=9)
fig.tight_layout(); fig.savefig(out,dpi=90)

# ---- replay the replanner's decisions over time on the real map ----
if len(sys.argv) > 3:
    Gg=np.array(GOAL); ge=(Gg[:,2]-lon0)*kE; gn=(Gg[:,1]-lat0)*kN
    # the leg: carrots after t_leg; route = carrot positions, thinned to 0.5 m
    t_leg=float(sys.argv[3]); k=Gg[:,0]>=t_leg
    R=[(ge[k][0],gn[k][0])]
    for x,y in zip(ge[k],gn[k]):
        if math.dist(R[-1],(x,y))>=0.5: R.append((x,y))
    R=np.array(R); cum=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(R,axis=0).T))])
    pl2=lp.LocalReplanner(R); si=0; pts=R.copy(); c2=cum.copy(); s_proj=0.0; plans=[]
    for t,rng,amin,inc,rmax in SC:
        if t<t_leg or t>G[-1,0]: continue
        x=np.interp(t,G[:,0],E); y=np.interp(t,G[:,0],N); yaw=math.radians(90-np.interp(t,H[:,0],hu))
        b=np.degrees(amin+np.arange(len(rng))*inc)
        pl2.observe(t,x,y,yaw,b,rng,rng<rmax-0.05)
        # projection as the node does (import its helper lazily)
        d=np.hypot(pts[:,0]-x,pts[:,1]-y); j=int(np.argmin(d)); s_proj=max(s_proj,float(c2[j]))
        new,note=pl2.check(t,x,y,pts,c2,s_proj)
        if note: print(f'{t-t0:6.1f} {note}')
        if new is not None:
            plans.append((t,new)); pts=np.asarray(new); c2=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(pts,axis=0).T))]); s_proj=0.0
        if len(plans)>=int(sys.argv[4]): break
    fig,ax=plt.subplots(figsize=(8,10))
    m=pl2.map; ext=[m.ox,m.ox+m.w*m.p['resolution_m'],m.oy,m.oy+m.h*m.p['resolution_m']]
    ax.imshow(np.where(m.seen,m.L,np.nan),origin='lower',extent=ext,cmap='RdBu_r',vmin=-2,vmax=3.5)
    ax.plot(R[:,0],R[:,1],'-',color='orange',lw=2,label='global route (from carrots)')
    tt=plans[-1][0] if plans else G[-1,0]; kk=G[:,0]<=tt
    ax.plot(E[kk],N[kk],'g-',lw=1,label='GPS track until last replan')
    for i,(t,n) in enumerate(plans): ax.plot(n[:,0],n[:,1],'--',lw=1.2,label=f'local replan t={t-t0:.0f}s')
    ax.set_xlim(E.min()-6,E.max()+6); ax.set_ylim(N.min()-4,N.max()+6); ax.set_aspect('equal'); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out.replace('.png','_replan.png'),dpi=90)
