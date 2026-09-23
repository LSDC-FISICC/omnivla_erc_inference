"""What RViz's rviz/local_planning.rviz shows, as a PNG, from a bag recorded with the
checkpoint_controller_node that publishes the leg topics (fixed frame leg_local):
local costmap (last), global route, every local route, the rover's path from TF
leg_local->rover_leg, and the /erc/free_space_leg hits placed through that TF.
For machines without RViz (the DGX has none). Only the LAST leg is drawn.

    source /opt/ros/jazzy/setup.bash
    python3 bag_leg_view.py <bag> out.png
"""
import sys, math, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
r=rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=sys.argv[1],storage_id=''),rosbag2_py.ConverterOptions('',''))
ty={t.name:t.type for t in r.get_all_topics_and_types()}
grid=None; routes=[]; glob=None; tf=[]; scans=[]; car=[]
while r.has_next():
    tp,d,t=r.read_next(); m=deserialize_message(d,get_message(ty[tp])); t*=1e-9
    if tp=='/erc/local_costmap': grid=m
    elif tp=='/erc/local_route': routes.append([(p.pose.position.x,p.pose.position.y) for p in m.poses])
    elif tp=='/erc/global_route': glob=[(p.pose.position.x,p.pose.position.y) for p in m.poses]
    elif tp=='/tf':
        for tr in m.transforms:
            if tr.child_frame_id=='rover_leg':
                q=tr.transform.rotation; tf.append((t,tr.transform.translation.x,tr.transform.translation.y,2*math.atan2(q.z,q.w)))
    elif tp=='/erc/free_space_leg': scans.append((t,m))
    elif tp=='/erc/carrot': car.append((m.point.x,m.point.y))
tf=np.array(tf)
fig,ax=plt.subplots(figsize=(11,5))
if grid is not None:
  g=np.array(grid.data,dtype=float).reshape(grid.info.height,grid.info.width); g[g<0]=np.nan
  res=grid.info.resolution; ox,oy=grid.info.origin.position.x,grid.info.origin.position.y
  ax.imshow(g,origin='lower',extent=[ox,ox+grid.info.width*res,oy,oy+grid.info.height*res],cmap='RdBu_r',vmin=0,vmax=100,alpha=.7)
if glob: ax.plot(*np.array(glob).T,'-',color='orange',lw=2,label='/erc/global_route')
for i,rt in enumerate(routes): ax.plot(*np.array(rt).T,'--',lw=1,label=f'/erc/local_route #{i+1}')
ax.plot(tf[:,1],tf[:,2],'g-',lw=1,label='TF leg_local->rover_leg')
pts=[]
for t,m in scans[::3]:
    i=np.searchsorted(tf[:,0],t); i=min(i,len(tf)-1); x,y,yaw=tf[i,1:]
    rr=np.array(m.ranges); a=m.angle_min+np.arange(len(rr))*m.angle_increment; k=rr<m.range_max-0.05
    pts.append(np.stack([x+rr[k]*np.cos(yaw+a[k]), y+rr[k]*np.sin(yaw+a[k])],1))
P=np.vstack(pts); ax.plot(P[:,0],P[:,1],'r.',ms=2,label='/erc/free_space_leg hits (via TF)')
ax.set_aspect('equal'); ax.set_xlim(tf[:,1].min()-6,tf[:,1].max()+6); ax.set_ylim(tf[:,2].min()-6,tf[:,2].max()+6); ax.legend(fontsize=6,loc='lower right'); ax.grid(alpha=.3)
ax.set_title('What RViz shows (fixed frame leg_local), rebuilt from the recorded bag',fontsize=9)
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=90)
