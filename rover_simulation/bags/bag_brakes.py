"""Contact sheet: the front camera just before every side-step brake, with the
/erc/free_space profile under it (red: nearer than 1.2 m). Answers "was it real?".

    source /opt/ros/jazzy/setup.bash
    python3 bag_brakes.py <bag> out.jpg
"""
import sys, re, numpy as np, cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
bag, out = sys.argv[1], sys.argv[2]
def reader():
    r=rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag,storage_id=''),rosbag2_py.ConverterOptions('',''))
    return r,{t.name:t.type for t in r.get_all_topics_and_types()}
r,types=reader(); t0=None; brakes=[]; scans=[]
while r.has_next():
    tp,data,t=r.read_next(); t*=1e-9; t0=t0 or t
    if tp=='/omnivla_debug':
        s=deserialize_message(data,get_message(types[tp])).data
        m=re.search(r'\[side-step\] brake ahead=([\d.]+)',s)
        if m: brakes.append((t,float(m.group(1))))
    elif tp=='/erc/free_space':
        m=deserialize_message(data,get_message(types[tp]))
        scans.append((t,np.array(m.ranges),m.angle_min,m.angle_increment))
st=np.array([s[0] for s in scans])
# frame ~0.5 s before the brake decision (what the profile was looking at)
targets=[t-0.3 for t,_ in brakes]
r,types=reader(); got={}
while r.has_next() and len(got)<len(targets):
    tp,data,t=r.read_next(); t*=1e-9
    if tp!='/erc/front_camera': continue
    for i,tt in enumerate(targets):
        if i not in got and t>=tt:
            m=deserialize_message(data,get_message(types[tp]))
            img=np.frombuffer(m.data,np.uint8).reshape(m.height,m.width,-1)
            if m.encoding.lower().startswith('rgb'): img=img[:,:,::-1]
            got[i]=img.copy()
tiles=[]
for i,(t,a) in enumerate(brakes):
    img=cv2.resize(got[i],(512,288))
    j=np.searchsorted(st,t); _,rr,amin,inc=scans[min(j,len(scans)-1)]
    ang=np.degrees(amin+np.arange(len(rr))*inc)
    # polar strip below the image: bar per bin, red if < 1.2 m
    strip=np.full((70,512,3),255,np.uint8)
    for b,f in zip(ang,rr):
        x=int(256-b/60*230)  # left-positive bearing -> image left
        h=int(min(f,3)/3*65)
        col=(0,0,220) if f<1.2 else (0,160,0)
        cv2.line(strip,(x,69),(x,69-h),col,4)
    tile=np.vstack([img,strip])
    cv2.putText(tile,f'#{i+1} t={t-t0:.0f}s ahead={a:.2f}m',(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
    tiles.append(tile)
while len(tiles)%3: tiles.append(np.full_like(tiles[0],255))
rows=[np.hstack(tiles[k:k+3]) for k in range(0,len(tiles),3)]
cv2.imwrite(out,np.vstack(rows))
print(len(brakes),'brakes ->',out)
