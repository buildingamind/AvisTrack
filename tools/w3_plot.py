import json, numpy as np, matplotlib.pyplot as plt
import matplotlib.patches as mpatches, matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, DayLocator
from datetime import datetime, timedelta

INTENDED_DUR_S, INTENDED_OTO_S = 120, 3600
CLUSTER_THRESH_S, REBOOT_THRESH_S, DRIFT_THRESH_S = 600, 7200, 4320
C_VALID,C_INVALID,C_NORMAL,C_CLUSTER,C_DRIFT,C_GAP,C_REF = (
    "#DFF2E1","#3A3A3A","#1976D2","#D32F2F","#FF9800","#78909C","#43A047")

with open(r"E:\Wave3\02_Global_Metadata\valid_intervals.json") as f: valid_raw=json.load(f)
with open(r"E:\Wave3\02_Global_Metadata\danger_intervals.json") as f: danger_raw=json.load(f)

def ts(t): return datetime.fromtimestamp(t)
valid_segs=[(ts(v[0]),ts(v[1])) for v in valid_raw]
d_start_ts=np.array([d[0] for d in danger_raw])
d_end_ts  =np.array([d[1] for d in danger_raw])
d_dur=d_end_ts-d_start_ts
d_start_dt=[ts(t) for t in d_start_ts]
n=len(danger_raw)
oto=np.diff(d_start_ts); oto_dt=d_start_dt[1:]

# 从 valid_intervals 计算真实的 reboot gap（录像中断期）
MIN_GAP_S = 120  # 忽略 < 2min 的极短间隙（文件切换）
valid_gaps = []
for i in range(len(valid_segs)-1):
    g_s = valid_segs[i][1]
    g_e = valid_segs[i+1][0]
    if (g_e - g_s).total_seconds() > MIN_GAP_S:
        valid_gaps.append((g_s, g_e))

def spans_valid_gap(dt_a, dt_b):
    """OTO 区间是否跨越了某个 valid gap（录像中断）"""
    for g_s, g_e in valid_gaps:
        if dt_a < g_e and dt_b > g_s:
            return True
    return False

# 标记每个事件是否在 valid gap 前（用于泳道着色）
pre_gap_set = set()
for g_s, _ in valid_gaps:
    for i in range(n-1, -1, -1):
        if d_start_dt[i] < g_s:
            pre_gap_set.add(i)
            break

def classify(i):
    if i>=n-1: return "normal"
    if oto[i]<CLUSTER_THRESH_S: return "cluster"
    return "normal"
cats=[classify(i) for i in range(n)]
for i in range(1,n):
    if oto[i-1]<CLUSTER_THRESH_S: cats[i]="cluster"
for i in pre_gap_set:
    if cats[i]=="normal": cats[i]="pre_reboot"
cat_color={"normal":C_NORMAL,"cluster":C_CLUSTER,"pre_reboot":C_DRIFT}

# 标记哪些 OTO 跨越了 valid gap（不应画为散点）
oto_spans_gap=[spans_valid_gap(d_start_dt[i], d_start_dt[i+1]) for i in range(n-1)]

exp_start_date=datetime(2025,10,30).date(); exp_end_date=datetime(2025,11,13).date()
days=[]; cur=exp_start_date
while cur<=exp_end_date: days.append(cur); cur+=timedelta(days=1)
n_days=len(days)
def day_idx(dt): return (dt.date()-exp_start_date).days
def hour_frac(dt): return dt.hour+dt.minute/60+dt.second/3600

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False})
SWIM_H=max(n_days*0.52,6)
fig=plt.figure(figsize=(22,SWIM_H+8)); fig.patch.set_facecolor("#F7F7F7")
gs=gridspec.GridSpec(3,1,figure=fig,height_ratios=[SWIM_H,4,4],hspace=0.42,
    left=0.07,right=0.78,top=0.94,bottom=0.05)
ax_swim=fig.add_subplot(gs[0]); ax_oto=fig.add_subplot(gs[1]); ax_dur=fig.add_subplot(gs[2])

# ── PANEL 1: SWIMLANE ──
ax=ax_swim; ax.set_facecolor(C_INVALID)
ax.set_xlim(0,24); ax.set_ylim(-0.5,n_days-0.5); ax.invert_yaxis()
for v_s,v_e in valid_segs:
    cur=v_s
    while cur.date()<=v_e.date():
        de=datetime.combine(cur.date()+timedelta(days=1),datetime.min.time())
        se=min(v_e,de); di=day_idx(cur)
        if 0<=di<n_days:
            h0=hour_frac(cur) if cur.date()==v_s.date() else 0.0
            h1=hour_frac(se)  if se<de else 24.0
            ax.barh(di,h1-h0,left=h0,height=0.82,color=C_VALID,zorder=1)
        cur=de
for i,dt in enumerate(d_start_dt):
    di=day_idx(dt)
    if 0<=di<n_days:
        ax.plot(hour_frac(dt),di,"|",color=cat_color.get(cats[i],C_NORMAL),
                markersize=11,markeredgewidth=2.2,zorder=3)
reboot_idx=np.where(oto>REBOOT_THRESH_S)[0]
for g_s,g_e in valid_gaps:
    gap_h=(g_e-g_s).total_seconds()/3600
    cur=g_s
    while cur.date()<=g_e.date():
        de=datetime.combine(cur.date()+timedelta(days=1),datetime.min.time())
        se=min(g_e,de); di=day_idx(cur)
        if 0<=di<n_days:
            h0=hour_frac(cur) if cur==g_s else 0.0
            h1=hour_frac(se)  if se<de    else 24.0
            ax.barh(di,h1-h0,left=h0,height=0.82,color="#FFCDD2",alpha=0.6,zorder=2)
            label_w=h1-h0
            if cur==g_s and label_w>0.3:
                label=f"⚡{gap_h*60:.0f}m" if gap_h<1 else f"⚡{gap_h:.1f}h"
                ax.text(h0+label_w/2,di,label,
                    ha="center",va="center",fontsize=7.5,color="#B71C1C",
                    fontweight="bold",zorder=4)
        cur=de
ax.set_yticks(range(n_days))
ax.set_yticklabels([f"{d.strftime('%m/%d')}  {d.strftime('%a')}" for d in days],fontsize=8.5)
ax.set_xticks(range(0,25,2))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0,25,2)],fontsize=8)
ax.set_xlabel("Time of Day",fontsize=10)
ax.set_title("Panel 1 — Danger Stimulus Swimlane  (each row = 1 calendar day)",
    fontsize=11,fontweight="bold",pad=7)
ax.grid(axis="x",alpha=0.18,color="gray",zorder=0)
leg1=[mpatches.Patch(color=C_VALID,  label="Valid recording"),
      mpatches.Patch(color=C_INVALID,label="No recording"),
      mpatches.Patch(color="#FFCDD2",alpha=0.8,label="Reboot gap (no stimuli)"),
      plt.Line2D([],[],marker="|",color=C_NORMAL, lw=0,ms=10,label="Danger — normal"),
      plt.Line2D([],[],marker="|",color=C_CLUSTER,lw=0,ms=10,label="Danger — clustering"),
      plt.Line2D([],[],marker="|",color=C_DRIFT,  lw=0,ms=10,label="Danger — before reboot")]
ax.legend(handles=leg1,bbox_to_anchor=(1.01,1),loc="upper left",borderaxespad=0,
          fontsize=8.2,framealpha=0.88,ncol=1)

# ── PANEL 2: OTO ──
ax=ax_oto

# 分类：跨 gap 的 OTO 不画散点；其余按 cluster/drift/normal 着色
colors_oto=[]
for i,v in enumerate(oto):
    if oto_spans_gap[i]:          colors_oto.append("skip")
    elif v<CLUSTER_THRESH_S:      colors_oto.append(C_CLUSTER)
    elif v>DRIFT_THRESH_S:        colors_oto.append(C_DRIFT)
    else:                         colors_oto.append(C_NORMAL)

mask_plotted = np.array([c!="skip" for c in colors_oto])
mask_cluster = (oto<CLUSTER_THRESH_S) & mask_plotted
mask_drift   = (oto>DRIFT_THRESH_S)  & mask_plotted
mask_ok      = mask_plotted & ~mask_cluster & ~mask_drift

# valid gap 灰色背景带（Panel 2）
for g_s,g_e in valid_gaps:
    gap_h=(g_e-g_s).total_seconds()/3600
    ax.axvspan(g_s,g_e,color=C_GAP,alpha=0.22,zorder=1)
    mid=g_s+(g_e-g_s)/2
    label=f"⚡reboot\n{gap_h*60:.0f}m" if gap_h<1 else f"⚡reboot\n{gap_h:.1f}h"
    ax.annotate(label,xy=(mid,DRIFT_THRESH_S/60*1.05),ha="center",va="bottom",fontsize=7.5,
        color="#37474F",fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=C_GAP,alpha=0.85),zorder=4)

for dt,v,c in zip(oto_dt,oto,colors_oto):
    if c!="skip": ax.scatter(dt,v/60,color=c,s=22,alpha=0.80,zorder=3)

# 滚动均值（只用非跨gap的点）
plotted_idx=[i for i,c in enumerate(colors_oto) if c!="skip"]
if len(plotted_idx)>10:
    pm_dts=[oto_dt[i] for i in plotted_idx]
    pm_vals=oto[plotted_idx]
    window=10; rm=np.convolve(pm_vals,np.ones(window)/window,mode="valid")
    ax.plot(pm_dts[window-1:],rm/60,color="#7B1FA2",lw=1.8,alpha=0.75,
            label=f"Rolling mean (n={window})",zorder=3)

ax.axhline(INTENDED_OTO_S/60,color=C_REF,  lw=1.8,ls="--",label="Intended 60 min",zorder=2)
ax.axhline(DRIFT_THRESH_S/60,color=C_DRIFT,lw=1.0,ls=":",alpha=0.7,
           label=f"Drift threshold {DRIFT_THRESH_S//60} min",zorder=2)
ax.set_ylabel("OTO Interval (min)",fontsize=10)
ax.set_title("Panel 2 — Onset-to-Onset Interval Over Time  [drift & clustering]",
    fontsize=11,fontweight="bold",pad=7)
ax.xaxis.set_major_formatter(DateFormatter("%m/%d")); ax.xaxis.set_major_locator(DayLocator())
ax.tick_params(axis="x",labelsize=8,rotation=25)
ax.set_ylim(0,max(oto[mask_plotted].max()/60*1.1, DRIFT_THRESH_S/60*1.3) if mask_plotted.any() else 150)
ax.grid(alpha=0.18,zorder=0)
leg2=[plt.Line2D([],[],marker="o",color=C_NORMAL, lw=0,ms=7,label="Normal OTO"),
      plt.Line2D([],[],marker="o",color=C_DRIFT,  lw=0,ms=7,label=f"Drifted (>{DRIFT_THRESH_S//60} min)"),
      plt.Line2D([],[],marker="o",color=C_CLUSTER,lw=0,ms=7,label=f"Clustered (<{CLUSTER_THRESH_S//60} min)"),
      mpatches.Patch(color=C_GAP,alpha=0.35,label="Reboot gap (valid_intervals)"),
      plt.Line2D([],[],color="#7B1FA2",lw=1.8,label="Rolling mean (n=10)"),
      plt.Line2D([],[],color=C_REF,lw=1.8,ls="--",label="Intended 60 min")]
ax.legend(handles=leg2,bbox_to_anchor=(1.01,1),loc="upper left",borderaxespad=0,
          fontsize=8.2,framealpha=0.88,ncol=1)
stats_txt=(f"Total OTO: {len(oto)}\n"
    f"  Normal ({CLUSTER_THRESH_S//60}-{DRIFT_THRESH_S//60} min): {int(mask_ok.sum())}\n"
    f"  Drifted (>{DRIFT_THRESH_S//60} min): {int(mask_drift.sum())}\n"
    f"  Clustered (<{CLUSTER_THRESH_S//60} min): {int(mask_cluster.sum())}\n"
    f"  Skipped (spans reboot gap): {int((~mask_plotted).sum())}\n"
    f"Mean OTO (plotted only): {oto[mask_plotted].mean()/60:.1f} min  (intended 60 min)")
ax.text(0.99,0.98,stats_txt,transform=ax.transAxes,ha="right",va="top",fontsize=8.2,
    color="#212121",bbox=dict(boxstyle="round,pad=0.4",fc="#FFFDE7",ec="#F9A825",alpha=0.92))

# ── PANEL 3: DURATION ──
ax=ax_dur
dur_colors=[]
for d in d_dur:
    dev=(d-INTENDED_DUR_S)/INTENDED_DUR_S
    if abs(dev)<=0.10: dur_colors.append(C_NORMAL)
    elif dev>0.50: dur_colors.append(C_CLUSTER)
    elif dev>0.10: dur_colors.append(C_DRIFT)
    else: dur_colors.append("#9C27B0")
ax.scatter(d_start_dt,d_dur,c=dur_colors,s=20,alpha=0.75,zorder=3)
for g_s,g_e in valid_gaps:
    ax.axvspan(g_s,g_e,color=C_GAP,alpha=0.22,zorder=1)
ax.axhline(INTENDED_DUR_S,      color=C_REF,  lw=1.8,ls="--",label=f"Intended {INTENDED_DUR_S}s",zorder=2)
ax.axhline(INTENDED_DUR_S*1.10, color=C_DRIFT,lw=1.0,ls=":",alpha=0.65,label="+10% threshold",zorder=2)
ax.set_ylabel("Duration (seconds)",fontsize=10); ax.set_xlabel("Date",fontsize=10)
ax.set_title("Panel 3 — Stimulus Duration Over Time",fontsize=11,fontweight="bold",pad=7)
ax.xaxis.set_major_formatter(DateFormatter("%m/%d")); ax.xaxis.set_major_locator(DayLocator())
ax.tick_params(axis="x",labelsize=8,rotation=25); ax.grid(alpha=0.18,zorder=0)
leg3=[plt.Line2D([],[],marker="o",color=C_NORMAL, lw=0,ms=7,label="Normal (+-10%)"),
      plt.Line2D([],[],marker="o",color=C_DRIFT,  lw=0,ms=7,label="Long (+10~50%)"),
      plt.Line2D([],[],marker="o",color=C_CLUSTER,lw=0,ms=7,label="Very long (>+50%)"),
      plt.Line2D([],[],marker="o",color="#9C27B0", lw=0,ms=7,label="Short (<-10%)"),
      plt.Line2D([],[],color=C_REF,lw=1.8,ls="--",label=f"Intended {INTENDED_DUR_S}s")]
ax.legend(handles=leg3,bbox_to_anchor=(1.01,1),loc="upper left",borderaxespad=0,
          fontsize=8.2,framealpha=0.88,ncol=1)
n_ok_dur=sum(1 for d in d_dur if abs(d-INTENDED_DUR_S)/INTENDED_DUR_S<=0.10)
ax.text(0.99,0.98,
    f"n={n}  |  Mean:{d_dur.mean():.0f}s  Median:{np.median(d_dur):.0f}s\n"
    f"Min:{d_dur.min():.0f}s  Max:{d_dur.max():.0f}s\n"
    f"Within +-10%: {n_ok_dur} ({100*n_ok_dur/n:.0f}%)",
    transform=ax.transAxes,ha="right",va="top",fontsize=8.2,color="#212121",
    bbox=dict(boxstyle="round,pad=0.4",fc="#FFFDE7",ec="#F9A825",alpha=0.92))

fig.suptitle(
    f"W3 Experiment — Danger Stimulus Irregularity Analysis\n"
    f"Oct 30 - Nov 13, 2025   |   {n} total events   |   "
    f"Intended: OTO={INTENDED_OTO_S//60} min,  Duration={INTENDED_DUR_S}s",
    fontsize=13,fontweight="bold",y=0.985)
OUT = r"E:\Wave3\w3_danger_analysis.png"
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="#F7F7F7")
print(f"Saved → {OUT}")
plt.show()
