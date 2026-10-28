"""Plot saved results only; this script never invokes an optimization."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
key=lambda r:(r['case_id'],r['capacity'],r['delay'],r['budget'])
adaptive=read('results/trb_joint_20261028/rows.json')
fixed={key(r):r for r in read('results/trb_joint_20261028/fixed_rows.json')}
before={key(r):r for r in read('results/trb_latency_20261028/rows.json')}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.titlesize':10,
                     'axes.labelsize':10,'legend.fontsize':8.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(6.5,2.7),sharey=True)
palette={'fixed':'#0072B2','adaptive':'#D55E00'}
for ax,delay in zip(axes,[1,2]):
    group=[r for r in adaptive if r['delay']==delay and r['budget']==1]
    for name,style in [('Fixed schedule','--'),('Adaptive requests','-')]:
        values=np.sort([before[key(r)]['attainable_adaptive']['waiting']-
                        (fixed[key(r)]['waiting'] if name=='Fixed schedule' else r['waiting']) for r in group])
        values[np.abs(values)<1e-10]=0
        role='fixed' if name=='Fixed schedule' else 'adaptive'
        ax.step(np.r_[-.05,values,1.55],np.r_[0,np.arange(1,len(values)+1)/len(values),1],
                where='post',color=palette[role],linestyle=style,linewidth=1.7,label=name)
    ax.set_title(f'{2*delay}-minute responses',fontsize=10)
    ax.set_xlim(-.05,1.55);ax.set_ylim(0,1.02)
    ax.set_xticks([0,.5,1,1.5]);ax.set_yticks([0,.25,.5,.75,1])
    ax.set_xlabel('Reduction (min/departure)')
    ax.grid(axis='y',color='#D9D9D9',linewidth=.55)
axes[0].set_ylabel('Fraction of combinations')
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=2,
           loc='lower center', bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.6, handlelength=2.4)
fig.tight_layout(rect=(0, .12, 1, 1), pad=.6, w_pad=1.4)
out=ROOT/'results/figures'
out.mkdir(parents=True,exist_ok=True)
fig.savefig(out/'joint_waiting_distribution.pdf',bbox_inches='tight')
fig.savefig(out/'joint_waiting_distribution.svg',bbox_inches='tight')
plt.close(fig)
print('Saved editable vector PDF and SVG from 400 saved conditions.')
