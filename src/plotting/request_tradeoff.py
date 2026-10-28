"""Plot stored trade-offs; does not import or execute experiment solvers."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
rows=json.loads((ROOT/'results/trb_precision_20261028/frontier_rows.json').read_text())
assert len(rows)==1200 and all(r['optimal'] for r in rows)
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9.5,'axes.titlesize':10,
                     'axes.labelsize':9.5,'legend.fontsize':8.5,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(7.1,2.95),sharey=True)
styles={1:dict(color='#0072B2',linestyle='-'),
        2:dict(color='#D55E00',linestyle='--'),
        4:dict(color='#009E73',linestyle=':')}
curves=[]
for ax,d in zip(axes,[1,2]):
    for w in [1,2,4]:
        group=[r for r in rows if r['delay']==d and r['width']==w]
        budgets=sorted({p['expected_queries'] for r in group for p in r['frontier']})
        times=[]
        for q in budgets:
            times.append(float(np.mean([min(p['waiting'] for p in r['frontier']
                                       if p['expected_queries']<=q+1e-10) for r in group])))
        label='Exact count' if w==1 else f'Up to {w} counts'
        ax.step(budgets,times,where='post',label=label,linewidth=1.9,**styles[w])
        curves.append(dict(delay=d,width=w,expected_checks=budgets,post_release_minutes=times))
    ax.set_title(f'{2*d}-minute responses',fontsize=10)
    ax.set_xlabel('Expected checks allowed per aircraft group')
    ax.grid(axis='y',color='#D9D9D9',linewidth=.55)
    ax.set_xlim(left=0)
axes[0].set_ylabel('Mean post-release time (min/departure)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=8.5, ncol=3,
           loc='lower center', bbox_to_anchor=(0.5, 0.0),
           columnspacing=1.5, handlelength=2.4)
fig.tight_layout(rect=(0, .12, 1, 1), pad=.6, w_pad=1.7)
out=ROOT/'results/figures'
out.mkdir(parents=True,exist_ok=True)
for suffix in ['pdf','svg']:
    fig.savefig(out/('precision_request_tradeoff.'+suffix),bbox_inches='tight')
(ROOT/'results/trb_precision_20261028/plot_curves.json').write_text(json.dumps(curves,indent=2),encoding='utf-8')
print('Saved vector PDF, editable SVG, and numerical plot curves.')
