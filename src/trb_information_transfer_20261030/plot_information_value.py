"""Plot saved results only; no experiments are imported or rerun."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/trb_information_transfer_20261030'
ART=OUT


def main():
    d=json.loads((OUT/'full.json').read_text(encoding='utf-8'))
    rs=[r for r in d['rows'] if all(v['reference_completes_group'] and v['takeoffs_preserved'] for v in r['methods'].values())]
    grouped=defaultdict(list)
    for r in rs:grouped[r['day']].append(r)
    days=sorted(grouped); totals=[]
    for day in days:
        rows=grouped[day]
        totals.append([len(rows),sum((r['methods']['none']['worst_postrelease']-r['methods']['three']['worst_postrelease'])*120/r['n'] for r in rows),
                       sum((r['methods']['three_independent']['worst_postrelease']-r['methods']['three']['worst_postrelease'])*120/r['n'] for r in rows)])
    a=np.array(totals);rng=np.random.default_rng(20261030)
    draws=rng.integers(0,len(days),size=(5000,len(days)));sums=a[draws].sum(axis=1)
    ci={name:np.quantile(sums[:,j]/sums[:,0],[.025,.975]).tolist() for name,j in [('three_request_gain',1),('shared_error_gain',2)]}
    style={'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
           'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}
    plt.rcParams.update(style)
    fig,axs=plt.subplots(1,2,figsize=(7.15,3.0),gridspec_kw={'width_ratios':[1.2,1]})
    colors=['#245A81','#BF6537']
    for delay,color in zip((1,2),colors):
        data=[r for r in rs if r['delay']==delay]
        xx=[];yy=[]
        for name,q in [('none',0),('one',1),('three',3),('six',6)]:
            xx.append(np.mean([r['methods'][name]['maximum_requests'] for r in data]))
            yy.append(np.mean([(r['methods']['none']['worst_postrelease']-r['methods'][name]['worst_postrelease'])*120/r['n'] for r in data]))
        axs[0].plot(xx,yy,'o-',lw=1.6,ms=4.5,color=color,label=f'{delay*2}-min answer')
        for x,y,q in zip(xx,yy,[0,1,3,6]):
            if q and (q!=1 or delay==1):axs[0].annotate(str(q),(x,y),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8,color=color)
        baseline=np.mean([(r['methods']['none']['worst_postrelease']-r['methods']['three_independent']['worst_postrelease'])*120/r['n'] for r in data])
        shared=np.mean([(r['methods']['none']['worst_postrelease']-r['methods']['three']['worst_postrelease'])*120/r['n'] for r in data])
        xpos=0 if delay==1 else 1
        axs[1].bar(xpos-.18,baseline,width=.33,color=color,alpha=.38,edgecolor=color,lw=.8)
        axs[1].bar(xpos+.18,shared,width=.33,color=color,edgecolor=color,lw=.8)
        axs[1].text(xpos-.18,baseline+1.1,f'{baseline:.1f}',ha='center',fontsize=8)
        axs[1].text(xpos+.18,shared+1.1,f'{shared:.1f}',ha='center',fontsize=8)
    axs[0].set(xlabel='Mean requests issued',ylabel='Additional gate holding (s/aircraft)',ylim=(-3,112),xlim=(-.15,5.2))
    axs[0].set_title('(a) Request budget and response time',pad=9)
    axs[1].set_title('(b) Three-request error accounting',pad=9)
    axs[1].set(xticks=[0,1],xticklabels=['2-min answer','4-min answer'],ylabel='Additional gate holding (s/aircraft)',ylim=(0,57))
    for ax in axs:
        ax.grid(axis='y',color='#D4D8DC',lw=.5,alpha=.65);ax.set_axisbelow(True)
        ax.tick_params(length=3,width=.6)
    from matplotlib.patches import Patch
    axs[0].legend(loc='upper left',frameon=False,fontsize=8)
    axs[1].legend(handles=[Patch(fc='#7F8C95',alpha=.38,label='Independent errors'),Patch(fc='#7F8C95',label='Shared allowance')],
                  loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=8,handlelength=1.2,columnspacing=1)
    fig.subplots_adjust(left=.083,right=.986,bottom=.245,top=.855,wspace=.43)
    ART.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg']:
        fig.savefig(ART/('information_value_transfer.'+ext),bbox_inches='tight',pad_inches=.035)
    fig.savefig(OUT/'information_value_transfer.png',dpi=180,bbox_inches='tight',pad_inches=.035)
    plt.close(fig)
    result=dict(date_clusters=len(days),paired_conditions=len(rs),bootstrap_repetitions=5000,
                interval_interpretation='Descriptive date-cluster resampling of the common preserved evaluation population',confidence_intervals_seconds=ci)
    (OUT/'uncertainty_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
