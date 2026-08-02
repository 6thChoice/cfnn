"""Summaries for strict realized-parameter benchmark records."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw',required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args()
    payload=json.loads(Path(a.raw).read_text()); df=pd.DataFrame(payload['records']); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    group=['task','family','target_budget','actual_parameters']
    summary=(df.groupby(group,as_index=False).agg(r2_mean=('r2','mean'),r2_sd=('r2','std'),r2_median=('r2','median'),n=('r2','size'),wall_mean=('wall_seconds','mean'),steps_median=('optimizer_steps','median')))
    summary.to_csv(out/'summary.csv',index=False); summary.to_json(out/'summary.json',orient='records',indent=2)
    # Paired differences against CFNN-Hybrid at the same task/budget/seed.
    base=df[df.family=='CFNN-Hybrid'][['task','target_budget','seed','r2']].rename(columns={'r2':'r2_cfnn'})
    pair=df[df.family!='CFNN-Hybrid'].merge(base,on=['task','target_budget','seed'],how='inner'); pair['delta_r2']=pair['r2']-pair['r2_cfnn']
    pair.to_csv(out/'paired_differences.csv',index=False)
    rows=[]
    for (t,f,b),g in pair.groupby(['task','family','target_budget']):
        vals=g.delta_r2.to_numpy(); rng=np.random.default_rng(17); boots=np.array([np.mean(rng.choice(vals,len(vals),replace=True)) for _ in range(5000)])
        rows.append({'task':t,'family':f,'target_budget':b,'n':len(vals),'delta_mean':float(vals.mean()),'ci_low':float(np.quantile(boots,.025)),'ci_high':float(np.quantile(boots,.975))})
    pd.DataFrame(rows).to_csv(out/'paired_bootstrap_ci.csv',index=False)
    fig,axes=plt.subplots(1,len(df.task.unique()),figsize=(14,4),constrained_layout=True,squeeze=False); axes=axes[0]
    for ax,(task,g) in zip(axes,df.groupby('task')):
        for fam,h in g.groupby('family'):
            q=h.groupby('target_budget',as_index=False).agg(p=('actual_parameters','median'),m=('r2','mean'),s=('r2','std'))
            ax.errorbar(q.p,q.m,yerr=q.s.fillna(0),marker='o',capsize=2,label=fam)
        ax.set_title(task); ax.set_xlabel('Actual trainable parameters'); ax.set_ylabel('Test $R^2$'); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7); fig.savefig(out/'strict_parameter_matched.pdf'); fig.savefig(out/'strict_parameter_matched.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
