"""Derive mean/std/median/IQR summaries and a compact figure from raw records."""
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--outdir', required=True)
    a=ap.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(json.loads(Path(a.raw).read_text()))
    if df.empty: raise SystemExit('raw result is empty')
    s=(df.groupby(['task','family','budget'],as_index=False)
         .agg(r2_mean=('r2','mean'),r2_std=('r2','std'),r2_median=('r2','median'),
              r2_iqr=('r2',lambda x:x.quantile(.75)-x.quantile(.25)),
              actual_parameters=('actual_parameters','median'),n=('r2','size')))
    s.to_csv(out/'summary_by_budget.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,(task,g) in zip(axes,df.groupby('task')):
        for family,h in g.groupby('family'):
            q=h.groupby('budget',as_index=False).agg(p=('actual_parameters','median'),m=('r2','mean'),sd=('r2','std'))
            ax.errorbar(q.p,q.m,yerr=q.sd.fillna(0),marker='o',capsize=2,label=family)
        ax.set_title(task.replace('_',' ')); ax.set_xlabel('Median trainable parameters'); ax.set_ylabel('Test $R^2$'); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7,loc='best'); fig.savefig(out/'independent_common_budget.pdf'); fig.savefig(out/'independent_common_budget.png',dpi=180); plt.close(fig)
    (out/'summary.json').write_text(s.to_json(orient='records',indent=2))
if __name__=='__main__': main()
