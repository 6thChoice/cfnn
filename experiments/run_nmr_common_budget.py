"""Common-budget NMR transfer audit with the corrected baseline identities."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torch import nn
from sklearn.metrics import r2_score
import nmr_data as D
from fair_baselines import build_mlp, build_gaussian_rbf, build_pykan, build_cofrnet, build_rational_nn, build_siren
from eis_models import CFNNHybridEIS
import nmr_peak_recovery as PR

FAMILIES=["CFNN-Hybrid","MLP","Gaussian RBF","KAN","CoFrNet-Standard","Rational activation NN","SIREN"]
BUDGETS=[64,256,512]
SEEDS=[42,123,456,789,1024]

def build(f,d,o,b,s):
    if f=="CFNN-Hybrid": return CFNNHybridEIS(d,o,max(1,b//40),3),{}
    if f=="MLP": return build_mlp(d,o,max(4,b//(d+o+2)),2)
    if f=="Gaussian RBF": return build_gaussian_rbf(d,o,max(4,b//(d+o+1)))
    if f=="KAN": return build_pykan(d,o,max(3,b//20),grid=5,seed=s)
    if f=="CoFrNet-Standard": return build_cofrnet(d,o,max(1,b//20),3)
    if f=="Rational activation NN": return build_rational_nn(d,o,max(4,b//(d+o+2)),2)
    if f=="SIREN": return build_siren(d,o,max(4,b//(d+o+2)),2)
    raise ValueError(f)

def fit(model,xt,yt,xv,yv,epochs,seed):
    torch.manual_seed(seed); np.random.seed(seed); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5); loss=nn.MSELoss(); best=float('inf'); best_epoch=1; stale=0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); z=loss(model(xt),yt); z.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        model.eval();
        with torch.no_grad(): v=loss(model(xv),yv).item()
        if v<best: best,best_epoch,stale=v,ep+1,0
        else: stale+=1
        if stale>=50: break
    return best_epoch,best

def refit(model,x,y,epochs,seed):
    torch.manual_seed(seed+1000003); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-5); loss=nn.MSELoss()
    for _ in range(max(1,epochs)):
        opt.zero_grad(); z=loss(model(x),y); z.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
    return model

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='experiment_refine/nmr_common_budget_results/raw.json'); ap.add_argument('--epochs',type=int,default=180); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    seeds=SEEDS[:1] if a.smoke else SEEDS; tasks={'ethanol':1000,'caffeine':1400,'strychnine':2400}; records=[]; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('[]')
    for task,n in tasks.items():
        ds=D.make_nmr_dataset(task,n,0.03,7,n_dense=2000 if task=='ethanol' else 2800 if task=='caffeine' else 4000)
        X,Y=ds['X'],ds['Y']; idx=np.arange(len(X)); val=idx[::4]; tr=np.setdiff1d(idx,val); te=np.arange(len(ds['Xeval']))
        ym,ys=Y[tr].mean(0,keepdims=True),Y[tr].std(0,keepdims=True)+1e-6; Yn=(Y-ym)/ys
        xt=torch.from_numpy(X[tr]); xv=torch.from_numpy(X[val]); yt=torch.from_numpy(Yn[tr]); yv=torch.from_numpy(Yn[val]); xe=torch.from_numpy(ds['Xeval'])
        for f in FAMILIES:
            for b in BUDGETS:
                for s in seeds:
                    t=time.time(); m,meta=build(f,1,1,b,s); ep,vm=fit(m,xt,yt,xv,yv,a.epochs,s); fm,_=build(f,1,1,b,s+1000003); fm=refit(fm,torch.cat([xt,xv]),torch.cat([yt,yv]),ep,s); fm.eval()
                    with torch.no_grad(): pred=fm(xe).numpy()*ys+ym
                    r2=float(r2_score(ds['Yeval'],pred)); ppm=(ds['Xeval'].ravel()+1)/2*(ds['ppm_hi']-ds['ppm_lo'])+ds['ppm_lo']; met=PR.peak_recovery_metrics(ds['true_peaks_ppm'],ppm,pred.ravel())
                    rec={'task':task,'family':f,'budget':b,'seed':s,'actual_parameters':sum(p.numel() for p in fm.parameters() if p.requires_grad),'r2':r2,'center_f1':float(met['f1']),'center_ppm_error':float(met['ppm_error_mean']),'selected_epochs':ep,'selection_val_mse':vm,'seconds':time.time()-t,'metadata':meta,'target_definition':'semi-synthetic spectra from measured chemical environments; metric matches environment centers'}
                    records.append(rec); out.write_text(json.dumps(records,indent=2)); print(task,f,b,s,r2,flush=True)
if __name__=='__main__': main()
