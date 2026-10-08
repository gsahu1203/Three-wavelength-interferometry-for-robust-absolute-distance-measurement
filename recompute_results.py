import importlib.util, json, platform, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import beta
P=Path(__file__).resolve().parent/'study.py'; sp=importlib.util.spec_from_file_location('study',P); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'; F=ROOT/'figures'
pro=np.array([1451.04,1570.96,1582.77])
designs={
 'Proposed alias-margin':pro,
 'Equal-span':np.array([1451.04,(1451.04+1582.77)/2,1582.77]),
 'Wide-span':np.array([1450.,1550.,1650.]),
 'Clustered':np.array([1549.5,1550.,1550.5]),
 'Arithmetic':np.array([1500.,1575.,1650.])}
# margin
mr=[]; curves=[]
for name,l in designs.items():
 mm,dm,de,no=m.alias_margin(l,D_um=100,step_um=.002); mr.append({'design':name,'lambda1_nm':l[0],'lambda2_nm':l[1],'lambda3_nm':l[2],'min_alias_margin_rad':mm,'closest_alias_delta_um':dm})
 take=np.arange(0,len(de),10); curves.append(pd.DataFrame({'design':name,'delta_um':de[take],'phase_space_distance_rad':no[take]}))
pd.DataFrame(mr).to_csv(D/'design_alias_margins.csv',index=False); pd.concat(curves).to_csv(D/'alias_margin_curves.csv',index=False)
# repeated
rows=[]; det0=None
for name,l in designs.items():
 for ri,seed in enumerate(range(31001,31011),1):
  met,det=m.simulate(l,N=10000,seed=seed,D_um=100,sigma_phi_rad=.015,sigma_lam_pm=.5,coarse_step_um=.05)
  rows.append({'design':name,'repeat':ri,'seed':seed,**met})
  if name=='Proposed alias-margin' and ri==1: det0=det
rep=pd.DataFrame(rows); rep.to_csv(D/'baseline_repeated_trials.csv',index=False); det0.to_csv(D/'proposed_example_10000_trials.csv',index=False)
agg=[]
for name,g in rep.groupby('design',sort=False):
 n=int(g.N.sum()); s=int(g.successes.sum()); lo,hi=m.cp_ci(s,n)
 agg.append({'design':name,'trials_total':n,'successes_total':s,'ambiguity_success_mean_pct':g.ambiguity_success_pct.mean(),'ambiguity_success_sd_pct':g.ambiguity_success_pct.std(ddof=1),'pooled_success_pct':100*s/n,'pooled_ci_low_pct':100*lo,'pooled_ci_high_pct':100*hi,'rmse_success_mean_nm':g.rmse_success_nm.mean(),'rmse_success_sd_nm':g.rmse_success_nm.std(ddof=1),'p95_abs_success_mean_nm':g.p95_abs_success_nm.mean(),'rmse_all_mean_nm':g.rmse_all_nm.mean()})
pd.DataFrame(agg).to_csv(D/'baseline_summary.csv',index=False)
# sweeps
for fn, vals, key, baseSeed in [
 ('phase_noise_sweep.csv',[5,10,15,20,30,40,50,75,100],'sigma_phi_mrad',42000),
 ('wavelength_uncertainty_sweep.csv',[0,.5,1,2,5,10,25,50,100,200,500],'sigma_lambda_pm',53000),
 ('range_sweep.csv',[25,50,75,100,125,150,200],'range_um',44000)]:
 rr=[]
 for v in vals:
  for name in ['Proposed alias-margin','Equal-span','Wide-span']:
   kw=dict(D_um=100,N=10000,seed=baseSeed+int(v*10),sigma_phi_rad=.015,sigma_lam_pm=.5,coarse_step_um=.05)
   if key=='sigma_phi_mrad': kw['sigma_phi_rad']=v/1000
   if key=='sigma_lambda_pm': kw['sigma_lam_pm']=v
   if key=='range_um': kw['D_um']=v; kw['seed']=baseSeed+int(v)
   met,_=m.simulate(designs[name],**kw); rr.append({key:v,'design':name,**met})
 pd.DataFrame(rr).to_csv(D/fn,index=False)
# convergence
rr=[]
for st in [.2,.1,.05,.025]:
 met,_=m.simulate(pro,N=20000,seed=45000+int(st*1000),D_um=100,sigma_phi_rad=.015,sigma_lam_pm=.5,coarse_step_um=st); rr.append({'coarse_step_um':st,**met})
pd.DataFrame(rr).to_csv(D/'codebook_convergence.csv',index=False)
# analytical
met,_=m.simulate(pro,N=50000,seed=46001,D_um=100,sigma_phi_rad=.015,sigma_lam_pm=0,coarse_step_um=.05)
k=4*np.pi/(pro*1e-9); pred=.015/np.sqrt(np.sum(k*k))*1e9
pd.DataFrame([{'sigma_phi_mrad':15,'analytical_local_sigma_nm':pred,'simulated_success_rmse_nm':met['rmse_success_nm'],'ambiguity_success_pct':met['ambiguity_success_pct'],'N':met['N']}]).to_csv(D/'analytical_validation.csv',index=False)
# ablation no refinement
rng=np.random.default_rng(47001); PI=np.pi; wrap=m.wrap; l=pro*1e-9;k=4*PI/l;grid=np.arange(0,100.0001,.05);code=wrap(grid[:,None]*1e-6*k);featc=np.r_[np.cos(code).T,np.sin(code).T];N=20000;dt=rng.uniform(0,100,N);lt=l[None,:]+rng.normal(0,.5e-12,(N,3));phi=wrap(dt[:,None]*1e-6*(4*PI/lt)+rng.normal(0,.015,(N,3)));feat=np.c_[np.cos(phi),np.sin(phi)];idx=np.empty(N,dtype=int)
for s in range(0,N,2500):idx[s:s+2500]=np.argmax(feat[s:s+2500]@featc,axis=1)
e=(grid[idx]-dt)*1000;ok=np.abs(e)<250
r1={'method':'Global codebook only','N':N,'ambiguity_success_pct':100*ok.mean(),'rmse_success_nm':np.sqrt(np.mean(e[ok]**2)),'p95_abs_success_nm':np.percentile(np.abs(e[ok]),95)}
r2,_=m.simulate(pro,N=N,seed=47001,D_um=100,sigma_phi_rad=.015,sigma_lam_pm=.5,coarse_step_um=.05);r2={'method':'Codebook + local LS refinement','N':N,'ambiguity_success_pct':r2['ambiguity_success_pct'],'rmse_success_nm':r2['rmse_success_nm'],'p95_abs_success_nm':r2['p95_abs_success_nm']};pd.DataFrame([r1,r2]).to_csv(D/'decoder_ablation.csv',index=False)
print(pd.read_csv(D/'design_alias_margins.csv').to_string(index=False));print(pd.read_csv(D/'baseline_summary.csv').to_string(index=False));print(pd.read_csv(D/'analytical_validation.csv').to_string(index=False));print(pd.read_csv(D/'decoder_ablation.csv').to_string(index=False))
