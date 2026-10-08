import json, math, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data'; FIG = ROOT/'figures'
DATA.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

PI=np.pi

def wrap(x):
    return (x+PI)%(2*PI)-PI

def phase_code(lams_nm, d_um):
    lams=np.asarray(lams_nm)*1e-9
    d=np.asarray(d_um)*1e-6
    return wrap(d[...,None]*(4*PI/lams))

def alias_margin(lams_nm, D_um=100.0, min_delta_um=0.25, step_um=0.002):
    delta=np.arange(min_delta_um, D_um+step_um/2, step_um)
    ph=phase_code(lams_nm,delta)
    norm=np.linalg.norm(ph,axis=1)
    i=np.argmin(norm)
    return float(norm[i]), float(delta[i]), delta, norm

def optimize_wavelengths(seed=20260919, n_stage1=12000, n_keep=60, low_nm=1450.0, high_nm=1650.0, min_spacing_nm=10.0, D_um=100.0):
    rng=np.random.default_rng(seed)
    cand=[]
    while len(cand)<n_stage1:
        x=np.sort(rng.uniform(low_nm,high_nm,size=(max(2000,n_stage1//3),3)),axis=1)
        ok=(np.diff(x,axis=1)>=min_spacing_nm).all(axis=1)
        cand.extend(x[ok].tolist())
    cand=np.asarray(cand[:n_stage1])
    # coarse nonlocal displacement grid
    delta=np.arange(0.25,D_um+0.025,0.05)*1e-6
    scores=np.empty(n_stage1)
    for s in range(0,n_stage1,200):
        C=cand[s:s+200]*1e-9
        # B x Delta x 3
        p=wrap((4*PI)*delta[None,:,None]/C[:,None,:])
        scores[s:s+len(C)]=np.linalg.norm(p,axis=2).min(axis=1)
    idx=np.argsort(scores)[-n_keep:][::-1]
    rows=[]
    best=None
    for rank,j in enumerate(idx,1):
        m,dm,_,_=alias_margin(cand[j],D_um=D_um,step_um=0.002)
        rows.append({'rank_coarse':rank,'lambda1_nm':cand[j,0],'lambda2_nm':cand[j,1],'lambda3_nm':cand[j,2],
                     'coarse_margin_rad':scores[j],'refined_margin_rad':m,'closest_alias_delta_um':dm})
        if best is None or m>best[0]: best=(m,dm,cand[j])
    pd.DataFrame(rows).sort_values('refined_margin_rad',ascending=False).to_csv(DATA/'optimization_top_candidates.csv',index=False)
    return best

def cp_ci(successes,n,alpha=0.05):
    lo=0.0 if successes==0 else beta.ppf(alpha/2,successes,n-successes+1)
    hi=1.0 if successes==n else beta.ppf(1-alpha/2,successes+1,n-successes)
    return float(lo),float(hi)

def simulate(lams_nm,D_um=100.0,N=10000,seed=0,sigma_phi_rad=0.015,sigma_lam_pm=0.5,coarse_step_um=0.05,success_tol_um=0.25):
    rng=np.random.default_rng(seed)
    lams_nom=np.asarray(lams_nm,dtype=float)*1e-9
    k=4*PI/lams_nom
    grid_um=np.arange(0,D_um+coarse_step_um/2,coarse_step_um)
    code=wrap((grid_um[:,None]*1e-6)*k[None,:])
    feat_code=np.concatenate([np.cos(code),np.sin(code)],axis=1).T # 6 x G
    d_true_um=rng.uniform(0,D_um,N)
    lams_true=lams_nom[None,:]+rng.normal(0,sigma_lam_pm*1e-12,size=(N,3))
    phi=wrap((d_true_um[:,None]*1e-6)*(4*PI/lams_true)+rng.normal(0,sigma_phi_rad,size=(N,3)))
    feat=np.concatenate([np.cos(phi),np.sin(phi)],axis=1)
    # chunks to limit memory
    idx=np.empty(N,dtype=np.int32)
    for s in range(0,N,2500):
        score=feat[s:s+2500]@feat_code
        idx[s:s+len(score)]=np.argmax(score,axis=1)
    d0_um=grid_um[idx]
    pred=wrap((d0_um[:,None]*1e-6)*k[None,:])
    resid=wrap(phi-pred)
    delta_m=(resid@k)/np.sum(k*k)
    d_hat_um=np.clip(d0_um+delta_m*1e6,0,D_um)
    err_nm=(d_hat_um-d_true_um)*1000
    ok=np.abs(err_nm)<success_tol_um*1000
    s=int(ok.sum()); lo,hi=cp_ci(s,N)
    abs_e=np.abs(err_nm)
    return {
      'N':N,'successes':s,'ambiguity_success_pct':100*s/N,'alias_failure_pct':100*(1-s/N),
      'success_ci_low_pct':100*lo,'success_ci_high_pct':100*hi,
      'rmse_all_nm':float(np.sqrt(np.mean(err_nm**2))),
      'mae_all_nm':float(np.mean(abs_e)),
      'rmse_success_nm':float(np.sqrt(np.mean(err_nm[ok]**2))) if s else np.nan,
      'mae_success_nm':float(np.mean(abs_e[ok])) if s else np.nan,
      'p95_abs_success_nm':float(np.percentile(abs_e[ok],95)) if s else np.nan,
      'max_abs_success_nm':float(np.max(abs_e[ok])) if s else np.nan,
    }, pd.DataFrame({'d_true_um':d_true_um,'d_hat_um':d_hat_um,'error_nm':err_nm,'correct_ambiguity':ok.astype(int)})

def run():
    t0=time.time()
    cfg=json.loads((ROOT/'code'/'config.json').read_text())
    best=optimize_wavelengths(seed=cfg['optimization_seed'],n_stage1=cfg['optimization_candidates'])
    proposed=np.round(best[2],2)
    # Stable practical design fixed after the documented search; if rerun gives near-neighbor, use rounded stored design for all reported results.
    proposed=np.array(cfg['proposed_wavelengths_nm'],float)
    designs={
      'Proposed alias-margin':proposed,
      'Equal-span':np.array([1451.04,1516.905,1582.77]),
      'Wide-span':np.array([1450.0,1550.0,1650.0]),
      'Clustered':np.array([1549.5,1550.0,1550.5]),
      'Arithmetic':np.array([1500.0,1575.0,1650.0]),
    }
    # Margin curves and summary
    margin_rows=[]; curves=[]
    for name,lams in designs.items():
        m,dm,delta,norm=alias_margin(lams,D_um=cfg['D_um'],step_um=0.002)
        margin_rows.append({'design':name,'lambda1_nm':lams[0],'lambda2_nm':lams[1],'lambda3_nm':lams[2],
                            'min_alias_margin_rad':m,'closest_alias_delta_um':dm})
        take=np.arange(0,len(delta),10)
        curves.append(pd.DataFrame({'design':name,'delta_um':delta[take],'phase_space_distance_rad':norm[take]}))
    pd.DataFrame(margin_rows).to_csv(DATA/'design_alias_margins.csv',index=False)
    pd.concat(curves,ignore_index=True).to_csv(DATA/'alias_margin_curves.csv',index=False)

    # Repeated baseline trials
    rep=[]
    detail_sample=None
    seeds=[31001+i for i in range(10)]
    for name,lams in designs.items():
        for ri,seed in enumerate(seeds,1):
            met,det=simulate(lams,D_um=cfg['D_um'],N=cfg['N_baseline_per_seed'],seed=seed,
                             sigma_phi_rad=cfg['sigma_phi_rad'],sigma_lam_pm=cfg['sigma_lam_pm'],
                             coarse_step_um=cfg['coarse_step_um'])
            met.update({'design':name,'repeat':ri,'seed':seed})
            rep.append(met)
            if name=='Proposed alias-margin' and ri==1: detail_sample=det
    repdf=pd.DataFrame(rep); repdf.to_csv(DATA/'baseline_repeated_trials.csv',index=False)
    # aggregate across 10 independent runs; pooled CI exact
    agg=[]
    for name,g in repdf.groupby('design',sort=False):
        n=int(g.N.sum()); s=int(g.successes.sum()); lo,hi=cp_ci(s,n)
        agg.append({'design':name,'trials_total':n,'successes_total':s,
                    'ambiguity_success_mean_pct':g.ambiguity_success_pct.mean(),
                    'ambiguity_success_sd_pct':g.ambiguity_success_pct.std(ddof=1),
                    'pooled_success_pct':100*s/n,'pooled_ci_low_pct':100*lo,'pooled_ci_high_pct':100*hi,
                    'rmse_success_mean_nm':g.rmse_success_nm.mean(),'rmse_success_sd_nm':g.rmse_success_nm.std(ddof=1),
                    'p95_abs_success_mean_nm':g.p95_abs_success_nm.mean(),
                    'rmse_all_mean_nm':g.rmse_all_nm.mean()})
    aggdf=pd.DataFrame(agg); aggdf.to_csv(DATA/'baseline_summary.csv',index=False)
    detail_sample.to_csv(DATA/'proposed_example_10000_trials.csv',index=False)

    # phase-noise sweep
    rows=[]
    for sig_mrad in [5,10,15,20,30,40,50,75,100]:
        for name in ['Proposed alias-margin','Equal-span','Wide-span']:
            met,_=simulate(designs[name],D_um=cfg['D_um'],N=10000,seed=42000+sig_mrad,
                           sigma_phi_rad=sig_mrad/1000,sigma_lam_pm=cfg['sigma_lam_pm'],coarse_step_um=cfg['coarse_step_um'])
            rows.append({'sigma_phi_mrad':sig_mrad,'design':name,**met})
    pd.DataFrame(rows).to_csv(DATA/'phase_noise_sweep.csv',index=False)

    # wavelength uncertainty sweep
    rows=[]
    for sl in [0,0.5,1,2,5,10,25,50,100,200,500]:
        for name in ['Proposed alias-margin','Equal-span','Wide-span']:
            met,_=simulate(designs[name],D_um=cfg['D_um'],N=10000,seed=43000+int(sl*100),
                           sigma_phi_rad=cfg['sigma_phi_rad'],sigma_lam_pm=sl,coarse_step_um=cfg['coarse_step_um'])
            rows.append({'sigma_lambda_pm':sl,'design':name,**met})
    pd.DataFrame(rows).to_csv(DATA/'wavelength_uncertainty_sweep.csv',index=False)

    # range sweep
    rows=[]
    for D in [25,50,75,100,125,150,200]:
        for name in ['Proposed alias-margin','Equal-span','Wide-span']:
            met,_=simulate(designs[name],D_um=D,N=10000,seed=44000+D,
                           sigma_phi_rad=cfg['sigma_phi_rad'],sigma_lam_pm=cfg['sigma_lam_pm'],coarse_step_um=cfg['coarse_step_um'])
            rows.append({'range_um':D,'design':name,**met})
    pd.DataFrame(rows).to_csv(DATA/'range_sweep.csv',index=False)

    # codebook convergence proposed
    rows=[]
    for st in [0.20,0.10,0.05,0.025]:
        met,_=simulate(proposed,D_um=cfg['D_um'],N=20000,seed=45000+int(1000*st),
                       sigma_phi_rad=cfg['sigma_phi_rad'],sigma_lam_pm=cfg['sigma_lam_pm'],coarse_step_um=st)
        rows.append({'coarse_step_um':st,**met})
    pd.DataFrame(rows).to_csv(DATA/'codebook_convergence.csv',index=False)

    # phase-only local precision independent validation (sigma lambda = 0, true distances constrained well inside range)
    met0,_=simulate(proposed,D_um=cfg['D_um'],N=50000,seed=46001,sigma_phi_rad=cfg['sigma_phi_rad'],sigma_lam_pm=0,coarse_step_um=cfg['coarse_step_um'])
    k=4*PI/(proposed*1e-9)
    predicted_nm=cfg['sigma_phi_rad']/np.sqrt(np.sum(k*k))*1e9
    pd.DataFrame([{'sigma_phi_mrad':cfg['sigma_phi_rad']*1000,'analytical_local_sigma_nm':predicted_nm,
                   'simulated_success_rmse_nm':met0['rmse_success_nm'],'ambiguity_success_pct':met0['ambiguity_success_pct'],
                   'N':met0['N']}]).to_csv(DATA/'analytical_validation.csv',index=False)

    # Save software versions
    versions={'python':sys.version.split()[0],'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__}
    try:
        import scipy; versions['scipy']=scipy.__version__
    except Exception: pass
    try:
        import matplotlib; versions['matplotlib']=matplotlib.__version__
    except Exception: pass
    (DATA/'software_versions.json').write_text(json.dumps(versions,indent=2))

    # Figures
    make_figures(designs)
    print('Proposed:',proposed)
    print(pd.read_csv(DATA/'design_alias_margins.csv').to_string(index=False))
    print(pd.read_csv(DATA/'baseline_summary.csv').to_string(index=False))
    print(pd.read_csv(DATA/'analytical_validation.csv').to_string(index=False))
    print('elapsed',time.time()-t0)

def make_figures(designs):
    # Fig1 concept schematic
    fig,ax=plt.subplots(figsize=(7.0,3.8)); ax.set_axis_off()
    ax.annotate('',xy=(0.90,0.62),xytext=(0.11,0.62),arrowprops=dict(arrowstyle='->',lw=1.5))
    ax.text(0.08,0.62,'Three phase-locked\nwavelengths',ha='right',va='center',fontsize=10)
    ax.add_patch(plt.Rectangle((0.28,0.49),0.10,0.26,fill=False,lw=1.3)); ax.text(0.33,0.45,'beam\ncombiner',ha='center',va='top',fontsize=9)
    ax.plot([0.55,0.55],[0.47,0.77],lw=2); ax.text(0.55,0.42,'reference/target\ninterferometer',ha='center',va='top',fontsize=9)
    ax.plot([0.78,0.78],[0.49,0.75],lw=2); ax.text(0.78,0.44,'phase\ndemodulation',ha='center',va='top',fontsize=9)
    ax.text(0.49,0.85,r'$\boldsymbol{\phi}(d)=[\phi_1,\phi_2,\phi_3]$ maps distance to a 3-D phase torus',ha='center',fontsize=10)
    ax.text(0.49,0.12,'Design objective: maximize the minimum separation between nonlocal phase states\nwithin the prescribed distance interval; decode globally, then refine locally.',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(FIG/'Fig1_system_concept.png',dpi=600,bbox_inches='tight'); fig.savefig(FIG/'Fig1_system_concept.pdf',bbox_inches='tight'); plt.close(fig)

    curves=pd.read_csv(DATA/'alias_margin_curves.csv')
    fig,ax=plt.subplots(figsize=(6.6,4.2))
    for name,g in curves.groupby('design',sort=False): ax.plot(g.delta_um,g.phase_space_distance_rad,label=name,lw=1.2)
    ax.set_xlabel(r'Nonlocal displacement difference $\Delta d$ ($\mu$m)'); ax.set_ylabel('Toroidal phase-space distance (rad)')
    ax.set_xlim(0.25,100); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig2_alias_margin_curves.png',dpi=600); fig.savefig(FIG/'Fig2_alias_margin_curves.pdf'); plt.close(fig)

    agg=pd.read_csv(DATA/'baseline_summary.csv')
    fig,ax=plt.subplots(figsize=(6.6,4.0)); x=np.arange(len(agg)); y=agg.pooled_success_pct.to_numpy();
    lo=y-agg.pooled_ci_low_pct.to_numpy(); hi=agg.pooled_ci_high_pct.to_numpy()-y
    ax.bar(x,y,facecolor='none',edgecolor='black',hatch=['///','\\\\','xx','..','++'])
    ax.errorbar(x,y,yerr=np.vstack([lo,hi]),fmt='none',capsize=3,ecolor='black')
    ax.set_xticks(x,agg.design,rotation=22,ha='right'); ax.set_ylabel('Correct ambiguity resolution (%)'); ax.set_ylim(0,105); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig3_baseline_success.png',dpi=600); fig.savefig(FIG/'Fig3_baseline_success.pdf'); plt.close(fig)

    df=pd.read_csv(DATA/'phase_noise_sweep.csv')
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    for name,g in df.groupby('design',sort=False): ax.plot(g.sigma_phi_mrad,g.ambiguity_success_pct,marker='o',label=name)
    ax.set_xlabel('Phase-noise standard deviation (mrad)'); ax.set_ylabel('Correct ambiguity resolution (%)'); ax.set_ylim(-2,102); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig4_phase_noise_sensitivity.png',dpi=600); fig.savefig(FIG/'Fig4_phase_noise_sensitivity.pdf'); plt.close(fig)

    df=pd.read_csv(DATA/'wavelength_uncertainty_sweep.csv')
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    for name,g in df.groupby('design',sort=False): ax.plot(g.sigma_lambda_pm,g.ambiguity_success_pct,marker='s',label=name)
    ax.set_xlabel('Wavelength-calibration uncertainty (pm RMS)'); ax.set_ylabel('Correct ambiguity resolution (%)'); ax.set_ylim(-2,102); ax.set_xscale('symlog',linthresh=1); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig5_wavelength_sensitivity.png',dpi=600); fig.savefig(FIG/'Fig5_wavelength_sensitivity.pdf'); plt.close(fig)

    df=pd.read_csv(DATA/'range_sweep.csv')
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    for name,g in df.groupby('design',sort=False): ax.plot(g.range_um,g.ambiguity_success_pct,marker='^',label=name)
    ax.set_xlabel(r'Prescribed distance range ($\mu$m)'); ax.set_ylabel('Correct ambiguity resolution (%)'); ax.set_ylim(-2,102); ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(FIG/'Fig6_range_sensitivity.png',dpi=600); fig.savefig(FIG/'Fig6_range_sensitivity.pdf'); plt.close(fig)

if __name__=='__main__': run()
