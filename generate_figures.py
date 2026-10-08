from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
# Concept schematic
fig,ax=plt.subplots(figsize=(7.0,3.8));ax.set_axis_off();ax.annotate('',xy=(.90,.62),xytext=(.11,.62),arrowprops=dict(arrowstyle='->',lw=1.5));ax.text(.08,.62,'Three phase-locked\nwavelengths',ha='right',va='center',fontsize=10);ax.add_patch(plt.Rectangle((.28,.49),.10,.26,fill=False,lw=1.3));ax.text(.33,.45,'beam\ncombiner',ha='center',va='top',fontsize=9);ax.plot([.55,.55],[.47,.77],lw=2);ax.text(.55,.42,'reference/target\ninterferometer',ha='center',va='top',fontsize=9);ax.plot([.78,.78],[.49,.75],lw=2);ax.text(.78,.44,'phase\ndemodulation',ha='center',va='top',fontsize=9);ax.text(.49,.85,r'$\boldsymbol{\phi}(d)=[\phi_1,\phi_2,\phi_3]$ maps distance to a 3-D phase torus',ha='center',fontsize=10);ax.text(.49,.12,'Design objective: maximize the minimum separation between nonlocal phase states\nwithin the prescribed distance interval; decode globally, then refine locally.',ha='center',fontsize=9);fig.tight_layout();fig.savefig(F/'Fig1_system_concept.png',dpi=600,bbox_inches='tight');fig.savefig(F/'Fig1_system_concept.pdf',bbox_inches='tight');plt.close(fig)
# Alias envelope
raw=pd.read_csv(D/'alias_margin_curves.csv');fig,ax=plt.subplots(figsize=(6.6,4.2))
for name,g in raw.groupby('design',sort=False):
 gg=g.assign(bin=np.floor(g.delta_um).astype(int)).groupby('bin',as_index=False).agg(delta_um=('delta_um','mean'),phase_space_distance_rad=('phase_space_distance_rad','min'))
 ax.plot(gg.delta_um,gg.phase_space_distance_rad,label=name,lw=1.4)
ax.set_xlabel(r'Nonlocal displacement difference $\Delta d$ ($\mu$m)');ax.set_ylabel(r'Worst phase-space separation per 1-$\mu$m bin (rad)');ax.set_xlim(.25,100);ax.set_ylim(0,1.6);ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/'Fig2_alias_margin_curves.png',dpi=600);fig.savefig(F/'Fig2_alias_margin_curves.pdf');plt.close(fig)
# Baseline
agg=pd.read_csv(D/'baseline_summary.csv');fig,ax=plt.subplots(figsize=(6.6,4));x=np.arange(len(agg));y=agg.pooled_success_pct.to_numpy();lo=y-agg.pooled_ci_low_pct.to_numpy();hi=agg.pooled_ci_high_pct.to_numpy()-y
ax.bar(x,y,facecolor='none',edgecolor='black',hatch=['///','\\\\','xx','..','++']);ax.errorbar(x,y,yerr=np.vstack([lo,hi]),fmt='none',capsize=3,ecolor='black');ax.set_xticks(x,agg.design,rotation=22,ha='right');ax.set_ylabel('Correct ambiguity resolution (%)');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(F/'Fig3_baseline_success.png',dpi=600);fig.savefig(F/'Fig3_baseline_success.pdf');plt.close(fig)
# Sweeps
for fn,xcol,xlab,out,mk,symlog in [('phase_noise_sweep.csv','sigma_phi_mrad','Phase-noise standard deviation (mrad)','Fig4_phase_noise_sensitivity','o',False),('wavelength_uncertainty_sweep.csv','sigma_lambda_pm','Wavelength-calibration uncertainty (pm RMS)','Fig5_wavelength_sensitivity','s',True),('range_sweep.csv','range_um',r'Prescribed distance range ($\mu$m)','Fig6_range_sensitivity','^',False)]:
 df=pd.read_csv(D/fn);fig,ax=plt.subplots(figsize=(6.4,4))
 for name,g in df.groupby('design',sort=False):ax.plot(g[xcol],g.ambiguity_success_pct,marker=mk,label=name)
 ax.set_xlabel(xlab);ax.set_ylabel('Correct ambiguity resolution (%)');ax.set_ylim(-2,102)
 if symlog:ax.set_xscale('symlog',linthresh=1)
 ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(F/(out+'.png'),dpi=600);fig.savefig(F/(out+'.pdf'));plt.close(fig)
