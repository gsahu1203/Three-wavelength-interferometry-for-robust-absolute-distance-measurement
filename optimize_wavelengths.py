"""Reproduce the wavelength-triplet search used in the manuscript.
The objective is the minimum nonlocal toroidal phase-space separation over 0.25-100 um.
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from pathlib import Path
PI=np.pi
ROOT=Path(__file__).resolve().parents[1]
def wrap(x): return (x+PI)%(2*PI)-PI
D=np.arange(0.25,100.0001,0.01)*1e-6

def objective(x):
    x=np.sort(np.asarray(x))
    spacing=np.diff(x)
    if np.min(spacing)<10.0:
        return 10.0+(10.0-np.min(spacing))
    p=wrap((4*PI)*D[:,None]/(x[None,:]*1e-9))
    return -float(np.linalg.norm(p,axis=1).min())

if __name__=='__main__':
    res=differential_evolution(objective,[(1450,1650)]*3,seed=20260919,maxiter=160,popsize=12,tol=1e-4,polish=False,updating='immediate')
    x=np.sort(res.x)
    Dr=np.arange(0.25,100.0001,0.002)*1e-6
    p=wrap((4*PI)*Dr[:,None]/(x[None,:]*1e-9)); n=np.linalg.norm(p,axis=1); i=int(np.argmin(n))
    out=pd.DataFrame([{'lambda1_nm':x[0],'lambda2_nm':x[1],'lambda3_nm':x[2],
                       'coarse_objective_margin_rad':-res.fun,'refined_margin_rad':n[i],
                       'closest_alias_delta_um':Dr[i]*1e6,'seed':20260919,'success':bool(res.success),
                       'message':str(res.message)}])
    out.to_csv(ROOT/'data'/'optimization_differential_evolution.csv',index=False)
    print(out.to_string(index=False))
