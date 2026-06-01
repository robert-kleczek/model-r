#!/usr/bin/env python3
"""
DRND_ST_LENSING_NFW_BASELINE.py

Immediate control test:
    baryon-only vs DRND-ST fixed a0 vs DRND-ST fixed a0 + screened environment
    vs NFW-only vs NFW+baryons.

Input CSV:
    lens_bin,R_kpc,DeltaSigma,DeltaSigma_err

Optional columns:
    logMbar or Mbar_Msun for mass binning.

Usage:
    python DRND_ST_LENSING_NFW_BASELINE.py ^
      --lensing-csv lensing_data.csv ^
      --bin-mode auto ^
      --n-bins 4 ^
      --outdir DRND_ST_NFW_BASELINE
"""

from __future__ import annotations

import argparse, csv, json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

G_SI=6.67430e-11
MSUN_KG=1.98847e30
PC_M=3.0856775814913673e16
KPC_M=1e3*PC_M
MPC_M=1e6*PC_M
C_M_S=299792458.0

def norm_col(s): return re.sub(r"[^a-z0-9]+","_",str(s).strip().lower()).strip("_")
def to_float(x,default=np.nan):
    try:
        if x is None: return default
        s=str(x).strip()
        if not s or s.lower() in {"nan","none","null","na","--"}: return default
        return float(s)
    except Exception: return default
def pick(row,names,default=np.nan):
    m={norm_col(k):v for k,v in row.items()}
    for n in names:
        k=norm_col(n)
        if k in m:
            v=to_float(m[k])
            if np.isfinite(v): return v
    return default
def pick_str(row,names,default=""):
    m={norm_col(k):v for k,v in row.items()}
    for n in names:
        k=norm_col(n)
        if k in m and m[k] is not None: return str(m[k]).strip()
    return default
def safe_json(o):
    if isinstance(o,dict): return {str(k):safe_json(v) for k,v in o.items()}
    if isinstance(o,list): return [safe_json(v) for v in o]
    if isinstance(o,np.generic): return o.item()
    if isinstance(o,float) and (math.isnan(o) or math.isinf(o)): return None
    return o
def write_csv(path,rows,preferred=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text("",encoding="utf-8"); return
    fields=sorted(set(k for r in rows for k in r.keys()))
    if preferred: fields=[f for f in preferred if f in fields]+[f for f in fields if f not in preferred]
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in fields})

def read_lensing_csv(path):
    rows=[]
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            lb=pick_str(row,["lens_bin","bin","sample"],"")
            R=pick(row,["R_kpc","R","rp","radius"])
            ds=pick(row,["DeltaSigma","delta_sigma","deltasigma","ds","signal"])
            err=pick(row,["DeltaSigma_err","delta_sigma_err","deltasigma_err","ds_err","err","error","sigma"])
            Mbar=pick(row,["Mbar_Msun","Mbar","mbar","baryonic_mass"])
            logM=pick(row,["logMbar","log_mbar","logM","log_mass"])
            if not np.isfinite(logM) and np.isfinite(Mbar) and Mbar>0: logM=math.log10(Mbar)
            if np.isfinite(R) and np.isfinite(ds) and np.isfinite(err) and R>0 and ds>0 and err>0:
                rows.append({"lens_bin_original":lb,"R_kpc":float(R),"DeltaSigma":float(ds),
                             "DeltaSigma_err":float(err),
                             "logMbar_input":float(logM) if np.isfinite(logM) else np.nan})
    if not rows: raise SystemExit("No usable rows")
    return rows

def assign_bins(rows,mode,n_bins):
    mode_used=mode
    existing=sorted(set(str(r.get("lens_bin_original","")).strip() for r in rows if str(r.get("lens_bin_original","")).strip()))
    if mode=="auto":
        if len(existing)>1: mode_used="existing"
        elif any(np.isfinite(r.get("logMbar_input",np.nan)) for r in rows): mode_used="mass"
        else: mode_used="signal"
    if mode_used=="existing":
        for r in rows: r["lens_bin"]=str(r.get("lens_bin_original","")).strip() or "all"
        return rows,mode_used
    if mode_used=="mass":
        key=np.array([r.get("logMbar_input",np.nan) for r in rows],float)
        if not np.all(np.isfinite(key)): raise SystemExit("mass binning requires logMbar/Mbar")
    elif mode_used=="radius": key=np.array([r["R_kpc"] for r in rows],float)
    elif mode_used=="signal": key=np.array([r["DeltaSigma"] for r in rows],float)
    else: raise SystemExit(f"Unknown bin mode {mode}")
    edges=np.unique(np.quantile(key,np.linspace(0,1,max(2,n_bins+1))))
    if len(edges)<=2:
        for r in rows: r["lens_bin"]="bin0"
        return rows,mode_used
    for r,v in zip(rows,key):
        idx=np.searchsorted(edges,v,side="right")-1
        idx=max(0,min(idx,len(edges)-2))
        r["lens_bin"]=f"{mode_used}_bin{idx}"
    return rows,mode_used

def a0_from_chi(H0,chi): return chi*C_M_S*(H0*1000/MPC_M)
def rhocrit(H0):
    H=H0*1000/MPC_M
    return 3*H*H/(8*math.pi*G_SI)
def nu_simple(x):
    x=np.maximum(np.asarray(x,float),1e-300)
    return 0.5+np.sqrt(0.25+1/x)
def mbar_enclosed(r,Mbar,Re,profile):
    r=np.asarray(r,float); M=float(Mbar); Re=max(float(Re),1e-5)
    if profile=="point": return np.full_like(r,M)
    if profile=="hernquist":
        a=Re/1.8153
        return M*r*r/np.maximum((r+a)**2,1e-300)
    x=r/Re
    return M*(1-np.exp(-x)*(1+x+0.5*x*x))
def density_from_enclosed(r_kpc,M_msun):
    rm=r_kpc*KPC_M
    Mkg=M_msun*MSUN_KG
    dMdr=np.gradient(Mkg,rm)
    return np.maximum(dMdr/(4*math.pi*np.maximum(rm*rm,1e-300)),0)
def sigma_projected(R,rgrid,rho):
    Rm=R*KPC_M; rm=rgrid*KPC_M
    mask=rm>Rm*(1+1e-8)
    if np.sum(mask)<10: return np.nan
    rr=rm[mask]; rh=rho[mask]
    integ=rh*rr/np.sqrt(np.maximum(rr*rr-Rm*Rm,1e-300))
    val=2*(np.trapezoid(integ,rr) if hasattr(np,"trapezoid") else np.trapz(integ,rr))
    return val*PC_M**2/MSUN_KG
def delta_sigma_from_density(R_eval,rgrid,rho):
    R_eval=np.asarray(R_eval,float)
    Sigma=np.array([sigma_projected(R,rgrid,rho) for R in R_eval])
    Rd=np.geomspace(max(np.min(R_eval)/50,1e-4),np.max(R_eval)*1.05,360)
    Sd=np.array([sigma_projected(R,rgrid,rho) for R in Rd])
    ok=np.isfinite(Sd); Rd=Rd[ok]; Sd=Sd[ok]
    if len(Rd)<20: return np.full_like(R_eval,np.nan)
    integ=Sd*Rd; cum=np.zeros_like(Rd)
    for i in range(1,len(Rd)): cum[i]=cum[i-1]+0.5*(integ[i]+integ[i-1])*(Rd[i]-Rd[i-1])
    Sbar=2*cum/np.maximum(Rd*Rd,1e-300)
    return np.interp(R_eval,Rd,Sbar)-Sigma
def baryon_ds(R,logM,logRe,profile):
    rgrid=np.geomspace(max(np.min(R)/200,1e-4),max(np.max(R)*200,1e4),1100)
    rho=density_from_enclosed(rgrid,mbar_enclosed(rgrid,10**logM,10**logRe,profile))
    return delta_sigma_from_density(R,rgrid,rho)
def drnd_ds(R,logM,logRe,a0,profile):
    rgrid=np.geomspace(max(np.min(R)/200,1e-4),max(np.max(R)*200,1e4),1100)
    Menc=mbar_enclosed(rgrid,10**logM,10**logRe,profile)
    rm=rgrid*KPC_M
    gbar=G_SI*(Menc*MSUN_KG)/np.maximum(rm*rm,1e-300)
    gst=gbar*nu_simple(gbar/a0)
    Meff=np.maximum.accumulate(gst*rm*rm/G_SI/MSUN_KG)
    rho=density_from_enclosed(rgrid,Meff)
    return delta_sigma_from_density(R,rgrid,rho)
def screen(R,Rt,q):
    R=np.asarray(R,float)
    return 1/(1+np.power(np.maximum(Rt/np.maximum(R,1e-12),1e-12),q))
def twohalo(R,logA,p,R0,Rt,q):
    return 10**logA*np.power(np.maximum(np.asarray(R,float)/R0,1e-12),-p)*screen(R,Rt,q)
def nfw_r200(logM200,H0):
    Mkg=10**logM200*MSUN_KG
    r=(3*Mkg/(4*math.pi*200*rhocrit(H0)))**(1/3)
    return r/KPC_M
def nfw_ds(R,logM200,logc,H0):
    c=10**logc
    r200=nfw_r200(logM200,H0)
    rs=r200/c
    M=10**logM200
    fc=math.log(1+c)-c/(1+c)
    rho_s=M/(4*math.pi*rs**3*fc)
    rgrid=np.geomspace(max(np.min(R)/300,1e-4),max(r200*50,np.max(R)*300,1e5),1200)
    x=rgrid/rs
    rho_msun_kpc3=rho_s/(np.maximum(x,1e-300)*(1+x)**2)
    rho=rho_msun_kpc3*MSUN_KG/(KPC_M**3)
    return delta_sigma_from_density(R,rgrid,rho)

def mass_size_prior(logM,logRe):
    pred=math.log10(5.0)+0.25*(logM-10)
    return ((logRe-pred)/0.35)**2
def c_prior(logc,logM200):
    pred=math.log10(8.0)-0.1*(logM200-12)
    return ((logc-pred)/0.25)**2
def chi2_from_res(res):
    return float(np.sum(res*res))

def predict_case(case,params,bins,data,args,a0):
    idx=0; preds=[]; rows=[]; bin_rows=[]; priors=[]
    for lb in bins:
        sub=data[lb]
        R=np.array([r["R_kpc"] for r in sub],float)
        if case=="baryon_only":
            logM,logRe=params[idx],params[idx+1]; idx+=2
            pred=baryon_ds(R,logM,logRe,args.profile_type)
            priors.append(math.sqrt(args.prior_weight*mass_size_prior(logM,logRe)))
            meta={"logMbar":logM,"Mbar_Msun":10**logM,"Re_kpc":10**logRe}
        elif case=="drnd_fixed_a0":
            logM,logRe=params[idx],params[idx+1]; idx+=2
            pred=drnd_ds(R,logM,logRe,a0,args.profile_type)
            priors.append(math.sqrt(args.prior_weight*mass_size_prior(logM,logRe)))
            meta={"logMbar":logM,"Mbar_Msun":10**logM,"Re_kpc":10**logRe}
        elif case=="drnd_fixed_a0_screen2h":
            logM,logRe,logA=params[idx],params[idx+1],params[idx+2]; idx+=3
            base=drnd_ds(R,logM,logRe,a0,args.profile_type)
            th=twohalo(R,logA,args.twohalo_slope,args.R0_kpc,args.Rt_kpc,args.screen_q)
            pred=base+th
            priors.append(math.sqrt(args.prior_weight*mass_size_prior(logM,logRe)))
            frac=th/np.maximum(pred,1e-30)
            inner=frac[R<=np.quantile(R,args.inner_quantile)]
            med=float(np.median(inner)) if len(inner) else 0.0
            if med>args.f_inner_max:
                priors.append(math.sqrt(args.inner_penalty_weight)*(med-args.f_inner_max)/0.1)
            meta={"logMbar":logM,"Mbar_Msun":10**logM,"Re_kpc":10**logRe,"twohalo_logA":logA,
                  "inner_twohalo_fraction":med,
                  "outer_twohalo_fraction":float(np.median(frac[R>=np.quantile(R,1-args.inner_quantile)]))}
        elif case=="nfw_only":
            logM200,logc=params[idx],params[idx+1]; idx+=2
            pred=nfw_ds(R,logM200,logc,args.H0)
            priors.append(math.sqrt(args.c_prior_weight*c_prior(logc,logM200)))
            meta={"logM200":logM200,"M200_Msun":10**logM200,"c200":10**logc}
        elif case=="nfw_plus_baryon":
            logM,logRe,logM200,logc=params[idx],params[idx+1],params[idx+2],params[idx+3]; idx+=4
            pred=baryon_ds(R,logM,logRe,args.profile_type)+nfw_ds(R,logM200,logc,args.H0)
            priors.append(math.sqrt(args.prior_weight*mass_size_prior(logM,logRe)))
            priors.append(math.sqrt(args.c_prior_weight*c_prior(logc,logM200)))
            meta={"logMbar":logM,"Mbar_Msun":10**logM,"Re_kpc":10**logRe,"logM200":logM200,"M200_Msun":10**logM200,"c200":10**logc}
        else: raise ValueError(case)
        bin_rows.append({"case":case,"lens_bin":lb,**meta})
        for i,r in enumerate(sub):
            preds.append(float(pred[i]) if np.isfinite(pred[i]) else np.nan)
            rows.append({"case":case,"lens_bin":lb,"R_kpc":r["R_kpc"],"DeltaSigma_obs":r["DeltaSigma"],
                         "DeltaSigma_err":r["DeltaSigma_err"],"DeltaSigma_pred":float(pred[i]) if np.isfinite(pred[i]) else np.nan,**meta})
    return np.array(preds,float),rows,bin_rows,np.array(priors,float)

def residuals(case,params,bins,data,args,a0):
    pred,_,_,priors=predict_case(case,params,bins,data,args,a0)
    obs=np.array([r["DeltaSigma"] for lb in bins for r in data[lb]],float)
    err=np.array([r["DeltaSigma_err"] for lb in bins for r in data[lb]],float)
    bad=np.where((~np.isfinite(pred)) | (pred<=0),1e4,0.0)
    if args.cov_space=="linear":
        res=(obs-pred)/np.maximum(err,1e-30)+bad
    else:
        sig=np.maximum(err/np.maximum(obs,1e-30)/math.log(10),0.03)
        res=(np.log10(np.maximum(obs,1e-30))-np.log10(np.maximum(pred,1e-30)))/sig+bad
    return res if args.no_prior else np.concatenate([res,priors])

def init_bounds(case,bins,args):
    p0=[]; bounds=[]
    for _ in bins:
        lm=11.0; lr=math.log10(5*(10**(lm-10))**0.25)
        if case in {"baryon_only","drnd_fixed_a0"}:
            p0 += [lm,lr]; bounds += [(8,12.8),(math.log10(0.2),math.log10(80))]
        elif case=="drnd_fixed_a0_screen2h":
            p0 += [lm,lr,0.0]; bounds += [(8,12.8),(math.log10(0.2),math.log10(80)),(-8,6)]
        elif case=="nfw_only":
            p0 += [12.0,math.log10(8.0)]; bounds += [(9,15),(math.log10(1),math.log10(40))]
        elif case=="nfw_plus_baryon":
            p0 += [lm,lr,12.0,math.log10(8.0)]
            bounds += [(8,12.8),(math.log10(0.2),math.log10(80)),(9,15),(math.log10(1),math.log10(40))]
    return np.array(p0,float),bounds

def fit_case(case,bins,data,args,a0):
    p0,bounds=init_bounds(case,bins,args)
    def obj(p): return chi2_from_res(residuals(case,p,bins,data,args,a0))
    opt=minimize(obj,p0,method="L-BFGS-B",bounds=bounds,options={"maxiter":args.maxiter,"ftol":1e-8})
    res=residuals(case,opt.x,bins,data,args,a0)
    chi2=chi2_from_res(res)
    pred_rows=[]; bin_rows=[]
    try:
        _,pred_rows,bin_rows,_=predict_case(case,opt.x,bins,data,args,a0)
    except Exception:
        pass
    n=sum(len(data[lb]) for lb in bins); k=len(opt.x); dof=max(n-k,1)
    return {"case":case,"chi2":chi2,"n_data":n,"k_params":k,"dof":dof,
            "chi2_per_dof":chi2/dof,"chi2_per_data":chi2/max(n,1),
            "AIC":chi2+2*k,"BIC":chi2+k*math.log(max(n,2)),
            "success":bool(opt.success),"message":str(opt.message),
            "pred_rows":pred_rows,"bin_rows":bin_rows}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lensing-csv",required=True)
    ap.add_argument("--cov-space",default="log",choices=["log","linear"])
    ap.add_argument("--bin-mode",default="auto",choices=["auto","existing","radius","signal","mass"])
    ap.add_argument("--n-bins",type=int,default=4)
    ap.add_argument("--H0",type=float,default=67.4)
    ap.add_argument("--chi",type=float,default=0.18)
    ap.add_argument("--profile-type",default="hernquist",choices=["hernquist","exponential_sphere","point"])
    ap.add_argument("--no-prior",action="store_true")
    ap.add_argument("--prior-weight",type=float,default=1.0)
    ap.add_argument("--c-prior-weight",type=float,default=1.0)
    ap.add_argument("--twohalo-slope",type=float,default=0.8)
    ap.add_argument("--R0-kpc",type=float,default=300.0)
    ap.add_argument("--Rt-kpc",type=float,default=200.0)
    ap.add_argument("--screen-q",type=float,default=3.0)
    ap.add_argument("--f-inner-max",type=float,default=0.3)
    ap.add_argument("--inner-quantile",type=float,default=0.35)
    ap.add_argument("--inner-penalty-weight",type=float,default=25.0)
    ap.add_argument("--maxiter",type=int,default=700)
    ap.add_argument("--outdir",default="DRND_ST_NFW_BASELINE")
    args=ap.parse_args()

    outdir=Path(args.outdir); tabledir=outdir/"tables"; plotdir=outdir/"plots"
    tabledir.mkdir(parents=True,exist_ok=True); plotdir.mkdir(parents=True,exist_ok=True)
    rows=read_lensing_csv(args.lensing_csv)
    rows,mode_used=assign_bins(rows,args.bin_mode,args.n_bins)
    write_csv(tabledir/"binned_lensing_data.csv",rows,preferred=["lens_bin","R_kpc","DeltaSigma","DeltaSigma_err","logMbar_input"])
    data={}
    for r in rows: data.setdefault(r["lens_bin"],[]).append(r)
    bins=sorted(data.keys())
    a0=a0_from_chi(args.H0,args.chi)

    cases=["baryon_only","drnd_fixed_a0","drnd_fixed_a0_screen2h","nfw_only","nfw_plus_baryon"]
    results=[]; all_bin_rows=[]
    for case in cases:
        print("[fit]",case)
        fit=fit_case(case,bins,data,args,a0)
        write_csv(tabledir/f"predictions_{case}.csv",fit["pred_rows"])
        all_bin_rows += fit["bin_rows"]
        results.append({k:v for k,v in fit.items() if k not in {"pred_rows","bin_rows"}})
    write_csv(tabledir/"bin_parameters.csv",all_bin_rows)

    best_aic=min(results,key=lambda r:r["AIC"])
    best_bic=min(results,key=lambda r:r["BIC"])
    comp=[]
    for r in sorted(results,key=lambda x:x["BIC"]):
        rr=dict(r)
        rr["delta_AIC_vs_best"]=rr["AIC"]-best_aic["AIC"]
        rr["delta_BIC_vs_best"]=rr["BIC"]-best_bic["BIC"]
        comp.append(rr)
    write_csv(tabledir/"model_comparison.csv",comp,preferred=["case","chi2","k_params","dof","chi2_per_dof","AIC","BIC","delta_AIC_vs_best","delta_BIC_vs_best","success"])

    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(range(len(comp)),[r["BIC"] for r in comp])
    ax.set_xticks(range(len(comp))); ax.set_xticklabels([r["case"] for r in comp],rotation=35,ha="right")
    ax.set_ylabel("BIC"); ax.set_title("DRND-ST vs NFW baseline")
    ax.grid(alpha=.25,axis="y"); fig.tight_layout()
    fig.savefig(plotdir/"model_comparison.png",dpi=180); plt.close(fig)

    by={r["case"]:r for r in comp}
    drnd=by.get("drnd_fixed_a0_screen2h"); nfw=by.get("nfw_plus_baryon")
    decision={
        "best_model_by_BIC":best_bic["case"],
        "best_model_by_AIC":best_aic["case"],
        "drnd_screened_beats_baryon_BIC": drnd["BIC"] < by["baryon_only"]["BIC"] if drnd else None,
        "drnd_screened_beats_nfw_plus_baryon_BIC": drnd["BIC"] < nfw["BIC"] if drnd and nfw else None,
        "delta_BIC_DRND_minus_NFWplusBaryon": drnd["BIC"]-nfw["BIC"] if drnd and nfw else None,
        "delta_AIC_DRND_minus_NFWplusBaryon": drnd["AIC"]-nfw["AIC"] if drnd and nfw else None,
        "drnd_chi2_per_dof": drnd["chi2_per_dof"] if drnd else None,
        "nfw_plus_baryon_chi2_per_dof": nfw["chi2_per_dof"] if nfw else None
    }
    summary={"model":"DRND-ST vs NFW lensing baseline",
             "input":vars(args)|{"n_points":len(rows),"bins":bins,"bin_mode_used":mode_used,"a0_fixed":a0},
             "model_comparison":comp,"decision":decision,
             "cautions":["Diagnostic spherical projection only.",
                         "NFW baseline lacks full HOD, CDM two-halo term, miscentering, satellites and survey covariance.",
                         "Use this as a control test before publication-grade likelihood."]}
    (outdir/"summary.json").write_text(json.dumps(safe_json(summary),indent=2),encoding="utf-8")
    print(json.dumps(safe_json({"outdir":str(outdir),"decision":decision,
                                "files":["summary.json","tables/model_comparison.csv","tables/bin_parameters.csv","plots/model_comparison.png"]}),indent=2))

if __name__=="__main__":
    main()
