### Simulate all the combinations of 3 values for each of the 14 parameters. 
# Computational time on 24 cores is 3 hours
# See prevalence_verona.ipynb for the variables explanation


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm
from scipy.optimize import fsolve
import matplotlib.dates as mdates
import datetime as dt
from lmfit import Parameters, Parameter, report_fit
from scipy import stats
import itertools
import multiprocessing as mp
import pickle

def datetime_range(start, events):
    diffs=np.diff(events)
    dates=[]
    dates.append(start)
    for i,d in enumerate(diffs):
        ne=dates[i]+timedelta(days=d)
        dates.append(ne)
    return dates

df_pre=pd.read_excel('csvs/verona.xlsx',0)
print(df_pre.tail(1))
df_post=pd.read_excel('csvs/verona.xlsx',1)
int_day=pd.to_datetime('2019-04-08')
print(int_day)
data=pd.concat([df_pre,df_post],ignore_index=True)
data['week']=pd.date_range("20180305", "20200504", freq='W-MON')
data.rename(columns={'N_CSKP_col_patients':'Sensitive',
                   'N-CRKP_col_patients':'Resistant',
                  'N_KPfree':'Free'},inplace=True)
data['n_patients']=data[['Sensitive','Resistant','Free']].sum(1)
data['intervention']=data.week>int_day
data['prevS']=data.Sensitive/data.n_patients
data['prevR']=data.Resistant/data.n_patients
data['prevF']=data.Free/data.n_patients

gelpre=44.27
gelpost=54.97

acc_pre=1357
acc_post=1421

occu_pre=.79
occu_post=.71
pdpre=14382
pdpost=13008


deg_pre=pdpre/acc_pre

days=7*114
dayspre=400


pr=0.006
ps=0.025
pf=1-ps-pr

B=46

Np=28 #data mean (not used)
F0=36
S0=0
R0=1

Nh=50/3 #circa 1/3 per turno
Hf0=Nh
Hs0=0
Hr0=0
print('Initial conditions: ',F0,S0,R0)
print('HCWs: ',Hf0)
#print('N to p ratio inverted: ',1/npr)

wrf=1/30 #blanqart
wsf=1/30 #blanqart
muh=1/(1/24) #sypsa


dres=1
dfpre=dres/10.05
dspre=dres/32.4
drpre=dres/60
dmeanpre=1/deg_pre
print('LOS mean pre: ',1/dmeanpre)

apre=acc_pre/dayspre


dmean=dmeanpre

nprCHI=1/12
nprICU=1/2
alpha=0.45
gamma=0.214
gel=gelpre
pdp=pdpre
ndays=dayspre


DOT_dict_pre={'BL':100.46/1000,'Cephal':47.66/1000,'Carbap':61.05/1000,'Floroq':21.64/1000}
Risk_dict ={'BL':2.28,'Cephal':4.52,'Carbap':3.99,'Floroq':1.75} #Li2020, Ofner-Agostini2009
DOT=sum(list(DOT_dict_pre.values()))
AR=sum([Risk_dict[tipo]*DOT_dict_pre[tipo] for tipo in DOT_dict_pre.keys()])/DOT
TD=DOT/dmean
Tfrac=TD*dmean
ATR=1+Tfrac*(AR-1)
print('Daily DOT per patient:',DOT)
print('Weighted average of risk:',AR)
print('Treatment duration: ',TD)
print('Antibiotic trasm risk increase: ',ATR)


intervenuto=False
tempo=np.linspace(1,days,days)


def differential_post(x,tempo,pars):
    F,S,R,Hf,Hs,Hr,In,Rcum,Rtrasm,Radm=x
           
    Kh=pars['Kh'].value
    h=pars['h'].value
    noth=1-h
    Tfrac=pars['Tfrac'].value
    AR=pars['AR'].value
    ATR=1+Tfrac*(AR-1)
    a=pars['a'].value
    pr=pars['pr'].value
    ps=pars['ps'].value
    pf=1-ps-pr
    wrf=pars['wrf'].value
    wsf=pars['wsf'].value
    alpha=pars['alpha'].value
    dmean=pars['dmean'].value
    muh=pars['muh'].value
    gamma=pars['gamma'].value
    
    dF=wrf*R+wsf*S-Kh*alpha*F*Hr*ATR-Kh*alpha*F*Hs-dmean*F+a*pf#*(B-S-F-R)/B
    dS=-wsf*S+Kh*alpha*F*Hs-dmean*S+a*ps#*(B-S-F-R)/B
    dR=+Kh*alpha*Hr*F*ATR-wrf*R-dmean*R+a*pr#*(B-S-F-R)/B
    dHf=-Kh*gamma*Hf*(S+R)*noth+muh*(Hs+Hr)+h*(Hs+Hr)*(F+S+R)*Kh
    dHs=Kh*gamma*Hf*S*noth-muh*Hs-h*Hs*(F+S+R)*Kh   
    dHr=Kh*gamma*R*Hf*noth-muh*Hr-h*Hr*(F+S+R)*Kh
    dIn=a#*(B-S-F-R)/B
    dRcum=Kh*alpha*Hr*F*ATR+a*pr#*(B-S-F-R)/B
    dRtrasm=Kh*alpha*Hr*F*ATR
    dRadm=a*pr#*(B-S-F-R)/B
    
    return dF,dS,dR,dHf,dHs,dHr,dIn,dRcum,dRtrasm,dRadm

def odesol(x0,t,pars,is_continue):
    q=pars['q'].value
    x0=list(x0)
    x0[3:6]=[val-val*q for val in x0[3:6]]
    #x0=tuple(x0)
    if is_continue:
        integ=odeint(differential_post,x0,t,mxstep=100000,args=(pars,))
    else:
        integ=odeint(differential,x0,t,mxstep=100000,args=(pars,))
    return integ

def residual(pars,ts,x0,data,interv):
    res=pd.DataFrame(odesol(x0,ts,pars,interv),columns=['Uncolonized','Sensitive','Resistant',
                                'HCW uncolonized','HCW sensitive',
                                'HCW resistant','New patient',
                                'Rcum','Rtrasm','Radm'])
    res['prevS']=res['Sensitive']/res[['Uncolonized','Sensitive','Resistant']].sum(1)
    res['prevR']=res['Resistant']/res[['Uncolonized','Sensitive','Resistant']].sum(1)
    res['prevF']=res['Uncolonized']/res[['Uncolonized','Sensitive','Resistant']].sum(1)
    data_fit=data[data.intervention==interv]
    return np.concatenate((data_fit.prevS.values-res.prevS.iloc[0::7].values,
                   data_fit.prevR.values-res.prevR.iloc[0::7].values))

x0= F0,S0,R0,Hf0,Hs0,Hr0,F0+S0+R0,R0,0,R0
print(x0)
# We directly set the fit results from the prevalence_verona notebook
h=0.85530#result.params['h'].value
Kh=(gelpre/1000*pdpre/0.004) / (h*Nh*pdpre )
q=0.10346#result.params['q'].value
a=3.3925
print('h:',h)
print('Kh:',Kh)
print('q:',q)
print('adm rate:',a)


# Prepare parameter intervals and all the combinations
percs=np.array([0.9,1,1.1])
wrfs=wrf*percs
wsfs=wsf*percs
Khs=Kh*percs
alphas=alpha*percs
Tfracs=Tfrac*percs
ARs=AR*percs
dmeans=dmean*percs
adms=a*percs
prs=pr*percs
pss=ps*percs
muhs=muh*percs
gammas=gamma*percs
hs=h*percs
qs=q*percs
#combs=itertools.product(wrfs,wsfs,Khs,alphas,Tfracs,ARs,drs,dss,dfs,adms,prs,pss,
#                        muhs,gammas,hs,qs)
combs=itertools.product(wrfs,wsfs,Khs,alphas,Tfracs,ARs,dmeans,
                        adms,prs,pss, muhs,gammas,hs,qs)

print('Preparing parameter space...')
s_t=time.time()
combs_mat_temp=np.row_stack([comb for comb in combs])
e_t=time.time()-s_t
ncombs=combs_mat_temp.shape[0]
print('No. simulations: ',ncombs)
print('Elapsed row stack: {} s'.format(e_t))

print('\nAdding identifier column...')
s_t=time.time()
combs_mat= np.empty((combs_mat_temp.shape[0],combs_mat_temp.shape[1]+1))
combs_mat[:,-1]=range(ncombs)
combs_mat[:,:-1] = combs_mat_temp
del combs_mat_temp
e_t=time.time()-s_t
print(combs_mat.shape)
print(' {} s'.format(e_t))

chunks=np.array_split(combs_mat,10000)
print('\nChunk size:', chunks[0].shape)
print(chunks[0][0:10,-3:])

data_fit=data[data.intervention==False]

def sim_chunk(chunk):
    final_res=pd.DataFrame(dtype=float, 
                           columns=['wrf','wsf','Kh','alpha','eps','xi','dmean',
                                   'a','pr','ps','muh','gamma','h','q',
                                   'pRf','pSf','pFf','pRmean','pSmean','pFmean',
                                   'hRf','hSf','hFf','hRmean','hSmean','hFmean',
                                    'fRprev','Rprev_mean','fSprev','Sprev_mean',
                                   'Frmse','Srmse','Rrmse','prevR_rmse'],
                          index=range(len(chunk)))
    j=int(chunk[-1,-1])
    for i in range(chunk.shape[0]):
        wrf,wsf,Kh,alpha,Tfrac,AR,dmean,a,pr,ps,muh,gamma,h,q,_= chunk[i,:]
        
        params = Parameters()
        params.add('q', value=q, min=0, max=1,vary=False)
        params.add('Kh', value=Kh, min=0, max=np.inf,vary=False)
        params.add('h', value=h, min=0., max=1,vary=False)
        params.add('Tfrac', value=Tfrac, min=0, max=np.inf,vary=False)
        params.add('AR', value=AR, min=min(Risk_dict.values()), max=max(Risk_dict.values()), vary=False)
        params.add('a',value=a,min=0,max=B,vary=False)
        params.add('wrf',value=wrf,min=0,max=np.inf,vary=False)
        params.add('wsf',value=wsf,min=0,max=np.inf,vary=False)
        params.add('alpha',value=alpha,min=0,max=1,vary=False)
        params.add('dmean',value=dmean,min=0,max=np.inf,vary=False)
        params.add('pr', value=pr, min=0, max=1,vary=False)
        params.add('ps',value=ps,min=0,max=1,vary=False)
        params.add('muh',value=muh,min=0,max=np.inf,vary=False)
        params.add('gamma',value=gamma,min=0,max=np.inf,vary=False)
                   
            #print('Parallel runs: started with alpha:',alpha)
        integ=odesol(x0,tempo[:400],params,is_continue=True)

        res=pd.DataFrame(integ,columns=['Uncolonized','Sensitive','Resistant',
                                'HCW uncolonized','HCW sensitive',
                                'HCW resistant','New patient',
                                'Rcum','Rtrasm','Radm'])
            
        fRp=res['Resistant'].iloc[-1]
            #dfRp[alpha].append(res['Resistant'].iloc[-1])
        fSp=res['Sensitive'].iloc[-1]
            #dfSp[alpha].append(res['Sensitive'].iloc[-1])
        fFp=res['Uncolonized'].iloc[-1]
        Rmean=res['Resistant'].iloc[28:].mean()
        Smean=res['Sensitive'].iloc[28:].mean()
        Fmean=res['Uncolonized'].iloc[28:].mean()
        
        fRh=res['HCW resistant'].iloc[-1]
            #dfRp[alpha].append(res['Resistant'].iloc[-1])
        fSh=res['HCW sensitive'].iloc[-1]
            #dfSp[alpha].append(res['Sensitive'].iloc[-1])
        fFh=res['HCW uncolonized'].iloc[-1]
        hRmean=res['HCW resistant'].iloc[28:].mean()
        hSmean=res['HCW sensitive'].iloc[28:].mean()
        hFmean=res['HCW uncolonized'].iloc[28:].mean()
        
        fRprev=res.Resistant.iloc[-1]/ res[['Uncolonized','Sensitive','Resistant']].sum(1).iloc[-1]
        Rprev_mean=np.mean(res.Resistant.iloc[0::7].values/
            res[['Uncolonized','Sensitive','Resistant']].sum(1).iloc[0::7].values)
        fSprev=res.Sensitive.iloc[-1]/ res[['Uncolonized','Sensitive','Resistant']].sum(1).iloc[-1]
        Sprev_mean=np.mean(res.Sensitive.iloc[0::7].values/
            res[['Uncolonized','Sensitive','Resistant']].sum(1).iloc[0::7].values)
        
        Frmse=np.sqrt(np.mean((data_fit.Free.values-res.Uncolonized.iloc[0::7].values)**2))
        Srmse=np.sqrt(np.mean((data_fit.Sensitive.values-res.Sensitive.iloc[0::7].values)**2))
        Rrmse=np.sqrt(np.mean((data_fit.Resistant.values-res.Resistant.iloc[0::7].values)**2))
        prev_rmse=np.sqrt(np.mean((
            data_fit.Resistant.values/data_fit[['Free','Sensitive','Resistant']].sum(1).values-
            res.Resistant.iloc[0::7].values/
            res[['Uncolonized','Sensitive','Resistant']].sum(1).iloc[0::7].values
        )**2))
        
        
        final_res.loc[i,:]= [wrf,wsf,Kh,alpha,Tfrac,AR,dmean,
                             a,pr,ps, muh,gamma,h,q,
                              fRp,fSp,fFp,Rmean,Smean,Fmean,
                              fRh,fSh,fFh,hRmean,hSmean,hFmean,
                             fRprev,Rprev_mean,fSprev,Sprev_mean,
                             Frmse,Srmse,Rrmse,prev_rmse]
    #final_res.to_parquet('data/sens_res_prev/chunk_'+str(j)+'.parquet')
    return None


s_t=time.time()    
pool=mp.Pool(24)
final_res=list(tqdm(pool.imap(sim_chunk,chunks),total=len(chunks)))
pool.close()
pool.join()


e_t=time.time()-s_t
print('Simulation time: {} hrs'.format(e_t/60/60))

        
    