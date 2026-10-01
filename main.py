import argparse
import json
import time
from pathlib import Path
import numpy as np
from common import BASE,finish,save_json,table,plt
from model import solve,evaluate,normalization,fronts,coverage,feasible


def main():
    p=argparse.ArgumentParser(description='Вариант 16, трек A: план закупок, NSGA-II')
    p.add_argument('--config',type=Path,default=BASE/'config.json');p.add_argument('--runs',type=int);p.add_argument('--seed',type=int);p.add_argument('--output',type=Path,default=BASE/'results')
    a=p.parse_args();c=json.loads(a.config.read_text(encoding='utf-8'))
    if a.runs is not None:c['runs']=a.runs
    if a.seed is not None:c['first_seed']=a.seed
    if c['runs']<1 or c['population']<4 or c['generations']<1 or c['first_seed']<0 or not all(0<=c[k]<=1 for k in ['mutation_probability','crossover_probability']):p.error('Некорректная конфигурация')
    data=json.loads((BASE/'data/procurement.json').read_text(encoding='utf-8'));scale=normalization(data)
    rows=[];traces={};all_fronts={};export=[];comparisons=[]
    for method in ['NSGA-II','Weighted-GA']:
        traces[method]=[];all_fronts[method]=[]
        for i in range(c['runs']):
            seed=c['first_seed']+i;start=time.perf_counter();pop,values,trace,calls=solve(c,data,seed,method)
            rows.append(dict(method=method,seed=seed,score=trace[-1],front_size=len(values),seconds=time.perf_counter()-start,evaluations=calls,feasible=int(all(feasible(x,data) for x in pop))))
            traces[method].append(trace);all_fronts[method].append(values)
            for x,v in zip(pop,values):export.append(dict(method=method,seed=seed,quantities=x.tolist(),cost=float(v[0]),risk=float(v[1]),storage=float(v[2])))
        print(method,'completed',flush=True)
    finish(a.output,c,rows,traces,'Best fixed weighted score (diagnostic only)')
    save_json(a.output/'fronts.json',export)
    for i in range(c['runs']):
        n,w=all_fronts['NSGA-II'][i],all_fronts['Weighted-GA'][i]
        comparisons.append(dict(seed=c['first_seed']+i,nsga_covers_weighted=coverage(n,w),weighted_covers_nsga=coverage(w,n),nsga_front_size=len(n),weighted_front_size=len(w)))
    table(a.output/'coverage.csv',comparisons)
    # Иллюстрация одного заранее фиксированного seed, а не удачно выбранного запуска.
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
    labels=['Cost','Shortage CVaR90','Expected inventory volume']
    for method in all_fronts:
        values=all_fronts[method][0]
        for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):ax.scatter(values[:,i],values[:,j],s=18,alpha=.7,label=method);ax.set(xlabel=labels[i],ylabel=labels[j]);ax.grid(alpha=.2)
    axes[0].legend();fig.savefig(a.output/'pareto.png',dpi=160);plt.close(fig)
    candidates=[r for r in export if r['method']=='NSGA-II' and r['seed']==c['first_seed']]
    values=np.array([[r['cost'],r['risk'],r['storage']] for r in candidates])
    cheap=int(np.argmin(values[:,0]));safe=int(np.argmin(values[:,1]));order=np.argsort((values/scale)@np.array([.25,.6,.15]))
    balanced=next((int(i) for i in order if i not in {cheap,safe}),int(order[0]))
    save_json(a.output/'representatives.json',[dict(role=role,**candidates[i]) for role,i in [('cheap',cheap),('low_shortage',safe),('balanced',balanced)]])
    table(a.output/'front_sizes_summary.csv',[dict(method=m,minimum=min(map(len,v)),mean=float(np.mean(list(map(len,v)))),median=float(np.median(list(map(len,v)))),std=float(np.std(list(map(len,v)),ddof=1)) if len(v)>1 else 0.,maximum=max(map(len,v))) for m,v in all_fronts.items()])

if __name__=='__main__':main()
