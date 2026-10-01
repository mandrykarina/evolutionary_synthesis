"""NSGA-II собственной реализации для целочисленного плана закупок."""
import numpy as np


def generate(seed=2650717):
    rng=np.random.default_rng(seed)
    n=12;mean=rng.integers(5,15,n)
    return dict(seed=seed,items=[f'Item-{i+1:02}' for i in range(n)],prices=rng.integers(8,35,n).tolist(),volumes=rng.integers(1,5,n).tolist(),importance=rng.integers(1,5,n).tolist(),demand=rng.poisson(mean,(60,n)).clip(0,20).tolist(),max_quantity=20,warehouse_capacity=340,budget=3200)


def repair(x,data):
    x=np.clip(np.rint(x),0,data['max_quantity']).astype(int)
    prices=np.array(data['prices']);volumes=np.array(data['volumes'])
    while x@prices>data['budget'] or x@volumes>data['warehouse_capacity']:
        positive=np.flatnonzero(x)
        # Удалять единицу с наибольшим нормированным расходом ресурсов.
        burden=prices[positive]/data['budget']+volumes[positive]/data['warehouse_capacity']
        x[positive[np.argmax(burden)]]-=1
    return x


def feasible(x,data):
    return bool(np.all(x>=0) and np.all(x<=data['max_quantity']) and np.all(x==np.floor(x)) and x@np.array(data['prices'])<=data['budget'] and x@np.array(data['volumes'])<=data['warehouse_capacity'])


def evaluate(pop,data):
    pop=np.atleast_2d(pop);demand=np.array(data['demand']);importance=np.array(data['importance']);volumes=np.array(data['volumes'])
    short=np.maximum(demand[None,:,:]-pop[:,None,:],0)@importance
    # Среднее худших 10% сценариев: эмпирический CVaR_0.9 при 60 сценариях.
    tail=max(1,int(np.ceil(.1*len(demand))))
    risk=np.sort(short,axis=1)[:,-tail:].mean(axis=1)
    storage=(np.maximum(pop[:,None,:]-demand[None,:,:],0)@volumes).mean(axis=1)
    return np.c_[pop@np.array(data['prices']),risk,storage]


def normalization(data):
    zero=np.zeros((1,len(data['items'])))
    return np.array([data['budget'],max(1.,evaluate(zero,data)[0,1]),data['warehouse_capacity']])


def fronts(values):
    """Недоминируемая сортировка O(M N²), минимизация всех критериев."""
    dominates=np.all(values[:,None,:]<=values[None,:,:],axis=2)&np.any(values[:,None,:]<values[None,:,:],axis=2)
    counts=dominates.sum(axis=0);remaining=np.ones(len(values),dtype=bool);result=[]
    while remaining.any():
        current=np.flatnonzero(remaining&(counts==0))
        if not len(current):raise RuntimeError('Нарушен порядок доминирования')
        result.append(current);remaining[current]=False;counts-=dominates[current].sum(axis=0)
    return result


def crowding(values):
    n=len(values);distance=np.zeros(n)
    if n<=2:return np.full(n,np.inf)
    for column in range(values.shape[1]):
        order=np.argsort(values[:,column],kind='stable');span=values[order[-1],column]-values[order[0],column]
        if span==0:continue
        distance[order[[0,-1]]]=np.inf
        distance[order[1:-1]]+=(values[order[2:],column]-values[order[:-2],column])/span
    return distance


def ranking(values):
    rank=np.empty(len(values),dtype=int);distance=np.zeros(len(values))
    for i,front in enumerate(fronts(values)):rank[front]=i;distance[front]=crowding(values[front])
    return rank,distance


def survive(pop,values,n):
    selected=[]
    for front in fronts(values):
        free=n-len(selected)
        if len(front)<=free:selected.extend(front.tolist())
        else:
            order=np.argsort(-crowding(values[front]),kind='stable');selected.extend(front[order[:free]].tolist());break
        if len(selected)==n:break
    return pop[selected],values[selected]


def solve(c,data,seed,method='NSGA-II'):
    rng=np.random.default_rng(seed);n=c['population'];d=len(data['items']);scale=normalization(data)
    pop=np.array([repair(x,data) for x in rng.integers(0,21,(n,d))]);values=evaluate(pop,data)
    weights=np.array([.25,.6,.15]);trace=[float(np.min((values/scale)@weights))]
    # Архив всех найденных недоминируемых точек для обеих схем.
    archive_pop=pop.copy();archive_values=values.copy()
    for _ in range(c['generations']):
        if method=='NSGA-II':
            rank,distance=ranking(values)
            def select():
                a,b=rng.integers(n,size=2)
                return a if (rank[a],-distance[a])<(rank[b],-distance[b]) else b
        else:
            fitness=(values/scale)@weights
            def select():
                candidates=rng.integers(n,size=3);return candidates[np.argmin(fitness[candidates])]
        children=[]
        for _ in range(n):
            a,b=pop[select()],pop[select()];child=a.copy()
            if rng.random()<c['crossover_probability']:child=np.where(rng.random(d)<.5,a,b)
            mask=rng.random(d)<c['mutation_probability'];steps=rng.choice([-3,-2,-1,1,2,3],d)
            children.append(repair(child+mask*steps,data))
        children=np.array(children);child_values=evaluate(children,data)
        both=np.concatenate([pop,children]);both_values=np.concatenate([values,child_values])
        if method=='NSGA-II':pop,values=survive(both,both_values,n)
        else:
            order=np.argsort((both_values/scale)@weights,kind='stable')[:n];pop,values=both[order],both_values[order]
        archive_pop=np.concatenate([archive_pop,children]);archive_values=np.concatenate([archive_values,child_values])
        # Для трёх целевых значений одинаковые точки представлены одним планом.
        _,unique=np.unique(archive_values,axis=0,return_index=True)
        archive_pop,archive_values=archive_pop[unique],archive_values[unique]
        first=fronts(archive_values)[0];archive_pop,archive_values=archive_pop[first],archive_values[first]
        trace.append(min(trace[-1],float(np.min((values/scale)@weights))))
    return archive_pop,archive_values,trace,n*(c['generations']+1)


def coverage(a,b):
    """Доля точек B, слабо доминируемых хотя бы одной точкой A."""
    return float(np.any(np.all(a[:,None,:]<=b[None,:,:],axis=2),axis=0).mean())
