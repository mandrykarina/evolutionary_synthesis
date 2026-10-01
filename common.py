"""Общие средства записи экспериментов; оптимизаторов здесь нет."""
import csv
import json
import platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows, metric="score"):
    out=[]
    for method in dict.fromkeys(r["method"] for r in rows):
        group=[r for r in rows if r["method"]==method]
        a=np.array([r[metric] for r in group])
        out.append(dict(method=method, runs=len(a), minimum=float(a.min()), mean=float(a.mean()), median=float(np.median(a)), std=float(a.std(ddof=1)) if len(a)>1 else 0., maximum=float(a.max()), mean_seconds=float(np.mean([r["seconds"] for r in group]))))
    return out

def curves(out, traces, ylabel, log=False, x_values=None):
    fig, ax=plt.subplots(figsize=(10,5.5), layout="constrained")
    for method, series in traces.items():
        a=np.array(series); x=np.arange(a.shape[1]) if x_values is None else np.asarray(x_values)
        line,=ax.plot(x,a.mean(axis=0),label=method,lw=2)
        ax.fill_between(x,a.min(axis=0),a.max(axis=0),alpha=.13,color=line.get_color())
    ax.set(xlabel="Generation" if x_values is None else "Objective evaluations",ylabel=ylabel)
    if log: ax.set_yscale("symlog",linthresh=1e-8)
    ax.grid(alpha=.2); ax.legend(); fig.savefig(out,dpi=160); plt.close(fig)

def finish(out, config, rows, traces, ylabel, log=False):
    out.mkdir(parents=True,exist_ok=True)
    table(out/"runs.csv",rows)
    summary=summarize(rows); table(out/"summary.csv",summary)
    save_json(out/"trajectories.json",traces)
    save_json(out/"experiment.json",dict(config=config,python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__,seed_policy="first_seed + run_index"))
    x_values = sorted(set(range(config["population"], config["budget"]+1, config["population"])) | {config["budget"]}) if "budget" in config else None
    curves(out/"convergence.png",traces,ylabel,log,x_values)
    lines=["# Результаты выполненной серии", "", "Минимум и максимум — значения метрики, направление оптимизации см. в отчёте. Стандартное отклонение выборочное (ddof=1). Полоса на графике — min–max, не доверительный интервал.", "", "| Метод | Запусков | Минимум | Среднее | Медиана | Std | Максимум | Время, с |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append("| {method} | {runs} | {minimum:.6g} | {mean:.6g} | {median:.6g} | {std:.6g} | {maximum:.6g} | {mean_seconds:.4f} |".format(**r))
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Results:",out)
    return summary

def tournament(rng, scores, count, size=3):
    candidates=rng.integers(len(scores),size=(count,size))
    return candidates[np.arange(count),np.argmin(scores[candidates],axis=1)]
