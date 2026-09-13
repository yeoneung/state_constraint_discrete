"""Compact contribution-oriented tables and figures from saved diagnostics."""
import csv
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT/"figures/aligned"
GEN = ROOT/"results/tables"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tex(name, content):
    (GEN/name).write_text(content+"\n", encoding="utf-8")


def save(fig, name):
    fig.savefig(FIG/(name+".pdf"), bbox_inches="tight")
    fig.savefig(FIG/(name+".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def mean_sd(values, digits=2):
    a = np.asarray(values)
    return f"{a.mean():.{digits}f} $\\pm$ {a.std(ddof=1):.{digits}f}"


def sci(value, upper=False):
    if upper and value > 0:
        unit = 10.**(np.floor(np.log10(value))-2)
        value = np.ceil(value/unit)*unit
    m, e = f"{value:.2e}".split("e")
    return m+r"\times10^{"+str(int(e))+"}"


def mechanism():
    evidence = read(ROOT/"reproduction/aligned/error_mechanism/summary.json")
    rows = evidence["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(5.12, 3.55), layout="constrained")
    summaries = []
    for col, (name, label) in enumerate([("boundary_decay_1d", "I"), ("directional_exit_1d", "II")]):
        selected = [r for r in rows if r["name"] == name]
        stages = np.arange(1, 9)
        def curve(key):
            return np.asarray([[next(r[key] for r in selected if r["seed"] == seed and r["stage"] == stage)
                for stage in range(8)] for seed in range(1, 6)])
        ax = axes[0, col]
        for key, color, text, style in [("next_policy_value_error", "#2166ac", r"$e_{n+1}$", "o-"),
            ("api_right_hand_side", "#b35806", r"$\gamma_h e_n+g_n/\lambda$", "s--")]:
            a = curve(key)
            for seed_curve in a:
                ax.semilogy(stages, seed_curve, color=color, alpha=.22, lw=.6)
            ax.semilogy(stages, a.mean(0), style, color=color, ms=2.7, lw=1.2, label=text)
        ax.set(title=f"{label}: policy error and bound", xlabel="Stage", ylabel="Full-grid maximum")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(alpha=.2)
        ax = axes[1, col]
        for key, color, text, style in [
            ("evaluation_rms", "#2166ac", "Evaluation", "o-"),
            ("iteration_rms", "#b35806", "Policy iteration", "s-"),
            ("mesh_rms", "#1b7837", "Mesh/viscosity", "-"),
            ("bias_proxy_rms", "#762a83", "Remaining bias", "-"),
            ("total_rms", "#222222", "Total", "--")]:
            ax.semilogy(stages, curve(key).mean(0), style, color=color, ms=2.4, lw=1.1, label=text)
        ax.set(title=f"{label}: separate error fields", xlabel="Stage", ylabel="Domain RMS")
        ax.grid(alpha=.2)
        ratios = [r["api_right_hand_side"]/r["next_policy_value_error"] for r in selected if r["next_policy_value_error"] > 1e-10]
        final = [r for r in selected if r["stage"] == 7]
        summaries.append({"benchmark": label, "median_bound_ratio": float(np.median(ratios)),
            "max_bound_ratio": float(max(ratios)), "fine_factor": final[0]["fine_factor"],
            "fine_difference_rms": final[0]["fine_successive_difference_rms"],
            **{k: float(np.mean([r[k] for r in final])) for k in
               ["evaluation_rms", "iteration_rms", "mesh_rms", "bias_proxy_rms", "total_rms"]}})
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, -.09),
        frameon=False, fontsize=7, ncol=3)
    save(fig, "error_mechanism")
    with (FIG/"error_mechanism.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (FIG/"error_mechanism_summary.json").write_text(json.dumps(summaries, indent=2)+"\n")
    tex("mechanism_results.tex",
        "The lower panels of Figure~\\ref{fig:error_mechanism} separate the error fields. "
        f"In II, the final mesh/viscosity difference and remaining-bias proxy have RMS "
        f"${sci(summaries[1]['mesh_rms'])}$ and ${sci(summaries[1]['bias_proxy_rms'])}$, "
        "yet their sum has smaller RMS because the signed errors partly cancel. "
        "The final neural error also includes evaluation and policy-iteration errors.")
    detail = [r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Final signed error-field RMS, averaged over the five PINN-PI seeds. The norms are not additive. The last column is a successive fine-mesh difference, not a rigorous continuum error bound.}\label{tab:mechanism_details}",
        r"\begin{tabular}{lrrrrrr}\toprule & Evaluation & PI & Mesh/viscosity & Bias proxy & Total & Fine change \\\midrule"]
    for s in summaries:
        detail.append(s["benchmark"]+" & "+" & ".join("$"+sci(s[k])+"$" for k in
            ["evaluation_rms", "iteration_rms", "mesh_rms", "bias_proxy_rms", "total_rms", "fine_difference_rms"])+r" \\")
    tex("mechanism_details.tex", "\n".join(detail+[r"\bottomrule\end{tabular}\end{table}"]))


def confinement():
    """Keep the main figure focused on confinement and resolved neural errors."""
    result = read(ROOT/"reproduction/aligned/confinement_diagnostic/summary.json")
    rows = read(ROOT/"reproduction/aligned/error_mechanism/summary.json")["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(5.3, 3.8), layout="constrained")
    leak = sorted(result["leakage"], key=lambda r:r["h"])
    for key, label, color, marker in [("distance", r"$p=\min\{\mathrm{dist},0.3\}$", "#2166ac", "o"),
                                       ("squared", r"$q=\min\{\mathrm{dist}^2,0.09\}$", "#b35806", "s")]:
        axes[0,0].loglog([r["h"] for r in leak], [r[key+"_over_h"] for r in leak],
                         marker=marker, color=color, ms=3, lw=1.1, label=label)
    axes[0,0].set(title="Inward-policy leakage", xlabel="$h$", ylabel="Penalty-only value / $h$")
    axes[0,0].legend(frameon=False, fontsize=6.4)
    for h, color, marker in zip(result["design"]["scales"],
            ["#333333", "#2166ac", "#b35806", "#1b7837", "#762a83"], ["o","s","^","v","D"]):
        selected = sorted([r for r in result["localization"] if r["h"] == h], key=lambda r:r["epsilon"])
        axes[0,1].loglog([r["epsilon"] for r in selected], [r["localization_over_h"] for r in selected],
                         marker=marker, color=color, ms=3, lw=1.1, label=f"$h={h:g}$")
    axes[0,1].set(title="Localization across penalties", xlabel=r"$\varepsilon$", ylabel=r"$E_{\mathrm{loc}}/h$")
    axes[0,1].legend(frameon=False, fontsize=5.8, loc="center left", ncol=2)
    from matplotlib.ticker import NullFormatter
    for ax in axes[0]:
        ax.set_xticks([.0025,.005,.01,.02,.04], labels=[".0025",".005",".01",".02",".04"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x",labelsize=7)
    for col,(name,label) in enumerate([("boundary_decay_1d","I"),("directional_exit_1d","II")]):
        selected = [r for r in rows if r["name"] == name]
        ax = axes[1,col]
        for key,color,title,style in [
                ("evaluation_rms", "#2166ac", "Evaluation", "o-"),
                ("iteration_rms", "#b35806", "Policy iteration", "s-"),
                ("mesh_rms", "#1b7837", "Mesh/viscosity", "-"),
                ("bias_proxy_rms", "#762a83", "Remaining bias", "-"),
                ("total_rms", "#222222", "Total", "--")]:
            means = [np.mean([r[key] for r in selected if r["stage"] == stage]) for stage in range(8)]
            ax.semilogy(np.arange(1,9), means, style, color=color, ms=2.4, lw=1.1, label=title)
        ax.set(title=f"{label}: separate error fields", xlabel="Stage", ylabel="Domain RMS")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    handles, labels = axes[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5,-.08), frameon=False, fontsize=7, ncol=3)
    save(fig,"confinement_and_errors")

    fig, axes = plt.subplots(1,2,figsize=(5.3,1.8),layout="constrained")
    for ax,(name,label) in zip(axes,[("boundary_decay_1d","I"),("directional_exit_1d","II")]):
        selected = [r for r in rows if r["name"] == name]
        for key,color,title,style in [("next_policy_value_error","#2166ac",r"$e_{n+1}$","o-"),
                                     ("api_right_hand_side","#b35806",r"$\gamma_he_n+g_n/\lambda$","s--")]:
            curves = np.asarray([[next(r[key] for r in selected if r["seed"] == seed and r["stage"] == stage)
                                  for stage in range(8)] for seed in range(1,6)])
            for curve in curves:
                ax.semilogy(np.arange(1,9),curve,color=color,alpha=.22,lw=.6)
            ax.semilogy(np.arange(1,9),curves.mean(0),style,color=color,ms=2.7,lw=1.2,label=title)
        ax.set(title=f"{label}: policy error and bound",xlabel="Stage",ylabel="Full-grid maximum")
        ax.legend(frameon=False,fontsize=7);ax.grid(alpha=.2)
    save(fig,"policy_recursion")
    max_ratio = max(r["localization_over_h"] for r in result["localization"])
    tex("confinement_results.tex",
        f"Figure~\\ref{{fig:error_mechanism}} (top) shows leakage divided by $h$ decreasing under refinement "
        f"and localization errors at most ${max_ratio:.3f}h$ over all {len(result['localization'])} pairs. "
        "The modest logarithmic margin illustrates uniform behavior over the tested penalty range; "
        "it does not test the conservative sufficient constant in the theorem. "
        "Supplementary Section~\\ref{supp:confinement_diagnostic} gives the outer-boundary and solver-error checks.")
    uncertainty = max(r["diagnostic_uncertainty"] for r in result["localization"])
    relative = max(r["diagnostic_uncertainty"]/r["localization_error"] for r in result["localization"])
    expansion = max(r["change_from_radius_40"] for r in result["outer_expansion"])
    tex("confinement_details.tex",
        f"The maximum combined boundary-width and residual estimate is ${sci(uncertainty)}$, "
        f"and it is less than ${np.ceil(relative*10000)/100:.2f}\\%$ of the measured localization difference in every case. "
        f"Increasing the outer radius from $40$ to $60$ at the three declared corner configurations "
        f"changes the localization difference by at most ${sci(expansion, upper=True)}$. "
        "These are double-precision checks without outward rounding.")


def exact_compact(manifest):
    labels = [("boundary_decay_1d", 1, "I"), ("directional_exit_1d", 1, "II")]
    labels += [("sum_cylinder_nd", d, f"III, $d={d}$") for d in [2, 5, 10, 20]]
    table = [r"\begin{table}[tbp]\centering\small",
        r"\caption{Relative RMS errors (\%), mean $\pm$ sample standard deviation over five paired seeds. The last column is the within-seed PI-minus-direct difference in percentage points; nominal optimization budgets are matched.}\label{tab:exact_compact}",
        r"\begin{tabular}{lrrr}\toprule Benchmark & PINN-PI & Direct Bellman & Paired difference \\\midrule"]
    data = [(read(ROOT/run/"config.json"), read(ROOT/run/"validation_float64.json")) for run in manifest["exact_runs"]]
    for name, dim, label in labels:
        values = {}
        for method in ["pi", "bellman"]:
            values[method] = np.asarray([next(m["relative_rms"]*100 for c, m in data
                if c["name"] == name and c["dim"] == dim and c["method"] == method and c["seed"] == seed)
                for seed in range(1, 6)])
        table.append(label+" & "+" & ".join(mean_sd(a) for a in
            [values["pi"], values["bellman"], values["pi"]-values["bellman"]])+r" \\")
    tex("exact_compact.tex", "\n".join(table+[r"\bottomrule\end{tabular}\end{table}"]))


def paired_obstacle():
    folder = ROOT/"reproduction/aligned/paired_obstacle"
    manifest = read(folder/"manifest.json")
    table = [r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{4pt}",
        r"\caption{Paired obstacle evaluation at the final stage. Residual is domain Bellman RMS; evaluation and policy errors are domain grid maxima against the frozen-policy value and midpoint-boundary Bellman value, respectively. Arrivals and segment violations use 200 common test starts at $\Delta t=0.00625$.}\label{tab:paired_obstacle}",
        r"\begin{tabular}{lrrrrr}\toprule Evaluator, seed & Residual & Eval. error & Policy error & Arrivals & Violations \\\midrule"]
    evidence = []
    for seed in manifest["seeds"]:
        for objective in ["raw", "grid"]:
            run = folder/f"{objective}_seed{seed}"
            metric = read(run/"validation_float64.json")
            policy = read(run/"hybrid_policy_validation.json")["stages"][-1]
            feedback = read(run/"final_feedback_value.json")
            audit = read(run/"test_audit/obstacle_audit.json")
            assert audit["heldout_sampling_seed"] == manifest["test_sampling_seed"]
            outcomes = [r["groups"]["heldout"] for r in audit["results"]]
            row = {"objective": objective, "seed": seed, "residual_rms": metric["domain_bellman_rms"],
                "evaluation_error": policy["neural_value_mismatch_domain_grid"],
                "policy_error": feedback["policy_value_error_domain_grid"],
                "arrivals": [r["goal_reached"] for r in outcomes],
                "violations": [r["strict_segment_violations"] for r in outcomes]}
            evidence.append(row)
            label = "Raw" if objective == "raw" else "Grid-assisted"
            table.append(f"{label}, {seed} & "+" & ".join(f"{row[k]:.3f}" for k in ["residual_rms", "evaluation_error", "policy_error"])
                +f" & {row['arrivals'][-1]}/200 & {row['violations'][-1]}"+r" \\")
    ref = read(folder/"reference/test_audit/obstacle_audit.json")
    last = ref["results"][-1]["groups"]["heldout"]
    table.append(f"Finite-grid reference & --- & --- & --- & {last['goal_reached']}/200 & {last['strict_segment_violations']}"+r" \\")
    tex("paired_obstacle.tex", "\n".join(table+[r"\bottomrule\end{tabular}\end{table}"]))
    (FIG/"paired_obstacle_summary.json").write_text(json.dumps(evidence, indent=2)+"\n")
    detail = [r"\begin{table}[tbp]\centering\small",
        r"\caption{All paired obstacle test outcomes. Triples correspond to $\Delta t=0.025,0.0125,0.00625$; every entry uses the same 200 starts. Times are measured per run while sharing the machine; grid time is diagnostic overhead for the raw arm.}\label{tab:paired_obstacle_details}",
        r"\begin{tabular}{lrrrr}\toprule Evaluator, seed & Arrivals & Violations & Neural time (s) & Grid time (s) \\\midrule"]
    for row in evidence:
        history = read(folder/f"{row['objective']}_seed{row['seed']}/history.json")
        label = "Raw" if row["objective"] == "raw" else "Grid-assisted"
        detail.append(f"{label}, {row['seed']} & "+"/".join(map(str, row["arrivals"]))+" & "+
            "/".join(map(str, row["violations"]))+f" & {sum(r['neural_training_seconds'] for r in history):.1f} & {sum(r['grid_seconds'] for r in history):.1f}"+r" \\")
    detail.append(r"\bottomrule\end{tabular}\end{table}")
    tex("paired_obstacle_details.tex", "\n".join(detail))
    raw = [r for r in evidence if r["objective"] == "raw"]
    grid = [r for r in evidence if r["objective"] == "grid"]
    same = all(len(set(r["arrivals"])) == len(set(r["violations"])) == 1 for r in evidence)
    text = ("Arrival and violation counts are unchanged at the two larger time steps. " if same else
        "Counts at all three time steps are provided in the supplement. ")
    grid_min = min(r['arrivals'][-1] for r in grid)
    grid_max = max(r['arrivals'][-1] for r in grid)
    grid_range = f"${grid_min}$" if grid_min == grid_max else f"${grid_min}$--${grid_max}$"
    text += (r"At $\Delta t=0.00625$, across the three seeds, raw-residual evaluation reaches the goal from "
        f"${min(r['arrivals'][-1] for r in raw)}$--${max(r['arrivals'][-1] for r in raw)}$ of the $200$ test starts, "
        f"whereas grid-assisted evaluation reaches it from {grid_range} starts. ")
    if max(r['residual_rms'] for r in raw) < min(r['residual_rms'] for r in grid) and \
       min(r['evaluation_error'] for r in raw) > max(r['evaluation_error'] for r in grid) and \
       min(r['policy_error'] for r in raw) > max(r['policy_error'] for r in grid):
        text += ("The raw-trained networks have smaller sampled Bellman residuals, but larger frozen-policy "
            "and deployed-policy value errors and fewer goal arrivals. ")
    text += (
        "This paired experiment compares evaluation procedures, including their finite-grid boundary treatment.")
    tex("paired_obstacle_results.tex", text)
    config = read(folder/"grid_seed1/config.json")
    import sys
    sys.path.insert(0, str(ROOT/"code/src"))
    from state_constrained.aligned.problem import Problem, RunConfig
    p = Problem(RunConfig(**config))
    c = p.base.config
    fig, axes = plt.subplots(1, 2, figsize=(5.3, 2.3), layout="constrained")
    for ax, objective, title in zip(axes, ["raw", "grid"], ["Raw residual, seed 1", "Grid-assisted, seed 1"]):
        paths = np.load(folder/f"{objective}_seed1/test_audit/rollout_dt_0.00625.npz")["paths"][:, :5]
        ax.add_patch(Circle((0, 0), c.workspace_radius, fill=False, ec="#333333", lw=.9))
        for center, radius in zip(c.centers, c.radii):
            ax.add_patch(Circle(center, radius, fc="#dddddd", ec="#777777", lw=.6))
        ax.add_patch(Circle(c.goal, c.success_radius, fill=False, ec="#19854a", lw=.8, ls="--"))
        ax.plot(*c.goal, marker="*", color="#19854a", ms=7)
        for i, color in enumerate(["#2166ac", "#b35806", "#1b7837", "#762a83", "#c51b7d"]):
            ax.plot(paths[:, i, 0], paths[:, i, 1], color=color, lw=1.)
            ax.plot(*paths[0, i], "o", ms=3, color=color)
            ax.plot(*paths[-1, i], "s", ms=2.5, color=color)
        inset = ax.inset_axes([.61,-.50,.36,.32])
        inset.set_facecolor("white")
        inset.add_patch(Circle(c.goal,c.success_radius,fill=False,ec="#19854a",lw=.8,ls="--"))
        inset.plot(*c.goal,marker="*",color="#19854a",ms=5)
        for i,color in enumerate(["#2166ac", "#b35806", "#1b7837", "#762a83", "#c51b7d"]):
            inset.plot(paths[:,i,0],paths[:,i,1],color=color,lw=.8)
            inset.plot(*paths[-1,i],"s",color=color,ms=3)
        inset.set(xlim=(.65,1.17),ylim=(.25,.80),aspect="equal",xticks=[],yticks=[])
        inset.set_title("Goal detail",fontsize=5.5,pad=1)
        ax.set(xlim=(-1.28, 1.28), ylim=(-1.28, 1.28), aspect="equal", xlabel="$x_1$", ylabel="$x_2$", title=title)
    save(fig, "paired_obstacle")


def build(manifest):
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    mechanism()
    confinement()
    exact_compact(manifest)
    paired_obstacle()



if __name__ == "__main__":
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    build(read(ROOT/"reproduction/aligned/PAPER_RUNS.json"))
