"""Generate paper tables and vector figures solely from declared run artifacts."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code/src"))
from state_constrained.aligned.train import load_run, _evaluate_chunks

LABELS = {("boundary_decay_1d", 1): "I", ("directional_exit_1d", 1): "II",
          **{("sum_cylinder_nd", d): f"III, d={d}" for d in [2, 5, 10, 20]}}
ORDER = list(LABELS)
FIG = ROOT/"figures/aligned"
GEN = ROOT/"results/tables"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tex(path, text):
    (GEN/path).write_text(text+"\n", encoding="utf-8")


def mean_sd(values, digits=3):
    values = np.asarray(values)
    return f"{values.mean():.{digits}f} $\\pm$ {values.std(ddof=1):.{digits}f}"


def sci(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return mantissa+r"\times10^{"+str(int(exponent))+"}"


def save(fig, name):
    fig.savefig(FIG/(name+".pdf"), bbox_inches="tight")
    fig.savefig(FIG/(name+".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def references():
    folder = ROOT/"reproduction/aligned/reference_studies"
    rows = read(folder/"summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), layout="constrained")
    for name, color, label in [("boundary_decay_1d", "#2166ac", "I"), ("directional_exit_1d", "#b35806", "II")]:
        for study, ax, key in [("mesh", axes[0, 0], "h"), ("penalty", axes[0, 1], "epsilon")]:
            selection = sorted([r for r in rows if r["name"] == name and r["study"] == study], key=lambda r: r["config"][key])
            ax.loglog([r["config"][key] for r in selection], [r["reference_rms"] for r in selection], "o-", ms=3, color=color, label=label)
        files = sorted(folder.glob(f"{name}_*_localization.npz"))
        data = [np.load(p) for p in files]
        reference = data[-1]
        mask = abs(reference["x"]) <= 2+1e-12
        radii, errors = [], []
        for path, arrays in zip(files, data):
            radii.append(read(path.with_suffix(".json"))["config"]["improvement_radius"])
            errors.append(float(abs(arrays["value"][mask]-reference["value"][mask]).max()))
        axes[1, 0].semilogy(radii, np.where(np.asarray(errors)>0, errors, np.nan), "o-", ms=3, color=color)
        selection = sorted([r for r in rows if r["name"] == name and r["study"] == "outer_box"], key=lambda r: r["config"]["training_radius"])
        axes[1, 1].semilogy([r["config"]["training_radius"] for r in selection],
            [max(r["boundary_bracket_width"], 1e-16) for r in selection], "o-", ms=3, color=color)
    titles = ["Mesh refinement ($\\epsilon=0.03$)", "Penalty refinement ($h=0.001$)",
              "Improvement-region effect", "Outer boundary comparison width"]
    xlabels = ["h", "$\\epsilon$", "L", "L'"]
    ylabels = ["RMS vs constrained value", "RMS vs constrained value", "Max difference from L=4", "Max upper-minus-lower"]
    for ax, title, xlabel, ylabel in zip(axes.flat, titles, xlabels, ylabels):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
    from matplotlib.ticker import NullFormatter
    for ax in axes[0]:
        ax.xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].legend(frameon=False)
    save(fig, "reference_studies")
    coupled = [r for r in rows if r["study"] == "coupled"]
    coarse = [r["reference_rms"] for r in coupled if r["config"]["epsilon"] == .1]
    fine = [r["reference_rms"] for r in coupled if r["config"]["epsilon"] == .003]
    slacks = [h["recursion_slack"] for r in rows for h in r.get("inexact_history", []) if "recursion_slack" in h]
    text = (f"The reference study contains {len(rows)} configurations. Along the coupled sequence "
        f"$h=\\varepsilon/2$, reducing $\\varepsilon$ from $0.1$ to $0.003$ changes the two "
        f"RMS errors from ${sci(min(coarse))}$--${sci(max(coarse))}$ to "
        f"${sci(min(fine))}$--${sci(max(fine))}$. Separate mesh and penalty sweeps are shown "
        f"in Figure~\\ref{{fig:reference_studies}}. Six deliberately inexact evaluations "
        f"use perturbation amplitudes $0.001$, $0.01$, and $0.05$; all {len(slacks)} recorded "
        f"recursion checks have nonnegative numerical slack (minimum ${sci(min(slacks))}$). "
        "The localization plot compares finite-lattice values with the largest improvement box; "
        "the outer-box plot reports the independently computed comparison width. Widths below "
        "$10^{-16}$ are displayed at the plotting floor.")
    tex("reference_results.tex", text)


def exact_results(manifest):
    groups = defaultdict(list)
    csv_rows = []
    for relative in manifest["exact_runs"]:
        run = ROOT/relative
        cfg, metric = read(run/"config.json"), read(run/"validation_float64.json")
        train, history = read(run/"metrics.json"), read(run/"history.json")
        key = (cfg["name"], cfg["dim"], cfg["method"])
        groups[key].append((run, cfg, metric, train, history))
        csv_rows.append({"run": relative, "benchmark": cfg["name"], "dim": cfg["dim"], "method": cfg["method"],
            "seed": cfg["seed"], "rms": metric["rms"], "relative_rms": metric["relative_rms"],
            "sample_max_error": metric["sample_max_error"], "domain_bellman_rms": metric["domain_bellman_rms"],
            "domain_evaluation_rms": metric["domain_evaluation_rms"], "box_bellman_sample_max": metric["box_bellman_sample_max"],
            "objective_evaluations": train["total_objective_evaluations"], "elapsed_seconds": train["elapsed_seconds"],
            "dtype": cfg["dtype"], "device": cfg["device"], "penalty_scale": cfg["penalty_scale"]})
    for key in ORDER:
        for method in ["pi", "bellman"]:
            if len(groups[(*key, method)]) != 5:
                raise ValueError(f"Exactly five declared runs required: {key} {method}")
    with (FIG/"exact_run_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader(); writer.writerows(csv_rows)
    table = [r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{3pt}",
        r"\caption{Final PINN-PI errors, mean $\pm$ sample standard deviation over five seeds. All metrics use double-precision evaluation at domain points separate from training samples.}\label{tab:aligned_exact}",
        r"\begin{tabular}{lrrr}\toprule Benchmark & RMS & Relative RMS (\%) & Sample maximum \\\midrule"]
    baseline = [r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{3pt}",
        r"\caption{Matched nominal optimization budgets: PINN-PI and direct Bellman residual minimization. Entries are seed means; detailed timing records are supplied in the machine-readable tables.}\label{tab:aligned_baseline}",
        r"\begin{tabular}{lrrrr}\toprule & \multicolumn{2}{c}{Relative RMS (\%)} & \multicolumn{2}{c}{Bellman RMS} \\ Benchmark & PI & Direct & PI & Direct \\\midrule"]
    protocol = [r"\begin{table}[tbp]\centering\small\setlength{\tabcolsep}{3pt}",
        r"\caption{Exact-benchmark protocol, common to paired methods. All MLPs have three hidden layers. Adam uses batches of $1024$; L-BFGS uses a fresh fixed batch of $2048$. The last column is the L-BFGS iteration budget per stage.}\label{tab:aligned_protocol}",
        r"\begin{tabular}{lrrrrrr}\toprule Benchmark & $(\varepsilon,h)$ & $\kappa$ & $(L,L')$ & Width & Adam updates & L-BFGS \\\midrule"]
    fig, axes = plt.subplots(2, 3, figsize=(7., 3.65), layout="constrained")
    learn, lax = plt.subplots(2, 3, figsize=(7., 3.65), layout="constrained")
    for idx, key in enumerate(ORDER):
        pi, direct = groups[(*key, "pi")], groups[(*key, "bellman")]
        label = LABELS[key]
        table.append(label+" & "+mean_sd([r[2]["rms"] for r in pi])+" & "+mean_sd([100*r[2]["relative_rms"] for r in pi], 2)+" & "+mean_sd([r[2]["sample_max_error"] for r in pi])+r" \\")
        baseline.append(label+" & "+" & ".join(f"{np.mean([r[2][field] for r in rows])*scale:.3f}" for rows, field, scale in
             [(pi, "relative_rms", 100), (direct, "relative_rms", 100), (pi, "domain_bellman_rms", 1), (direct, "domain_bellman_rms", 1)])+r" \\")
        cfg = pi[0][1]
        protocol.append(f"{label} & ({cfg['epsilon']:g},{cfg['h']:g}) & {cfg['penalty_scale']:g} & ({cfg['improvement_radius']:g},{cfg['training_radius']:g}) & {cfg['width']} & ${cfg['policy_iterations']}\\times {cfg['steps_per_policy']}$ & {cfg['lbfgs_steps']} "+r" \\")
        axis = np.linspace(-2 if key[1] == 1 else -1, 2 if key[1] == 1 else 1, 801)
        x = torch.from_numpy(axis[:, None]/key[1]).repeat(1, key[1])
        values = []
        for run, *_ in pi:
            p, model, _, _ = load_run(run/"checkpoint_final.pt", dtype="float64")
            values.append(_evaluate_chunks(model, x).numpy().ravel())
        values = np.asarray(values)
        exact = p.exact(x).numpy().ravel()
        ax = axes.flat[idx]
        ax.plot(axis, exact, color="#b35806", lw=1.4, label="Exact")
        ax.plot(axis, values.mean(0), color="#2166ac", lw=1.2, label="PINN-PI")
        ax.fill_between(axis, values.mean(0)-values.std(0, ddof=1), values.mean(0)+values.std(0, ddof=1), color="#2166ac", alpha=.18)
        ax.set(title=label, xlabel="x" if key[1] == 1 else "S", ylabel="Value")
        np.savez_compressed(FIG/f"solution_{idx}.npz", coordinate=axis, prediction=values, exact=exact)
        curves = np.asarray([[h["relative_rms"] for h in r[4]] for r in pi])
        residuals = np.asarray([[h["domain_bellman_rms"] for h in r[4]] for r in pi])
        steps = np.arange(1, curves.shape[1]+1)
        ax = lax.flat[idx]
        ax.semilogy(steps, curves.mean(0), "o-", ms=2.5, color="#2166ac", label="Relative RMS")
        ax.semilogy(steps, residuals.mean(0), "s--", ms=2.5, color="#b35806", label="Bellman RMS")
        ax.set(title=label, xlabel="Stage", ylabel="Domain diagnostic")
        ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False, fontsize=7)
    lax.flat[0].legend(frameon=False, fontsize=6.5)
    save(fig, "exact_solutions"); save(learn, "learning_curves")
    end = r"\bottomrule\end{tabular}\end{table}"
    tex("exact_results.tex", "\n".join(table+[end]))
    tex("baseline_results.tex", "\n".join(baseline+[end])+"\n"+
        "The comparison does not show a uniform advantage for either method. Actual objective evaluations, "
        "elapsed times, all seed results, and finite-sample maxima are supplied in the machine-readable tables. "
        "Small residuals for a fixed penalized operator alone do not remove its penalty and discretization bias.")
    tex("protocol_table.tex", "\n".join(protocol+[end])+"\n"+
        "One-dimensional training uses float64 on CPU (eight threads per run); the cylindrical runs use float32 "
        "on GPU. All table diagnostics are reevaluated in float64. TF32 is disabled. "
        "The learning-rate floor is one tenth of the stage's initial rate. No stage is selected by test error.")


def main():
    torch.set_num_threads(2)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    FIG.mkdir(parents=True, exist_ok=True); GEN.mkdir(parents=True, exist_ok=True)
    manifest = read(ROOT/"reproduction/aligned/PAPER_RUNS.json")
    references(); exact_results(manifest)
    from build_focused_results import build
    build(manifest)
    evidence = {"generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "focused_generator_sha256": hashlib.sha256(Path(__file__).with_name("build_focused_results.py").read_bytes()).hexdigest(),
                "confinement_diagnostic_sha256": hashlib.sha256((ROOT/"reproduction/aligned/confinement_diagnostic/summary.json").read_bytes()).hexdigest(),
                "paper_runs": manifest,
                "artifacts": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(FIG.glob("*")) if p.is_file() and p.name != "provenance.json"}}
    (FIG/"provenance.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(FIG)


if __name__ == "__main__":
    main()
