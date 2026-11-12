"""Aggregate Week 3 runs. Usage: python aggregate_w3.py --root DIR
Reads DIR/<config>_s<seed>/run_info.json and summary.csv. Unfinished runs are listed and left out.
Std is over seeds (ddof=1). Peak GPU memory is reported as the max over seeds, because a run resumed after its last epoch
only measures the final evaluation and under-reports."""
import argparse, json, re
from pathlib import Path
import pandas as pd

ORG = ("Nucleus", "Mitochondria", "Tubulin", "Actin")
CFG = ("base", "cos", "cosfilm")


def load(root):
    rows, unfinished = [], []
    for d in sorted(Path(root).glob("*_s*")):
        m = re.fullmatch(r"(\w+)_s(\d+)", d.name)
        if not m:
            continue
        if not (d / "run_info.json").exists() or not (d / "summary.csv").exists():
            unfinished.append(d.name)
            continue
        info = json.load(open(d / "run_info.json"))
        s = pd.read_csv(d / "summary.csv").set_index("organelle")
        r = {"config": m.group(1), "seed": int(m.group(2)), "best_epoch": info["best_epoch"], "macro_ssim": info["macro_ssim"],
             "macro_pcc": info["macro_pcc"], "ssim_unknown": info.get("macro_ssim_unknown_modality"),
             "pcc_unknown": info.get("macro_pcc_unknown_modality"), "train_h": info["train_seconds"] / 3600,
             "infer_s": info["inference_seconds_per_image"], "peak_gb": info["peak_gpu_memory_gb"]}
        for o in ORG:
            r[f"ssim_{o}"], r[f"pcc_{o}"] = s.loc[o, "ssim_mean"], s.loc[o, "pcc_mean"]
        rows.append(r)
    return pd.DataFrame(rows), unfinished


def ms(x, nd=4):
    x = x.dropna()
    if not len(x):
        return "n/a"
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f} (n={len(x)})" if len(x) > 1 else f"{x.mean():.{nd}f} (n=1)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df, unfinished = load(a.root)
    out = Path(a.out or a.root)
    df.to_csv(out / "w3_runs.csv", index=False)
    L = []
    if unfinished:
        L.append("Unfinished (left out): " + ", ".join(unfinished))
    cfgs = [c for c in CFG if (df.config == c).any()]
    L += ["", "Every run", df[["config", "seed", "best_epoch", "macro_ssim", "macro_pcc", "ssim_unknown", "pcc_unknown"]].round(4).to_string(index=False)]
    t = {c: {"macro_ssim": ms(df[df.config == c].macro_ssim), "macro_pcc": ms(df[df.config == c].macro_pcc),
             "ssim_label_withheld": ms(df[df.config == c].ssim_unknown), "pcc_label_withheld": ms(df[df.config == c].pcc_unknown),
             "train_h": ms(df[df.config == c].train_h, 2), "infer_s_per_image": ms(df[df.config == c].infer_s, 3),
             "peak_gpu_gb_max": f"{df[df.config == c].peak_gb.max():.2f}"} for c in cfgs}
    L += ["", "Per configuration, mean ± std over seeds", pd.DataFrame(t).T.to_string()]
    for k in ("ssim", "pcc"):
        L += ["", f"Per organelle {k.upper()}", pd.DataFrame({o: {c: ms(df[df.config == c][f'{k}_{o}']) for c in cfgs} for o in ORG}).to_string()]
    p = df.pivot(index="seed", columns="config", values=["macro_ssim", "macro_pcc"])
    L.append("")
    for x, y in (("cos", "base"), ("cosfilm", "cos"), ("cosfilm", "base")):
        for m in ("macro_ssim", "macro_pcc"):
            if (m, x) in p and (m, y) in p:
                d = (p[(m, x)] - p[(m, y)]).dropna()
                if len(d):
                    L.append(f"{x} minus {y}, {m}: " + ", ".join(f"seed {s}: {v:+.4f}" for s, v in d.items()) + f" | mean {d.mean():+.4f}, gains {(d > 0).sum()} of {len(d)}")
    text = "\n".join(L)
    (out / "w3_summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
