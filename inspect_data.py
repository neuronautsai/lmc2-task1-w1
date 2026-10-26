"""Build the data manifest from the organizer TSVs, scan OME-TIFF metadata and run consistency checks.

Usage:
  python inspect_data.py manifest --info DIR_WITH_TSVS --out OUT_DIR   (writes OUT_DIR/manifest.csv, before download)
  python inspect_data.py scan --manifest manifest.csv --out OUT_DIR    (after download, writes data_manifest_full.csv)
"""
import argparse
import glob
import os
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import tifffile

TRANS = {"BF", "PC", "DIC"}
FLUO = {"Nucleus", "Mitochondria", "Tubulin", "Actin"}


def study_of(path):
    return int(re.match(r"Study_(\d+)", os.path.basename(path)).group(1))


def build_manifest(info):
    allf, bests = [], []
    for f in sorted(glob.glob(info + "/Study_*_all_files.tsv")):
        d = pd.read_csv(f, sep="\t")
        d.insert(0, "study", study_of(f))
        d["channel"] = d.channel.replace("Phase", "PC")
        d["filename"] = d.filename.str.replace("_Phase_", "_PC_")
        allf.append(d)
    for f in sorted(glob.glob(info + "/Study_*_best_focus.tsv")):
        d = pd.read_csv(f, sep="\t")
        rows = []
        for _, r in d.iterrows():
            for c in d.columns:
                if c.endswith("_best_z"):
                    ch = c[:-7]
                    nz = r.get(ch + "_nZ", r.get("nZ"))
                    rows.append({"study": study_of(f), "image_id": r["image_id"], "channel": ch, "best_z": r[c], "nZ": nz})
        bests.append(pd.DataFrame(rows))
    m = pd.concat(allf, ignore_index=True).merge(pd.concat(bests, ignore_index=True),
                                                 on=["study", "image_id", "channel"], how="left")
    m["kind"] = m.channel.map(lambda c: "transmitted" if c in TRANS else ("fluorescence" if c in FLUO else "other"))
    m["needed"] = (m.kind == "transmitted") | ((m.kind == "fluorescence") & (m.z == m.best_z))
    return m


def meta(p):
    out = {"filename": os.path.basename(p)}
    try:
        with tifffile.TiffFile(p) as t:
            s = t.series[0]
            out.update(shape="x".join(map(str, s.shape)), axes=s.axes, dtype=str(s.dtype))
            ome = t.ome_metadata or ""
            for k in ("PhysicalSizeX", "PhysicalSizeXUnit", "PhysicalSizeY"):
                mm = re.search(k + r'="([^"]*)"', ome)
                out[k] = mm.group(1) if mm else ""
    except Exception as e:
        out["error"] = repr(e)
    return out


def scan(manifest, out):
    m = pd.read_csv(manifest)
    have = {}
    for d in sorted(glob.glob(f"{out}/Study_*")):
        have[study_of(d)] = {x for x in os.listdir(d) if x.endswith(".ome.tiff")}
    m["on_drive"] = [fn in have.get(s, ()) for s, fn in zip(m.study, m.filename)]
    print(pd.crosstab(m.needed, m.on_drive))
    os.makedirs(f"{out}/_meta", exist_ok=True)
    for s in sorted(have):
        outp = f"{out}/_meta/Study_{s:02d}.csv"
        if os.path.exists(outp):
            continue
        paths = [f"{out}/Study_{s:02d}/{fn}" for fn in sorted(have[s])]
        with ThreadPoolExecutor(16) as ex:
            rows = list(ex.map(meta, paths))
        pd.DataFrame(rows).to_csv(outp, index=False)
    meta_all = pd.concat([pd.read_csv(f).assign(study=study_of(f)) for f in sorted(glob.glob(f"{out}/_meta/Study_*.csv"))],
                         ignore_index=True)
    print(len(meta_all), "rows, errors:", meta_all["error"].notna().sum() if "error" in meta_all else 0)
    print(meta_all.groupby(["axes", "dtype"]).size())
    print("distinct shapes:", meta_all["shape"].nunique())
    full = m.merge(meta_all.drop(columns="error", errors="ignore"), on=["study", "filename"], how="left")
    d = full[full.on_drive]
    mis = d.groupby(["study", "image_id"])["shape"].nunique()
    mp = d.groupby(["study", "image_id"])["PhysicalSizeX"].nunique()
    print("images with shape mismatch across channels:", (mis > 1).sum(), "/", len(mis))
    print("images with pixel size mismatch across channels:", (mp > 1).sum(), "/", len(mp))
    full.to_csv(f"{out}/data_manifest_full.csv", index=False)
    print("manifest:", full.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("manifest")
    a1.add_argument("--info", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("scan")
    a2.add_argument("--manifest", required=True)
    a2.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "manifest":
        os.makedirs(a.out, exist_ok=True)
        build_manifest(a.info).to_csv(f"{a.out}/manifest.csv", index=False)
    else:
        scan(a.manifest, a.out)
