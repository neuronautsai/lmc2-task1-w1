"""Download LMC2 Task 1 data from the official Seafile share and keep only the files needed.

Kept per study: all transmitted-light slices (BF, PC, DIC) and the best-z slice of each fluorescence channel.
Needs: 7z on PATH, requests, pandas. Run info download and manifest build first (see build_manifest in inspect_data.py).
Usage: python download_data.py --base-url URL_TEMPLATE --root /content/lmc2 --out /content/drive/MyDrive/LMC2/task1 --manifest manifest.csv
"""
import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess

import pandas as pd
import requests

# Direct-download URL template of the organizers' Seafile share, with {} for the file name.
# Not stored in this repository, pass it with --base-url (format: https://HOST/d/SHARE_ID/files/?p=%2F{}&dl=1)
BASE = None
INFO_ZIP = "Study_info_filenames_and_optimal_focus_V3.zip"
# Split archives (read from the Seafile folder listing): number of .zNN parts per study
PARTS = {26: 2, 27: 8, 74: 5, 75: 4}
# Studies whose archive does not match the TSV listing (Study_59: TSV lists 684 files, archive has 107)
NON_STRICT = {59}


def fetch(name, root):
    dst = os.path.join(root, name)
    with requests.get(BASE.format(name), stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dst


def part_names(s):
    return [f"Study_{s:02d}.z{n:02d}" for n in range(1, PARTS.get(s, 0) + 1)] + [f"Study_{s:02d}.zip"]


def process_study(s, need, root, out, work):
    tag = f"Study_{s:02d}"
    dst = os.path.join(out, tag)
    marker = os.path.join(dst, "done.json")
    if os.path.exists(marker):
        print(tag, "already done")
        return
    for n in part_names(s):
        fetch(n, root)
    zp = os.path.join(root, f"{tag}.zip")
    lst = subprocess.run(["7z", "l", "-slt", zp], capture_output=True, text=True).stdout
    paths = [l[7:] for l in lst.splitlines() if l.startswith("Path = ") and ".ome.tiff" in l]
    prefix = os.path.dirname(paths[0])
    in_zip = {os.path.basename(p) for p in paths}
    nd = need[need.study == s]
    want = [fn for fn in nd.filename if fn in in_zip] if s in NON_STRICT else list(nd.filename)
    with open(os.path.join(work, f"{tag}_list.txt"), "w") as f:
        for fn in want:
            f.write((prefix + "/" if prefix else "") + fn + "\n")
    shutil.rmtree(os.path.join(work, tag), ignore_errors=True)
    subprocess.run(["7z", "x", "-y", f"-o{os.path.join(work, tag)}", zp, f"-i@{os.path.join(work, tag + '_list.txt')}"],
                   capture_output=True)
    got = glob.glob(os.path.join(work, tag, "**", "*.ome.tiff"), recursive=True)
    print(tag, "listed:", len(nd), "extracted:", len(got))
    if len(got) != len(want):
        print("COUNT MISMATCH, not moved")
        return
    os.makedirs(dst, exist_ok=True)
    for p in got:
        shutil.move(p, os.path.join(dst, os.path.basename(p)))
    info = {"study": s, "files": len(got), "listed": len(nd), "date": datetime.date.today().isoformat()}
    if s in NON_STRICT:
        info["note"] = "archive holds fewer files than the TSV lists, channel Phase in TSV is PC in the archive"
    json.dump(info, open(marker, "w"))
    shutil.rmtree(os.path.join(work, tag), ignore_errors=True)
    for n in part_names(s):
        pth = os.path.join(root, n)
        if os.path.exists(pth):
            os.remove(pth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="Seafile direct-download URL template with {} for the file name")
    ap.add_argument("--root", required=True, help="scratch dir for archives")
    ap.add_argument("--out", required=True, help="output dir (Drive)")
    ap.add_argument("--manifest", required=True, help="manifest csv with study, filename, kind, z, best_z")
    ap.add_argument("--work", default="/content/work")
    a = ap.parse_args()
    global BASE
    BASE = a.base_url
    for d in (a.root, a.out, a.work):
        os.makedirs(d, exist_ok=True)
    m = pd.read_csv(a.manifest)
    need = m[(m.kind == "transmitted") | ((m.kind == "fluorescence") & (m.z == m.best_z))]
    order = [s for s in range(1, 88) if s not in PARTS] + [26, 74, 75, 27]
    for s in order:
        if os.path.exists(os.path.join(a.out, f"Study_{s:02d}", "done.json")):
            continue
        try:
            process_study(s, need, a.root, a.out, a.work)
        except Exception as e:
            print(f"Study_{s:02d} ERROR:", repr(e))
            for p in glob.glob(os.path.join(a.root, f"Study_{s:02d}.z*")):
                os.remove(p)
            shutil.rmtree(os.path.join(a.work, f"Study_{s:02d}"), ignore_errors=True)


if __name__ == "__main__":
    main()
