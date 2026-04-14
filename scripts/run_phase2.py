import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

from src.utils.config import load_config


BASE = Path("D:/deepfake_detection")
NB1 = BASE / "notebooks/01_train_atn.ipynb"
NB2 = BASE / "notebooks/02_setup_deepsafe.ipynb"
NB3 = BASE / "notebooks/03_inference_test.ipynb"
REGISTRY = BASE / "models/registry.json"
REFACE_ATN = BASE / "models/reface_atn.pth"
DEEPSAFE = BASE / "models/deepsafe.pth"
CONFIG_PATH = BASE / "config/default.yaml"


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(
        [
            f
            for f in directory.rglob("*")
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def run_preflight() -> bool:
    script = BASE / "scripts/preflight_check.py"
    proc = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return proc.returncode == 0


def run_data_download() -> bool:
    script = BASE / "scripts/download_data.py"
    proc = subprocess.run([sys.executable, str(script)], check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    return proc.returncode == 0


def _extract_error_cell(notebook_path: Path):
    if not notebook_path.exists():
        return None, None
    with open(notebook_path, encoding="utf-8") as f:
        nb = json.load(f)
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                msg = output.get("evalue", "unknown error")
                src = "".join(cell.get("source", []))
                return idx, f"{output.get('ename', 'Error')}: {msg}\n{src}"
    return None, None


def execute_notebook(notebook_path: Path, timeout: int, label: str) -> bool:
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(notebook_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr)
    if proc.returncode != 0:
        cell_no, details = _extract_error_cell(notebook_path)
        if cell_no:
            print(f"[ERROR] Notebook {notebook_path.name} failed at cell {cell_no}: {details.splitlines()[0]}")
            print("[ERROR CELL SOURCE]")
            print("\n".join(details.splitlines()[1:]))
        else:
            print(f"[ERROR] Notebook {notebook_path.name} failed: nbconvert returned non-zero exit code")
        print(f"[{label}] ✗ Failed: notebook execution failed")
        return False
    return True


def verify_nb1_outputs() -> bool:
    with open(NB1, encoding="utf-8") as f:
        nb = json.load(f)
    cells_with_output = sum(1 for c in nb["cells"] if c.get("outputs"))
    print(f"[NB1] {cells_with_output}/{len(nb['cells'])} cells have outputs")
    if REFACE_ATN.exists():
        size_mb = REFACE_ATN.stat().st_size / (1024 ** 2)
        print(f"[NB1] PASS reface_atn.pth saved ({size_mb:.1f} MB)")
        return True
    print("[NB1] FAIL reface_atn.pth NOT found - notebook may have failed")
    return False


def verify_nb2_outputs() -> bool:
    if DEEPSAFE.exists():
        size_mb = DEEPSAFE.stat().st_size / (1024 ** 2)
        print(f"[NB2] PASS deepsafe.pth saved ({size_mb:.1f} MB)")
        return True
    print("[NB2] FAIL deepsafe.pth NOT found")
    return False


def update_registry_sha256() -> bool:
    with open(REGISTRY, encoding="utf-8") as f:
        registry = json.load(f)

    model_files = {
        "reface_atn": REFACE_ATN,
        "deepsafe": DEEPSAFE,
    }

    ok = True
    model_entries = registry.get("models", {})
    if not isinstance(model_entries, dict):
        print("[SHA256] FAIL registry.json has invalid schema: expected models object")
        return False

    for name, model_path in model_files.items():
        entry = model_entries.get(name)
        if not isinstance(entry, dict):
            ok = False
            print(f"[SHA256] FAIL {name}: entry missing from registry")
            continue

        if model_path.exists():
            real_hash = sha256(model_path)
            entry["sha256"] = real_hash
            print(f"[SHA256] {name}: {real_hash}")
        else:
            ok = False
            print(f"[SHA256] FAIL {name}: file not found, hash not updated")

    with open(REGISTRY, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
        f.write("\n")

    print("\n[registry.json] PASS Updated with real SHA256 hashes")
    return ok


def print_nb3_summary():
    with open(NB3, encoding="utf-8") as f:
        nb = json.load(f)

    last_cell = nb["cells"][-1]
    summary_found = False
    for output in last_cell.get("outputs", []):
        if output.get("output_type") in ("stream", "execute_result", "display_data"):
            text = output.get("text", output.get("data", {}).get("text/plain", ""))
            if isinstance(text, list):
                text = "".join(text)
            if text:
                summary_found = True
                print("[NB3 SUMMARY]\n" + text)
    if not summary_found:
        print("[NB3 SUMMARY] No printable output found in the last cell.")


def final_report():
    items = [
        ("data/real images", count_images(BASE / "data/real"), "≥ 5000"),
        ("data/fake images", count_images(BASE / "data/fake"), "≥ 1000"),
        ("data/atn_train images", count_images(BASE / "data/atn_train"), "≥ 3500"),
        ("data/atn_val images", count_images(BASE / "data/atn_val"), "≥ 900"),
        ("models/reface_atn.pth exists", REFACE_ATN.exists(), True),
        ("models/deepsafe.pth exists", DEEPSAFE.exists(), True),
        ("registry.json updated", REGISTRY.exists(), True),
        ("01_train_atn.ipynb executed", NB1.exists(), True),
        ("02_setup_deepsafe.ipynb done", NB2.exists(), True),
        ("03_inference_test.ipynb done", NB3.exists(), True),
    ]

    print(f"\n{'Item':<40} {'Value':<12} {'Target':<10} {'Status':>6}")
    print("-" * 72)
    flags = []
    for name, value, target in items:
        if isinstance(target, bool):
            ok = value == target
            display_val = "Yes" if value else "No"
            display_tgt = "Present"
        else:
            threshold = int(target.replace("≥ ", "").replace(",", ""))
            ok = int(value) >= threshold
            display_val = str(value)
            display_tgt = target
        flags.append(ok)
        status = "OK" if ok else "NO"
        print(f"{name:<40} {display_val:<12} {display_tgt:<10} {status:>6}")

    if all(flags):
        print("\nALL DONE - Models ready for AFS pipeline.")
    else:
        print("\nSome items incomplete - check failures above.")


def main() -> int:
    try:
        config = load_config(str(CONFIG_PATH))
        ci_cfg = config.get("ci", {})
        gate_label = ci_cfg.get("gate_label", "unspecified")
        counts_as_prod = bool(ci_cfg.get("counts_as_production_evidence", False))
        print(f"[CI GATE: {gate_label}]")
        if not counts_as_prod:
            print(
                "WARNING: This run uses a permissive smoke profile. "
                "Results do not constitute production validation."
            )

        preflight_ok = run_preflight()
        if preflight_ok:
            print("[STEP 1] PASS Done")
        else:
            print("[STEP 1] FAIL Failed: Blockers found in strict preflight")
            step2_ok = run_data_download()
            if step2_ok:
                print("[STEP 2] PASS Done")
            else:
                print("[STEP 2] FAIL Failed: Data acquisition targets not met")
                return 1

            preflight_after = run_preflight()
            if preflight_after:
                print("[STEP 2D] PASS Done")
            else:
                print("[STEP 2D] FAIL Failed: Preflight still reports blockers")
                return 1

        if execute_notebook(NB1, 7200, "STEP 3") and verify_nb1_outputs():
            print("[STEP 3] PASS Done")
        else:
            print("[STEP 3] FAIL Failed: Notebook 1 verification failed")
            return 1

        if execute_notebook(NB2, 3600, "STEP 4") and verify_nb2_outputs():
            print("[STEP 4] PASS Done")
        else:
            print("[STEP 4] FAIL Failed: Notebook 2 verification failed")
            return 1

        if update_registry_sha256():
            print("[STEP 5] PASS Done")
        else:
            print("[STEP 5] FAIL Failed: One or more model hashes could not be updated")
            return 1

        if execute_notebook(NB3, 1800, "STEP 6"):
            print("[NB3] PASS Inference test notebook executed")
            print_nb3_summary()
            print("[STEP 6] PASS Done")
        else:
            print("[STEP 6] FAIL Failed: Notebook 3 execution failed")
            return 1

        final_report()
        print("[STEP 7] PASS Done")
        return 0

    except Exception as exc:
        print(f"[FATAL] {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())