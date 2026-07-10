"""Manual driver for the from-scratch run stage (no setup loop): run each claim's
entrypoint via paper-reprise's own from-scratch executor, write env_snapshot.json,
then leave grading/report to `paper-reprise report <run_dir>`.

Open the RunDir with an ABSOLUTE path: the executor activates env/bin on PATH
relatively and run_eval.sh cd's into impl/, so a relative run dir breaks resolution.
"""
import json
import sys
from pathlib import Path

from paper_reprise.fromscratch import make_fromscratch_run_executor
from paper_reprise.rundir import RunDir

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
rd = RunDir.open(ROOT)
spec = rd.read_spec()
arts = {a.id: a for a in spec.artifacts}

executor = make_fromscratch_run_executor()

only = set(sys.argv[2:])  # optional subset of claim ids
for c in spec.claims:
    if only and c.id not in only:
        continue
    cdir = rd.claim_dir(c.id)
    art = arts[c.artifact]
    print(f"[drive] running {c.id} (method={art.method}) ...", flush=True)
    try:
        meta = executor(c, art, cdir)
        print(f"[drive] {c.id}: ok in {meta['minutes']:.1f} min -> {meta['stdout_path']}",
              flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[drive] {c.id}: FAILED: {e}", flush=True)

# Environment snapshot for the report's Environment line (CUDA -> torch -> ... -> lm_eval).
# Query the RUN env's python (torch/lm_eval live there, not in the driver's venv).
import subprocess
try:
    env_py = str((ROOT / "env" / "bin" / "python").resolve())
    probe = (
        "import json,torch,transformers,lm_eval;"
        "print(json.dumps({'torch':torch.__version__,"
        "'transformers':transformers.__version__,"
        "'lm_eval':getattr(lm_eval,'__version__','unknown'),"
        "'cuda':torch.version.cuda or 'unknown',"
        "'rocm':getattr(torch.version,'hip',None) or 'unknown','pip_freeze':''}))"
    )
    out = subprocess.check_output([env_py, "-c", probe], text=True).strip()
    snap = json.loads(out.splitlines()[-1])
    (ROOT / "env_snapshot.json").write_text(json.dumps(snap, indent=2))
    print(f"[drive] wrote env_snapshot.json: {snap['torch']} / cuda {snap['cuda']}")
except Exception as e:  # noqa: BLE001
    print(f"[drive] env_snapshot skipped: {e}")

print("[drive] done.")
