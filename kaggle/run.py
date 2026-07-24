#!/usr/bin/env python
"""Kaggle-side runner: run one target, tee the full log, push results to GitHub.
Invoked by kaggle/notebook_cell.py from inside the freshly cloned repo (whose
git credential helper already holds the token, so `git push` just works)."""
import os, sys, subprocess, datetime, json

# ---- CE-PINN targets: map target name -> script (+ optional env) ----
TARGETS = {
    # --- KdV ---
    "kdv1_vanilla":   {"script": "experiments/kdv_single/train_kdv_single.py",
                       "env": {"VARIANT": "vanilla"}},
    "kdv1_hard_bc":   {"script": "experiments/kdv_single/train_kdv_single.py",
                       "env": {"VARIANT": "hard_bc"}},
    "kdv1_ce_poly":   {"script": "experiments/kdv_single/train_kdv_single.py",
                       "env": {"VARIANT": "ce_poly"}},
    "kdv1_ce_full":   {"script": "experiments/kdv_single/train_kdv_single.py",
                       "env": {"VARIANT": "ce_full"}},
    "kdv2_ce_poly":   {"script": "experiments/kdv_two_soliton/train_kdv_two.py",
                       "env": {"VARIANT": "ce_poly"}},
    "kdv2_vanilla":   {"script": "experiments/kdv_two_soliton/train_kdv_two.py",
                       "env": {"VARIANT": "vanilla"}},

    # --- RLW (main contribution) ---
    "rlw1_vanilla":   {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "vanilla"}},
    "rlw1_hard_bc":   {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "hard_bc"}},
    "rlw1_ce_poly":   {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "ce_poly"}},
    "rlw1_ce_helm":   {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "ce_helm"}},
    "rlw1_ce_full":   {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "ce_full"}},
    "rlw2_ce_helm":   {"script": "experiments/rlw_two_soliton/train_rlw_two.py",
                       "env": {"VARIANT": "ce_helm"}},
    "rlw2_vanilla":   {"script": "experiments/rlw_two_soliton/train_rlw_two.py",
                       "env": {"VARIANT": "vanilla"}},
    "rlw3_ce_helm":   {"script": "experiments/rlw_bore/train_rlw_bore.py",
                       "env": {"VARIANT": "ce_helm"}},
    "rlw3_vanilla":   {"script": "experiments/rlw_bore/train_rlw_bore.py",
                       "env": {"VARIANT": "vanilla"}},

    # --- KP (2D) ---
    "kp1_ce_poly":    {"script": "experiments/kp_line/train_kp_line.py",
                       "env": {"VARIANT": "ce_poly"}},
    "kp1_vanilla":    {"script": "experiments/kp_line/train_kp_line.py",
                       "env": {"VARIANT": "vanilla"}},
    "kp2_ce_poly":    {"script": "experiments/kp_lump/train_kp_lump.py",
                       "env": {"VARIANT": "ce_poly"}},
    "kp2_vanilla":    {"script": "experiments/kp_lump/train_kp_lump.py",
                       "env": {"VARIANT": "vanilla"}},

    # --- Smoke tests (quick sanity check) ---
    "smoke_kdv":      {"script": "experiments/kdv_single/train_kdv_single.py",
                       "env": {"VARIANT": "ce_poly", "SMOKE": "1"}},
    "smoke_rlw":      {"script": "experiments/rlw_single/train_rlw_single.py",
                       "env": {"VARIANT": "ce_helm", "SMOKE": "1"}},
}
BAD_GPU = "P100"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kw)


def current_branch():
    r = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return (r.stdout or "").strip() or "main"


def push(outdir, target, runid, state):
    branch = current_branch()
    sh(["git", "config", "user.email", "kaggle-bot@auto.local"])
    sh(["git", "config", "user.name", "kaggle-bot"])
    sh(["git", "add", "-A", "results"])
    sh(["git", "commit", "-m", f"results({target}/{runid}): {state}"])
    for attempt in range(5):
        sh(["git", "fetch", "origin", branch])
        sh(["git", "pull", "--rebase", "--autostash", "origin", branch])
        if sh(["git", "push", "origin", f"HEAD:{branch}"]).returncode == 0:
            print(f"\nPUSHED -> results/{target}/{runid} (state={state})", flush=True)
            return
        print(f"push attempt {attempt+1} failed; retrying after rebase...", flush=True)
    print("PUSH FAILED after retries. Kaggle kernel output still has the files.", flush=True)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        print("usage: python kaggle/run.py <target> ; targets:", sorted(TARGETS)); sys.exit(2)
    target = sys.argv[1]
    script = TARGETS[target]["script"]
    target_env = TARGETS[target]["env"]
    smoke = target_env.get("SMOKE", os.environ.get("SMOKE", "0")) == "1"

    ts = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    runid = ts + ("-smoke" if smoke else "")
    outdir = os.path.join("results", target, runid)
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(outdir, "run.log")

    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
    except Exception as e:
        gpu = f"(nvidia-smi failed: {e})"
    print("GPU:", gpu, flush=True)

    status = {"target": target, "runid": runid, "gpu": gpu, "smoke": smoke,
              "started_utc": ts, "state": "running"}
    json.dump(status, open(os.path.join(outdir, "status.json"), "w"), indent=2)

    with open(logpath, "w", encoding="utf-8") as logf:
        logf.write(f"target={target} runid={runid} gpu={gpu} smoke={smoke}\n"); logf.flush()
        if BAD_GPU and BAD_GPU in gpu:
            msg = (f"ABORTED: GPU is {gpu} (unsupported). "
                   "Set Accelerator = 'GPU T4 x2' in notebook settings, then Save & Run All.\n")
            print(msg, flush=True); logf.write(msg); status["state"] = "aborted_bad_gpu"
        else:
            env = dict(os.environ, OUT_DIR=outdir, **target_env)
            p = subprocess.Popen([sys.executable, "-u", script], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                logf.write(line); logf.flush()
            rc = p.wait()
            status["returncode"] = rc
            status["state"] = "done" if rc == 0 else f"error_rc{rc}"

    status["ended_utc"] = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    json.dump(status, open(os.path.join(outdir, "status.json"), "w"), indent=2)
    print("STATE:", status["state"], flush=True)
    push(outdir, target, runid, status["state"])


if __name__ == "__main__":
    main()
