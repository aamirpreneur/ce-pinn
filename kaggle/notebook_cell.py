import os, subprocess
from kaggle_secrets import UserSecretsClient

# ---- change these per run ------------------------------------------------
GITHUB_REPO = "aamirpreneur/ce-pinn"
TARGET      = "smoke_rlw"       # any key from TARGETS in kaggle/run.py
# os.environ["SMOKE"] = "1"     # uncomment for fast end-to-end test
# --------------------------------------------------------------------------

TOKEN = UserSecretsClient().get_secret("GITHUB_TOKEN")

with open(os.path.expanduser("~/.git-credentials"), "w") as f:
    f.write(f"https://x-access-token:{TOKEN}@github.com\n")
subprocess.run(["git", "config", "--global", "credential.helper", "store"], check=True)
subprocess.run(["git", "config", "--global", "user.email", "kaggle-bot@auto.local"], check=True)
subprocess.run(["git", "config", "--global", "user.name", "kaggle-bot"], check=True)

os.chdir("/kaggle/working")
subprocess.run(["rm", "-rf", "/kaggle/working/repo"])
subprocess.run(["git", "clone", f"https://github.com/{GITHUB_REPO}.git",
                "/kaggle/working/repo"], check=True)
os.chdir("/kaggle/working/repo")

subprocess.run(["python", "kaggle/run.py", TARGET], check=False)
