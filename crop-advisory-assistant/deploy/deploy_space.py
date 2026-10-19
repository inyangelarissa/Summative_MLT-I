import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="your Hugging Face username (or organisation)")
    ap.add_argument("--space_name", default="crop-advisory-assistant")
    ap.add_argument("--adapter", required=True, help="LoRA adapter folder, relative to the repo root")
    ap.add_argument("--no_context", action="store_true")
    ap.add_argument("--no_climate", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="build the folder only, do not upload")
    args = ap.parse_args()

    space_dir = ROOT / "space"
    cmd = [sys.executable, str(ROOT / "deploy" / "prepare_space.py"), "--space_dir", str(space_dir),
           "--adapter", args.adapter]
    if args.no_context:
        cmd.append("--no_context")
    if args.no_climate:
        cmd.append("--no_climate")
    subprocess.run(cmd, check=True, cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1"))
    if args.dry_run:
        print("Dry run finished: nothing uploaded.")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo_id = f"{args.user}/{args.space_name}"
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=args.private)
    api.upload_folder(folder_path=str(space_dir), repo_id=repo_id, repo_type="space",
                      commit_message="Deploy crop advisory assistant")
    print(f"Uploaded. Your Space: https://huggingface.co/spaces/{repo_id}")
    print("It builds for a few minutes; watch the 'Logs' tab on that page.")


if __name__ == "__main__":
    main()
