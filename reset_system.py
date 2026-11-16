"""Full factory reset before a demo: deletes ALL employees, face photos, attendance, alerts,
snapshots, Excel reports, logs and recordings. Models and configuration are kept.

    python reset_system.py          asks for confirmation
    python reset_system.py --yes    no prompt
"""
import argparse
import glob
import os
import shutil
import socket
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))


def app_is_running(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    ap = argparse.ArgumentParser(description="Delete every record for a fresh start")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    os.chdir(ROOT)
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    app = cfg.get("app", {})
    data_dir = os.path.abspath(app.get("data_dir", "data"))
    targets = {
        "Database (employees, attendance, alerts, settings)": glob.glob(os.path.join(data_dir, "attendance.db*")),
        "Employee face photos": [os.path.join(data_dir, "faces")],
        "Alert snapshots": [os.path.join(data_dir, "snapshots")],
        "Excel reports": [os.path.abspath(app.get("reports_dir", "reports"))],
        "Logs": [os.path.abspath(app.get("log_dir", "logs"))],
        "Recorded videos": [os.path.abspath(cfg.get("recording", {}).get("dir", "outputs"))],
    }

    port = int(cfg.get("web", {}).get("port", 8000))
    if app_is_running(port):
        print(f"\n[STOP] The attendance app is still running (port {port} is in use).")
        print("       Close it first (press Q in the camera window or Ctrl+C), then run this again.\n")
        sys.exit(1)

    print("\n" + "=" * 64)
    print("  FULL RESET - the following will be permanently deleted:")
    print("=" * 64)
    for label, paths in targets.items():
        existing = [p for p in paths if os.path.exists(p)]
        print(f"  - {label}")
        for p in existing:
            print(f"      {p}")
    print("\n  Kept: models/, configs/, code")
    print("=" * 64)

    if not args.yes and input("\nType RESET to delete everything: ").strip() != "RESET":
        print("Cancelled - nothing was deleted.")
        return

    failed = []
    for paths in targets.values():
        for p in paths:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.exists(p):
                    os.remove(p)
            except OSError as e:
                failed.append(f"{p}: {e.strerror}")

    for d in (data_dir, targets["Excel reports"][0], targets["Logs"][0]):
        os.makedirs(d, exist_ok=True)

    if failed:
        print("\nSome items could not be deleted (open in another program?):")
        for line in failed:
            print("  " + line)
        sys.exit(1)
    print("\nDone. The system is empty - start it with:  python main.py")
    print("Then register employees at http://localhost:%d (Register Employee tab).\n" % port)


if __name__ == "__main__":
    main()
