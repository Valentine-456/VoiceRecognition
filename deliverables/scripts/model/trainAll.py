import subprocess
import sys
from pathlib import Path

CONFIG_DIR = Path("src/config")

def main():
    configs = sorted(CONFIG_DIR.glob("*.yaml"))
    print(f"Found {len(configs)} configs:\n")

    for cfg in configs:
        print("=" * 20)
        print(f"Running training for {cfg.name}")
        print("=" * 20)

        subprocess.run(
            [sys.executable, "-m", "scripts.model.train", str(cfg.name)],
            check=True,
        )

    print("\nAll experiments finished.")

if __name__ == "__main__":
    main()
