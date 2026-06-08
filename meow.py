import os
import sys
import subprocess

TARGET_DIR = os.environ.get("MEOW_TARGET_DIR", "meow_decoder_only")

_KNOWN_TARGETS = [
    "meow",
    "meow_decoder_only",
    "meow_lgbm",
    "meow_lstm",
    "meow_xg",
]

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(root_dir, TARGET_DIR)
    target_script = os.path.join(target_dir, "meow.py")

    if not os.path.isfile(target_script):
        print(
            f"Error: Target '{TARGET_DIR}' not found.\n"
            f"  Looked for: {target_script}\n"
            f"  Available targets: {', '.join(_KNOWN_TARGETS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    result = subprocess.run(
        [sys.executable, target_script] + sys.argv[1:],
        cwd=root_dir,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
