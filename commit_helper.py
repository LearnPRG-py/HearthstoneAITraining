import os
import re
import subprocess
import sys

# This is a full helper script to help contributors commit changes.

print(
    "Thank you for contributing to the Hearthstone AI. this script will guide \
    you through the process of committing your changes."
)

print("[1/7] - Checking for libraries and versions.")

print("[1/6] - Checking for git...", end="")
try:
    subprocess.run(
        ["git", "--version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Git is installed!")
except:
    print("Git is not installed. Please install git to proceed.")
    exit(1)

py = input(
    "[2/6] - Checking for python at python3. Enter to continue or type a different path to change the python call: "
).strip()

if py == "":
    py = "python3"

try:
    subprocess.run(
        [py, "--version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Using python at:", py, ". Python is installed.")
except:
    print("Python is not installed. Please install python to proceed.")
    exit(1)

print("[3/6] - Checking for pip...", end="\r")
try:
    subprocess.run(
        [py, "-m", "pip", "--version"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("Pip is installed!")
except:
    print("Pip is not installed. Please install pip to proceed.")
    exit(1)

print("[4/6] - Checking for requirements.txt...", end="\r")
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    print("requirements.txt not found. Please create a requirements.txt file.")
    exit(1)

not_installed = []
for req in requirements:
    print(f"[5/6] - Checking for {req}...", end="\r")
    try:
        os.system(
            py + " -m pip show " + req, output=open(os.devnull, "w")
        )  # Suppress output
    except:
        print(f"{req} is not installed. Please install {req} to proceed.")
        not_installed.append(req)

if not_installed:
    print(f"The following packages are not installed: {', '.join(not_installed)}")
    print("[6/6] - Installing missing packages...", end="\r")
    for req in not_installed:
        os.system(py + " -m pip install " + req)
else:
    print("[6/6] - All packages are installed.", end="\r")

print("[2/7] - Checking libraries")

libraries = [
    ("tensorflow", "tensorflow"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib.pyplot", "matplotlib"),
    ("sklearn.model_selection", "scikit-learn"),
    ("black", "black"),
]

for lib, package in libraries:
    print(f"Checking {lib}...", end="")

    try:
        __import__(lib)
        print("Done")
    except ImportError:
        print("Failed")
        print(f"{package} is not installed. Please install {package} to proceed.")
        exit(1)

print("[2/7] - All libraries verified")
print("[3/7] - Checking uncommitted changes...")

import subprocess
import sys

ALLOWED_FILES = {"model.py"}

try:
    result = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
    )

    changed_files = []

    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        # Git porcelain format: XY filename
        file_path = line[3:].strip()
        changed_files.append(file_path)

    invalid_files = [file for file in changed_files if file not in ALLOWED_FILES]

    if invalid_files:
        print("Failed")
        print("The following files have uncommitted changes:")
        for file in invalid_files:
            print(f" - {file}")

        print("\nOnly model.py is allowed to have changes.")
        sys.exit(1)

    print("Done")

except subprocess.CalledProcessError:
    print("Failed")
    print("Git repository check failed.")
    sys.exit(1)

print("[4/7] - Black formatting...")
try:
    os.system(py + " -m black .")
    print("Done")
except:
    print("Failed")
    print("Black formatting failed. Please fix the issues and try again.")
    exit(1)

print("[5/7] - Running tests...")
import train

# Empty training_log.txt
with open("training_log.txt", "w") as f:
    f.write("")

print("Running training with reduced epochs and dataset for testing purposes...")
test_loss, test_acc = train.train_callback(
    epochs=5, batch_size=256, reduced_training=True, CI=True
)

print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

print("[6/7] - Checking test results...")

last_commit = subprocess.check_output(["git", "log", "-1", "--pretty=%B"], text=True)
match = re.search(r"data=\(([\d.]+),([\d.]+)\)", last_commit)
if not match:
    print("[ERROR] Could not find previous benchmark data in latest commit.")
    exit(1)

best_acc = float(match.group(1))
best_loss = float(match.group(2))
print(f"Previous best accuracy: {best_acc}")
print(f"Previous best loss: {best_loss}")

if test_acc < best_acc:
    print("Failed")
    print(f"Test accuracy {test_acc} is lower than " f"the best accuracy {best_acc}.")
    exit(1)

if test_loss > best_loss:
    print("Failed")
    print(f"Test loss {test_loss} is higher than " f"the best loss {best_loss}.")
    exit(1)

print("Done! Congrats on a new best model! 🚀🎉")

old_best_acc = best_acc
old_best_loss = best_loss
new_best_acc = test_acc
new_best_loss = test_loss

print("[7/7] - Committing changes...")


def run(cmd, check=True, capture=False):
    print(">", " ".join(cmd))

    if capture:
        return subprocess.check_output(cmd, text=True).strip()

    return subprocess.run(cmd, check=check)


try:
    run(["git", "stash", "push", "-u", "-m", "auto-temp-stash"])
    run(["git", "fetch", "origin"])
    run(["git", "checkout", "staging"])
    run(["git", "reset", "--hard", "origin/staging"])
    result = subprocess.run(["git", "stash", "pop"])
    if result.returncode != 0:
        print("[!] Merge conflicts detected...")
        conflicted = run(
            ["git", "diff", "--name-only", "--diff-filter=U"], capture=True
        ).splitlines()
        for file in conflicted:
            run(["git", "checkout", "--ours", file])
            run(["git", "add", file])
        print("[+] Conflicts resolved using local changes.")
    run(["git", "add", "."])
    with open("training_log.txt", "r") as f:
        training_log = f.read()
    commit_message = (
        f"Hearthstone AI Improvement: "
        f"{old_best_acc:.4f}->{new_best_acc:.4f} "
        f"(Loss: {old_best_loss:.4f}->{new_best_loss:.4f})\n"
        f"data=({new_best_acc:.4f},{new_best_loss:.4f})\n\n"
        f"{training_log}"
    )
    run(["git", "commit", "-m", commit_message])
    run(["git", "push", "origin", "HEAD:staging"])
    print("[+] Push complete.")
    print("[+] Change has been successfully rebased and submitted as (git ID)")

except subprocess.CalledProcessError as e:
    print(f"[ERROR] Command failed: {e}")
    sys.exit(1)
