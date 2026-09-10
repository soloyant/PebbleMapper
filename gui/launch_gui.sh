#!/usr/bin/env bash
# =============================================================================
#  PebbleMapper - macOS / Linux Launcher
# =============================================================================
#  Run from a terminal in the repo root:    ./launch_gui.sh
#  On first use, you may need:              chmod +x launch_gui.sh
#
#  macOS users: to make this double-clickable from Finder, see the README's
#  Graphical Interface section for the Automator wrapper recipe.
# =============================================================================
set -e

# === BRANDING: product display name (single edit point for this script) ======
#  Keep in sync with functions/branding.py APP_NAME. The banners below read it.
APP_NAME="PebbleMapper"
# =============================================================================

# --- Locate the repo root. The script is normally placed at the repo root,
# --- but if it ends up inside gui/ we transparently walk up one level so
# --- the user doesn't have to figure out where it should live.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" = "gui" ]; then
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
else
    REPO_ROOT="$SCRIPT_DIR"
fi
cd "$REPO_ROOT"

# --- Sanity-check the package layout. Both gui/ and functions/ must be
# --- importable Python packages, i.e. they must contain __init__.py.
for pkg in gui functions; do
    if [ ! -f "$REPO_ROOT/$pkg/__init__.py" ]; then
        echo
        echo "  ERROR: Missing $REPO_ROOT/$pkg/__init__.py"
        echo "  Create an empty file at that path so '$pkg' is a Python package:"
        echo
        echo "      touch $REPO_ROOT/$pkg/__init__.py"
        echo
        exit 1
    fi
done

if [ ! -f "$REPO_ROOT/gui/app.py" ]; then
    echo
    echo "  ERROR: Missing $REPO_ROOT/gui/app.py"
    echo "  Has the repo been fully cloned?"
    echo
    exit 1
fi

# --- Locate conda's shell hook ---
CONDA_SH=""
for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/opt/miniconda3/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/usr/local/miniconda3/etc/profile.d/conda.sh" \
    "/usr/local/anaconda3/etc/profile.d/conda.sh" \
    "/opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh"
do
    if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
    fi
done

if [ -z "$CONDA_SH" ]; then
    echo
    echo "  ERROR: Could not find conda's shell hook in any standard location."
    echo "  Either edit this script and set CONDA_SH manually, or activate the env"
    echo "  yourself and run:"
    echo
    echo "      conda activate maskrcnn"
    echo "      cd \"$REPO_ROOT\""
    echo "      python -m gui.app"
    echo
    exit 1
fi

# --- Activate the env ---
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate maskrcnn || {
    echo
    echo "  ERROR: Failed to activate the 'maskrcnn' conda environment."
    echo "  Have you run 'conda env create -f environment.yml' yet?"
    echo
    exit 1
}

# --- Confirm nicegui is installed ---
if ! python -c "import nicegui" 2>/dev/null; then
    echo
    echo "  ERROR: nicegui is not installed in the maskrcnn env."
    echo "  Re-run 'conda env create -f environment.yml' to install it."
    echo
    exit 1
fi

# --- HEIC/HEIF photographs need the pillow-heif plugin; warn, do not stop ---
if ! python -c "import pillow_heif" 2>/dev/null; then
    echo
    echo "  NOTE: HEIC/HEIF photographs will not open: pillow-heif is not installed."
    echo "  To add it:   conda activate maskrcnn && pip install pillow-heif"
    echo
fi

# --- Make sure the repo root is on sys.path so 'from functions.X import Y'
# --- resolves. python -m gui.app implicitly does this when run from the
# --- repo root, but setting PYTHONPATH explicitly is belt-and-suspenders.
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# --- Launch ---
echo
echo "  Starting $APP_NAME GUI..."
echo "  The browser will open automatically. To stop the server, press Ctrl+C in this terminal."
echo

# -u: unbuffered stdout so the "Server starting at http://..." line shows up
# immediately rather than waiting for buffer flush.
# -m gui.app: run gui as a package so package-relative imports work.
exec python -u -m gui.app
