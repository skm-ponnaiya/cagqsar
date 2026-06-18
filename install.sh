#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define installation directories
INSTALL_DIR="/opt/cagqsar"
BIN_LINK="/usr/local/bin/cagqsar"

# Check if script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Error: Please run this installation script as root (e.g., sudo ./install.sh)"
  exit 1
fi

echo "============================================="
echo " Installing CAG-QSAR Globally on Linux/WSL"
echo "============================================="

# 1. Install system prerequisites if missing
echo "Checking system prerequisites..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Installing..."
    apt-get update && apt-get install -y python3 python3-pip python3-venv
else
    echo "Python3 is already installed."
fi

# Ensure python3-venv package is present
if ! python3 -c "import venv" &> /dev/null; then
    echo "python3-venv package is missing. Installing..."
    apt-get update && apt-get install -y python3-venv
fi

# 2. Setup clean system-level directory
echo "Creating installation directory at ${INSTALL_DIR}..."
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"

# 3. Copy source files
echo "Copying source files to ${INSTALL_DIR}..."
# Get current directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cp -r "${DIR}"/* "${INSTALL_DIR}/"

# 4. Create virtual environment inside /opt
echo "Creating isolated virtual environment..."
python3 -m venv "${INSTALL_DIR}/venv"

# 5. Install requirements and package inside virtualenv
echo "Installing Python dependencies (RDKit, PyTorch CPU, XGBoost, scikit-learn)..."
"${INSTALL_DIR}/venv/bin/pip" install --upgrade pip
"${INSTALL_DIR}/venv/bin/pip" install torch --extra-index-url https://download.pytorch.org/whl/cpu
"${INSTALL_DIR}/venv/bin/pip" install "${INSTALL_DIR}"

# 6. Create symbolic link in /usr/local/bin
echo "Creating symbolic link to ${BIN_LINK}..."
rm -f "${BIN_LINK}"
ln -s "${INSTALL_DIR}/venv/bin/cagqsar" "${BIN_LINK}"

echo "============================================="
echo " CAG-QSAR successfully installed system-wide!"
echo " You can now run the tool using command:"
echo "     cagqsar --help"
echo "============================================="
