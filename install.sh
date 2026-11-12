#!/bin/bash
set -e

# LagRBF Installation Script
# This script automates the installation of LagRBF and its dependencies.

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
PETSC_VERSION="3.25.3"
PYTHON_VENV=""
LAGRBF_DIR="$(pwd)"
SKIP_PETSC=false
SKIP_PYTHON=false
PYTHON_EXECUTABLE="python3.13"
BUILD_JOBS=4
EXTRAS=""

# Help function
show_help() {
    cat << 'EOF'
LagRBF Installation Script

Usage: ./install.sh [OPTIONS]

Options:
    -h, --help              Show this help message
    -p, --petsc-dir DIR     Path to PETSc installation directory
    -a, --petsc-arch ARCH   PETSc architecture
    -v, --python-venv VENV  Path to Python virtual environment
    -j, --jobs N            Number of parallel build jobs (default: 4)
    -e, --extras EXTRAS     Optional Python dependencies: analytical, dev, or all
    --skip-python           Skip Python package installation
    --python-exe EXE        Python executable to use (default: python3.13)

Examples:
    # Full automated installation with new venv
    ./install.sh --petsc-dir ~/Software/petsc --python-venv ~/.venv/lagrbf

    # Installation with optional dependencies (analytical tools)
    ./install.sh --petsc-dir path/to/petsc --python-venv lagrbfpyvenv --extras analytical

    # Installation with all optional dependencies (dev tools + analytical)
    ./install.sh --petsc-dir path/to/petsc --python-venv lagrbfpyvenv --extras all

    # Build within an existing Python virtual environment with PETSC_DIR and PETSC_ARCH already set
    ./install.sh

    # Custom build parallelism
    ./install.sh --jobs 8
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--petsc-dir)
            PETSC_DIR="$2"
            shift 2
            ;;
        -a|--petsc-arch)
            PETSC_ARCH="$2"
            shift 2
            ;;
        -v|--python-venv)
            PYTHON_VENV="$2"
            shift 2
            ;;
        -j|--jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        -e|--extras)
            EXTRAS="$2"
            shift 2
            ;;
        --skip-python)
            SKIP_PYTHON=true
            shift
            ;;
        --python-exe)
            PYTHON_EXECUTABLE="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Normalize paths to absolute paths
if [ -n "$PYTHON_VENV" ] && [ ! -d "$PYTHON_VENV" ]; then
    # If the path doesn't exist yet and is relative, convert to absolute
    if [[ "$PYTHON_VENV" != /* ]]; then
        PYTHON_VENV="$(cd "$(dirname "$PYTHON_VENV")" 2>/dev/null && pwd)/$(basename "$PYTHON_VENV")" || PYTHON_VENV="$(pwd)/$PYTHON_VENV"
    fi
fi

if [ -n "$PYTHON_VENV" ] && [ -d "$PYTHON_VENV" ]; then
    # If venv exists, convert to absolute path
    PYTHON_VENV="$(cd "$PYTHON_VENV" && pwd)"
fi

if [ -n "$PETSC_DIR" ] && [ -d "$PETSC_DIR" ]; then
    PETSC_DIR="$(cd "$PETSC_DIR" && pwd)"
fi

# ============================================================================
# Validation Functions
# ============================================================================

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Command '$1' not found. Please install it."
        return 1
    fi
    return 0
}

check_python() {
    if ! check_command "$PYTHON_EXECUTABLE"; then
        log_error "Python executable '$PYTHON_EXECUTABLE' not found"
        exit 1
    fi

    local py_version=$($PYTHON_EXECUTABLE --version 2>&1 | awk '{print $2}')
    log_info "Python version: $py_version"
    
    # Check minimum version (3.11+)
    if ! $PYTHON_EXECUTABLE -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
        log_error "Python 3.11+ required, found: $py_version"
        exit 1
    fi
}

check_petsc() {
    if [ "$SKIP_PETSC" = true ]; then
        log_info "Skipping PETSc validation (--skip-petsc)"
        return 0
    fi

    if [ -z "$PETSC_DIR" ]; then
        log_error "PETSC_DIR not specified. Use: ./install.sh -p /path/to/petsc"
        exit 1
    fi

    if [ ! -d "$PETSC_DIR" ]; then
        log_error "PETSC_DIR does not exist: $PETSC_DIR"
        exit 1
    fi

    if [ -z "$PETSC_ARCH" ]; then
        log_error "PETSC_ARCH environment variable not set"
        log_info "Set it from your PETSc directory:"
        log_info "  export PETSC_ARCH=arch-linux-openmpi-opt-32bits"
        exit 1
    fi

    local petsc_lib="$PETSC_DIR/$PETSC_ARCH/lib"
    if [ ! -f "$petsc_lib/libpetsc.so" ] && [ ! -f "$petsc_lib/libpetsc.a" ]; then
        log_error "PETSc library not found at: $petsc_lib"
        exit 1
    fi

    log_success "PETSc found: $PETSC_DIR/$PETSC_ARCH"
}

# ============================================================================
# Setup Functions
# ============================================================================

setup_python_venv() {
    if [ -z "$PYTHON_VENV" ]; then
        log_warning "No Python virtual environment specified. Using system Python."
        log_warning "Recommended: ./install.sh -v ~/.venv/lagrbf"
        return 0
    fi

    if [ ! -d "$PYTHON_VENV" ]; then
        log_info "Creating Python virtual environment at: $PYTHON_VENV"
        $PYTHON_EXECUTABLE -m venv "$PYTHON_VENV"
        log_success "Virtual environment created"
    else
        log_info "Using existing virtual environment: $PYTHON_VENV"
    fi

    log_info "Activating virtual environment..."
    # shellcheck disable=SC1090
    source "$PYTHON_VENV/bin/activate"

    log_info "Upgrading pip, setuptools, wheel..."
    pip install --upgrade pip setuptools wheel

    log_success "Python environment ready"
}

setup_petsc_env() {
    if [ "$SKIP_PETSC" = true ]; then
        return 0
    fi

    log_info "Setting PETSc environment variables..."
    export PETSC_DIR="$PETSC_DIR"
    export PETSC_ARCH="$PETSC_ARCH"
    export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH"

    log_success "PETSc environment configured"
}

build_lagrbf_c() {
    log_info "Building LagRBF C library..."
    cd "$LAGRBF_DIR"

    if [ ! -f "Makefile" ]; then
        log_error "Makefile not found in $LAGRBF_DIR"
        exit 1
    fi

    if make all -j "$BUILD_JOBS" 2>&1 | tee build.log; then
        log_success "C library built successfully"
    else
        log_error "Build failed. See build.log for details."
        exit 1
    fi
}

install_python_package() {
    if [ "$SKIP_PYTHON" = true ]; then
        log_warning "Skipping Python package installation (--skip-python)"
        return 0
    fi

    log_info "Installing petsc4py from PETSc build..."
    if [ -z "$PETSC_DIR" ]; then
        log_error "PETSC_DIR not set. Cannot install petsc4py."
        exit 1
    fi

    if [ -z "$PETSC_ARCH" ]; then
        log_error "PETSC_ARCH not set. Cannot install petsc4py."
        exit 1
    fi

    PETSC4PY_PATH="$PETSC_DIR/src/binding/petsc4py"
    if [ ! -d "$PETSC4PY_PATH" ]; then
        log_error "petsc4py not found at: $PETSC4PY_PATH"
        log_error "Make sure PETSC_DIR and PETSC_ARCH are set correctly."
        exit 1
    fi

    if pip install "$PETSC4PY_PATH" 2>&1 | tee pip_petsc4py.log; then
        log_success "petsc4py installed from PETSc build"
    else
        log_error "petsc4py installation failed. See pip_petsc4py.log for details."
        exit 1
    fi

    log_info "Installing LagRBF Python package..."
    cd "$LAGRBF_DIR/python_api"

    # Build pip install command with optional extras
    PIP_INSTALL_CMD="pip install -e ."
    if [ -n "$EXTRAS" ]; then
        PIP_INSTALL_CMD="pip install -e '.[${EXTRAS}]'"
        log_info "Installing with extras: $EXTRAS"
    fi

    if eval "$PIP_INSTALL_CMD" 2>&1 | tee pip_install.log; then
        log_success "Python package installed"
    else
        log_error "Python package installation failed. See pip_install.log for details."
        exit 1
    fi
}

verify_installation() {
    log_info "Verifying installation..."

    if $PYTHON_EXECUTABLE -c "import lagrbfpy; print('✓ lagrbfpy imported')" 2>&1; then
        log_success "lagrbfpy module imports successfully"
    else
        log_error "Failed to import lagrbfpy"
        exit 1
    fi

    if $PYTHON_EXECUTABLE -c "from lagrbfpy import LagRBFCtx, PointSwarm; print('✓ Core classes available')" 2>&1; then
        log_success "Core LagRBF classes available"
    else
        log_error "Core classes not available"
        exit 1
    fi

    # Test optional dependencies
    if $PYTHON_EXECUTABLE -c "from lagrbfpy import HarmonicFlow; print('✓ Sphericart available')" 2>&1; then
        log_success "Optional sphericart dependency available"
    else
        log_warning "Sphericart not available (optional). Install with: pip install sphericart"
    fi
}

configure_venv_environment() {
    if [ -z "$PYTHON_VENV" ]; then
        log_warning "Virtual environment path not specified; skipping automatic environment setup"
        log_info "To configure environment variables manually, add to your shell profile:"
        log_info "  export PETSC_DIR=$PETSC_DIR"
        log_info "  export PETSC_ARCH=$PETSC_ARCH"
        log_info "  export LD_LIBRARY_PATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
        return 0
    fi

    log_info "Configuring virtual environment with PETSc settings..."

    ACTIVATE_SCRIPT="$PYTHON_VENV/bin/activate"
    LAGRBF_ENV_FILE="$PYTHON_VENV/bin/lagrbf_env.sh"

    # Verify venv exists
    if [ ! -f "$ACTIVATE_SCRIPT" ]; then
        log_error "Activation script not found at: $ACTIVATE_SCRIPT"
        log_warning "Virtual environment may not be properly initialized"
        return 1
    fi

    # Create lagrbf_env.sh with PETSc environment variables
    cat > "$LAGRBF_ENV_FILE" << ENVEOF
# LagRBF PETSc Environment Configuration
# Auto-generated by install.sh - do not edit manually
export PETSC_DIR=$PETSC_DIR
export PETSC_ARCH=$PETSC_ARCH
export LD_LIBRARY_PATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH
ENVEOF

    if [ ! -f "$LAGRBF_ENV_FILE" ]; then
        log_error "Failed to create $LAGRBF_ENV_FILE"
        return 1
    fi

    chmod 644 "$LAGRBF_ENV_FILE"
    log_success "Created lagrbf_env.sh"

    # Check if activation hook already exists (using safer method)
    if [ -f "$ACTIVATE_SCRIPT" ] && grep -q "lagrbf_env.sh" "$ACTIVATE_SCRIPT"; then
        log_info "Activation hook already configured"
        return 0
    fi

    # Add source command to the end of the activate script
    cat >> "$ACTIVATE_SCRIPT" << 'HOOKEOF'

# Load LagRBF PETSc environment (auto-added by install.sh)
if [ -f "${BASH_SOURCE%/*}/lagrbf_env.sh" ]; then
    source "${BASH_SOURCE%/*}/lagrbf_env.sh" 2>/dev/null
elif [ -f "$(dirname "$0")/lagrbf_env.sh" ]; then
    source "$(dirname "$0")/lagrbf_env.sh" 2>/dev/null
fi
HOOKEOF

    if [ $? -eq 0 ]; then
        log_success "Added environment configuration to venv activation script"
        log_info "Environment variables will be automatically set when you activate the venv:"
        log_info "  source $PYTHON_VENV/bin/activate"
    else
        log_error "Failed to modify activation script"
        return 1
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         LagRBF Installation Script                         ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""

    log_info "Installation directory: $LAGRBF_DIR"
    echo ""

    # Pre-flight checks
    log_info "Performing pre-flight checks..."
    check_command "git"
    check_command "make"
    check_command "mpicc"
    check_python
    check_petsc
    log_success "Pre-flight checks passed"
    echo ""

    # Setup Python environment
    setup_python_venv
    echo ""

    # Setup PETSc environment
    setup_petsc_env
    echo ""

    # Build C library
    build_lagrbf_c
    echo ""

    # Install Python package
    install_python_package
    echo ""

    # Verify installation
    verify_installation
    echo ""

    # Configure venv environment
    configure_venv_environment
    echo ""

    # Summary
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║             Installation Completed Successfully!           ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    log_info "Next steps:"
    echo "  1. Activate the virtual environment:"
    if [ -n "$PYTHON_VENV" ]; then
        echo "     source $PYTHON_VENV/bin/activate"
        echo "     (PETSc environment variables will be loaded automatically)"
    else
        echo "     (activate your existing virtual environment)"
        echo "     Then manually set:"
        echo "       export PETSC_DIR=$PETSC_DIR"
        echo "       export PETSC_ARCH=$PETSC_ARCH"
        echo "       export LD_LIBRARY_PATH=\$PETSC_DIR/\$PETSC_ARCH/lib:\$LD_LIBRARY_PATH"
    fi
    echo ""
    echo "  2. Run tests:"
    echo "     python python_api/scripts/test_interpolation.py"
    echo ""
    echo "  3. Try examples:"
    echo "     python python_api/scripts/test_advection.py"
    echo ""
    log_success "Happy computing!"
}

# Run main function
main
