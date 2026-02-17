"""
Fallback import handling for the Rust core extension.
"""

try:
    from fraudboost_rust_core import RustBooster
    _HAS_RUST = True
    _RUST_ERROR = None
except ImportError as e:
    _HAS_RUST = False
    _RUST_ERROR = str(e)
    RustBooster = None

def has_rust_backend() -> bool:
    """Check if Rust backend is available."""
    return _HAS_RUST

def get_rust_import_error() -> str:
    """Get the import error message if Rust backend is not available."""
    return _RUST_ERROR or "No error recorded"

def require_rust_backend():
    """Raise an error if Rust backend is not available."""
    if not _HAS_RUST:
        raise ImportError(f"Rust backend not available: {_RUST_ERROR}")