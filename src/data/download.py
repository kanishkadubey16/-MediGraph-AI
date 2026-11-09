"""
src/data/download.py
--------------------
PrimeKG dataset acquisition for MediGraph AI.

Responsibilities:
  - Explain how to download PrimeKG manually
  - Attempt programmatic download via Harvard Dataverse API (network required)
  - File existence checks and integrity verification
  - Alternative download via PyTDC if installed

Usage:
    python src/data/download.py [--output data/raw/kg.csv]

Author: Member 1 — Data Engineering
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root bootstrap (so this script works from any working directory)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — PrimeKG source
# ---------------------------------------------------------------------------

PRIMEKG_DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_DATAVERSE_PAGE = (
    "https://dataverse.harvard.edu/dataset.xhtml"
    "?persistentId=doi:10.7910/DVN/IXA7BM"
)

# Expected columns in a valid PrimeKG kg.csv (12 columns)
EXPECTED_COLUMNS = {
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
}

# Known approximate file size range in bytes (~450 MB – 600 MB)
MIN_FILE_SIZE_BYTES = 100 * 1024 * 1024   # 100 MB minimum sanity check


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def kg_csv_path(cfg: dict | None = None) -> Path:
    """Return the expected absolute path for kg.csv given the config."""
    if cfg is None:
        cfg = load_config()
    return Path(cfg["paths"]["data_raw"]) / "kg.csv"


def check_file_exists(cfg: dict | None = None) -> bool:
    """
    Return True if data/raw/kg.csv exists and passes basic sanity checks.

    Parameters
    ----------
    cfg : dict, optional
        Loaded config dict. Loaded automatically if not provided.

    Returns
    -------
    bool
    """
    path = kg_csv_path(cfg)
    if not path.exists():
        log.warning("[MediGraph] kg.csv not found at '%s'.", path)
        return False
    size = path.stat().st_size
    if size < MIN_FILE_SIZE_BYTES:
        log.warning(
            "[MediGraph] kg.csv exists but is suspiciously small (%d bytes). "
            "The file may be incomplete or corrupted.",
            size,
        )
        return False
    log.info("[MediGraph] kg.csv found at '%s' (%.1f MB).", path, size / 1e6)
    return True


def validate_columns(path: Path) -> bool:
    """
    Read the header of kg.csv and confirm expected columns are present.

    Parameters
    ----------
    path : Path
        Path to kg.csv.

    Returns
    -------
    bool
        True if all expected columns are present.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[MediGraph] kg.csv not found at '{path}'. "
            f"Run `python src/data/download.py` to get download instructions."
        )
    import pandas as pd
    header = pd.read_csv(path, nrows=0)
    actual_cols = set(header.columns)
    missing = EXPECTED_COLUMNS - actual_cols
    if missing:
        log.error(
            "[MediGraph] kg.csv is missing expected columns: %s. "
            "Actual columns found: %s",
            sorted(missing),
            sorted(actual_cols),
        )
        return False
    log.info("[MediGraph] kg.csv column validation passed. Columns: %s", sorted(actual_cols))
    return True


def print_manual_instructions(output_path: Path) -> None:
    """Print clear manual download instructions to stdout."""
    separator = "=" * 70
    print(f"\n{separator}")
    print("  PRIMEKG DOWNLOAD INSTRUCTIONS")
    print(separator)
    print()
    print("PrimeKG (Precision Medicine Knowledge Graph) must be downloaded")
    print("manually from Harvard Dataverse due to terms-of-use acceptance.")
    print()
    print("OPTION 1 — Direct file download (fastest, ~500 MB):")
    print()
    print(f"  wget -O {output_path} \\")
    print(f"    '{PRIMEKG_DATAVERSE_URL}'")
    print()
    print("  OR with curl:")
    print()
    print(f"  curl -L '{PRIMEKG_DATAVERSE_URL}' -o {output_path}")
    print()
    print("OPTION 2 — Browser download:")
    print()
    print(f"  1. Visit: {PRIMEKG_DATAVERSE_PAGE}")
    print("  2. Download the file named 'kg.csv'")
    print(f"  3. Place it at: {output_path}")
    print()
    print("OPTION 3 — PyTDC (Python Therapeutics Data Commons):")
    print()
    print("  pip install PyTDC")
    print("  python -c \"from tdc.resource import PrimeKG; PrimeKG(path='data/raw')\"")
    print("  # Note: TDC saves as 'kg.csv' in the specified path")
    print()
    print("CITATION (required if you publish results):")
    print()
    print("  Chandak P, Huang K, Zitnik M. (2023). Building a Knowledge Graph")
    print("  to Enable Precision Medicine. Scientific Data, 10, 67.")
    print("  https://doi.org/10.1038/s41597-023-01960-3")
    print()
    print("LICENSE: CC BY 4.0 — attribution required.")
    print(separator)
    print()


def attempt_programmatic_download(output_path: Path) -> bool:
    """
    Attempt to download kg.csv programmatically from Harvard Dataverse.

    This requires network access (not available in sandboxed environments).
    Returns True if download succeeded, False otherwise.

    Parameters
    ----------
    output_path : Path
        Where to save the downloaded file.

    Returns
    -------
    bool
    """
    try:
        import urllib.request
        import urllib.error
    except ImportError:
        log.error("[MediGraph] urllib not available. Use manual download.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(
        "[MediGraph] Attempting programmatic download from Harvard Dataverse..."
    )
    log.info("[MediGraph] URL: %s", PRIMEKG_DATAVERSE_URL)
    log.info("[MediGraph] Destination: %s", output_path)
    log.warning(
        "[MediGraph] This file is ~500 MB. Download may take several minutes."
    )

    try:
        def _progress_hook(block_count: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                pct = min(100.0, block_count * block_size * 100.0 / total_size)
                downloaded_mb = block_count * block_size / 1e6
                total_mb = total_size / 1e6
                print(
                    f"\r  Downloading: {downloaded_mb:.1f} / {total_mb:.1f} MB"
                    f"  ({pct:.1f}%)",
                    end="",
                    flush=True,
                )

        urllib.request.urlretrieve(
            PRIMEKG_DATAVERSE_URL,
            filename=str(output_path),
            reporthook=_progress_hook,
        )
        print()  # newline after progress
        size = output_path.stat().st_size
        log.info(
            "[MediGraph] Download complete. File saved at '%s' (%.1f MB).",
            output_path,
            size / 1e6,
        )
        return True

    except urllib.error.URLError as exc:
        log.error(
            "[MediGraph] Network download failed: %s. "
            "If you are in a network-restricted environment, use manual download.",
            exc,
        )
        # Clean up partial file
        if output_path.exists() and output_path.stat().st_size < MIN_FILE_SIZE_BYTES:
            output_path.unlink(missing_ok=True)
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("[MediGraph] Unexpected download error: %s", exc)
        if output_path.exists():
            output_path.unlink(missing_ok=True)
        return False


def attempt_tdc_download(output_path: Path) -> bool:
    """
    Attempt to download PrimeKG via PyTDC if installed.

    Parameters
    ----------
    output_path : Path
        Target directory (data/raw/). TDC saves kg.csv inside it.

    Returns
    -------
    bool
    """
    try:
        from tdc.resource import PrimeKG  # type: ignore[import]
        log.info("[MediGraph] PyTDC found. Downloading PrimeKG via TDC...")
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        PrimeKG(path=str(output_dir))
        tdc_path = output_dir / "kg.csv"
        if tdc_path.exists():
            log.info(
                "[MediGraph] TDC download complete at '%s' (%.1f MB).",
                tdc_path,
                tdc_path.stat().st_size / 1e6,
            )
            return True
        log.warning("[MediGraph] TDC download completed but kg.csv not found.")
        return False
    except ImportError:
        log.info("[MediGraph] PyTDC not installed. Skipping TDC download.")
        return False
    except Exception as exc:  # noqa: BLE001
        log.error("[MediGraph] TDC download failed: %s", exc)
        return False


def ensure_dataset(cfg: dict | None = None, attempt_download: bool = True) -> Path:
    """
    Ensure data/raw/kg.csv is present and valid.

    If the file is missing:
      1. Try PyTDC download (if installed)
      2. Try direct Harvard Dataverse download
      3. Print manual instructions and raise FileNotFoundError

    Parameters
    ----------
    cfg : dict, optional
    attempt_download : bool
        If False, only checks existence without attempting download.

    Returns
    -------
    Path
        Absolute path to kg.csv.

    Raises
    ------
    FileNotFoundError
        If the file cannot be obtained.
    """
    if cfg is None:
        cfg = load_config()

    path = kg_csv_path(cfg)

    if check_file_exists(cfg):
        validate_columns(path)
        return path

    log.warning("[MediGraph] kg.csv not found. Attempting to acquire dataset...")
    print_manual_instructions(path)

    if not attempt_download:
        raise FileNotFoundError(
            f"[MediGraph] PrimeKG kg.csv not found at '{path}'. "
            f"See download instructions above or run: python src/data/download.py"
        )

    # Try TDC first (more reliable, handles authentication)
    if attempt_tdc_download(path):
        validate_columns(path)
        return path

    # Try direct download
    if attempt_programmatic_download(path):
        validate_columns(path)
        return path

    raise FileNotFoundError(
        f"[MediGraph] Could not acquire PrimeKG kg.csv. "
        f"Please download manually and place at '{path}'. "
        f"See printed instructions above."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download or verify PrimeKG dataset for MediGraph AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/data/download.py --check
  python src/data/download.py --download
  python src/data/download.py --instructions
        """,
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for kg.csv (default: data/raw/kg.csv from config)",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        action="store_true",
        help="Check if kg.csv exists and validate its columns",
    )
    group.add_argument(
        "--download",
        action="store_true",
        help="Attempt programmatic download of PrimeKG",
    )
    group.add_argument(
        "--instructions",
        action="store_true",
        help="Print manual download instructions and exit",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = load_config()

    if args.output:
        # Override config path
        cfg["paths"]["data_raw"] = str(args.output.parent)

    path = kg_csv_path(cfg)

    if args.check:
        exists = check_file_exists(cfg)
        if exists:
            validate_columns(path)
            print(f"\n✓ Dataset ready at: {path}")
        else:
            print(f"\n✗ Dataset not found at: {path}")
            print_manual_instructions(path)
        sys.exit(0 if exists else 1)

    elif args.instructions:
        print_manual_instructions(path)
        sys.exit(0)

    else:
        # Default / --download: try to ensure the file exists
        try:
            final_path = ensure_dataset(cfg, attempt_download=True)
            print(f"\n✓ Dataset ready at: {final_path}")
        except FileNotFoundError as exc:
            print(f"\n✗ {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()

