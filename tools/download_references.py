r"""Download every PDF listed in referencias/fuentes.csv into its sub-folder.

    python tools\download_references.py
    python tools\download_references.py --only bolivia

Safe to run again: files that are already valid PDFs are skipped. Anything that fails
is listed at the end (and in referencias/descargas_log.csv) so it can be opened by hand.
"""

from __future__ import annotations

import argparse
import csv
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
}
PDF_MAGIC = b"%PDF"
MIN_BYTES = 20_000          # anything smaller is almost surely an error page


def is_pdf(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return fh.read(4) == PDF_MAGIC and path.stat().st_size >= MIN_BYTES
    except OSError:
        return False


INSECURE = False            # set by --insecure: skip the certificate check (public PDFs only)


def fetch(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers=HEADERS)
    context = ssl._create_unverified_context() if INSECURE else None
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read()


def download(url: str, dest: Path, attempts: int = 3) -> str:
    """Return 'ok' or a short reason for the failure."""
    reason = "unknown"
    for attempt in range(1, attempts + 1):
        try:
            data = fetch(url)
        except urllib.error.HTTPError as exc:
            reason = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            reason = f"network: {exc}"
        else:
            if data[:4] != PDF_MAGIC:
                return "not a PDF (the site sent a web page; open the link by hand)"
            if len(data) < MIN_BYTES:
                return "file too small"
            dest.write_bytes(data)
            return "ok"
        time.sleep(2 * attempt)
    return reason


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=Path("referencias") / "fuentes.csv")
    ap.add_argument("--only", default=None, help="download just this sub-folder, e.g. bolivia")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between downloads (be polite)")
    ap.add_argument("--insecure", action="store_true",
                    help="skip the certificate check for sites Windows does not trust (public PDFs only)")
    a = ap.parse_args()
    global INSECURE
    INSECURE = a.insecure

    root = a.list.parent
    with a.list.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.DictReader(fh) if a.only in (None, r["folder"])]

    log: list[tuple[str, str, str]] = []
    for i, row in enumerate(rows, start=1):
        dest = root / row["folder"] / row["filename"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if is_pdf(dest):
            status = "already there"
        else:
            status = download(row["url"], dest)
            time.sleep(a.pause)
        print(f"[{i:>2}/{len(rows)}] {status:<14} {row['folder']}/{row['filename']}")
        log.append((f"{row['folder']}/{row['filename']}", status, row["url"]))

    with (root / "descargas_log.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(("file", "status", "url"))
        writer.writerows(log)

    failed = [r for r in log if r[1] not in ("ok", "already there")]
    print(f"\nDone: {len(log) - len(failed)} of {len(log)} PDFs are in {root}.")
    if failed:
        print("Open these by hand (copy the link into the browser, save the PDF in the folder shown):")
        for name, status, url in failed:
            print(f"  - {name}\n      {status}\n      {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
