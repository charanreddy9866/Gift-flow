"""Process a folder of scans from the command line and print the review queue.

Usage:  python scripts/run_batch.py samples 6625.00
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giftflow.config import load_settings  # noqa: E402
from giftflow.db import connect, init_db, loads  # noqa: E402
from giftflow.llm import get_extractor  # noqa: E402
from giftflow.pipeline import create_batch, process_batch  # noqa: E402


def main(folder: str, deposit_total: float) -> None:
    s = load_settings()
    conn = connect(s.db_path)
    init_db(conn)
    batch = create_batch(conn, f"CLI: {folder}", deposit_total, actor="cli")
    paths = sorted(Path(folder).glob("*.png")) + sorted(Path(folder).glob("*.jpg"))
    process_batch(conn, batch, paths, get_extractor(s.llm_provider, s.openai_model), s)
    print(f"Batch #{batch}: {len(paths)} documents  (AI mode: {s.llm_provider})\n")
    for r in conn.execute("SELECT filename, proposal, flags FROM documents WHERE batch_id=?", (batch,)):
        p, flags = loads(r["proposal"], {}), loads(r["flags"], [])
        amount = p.get("amount") if p.get("doc_type") == "check" else p.get("stated_value")
        mark = "BLOCK" if any(f["severity"] == "block" for f in flags) else ("WARN " if flags else "OK   ")
        print(f"{mark} {r['filename']:<34} {str(p.get('donor_name')):<26} "
              f"{'$%.2f' % amount if amount else '-':>10}  {p.get('suggested_fund')}/{p.get('suggested_campaign')}")
        for f in flags:
            print(f"        {f['severity']:<5} {f['code']}: {f['message']}")
    print("\nNext: review and approve in the app ->  streamlit run app.py")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "samples", float(sys.argv[2]) if len(sys.argv) > 2 else 6625.00)
