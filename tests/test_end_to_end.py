"""Full flow on the synthetic batch: intake -> review -> post -> reconcile."""
import pytest

from giftflow.db import loads
from giftflow.llm import MockExtractor
from giftflow.pipeline import create_batch, process_batch
from giftflow.posting import export_gift_csv, post_batch, reconcile
from giftflow.review import ApprovalBlocked, accuracy_metrics, approve, create_donor, reject


def _doc(conn, name):
    return conn.execute("SELECT * FROM documents WHERE filename LIKE ?", (f"{name}%",)).fetchone()


def _flags(row):
    return {f["code"] for f in loads(row["flags"], [])}


def test_full_batch(conn, settings, samples):
    batch = create_batch(conn, "test", 6625.00)
    process_batch(conn, batch, sorted(samples.glob("*.png")), MockExtractor(), settings)

    assert "amount_mismatch" in _flags(_doc(conn, "05_"))
    assert "duplicate_scan" in _flags(_doc(conn, "06_"))
    assert "donor_new" in _flags(_doc(conn, "07_"))
    assert "donor_required" in _flags(_doc(conn, "08_"))
    assert "gik_no_value" in _flags(_doc(conn, "09_"))
    assert loads(_doc(conn, "02_")["proposal"])["suggested_donor_id"] == 2   # Bob -> Robert Martinez

    # bank numbers never stored
    for r in conn.execute("SELECT ocr_text FROM documents"):
        assert "063100277" not in r["ocr_text"] and "004512378901" not in r["ocr_text"]

    for name in ["01_", "02_", "03_", "04_", "09_", "10_"]:
        approve(conn, _doc(conn, name)["id"], {}, "tester", settings)

    with pytest.raises(ApprovalBlocked):
        approve(conn, _doc(conn, "05_")["id"], {}, "tester", settings)
    approve(conn, _doc(conn, "05_")["id"], {"amount": 150.0}, "tester", settings)   # written amount is the legal amount

    approve(conn, _doc(conn, "08_")["id"], {"suggested_donor_id": 10, "check_number": "88"}, "tester", settings)
    kid = create_donor(conn, "Kevin O'Brien", "45 Elm St, Plantation, FL 33317", "tester")
    approve(conn, _doc(conn, "07_")["id"], {"suggested_donor_id": kid}, "tester", settings)

    with pytest.raises(ApprovalBlocked):
        approve(conn, _doc(conn, "06_")["id"], {}, "tester", settings)
    reject(conn, _doc(conn, "06_")["id"], "tester", "duplicate scan")

    res = post_batch(conn, batch, "tester")
    assert len(res["posted"]) == 9 and not res["errors"]
    assert post_batch(conn, batch, "tester")["posted"] == []          # idempotent

    rec = reconcile(conn, batch)
    assert rec["status"] == "balanced" and rec["difference"] == 0

    m = accuracy_metrics(conn)
    assert m["reviewed"] == 9 and "amount" in m["corrections_by_field"]
    assert "ConstituentID" in export_gift_csv(conn, batch)
    actions = {r["action"] for r in conn.execute("SELECT action FROM audit_log")}
    assert {"document_processed", "document_approved", "donor_created", "gift_posted"} <= actions


def test_same_check_cannot_post_twice_across_batches(conn, settings, samples):
    b1 = create_batch(conn, "a", 250)
    process_batch(conn, b1, [samples / "01_check_robert_smith.png"], MockExtractor(), settings)
    approve(conn, _doc(conn, "01_")["id"], {}, "t", settings)
    post_batch(conn, b1, "t")

    b2 = create_batch(conn, "b", 250)
    conn.execute("UPDATE documents SET filename='old_' || filename")  # keep names unique for the lookup
    process_batch(conn, b2, [samples / "01_check_robert_smith.png"], MockExtractor(), settings)
    row = conn.execute("SELECT * FROM documents WHERE batch_id=?", (b2,)).fetchone()
    assert "duplicate_posted" in _flags(row)


def test_works_without_tesseract(conn, settings, samples, tmp_path, monkeypatch):
    """No Tesseract: samples fall back to their .txt sidecars; a bare image is flagged, not a crash."""
    import shutil
    import pytesseract
    monkeypatch.setattr(pytesseract.pytesseract, "tesseract_cmd", "/nonexistent/tesseract")
    bare = tmp_path / "bare.png"
    shutil.copy(samples / "10_gik_linda_park.png", bare)
    b = create_batch(conn, "no-ocr", 0)
    process_batch(conn, b, [samples / "01_check_robert_smith.png", bare], MockExtractor(), settings)
    rows = {r["filename"]: r for r in conn.execute("SELECT * FROM documents WHERE batch_id=?", (b,))}
    assert loads(rows["01_check_robert_smith.png"]["proposal"])["amount"] == 250.0
    assert "ocr_failed" in _flags(rows["bare.png"])


def test_creating_donor_selects_it_and_clears_the_flag(conn, settings, samples):
    from giftflow.review import create_donor_for_document

    batch = create_batch(conn, "test", 6625.00)
    process_batch(conn, batch, sorted(samples.glob("*.png")), MockExtractor(), settings)
    doc = _doc(conn, "07_")
    assert "donor_new" in _flags(doc)
    assert "coding_low_conf" not in _flags(doc)          # "General donation" is coded by rule
    assert loads(doc["proposal"])["suggested_fund"] == "GENERAL"

    kid = create_donor_for_document(conn, doc["id"], "Kevin O'Brien", "45 Elm St, Plantation, FL 33317",
                                    "tester", settings)
    doc = _doc(conn, "07_")
    assert loads(doc["proposal"])["suggested_donor_id"] == kid
    assert "donor_new" not in _flags(doc)
    approve(conn, doc["id"], {}, "tester", settings)      # nothing left blocking
