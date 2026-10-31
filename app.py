"""GiftFlow review app.

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from giftflow.config import load_settings
from giftflow.db import allowed_codes, connect, init_db, loads
from giftflow.llm import get_extractor
from giftflow.pipeline import create_batch, process_batch
from giftflow.posting import export_gift_csv, post_batch, reconcile
from giftflow.review import ApprovalBlocked, accuracy_metrics, approve, create_donor_for_document, reject

ROOT = Path(__file__).parent
SAMPLES = ROOT / "samples"
UPLOADS = ROOT / "uploads"
SAMPLE_DEPOSIT = 6625.00

st.set_page_config(page_title="GiftFlow", page_icon="🧾", layout="wide")
settings = load_settings()


@st.cache_resource
def get_conn():
    conn = connect(settings.db_path)
    init_db(conn)
    return conn


conn = get_conn()


def reset_demo_data() -> None:
    """Clear batches, documents, gifts and the audit log; keep the 10 seed donors."""
    for t in ["gifts", "documents", "batches", "audit_log"]:
        conn.execute(f"DELETE FROM {t}")
    conn.execute("DELETE FROM donors WHERE id > 10")
    conn.commit()


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("GiftFlow")
    st.caption("AI-assisted intake for donation checks and gift-in-kind forms")
    mode = "Offline demo (mock AI)" if settings.llm_provider != "openai" else f"OpenAI · {settings.openai_model}"
    st.markdown(f"**AI mode:** {mode}")
    reviewer = st.text_input("Reviewer name", value="reviewer")
    batches = conn.execute("SELECT id, name FROM batches ORDER BY id DESC").fetchall()
    batch_labels = {f"#{b['id']} · {b['name']}": b["id"] for b in batches}
    batch_id = batch_labels[st.selectbox("Batch", list(batch_labels))] if batch_labels else None
    st.divider()
    st.markdown(
        "**Flow**\n\n1. Scan\n2. OCR (bank strip cropped, numbers redacted)\n3. AI extracts + looks up donor\n"
        "4. Rules code fund/campaign\n5. Code validates\n6. **Person approves**\n7. Post (idempotent)\n"
        "8. Reconcile with deposit"
    )
    if st.button("Reset demo data", type="secondary"):
        reset_demo_data()
        st.rerun()

tab_intake, tab_review, tab_post, tab_accuracy, tab_audit = st.tabs(
    ["1 · Intake", "2 · Review", "3 · Post & reconcile", "4 · Accuracy", "5 · Audit log"])

# ------------------------------------------------------------------ intake
with tab_intake:
    st.subheader("Start a batch")
    st.write("Scan a day's checks and in-kind forms as one batch. The deposit total comes from the bank deposit slip.")
    use_samples = st.toggle("Use the sample batch (10 synthetic documents)", value=True)
    c1, c2 = st.columns(2)
    name = c1.text_input("Batch name", value="Sample deposit" if use_samples else "")
    deposit = c2.number_input("Deposit slip total ($)", min_value=0.0, step=0.01,
                              value=SAMPLE_DEPOSIT if use_samples else 0.0)
    if use_samples:
        st.caption("Running the sample batch starts the demo fresh, so earlier runs don't show up as duplicate scans.")
    uploads = None if use_samples else st.file_uploader(
        "Scanned images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if st.button("Run intake", type="primary", disabled=not name):
        if use_samples:
            reset_demo_data()
            if not any(SAMPLES.glob("*.png")):
                from scripts.generate_samples import main as gen
                gen(str(SAMPLES))
            paths = sorted(SAMPLES.glob("*.png"))
        else:
            paths = []
        new_id = create_batch(conn, name, deposit, actor=reviewer)
        if uploads:
            folder = UPLOADS / str(new_id)
            folder.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                p = folder / f.name
                p.write_bytes(f.getbuffer())
                paths.append(p)
        if not paths:
            st.error("Add at least one image.")
        else:
            extractor = get_extractor(settings.llm_provider, settings.openai_model)
            with st.spinner(f"Processing {len(paths)} documents…"):
                process_batch(conn, new_id, paths, extractor, settings)
            st.success(f"Batch #{new_id} processed. Select it in the sidebar and open Review.")
            st.rerun()

    if batch_id:
        rows = conn.execute("SELECT * FROM documents WHERE batch_id=? ORDER BY id", (batch_id,)).fetchall()
        table = []
        for r in rows:
            p = loads(r["proposal"], {})
            fl = loads(r["flags"], [])
            table.append({
                "file": r["filename"], "type": r["doc_type"], "donor": p.get("donor_name"),
                "amount": p.get("amount") if r["doc_type"] == "check" else p.get("stated_value"),
                "OCR conf": r["ocr_confidence"],
                "blocking": sum(f["severity"] == "block" for f in fl),
                "warnings": sum(f["severity"] == "warn" for f in fl),
                "status": r["status"],
            })
        if table:
            st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)

# ------------------------------------------------------------------ review
with tab_review:
    if not batch_id:
        st.info("Run an intake first.")
    else:
        funds, campaigns = allowed_codes(conn)
        donors = conn.execute("SELECT id, name, address FROM donors ORDER BY name").fetchall()
        donor_label = {d["id"]: f"{d['name']} — {d['address'] or 'no address'}" for d in donors}
        rows = conn.execute("SELECT * FROM documents WHERE batch_id=? AND status IN ('needs_review','approved') "
                            "ORDER BY id", (batch_id,)).fetchall()

        def priority(r):
            fl = loads(r["flags"], [])
            return (r["status"] != "needs_review",
                    -sum(f["severity"] == "block" for f in fl), -len(fl), r["id"])

        rows = sorted(rows, key=priority)
        pending = sum(r["status"] == "needs_review" for r in rows)
        st.subheader(f"Review queue · {pending} waiting")
        st.caption("Flagged items first. Nothing is posted until a person approves it.")

        for r in rows:
            p = loads(r["final"]) or loads(r["proposal"], {})
            fl = loads(r["flags"], [])
            blocks = sum(f["severity"] == "block" for f in fl)
            icon = "✅" if r["status"] == "approved" else ("⛔" if blocks else ("⚠️" if fl else "🟢"))
            amount = p.get("amount") if r["doc_type"] == "check" else p.get("stated_value")
            title = f"{icon} {r['filename']} · {p.get('donor_name') or '?'} · " + (
                f"${amount:,.2f}" if amount else "no amount")
            with st.expander(title, expanded=(r["status"] == "needs_review" and bool(fl))):
                left, right = st.columns([1.1, 1])
                with left:
                    if r["image_path"] and Path(r["image_path"]).exists():
                        st.image(r["image_path"], width="stretch")
                    with st.popover("OCR text (bank numbers removed)"):
                        st.code(r["ocr_text"] or "", language=None)
                        st.caption(f"Mean OCR confidence: {r['ocr_confidence']}")
                    with st.popover("AI function calls"):
                        st.json(loads(r["tool_calls"], []))
                    st.caption("AI reasoning: " + (loads(r["extraction"], {}).get("reasoning") or "—"))
                with right:
                    for f in fl:
                        (st.error if f["severity"] == "block" else st.warning)(f"**{f['code']}** — {f['message']}")
                    if r["status"] == "approved":
                        st.success(f"Approved by {r['reviewer']}. Corrections: {', '.join(loads(r['corrections'], [])) or 'none'}")
                        continue

                    cands = loads(r["candidates"], [])
                    order = [c["donor_id"] for c in cands] + [d["id"] for d in donors if d["id"] not in {c["donor_id"] for c in cands}]
                    scores = {c["donor_id"]: c["score"] for c in cands}
                    options = [None] + order
                    current = p.get("suggested_donor_id")
                    with st.form(f"form_{r['id']}"):
                        donor_id = st.selectbox(
                            "Donor", options, index=options.index(current) if current in options else 0,
                            format_func=lambda i, sc=scores: "— pick a donor —" if i is None else (
                                donor_label[i] + (f"  ·  match {sc[i]:.0f}" if i in sc else "")),
                            # key includes the suggestion, so the box resets when a donor is created
                            key=f"donor_{r['id']}_{current}")
                        c1, c2 = st.columns(2)
                        if r["doc_type"] == "check":
                            amt = c1.number_input("Amount ($)", value=p.get("amount"), min_value=0.0, step=0.01,
                                                  key=f"amt_{r['id']}")
                            chk = c2.text_input("Check number", value=p.get("check_number") or "", key=f"chk_{r['id']}")
                            words = st.text_input("Written amount", value=p.get("amount_words") or "", key=f"w_{r['id']}")
                            memo = st.text_input("Memo", value=p.get("memo") or "", key=f"memo_{r['id']}")
                        else:
                            item = st.text_input("Item description", value=p.get("item_description") or "", key=f"it_{r['id']}")
                            qty = c1.number_input("Quantity", value=p.get("quantity"), min_value=0, step=1, key=f"q_{r['id']}")
                            val = c2.number_input("Value stated by donor ($)", value=p.get("stated_value"),
                                                  min_value=0.0, step=0.01, key=f"v_{r['id']}")
                        date_txt = st.text_input("Date (YYYY-MM-DD)", value=p.get("date") or "", key=f"d_{r['id']}")
                        c3, c4 = st.columns(2)
                        fund = c3.selectbox("Fund", funds, index=funds.index(p["suggested_fund"]) if p.get("suggested_fund") in funds else 0, key=f"f_{r['id']}")
                        camp = c4.selectbox("Campaign", campaigns, index=campaigns.index(p["suggested_campaign"]) if p.get("suggested_campaign") in campaigns else 0, key=f"c_{r['id']}")
                        b1, b2 = st.columns(2)
                        do_approve = b1.form_submit_button("Approve", type="primary")
                        do_reject = b2.form_submit_button("Reject (rescan / not a gift)")

                    if do_approve:
                        final = {"suggested_donor_id": donor_id, "date": date_txt or None,
                                 "suggested_fund": fund, "suggested_campaign": camp}
                        if donor_id:
                            final["donor_name"] = p.get("donor_name")
                        if r["doc_type"] == "check":
                            final.update(amount=amt, check_number=chk or None, amount_words=words or None, memo=memo or None)
                        else:
                            final.update(item_description=item or None, quantity=int(qty) if qty else None, stated_value=val)
                        try:
                            approve(conn, r["id"], final, reviewer, settings)
                            st.rerun()
                        except ApprovalBlocked as e:
                            for f in e.flags:
                                if f.severity == "block":
                                    st.error(f"Can't approve yet: {f.message}")
                    if do_reject:
                        reject(conn, r["id"], reviewer, "rejected in review")
                        st.rerun()

                    if any(f["code"] == "donor_new" for f in fl):
                        with st.form(f"newdonor_{r['id']}"):
                            st.markdown("**Create donor** (only after confirming this is a new donor)")
                            nn = st.text_input("Name", value=p.get("donor_name") or "", key=f"nn_{r['id']}")
                            na = st.text_input("Address", value=p.get("address") or "", key=f"na_{r['id']}")
                            if st.form_submit_button("Create donor"):
                                if not nn.strip():
                                    st.error("Enter the donor's name.")
                                else:
                                    create_donor_for_document(conn, r["id"], nn, na or None, reviewer, settings)
                                    st.rerun()

# ------------------------------------------------------------------ post & reconcile
with tab_post:
    if not batch_id:
        st.info("Run an intake first.")
    else:
        rec = reconcile(conn, batch_id)
        st.subheader("Reconciliation against the deposit slip")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Deposit slip", f"${rec['deposit_total']:,.2f}")
        m2.metric("Posted checks", f"${rec['posted_checks']:,.2f}")
        m3.metric("Approved, not posted", f"${rec['approved_not_posted']:,.2f}")
        m4.metric("Difference", f"${rec['difference']:,.2f}", rec["status"], delta_color="off")
        if rec["status"] == "balanced":
            st.success("Balanced: approved and posted checks match the bank deposit.")
        else:
            st.warning(f"Not balanced yet ({rec['status']} by ${abs(rec['difference']):,.2f}). Finish reviewing, or look for a misread or duplicate.")

        approved = conn.execute("SELECT COUNT(*) FROM documents WHERE batch_id=? AND status='approved'", (batch_id,)).fetchone()[0]
        if st.button(f"Post {approved} approved gift(s)", type="primary", disabled=approved == 0):
            res = post_batch(conn, batch_id, reviewer)
            st.write(res)
            st.rerun()

        gifts = conn.execute(
            """SELECT g.id, dn.name AS donor, g.gift_type, g.amount, g.gift_date, g.check_number, g.fund, g.campaign,
                      g.description, g.posted_at FROM gifts g JOIN documents d ON d.id=g.document_id
                      JOIN donors dn ON dn.id=g.donor_id WHERE d.batch_id=? ORDER BY g.id""", (batch_id,)).fetchall()
        if gifts:
            st.dataframe(pd.DataFrame([dict(g) for g in gifts]), width="stretch", hide_index=True)
            st.download_button("Download gift import CSV", export_gift_csv(conn, batch_id),
                               file_name=f"gift_import_batch_{batch_id}.csv", mime="text/csv")

# ------------------------------------------------------------------ accuracy
with tab_accuracy:
    m = accuracy_metrics(conn)
    st.subheader("How often the AI's proposal is accepted without changes")
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviewed", m["reviewed"])
    c2.metric("Accepted unchanged", m["accepted_unchanged"])
    c3.metric("Acceptance rate", f"{m['acceptance_rate']:.0%}" if m["acceptance_rate"] is not None else "—")
    if m["corrections_by_field"]:
        st.caption("Fields reviewers corrected most. These show where to improve prompts, rules or OCR.")
        st.bar_chart(pd.Series(m["corrections_by_field"], name="corrections"))

# ------------------------------------------------------------------ audit
with tab_audit:
    log = conn.execute("SELECT ts, actor, action, entity, entity_id, details FROM audit_log ORDER BY id DESC LIMIT 500").fetchall()
    st.subheader("Audit log")
    st.caption("Who did what, and when: every AI step, approval, correction and posting.")
    if log:
        st.dataframe(pd.DataFrame([dict(x) for x in log]), width="stretch", hide_index=True)
