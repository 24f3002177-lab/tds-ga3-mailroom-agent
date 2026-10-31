import asyncio
import base64
import json
import traceback

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import decide
import storage

# Smaller chunks, more workers: keeps each request inside the time limit.
decide.CHUNK = 6
decide.WORKERS = 2

PROFILE = "ga5-mailroom-action-gate/v2"
MAX_BODY = 6 * 1024 * 1024

app = FastAPI(title="Mailroom Agent", redirect_slashes=False)


class Reject(Exception):
    """args=(http_status,error_code)"""


def error(status, code):
    return JSONResponse(status_code=status, content={"error": {"code": code}})


def b64_decode(text):
    text = text.strip()
    text += "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text.encode("ascii"))


def load_key(jwk):
    raw = b64_decode(jwk["x"])
    if len(raw) != 32:
        raise ValueError("bad key length")
    return Ed25519PublicKey.from_public_bytes(raw)


def verify_receipt(key, evaluation_id, digest, receipt):
    try:
        inner = {k: v for k, v in receipt.items() if k != "receiptSignature"}
        message = storage.canonical_json({
            "profile": PROFILE,
            "evaluationId": evaluation_id,
            "inputDigest": digest,
            "receipt": inner,
        }).encode("utf-8")
        key.verify(b64_decode(receipt["receiptSignature"]), message)
        return True
    except (InvalidSignature, ValueError, Exception):
        return False


# ---------------- propose ----------------

def check_propose(body):
    v = body.get("receiptVerifier")
    if not isinstance(v, dict) or v.get("algorithm") != "Ed25519":
        raise Reject(422, "INVALID_VERIFIER")
    jwk = v.get("publicKeyJwk")
    if (not isinstance(jwk, dict) or jwk.get("kty") != "OKP"
            or jwk.get("crv") != "Ed25519" or not isinstance(jwk.get("x"), str)):
        raise Reject(422, "INVALID_VERIFIER")
    try:
        load_key(jwk)
    except Exception:
        raise Reject(422, "INVALID_VERIFIER_KEY")

    allowed = body.get("allowedActions")
    if allowed is not None and (
        not isinstance(allowed, list) or any(not isinstance(a, str) for a in allowed)
    ):
        raise Reject(422, "INVALID_ALLOWED_ACTIONS")

    dossiers = body.get("dossiers")
    if not isinstance(dossiers, list) or not dossiers:
        raise Reject(422, "INVALID_DOSSIERS")
    seen = set()
    for d in dossiers:
        if not isinstance(d, dict) or not isinstance(d.get("dossierId"), str) or not d["dossierId"]:
            raise Reject(422, "INVALID_DOSSIERS")
        if d["dossierId"] in seen:
            raise Reject(422, "DUPLICATE_DOSSIER_ID")
        seen.add(d["dossierId"])
        sources = d.get("sources")
        if not isinstance(sources, list):
            raise Reject(422, "INVALID_DOSSIERS")
        for s in sources:
            if not isinstance(s, dict) or not isinstance(s.get("lines"), list):
                raise Reject(422, "INVALID_DOSSIERS")
            for line in s["lines"]:
                if (not isinstance(line, dict) or not isinstance(line.get("lineId"), str)
                        or not isinstance(line.get("text"), str)):
                    raise Reject(422, "INVALID_DOSSIERS")


def handle_propose(body):
    check_propose(body)
    eid = body["evaluationId"]
    req_hash = storage.sha256_hex(storage.canonical_json(body))
    dossiers = body["dossiers"]
    allowed = set(body.get("allowedActions") or decide.ACTIONS)

    with storage.eval_lock(eid):
        existing = storage.get_evaluation(eid)
        if existing:
            if existing["requestHash"] != req_hash:
                raise Reject(409, "EVALUATION_CONFLICT")
            return existing["propose"]
        decisions = decide.decide_dossiers(dossiers)
        proposals = []
        for d in dossiers:
            dec = decisions[d["dossierId"]]
            if dec["action"] not in allowed:
                dec = decide.fallback(d)
            proposals.append({
                "dossierId": d["dossierId"],
                "callId": storage.call_id_for(storage.dossier_fingerprint(d)),
                "action": dec["action"],
                "target": dec["target"],
                "payload": dec["payload"],
                "evidence": dec["evidence"],
            })

        digest = storage.input_digest(dossiers)
        response = {
            "profile": PROFILE,
            "evaluationId": eid,
            "status": "awaiting_receipts",
            "inputDigest": digest,
            "proposals": proposals,
        }
        storage.create_evaluation(
            eid, req_hash, digest, body["receiptVerifier"], response
        )
        return response


# ---------------- commit ----------------

def handle_commit(body):
    eid = body["evaluationId"]
    digest = body.get("inputDigest")
    receipts = body.get("receipts")
    if not isinstance(digest, str) or not isinstance(receipts, list) or not receipts:
        raise Reject(422, "INVALID_COMMIT")
    for r in receipts:
        if not isinstance(r, dict):
            raise Reject(422, "INVALID_COMMIT")
        for k in ("dossierId", "callId", "action", "proposalDigest",
                  "receiptId", "receiptSignature"):
            if not isinstance(r.get(k), str) or not r[k]:
                raise Reject(422, "INVALID_COMMIT")
        if not isinstance(r.get("accepted"), bool):
            raise Reject(422, "INVALID_COMMIT")

    req_hash = storage.sha256_hex(storage.canonical_json(body))

    with storage.eval_lock(eid):
        ev = storage.get_evaluation(eid)
        if ev is None:
            raise Reject(409, "UNKNOWN_EVALUATION")

        if ev["status"] == "completed":
            if ev["commitHash"] == req_hash:
                return ev["commit"]["response"]
            raise Reject(409, "COMMIT_CONFLICT")

        if digest != ev["inputDigest"]:
            raise Reject(409, "INPUT_DIGEST_MISMATCH")

        proposals = {p["dossierId"]: p for p in ev["propose"]["proposals"]}
        if len(receipts) != len(proposals):
            raise Reject(400, "RECEIPT_SET_MISMATCH")

        seen_d, seen_r, seen_s = set(), set(), set()
        for r in receipts:
            p = proposals.get(r["dossierId"])
            if p is None or r["dossierId"] in seen_d:
                raise Reject(400, "RECEIPT_SET_MISMATCH")
            if r["receiptId"] in seen_r or r["receiptSignature"] in seen_s:
                raise Reject(400, "DUPLICATE_RECEIPT")
            seen_d.add(r["dossierId"])
            seen_r.add(r["receiptId"])
            seen_s.add(r["receiptSignature"])
            if (r["callId"] != p["callId"] or r["action"] != p["action"]
                    or r["proposalDigest"] != storage.proposal_digest(p)):
                raise Reject(409, "RECEIPT_PROPOSAL_MISMATCH")

        key = load_key(ev["verifier"]["publicKeyJwk"])
        for r in receipts:
            if not verify_receipt(key, eid, digest, r):
                raise Reject(400, "INVALID_RECEIPT_SIGNATURE")

        by_dossier = {r["dossierId"]: r for r in receipts}
        outcomes = []
        for p in ev["propose"]["proposals"]:
            r = by_dossier[p["dossierId"]]
            outcomes.append({
                "dossierId": p["dossierId"],
                "callId": p["callId"],
                "action": p["action"],
                "proposalDigest": r["proposalDigest"],
                "receiptId": r["receiptId"],
                "status": "executed" if r["accepted"] else "rejected",
            })
        response = {
            "profile": PROFILE,
            "evaluationId": eid,
            "status": "completed",
            "inputDigest": digest,
            "outcomes": outcomes,
        }
        saved = storage.complete_evaluation(
            eid, req_hash, {"response": response, "receipts": receipts}
        )
        if not saved:
            again = storage.get_evaluation(eid)
            if again and again["commitHash"] == req_hash:
                return again["commit"]["response"]
            raise Reject(409, "COMMIT_CONFLICT")
        return response


# ---------------- routes ----------------

@app.get("/")
def home():
    return {"message": "Mailroom Agent", "status": "running"}


@app.post("/{full_path:path}")
async def mailroom(request: Request):
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return error(413, "BODY_TOO_LARGE")
    try:
        body = json.loads(raw)
    except Exception:
        return error(400, "INVALID_JSON")
    if not isinstance(body, dict):
        return error(400, "INVALID_REQUEST")
    if body.get("profile") != PROFILE:
        return error(400, "INVALID_PROFILE")
    eid = body.get("evaluationId")
    if not isinstance(eid, str) or not eid.strip() or len(eid) > 256:
        return error(400, "INVALID_EVALUATION_ID")

    operation = body.get("operation")
    try:
        if operation == "propose":
            result = await asyncio.to_thread(handle_propose, body)
        elif operation == "commit":
            result = await asyncio.to_thread(handle_commit, body)
        else:
            return error(400, "INVALID_OPERATION")
    except Reject as r:
        try:
            status, code = int(r.args[0]), str(r.args[1])
        except Exception:
            status, code = 400, "REJECTED"
        print("REJECT",operation, status, code, flush=True)
        return JSONResponse(
            status_code=status, content={"error": {"code": code}}
        )
    except Exception:
        traceback.print_exc()
        return JSONResponse(
            status_code=500, content={"error": {"code": "INTERNAL_ERROR"}}
        )
    return JSONResponse(status_code=200, content=result)
