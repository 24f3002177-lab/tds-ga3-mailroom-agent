import hashlib
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

DB_PATH = Path(file).with_name("mailroom.db")
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


# ---------- hashing helpers ----------

def canonical_json(value):
    """JSON with sorted keys and no spaces, the grader's exact format."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_digest(dossiers):
    return sha256_hex(canonical_json(dossiers))


def proposal_digest(p):
    view = {
        "dossierId": p["dossierId"],
        "callId": p["callId"],
        "action": p["action"],
        "target": p.get("target"),
        "payload": p.get("payload"),
        "evidence": sorted(p.get("evidence") or []),
    }
    return sha256_hex(canonical_json(view))


def dossier_fingerprint(dossier):
    return sha256_hex(canonical_json(dossier))


def call_id_for(fingerprint):
    # Same dossier content always gives the same callId.
    return "call:" + fingerprint[:40]


# ---------- locks ----------

def eval_lock(evaluation_id):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(evaluation_id, threading.Lock())


# ---------- database ----------

def connect():
    conn = sqlite3.connect(str(DB_PATH), timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_database():
    with closing(connect()) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                evaluation_id TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                input_digest TEXT NOT NULL,
                verifier_json TEXT NOT NULL,
                status TEXT NOT NULL,
                propose_json TEXT NOT NULL,
                commit_hash TEXT,
                commit_json TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS decisions (
                fingerprint TEXT PRIMARY KEY,
                decision_json TEXT NOT NULL
            )
        """)


def get_evaluation(evaluation_id):
    with closing(connect()) as c:
        row = c.execute(
            "SELECT * FROM evaluations WHERE evaluation_id = ?",
            (evaluation_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "evaluationId": row["evaluation_id"],
        "requestHash": row["request_hash"],
        "inputDigest": row["input_digest"],
        "verifier": json.loads(row["verifier_json"]),
        "status": row["status"],
        "propose": json.loads(row["propose_json"]),
        "commitHash": row["commit_hash"],
        "commit": json.loads(row["commit_json"]) if row["commit_json"] else None,
    }


def create_evaluation(evaluation_id, request_hash, digest, verifier, propose_response):
    """Save a new evaluation. Returns False if it already existed."""
    with closing(connect()) as c:
        cur = c.execute(
            """INSERT OR IGNORE INTO evaluations
               (evaluation_id, request_hash, input_digest, verifier_json,
                status, propose_json)
               VALUES (?, ?, ?, ?, 'awaiting_receipts', ?)""",
            (
                evaluation_id, request_hash, digest,
                canonical_json(verifier), canonical_json(propose_response),
            ),
        )
        return cur.rowcount == 1


def complete_evaluation(evaluation_id, commit_hash, commit_response):
    """Mark completed, but only if it was still awaiting receipts."""
    with closing(connect()) as c:
        cur = c.execute(
            """UPDATE evaluations
               SET status = 'completed', commit_hash = ?, commit_json = ?
               WHERE evaluation_id = ? AND status = 'awaiting_receipts'""",
            (commit_hash, canonical_json(commit_response), evaluation_id),
        )
        return cur.rowcount == 1


def get_decision(fingerprint):
    with closing(connect()) as c:
        row = c.execute(
            "SELECT decision_json FROM decisions WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
    return json.loads(row["decision_json"]) if row else None


def put_decision(fingerprint, decision):
    with closing(connect()) as c:
        c.execute(
            "INSERT OR IGNORE INTO decisions (fingerprint, decision_json) VALUES (?, ?)",
            (fingerprint, canonical_json(decision)),
        )


initialize_database()
