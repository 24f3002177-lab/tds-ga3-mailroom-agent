def get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1"),
            timeout=42,
            max_retries=0,
        )
    return _client


def dossier_text(d):
    parts = [str(d.get("mailbox", "")), str(d.get("objective", ""))]
    for s in d.get("sources", []):
        parts.append(str(s.get("sourceId", "")))
        parts.append(str(s.get("title", "")))
        for line in s.get("lines", []):
            parts.append(str(line.get("text", "")))
    return "\n".join(parts)


def line_ids(d):
    return {
        line["lineId"]
        for s in d.get("sources", [])
        for line in s.get("lines", [])
    }


def build_body(action, a, mailbox):
    if action == "create_draft":
        target = {"kind": "draft_queue", "id": "mailbox:" + mailbox}
        payload = {
            "recipient": a["recipient"], "referenceId": a["referenceId"],
            "status": a["status"], "template": "order_status",
        }
    elif action == "update_internal_record":
        target = {"kind": "case_record", "id": a["caseId"]}
        payload = {
            "field": "delivery_window", "sourceEventId": a["sourceEventId"],
            "value": a["value"],
        }
    elif action == "send_approved_notice":
        target = {"kind": "email", "id": a["recipient"]}
        payload = {
            "referenceId": a["referenceId"], "status": a["status"],
            "template": "approved_delivery_notice",
        }
    elif action == "request_confirmation":
        target = {"kind": "approval_queue", "id": a["team"]}
        payload = {
            "claimedSender": a["claimedSender"], "questionCode": "VERIFY_REQUEST",
            "referenceId": a["referenceId"],
        }
    elif action == "quarantine_item":
        target = {"kind": "security_queue", "id": "mailroom"}
        payload = {"artifactId": a["artifactId"], "reasonCode": "INDIRECT_PROMPT_INJECTION"}
    else:
        target = None
        payload = {"reasonCode": a["reasonCode"], "referenceId": a["referenceId"]}
    return target, payload


def clean_decision(dec, dossier):
    """Check one AI answer. Return a safe decision, or None if anything is off."""
    try:
        if not isinstance(dec, dict):
            return None
        action = dec.get("action")
        if action not in ARG_SPEC:
            return None
        args = dec.get("args") or {}
        required, grounded = ARG_SPEC[action]
        a = {}
        for k in required:
            v = args.get(k)
            if not isinstance(v, str) or not v.strip() or len(v) > 200:
                return None
            a[k] = v.strip()
        if any(BAD_RE.search(v) for v in a.values()):
            return None
        text = dossier_text(dossier).lower()
        for k in grounded:
            if a[k].lower() not in text:
                return None
        if action == "no_action" and a["reasonCode"] not in REASONS:
            return None
        evidence = dec.get("evidence")
        ids = line_ids(dossier)
        if not isinstance(evidence, list) or not evidence:
            return None
        if len(set(evidence)) != len(evidence) or any(e not in ids for e in evidence):
            return None
        target, payload = build_body(action, a, str(dossier.get("mailbox", "")))
        return {
            "action": action, "target": target,
            "payload": payload, "evidence": sorted(evidence),
        }
    except Exception:
        return None


def fallback(dossier):
    """Harmless default: do nothing."""
    evidence = []
    try:
        evidence = [dossier["sources"][0]["lines"][0]["lineId"]]
    except Exception:
        pass
    return {
        "action": "no_action", "target": None,
        "payload": {"reasonCode": "INFORMATIONAL", "referenceId": dossier.get("dossierId", "")},
        "evidence": evidence,
    }
def ask_chunk(chunk):
    """chunk: list of (fingerprint, dossier). Returns {dossierId: raw answer}."""
    try:
        blocks = [
            "=== DOSSIER ===\n" + json.dumps(d, ensure_ascii=False)
            for _fp, d in chunk
        ]
        response = get_client().chat.completions.create(
            model=AI_MODEL,
            reasoning_effort="low",
            max_completion_tokens=12000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "\n\n".join(blocks)},
            ],
        )
        data = json.loads(response.choices[0].message.content)
        return {
            x.get("dossierId"): x
            for x in data.get("decisions", []) if isinstance(x, dict)
        }
    except Exception as exc:
        print("AI CALL FAILED:", repr(exc))
        return {}


def decide_dossiers(dossiers):
    """Return {dossierId: decision}. Uses the saved cache where it can."""
    result = {}
    todo = []
    for d in dossiers:
        fp = storage.dossier_fingerprint(d)
        cached = storage.get_decision(fp)
        if cached:
            result[d["dossierId"]] = cached
        else:
            todo.append((fp, d))

    if todo:
        kinds = sorted({(str(s.get("kind")), str(s.get("provenance")))
                        for _fp, d in todo for s in d.get("sources", [])})
        print("KINDS AND PROVENANCE SEEN:", kinds)
        print("FIRST DOSSIER SAMPLE:", json.dumps(todo[0][1], ensure_ascii=False)[:3000])

        chunks = [todo[i:i + CHUNK] for i in range(0, len(todo), CHUNK)]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            outputs = list(pool.map(ask_chunk, chunks))

        counts = Counter()
        for chunk, answers in zip(chunks, outputs):
            for fp, d in chunk:
                clean = clean_decision(answers.get(d["dossierId"]), d)
                if clean is None:
                    print("BAD OR MISSING AI DECISION for", d["dossierId"], "- using safe fallback")
                    result[d["dossierId"]] = fallback(d)
                    counts["FALLBACK"] += 1
                else:
                    storage.put_decision(fp, clean)
                    result[d["dossierId"]] = clean
                    counts[clean["action"]] += 1
        print("DECISION COUNTS:", dict(counts))
    return result
