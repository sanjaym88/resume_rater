from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from groq import Groq
import tempfile
import time
import json
import re
import os
import sys

# Windows consoles default to a codepage (e.g. cp1252) that can't encode a lot
# of the punctuation LLMs like to use (em dashes, non-breaking hyphens, curly
# quotes). Without this, printing that output crashes the request instead of
# just logging it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

client = Groq()
message_store = {}

# Flip this to False once things are stable to silence all the debug prints
# without having to delete them one by one.
DEBUG = True


def log(*args):
    if DEBUG:
        print(*args)


# The prompt is pinned server-side (not user-uploadable) so scores stay
# comparable across requests. Only the calibration clause changes per mode —
# everything else (rubric, evidence rules, JSON schema) stays identical.
PROMPT_TEMPLATE = """You are an AI-powered applicant tracking system (ATS) similar to modern semantic screening tools used by companies like Workday and LinkedIn. Unlike a legacy keyword-only ATS, you understand context: you can recognize when a candidate demonstrates a required skill through related, described work, even if they don't use the exact keyword from the job description. However, you do not give credit for skills that are not genuinely evidenced anywhere in the resume — inference must be reasonable and defensible, not generous guessing.

Your task: evaluate this candidate's resume against the job description and produce a score from 0 to 10.

Job Description:
{jd}

Candidate Resume:
{resume}

Follow this process:

STEP 1 — Extract requirements: Identify the core technical requirements, tools, and qualifications the JD is actually asking for. Distinguish "must-have" requirements from "nice-to-have" ones if the JD implies a difference.

STEP 2 — Match with context, not just keywords: For each requirement, examine the FULL resume (summary, experience, and projects together, not just the skills list) and determine if the candidate demonstrates it — either explicitly (the exact tool/skill is named) or through clearly described equivalent work (e.g., "built a REST API with FastAPI" reasonably demonstrates "API development experience" even if the JD's exact phrase isn't used). Do not credit vague or unrelated experience as a match — the connection must be genuine and specific, not a stretch.

STEP 3 — Weigh must-haves heavily: A resume missing multiple core, must-have requirements should score low even if it has strong nice-to-haves. A resume meeting most must-haves through clear, specific evidence should score well even with some gaps in nice-to-haves.

STEP 4 — Calibration for this evaluation: <<HARSHNESS_CLAUSE>>

STEP 5 — Assign the score using this rubric:
0–2: Little to no evidence of the core requirements, even considering context.
3–4: Some foundational or adjacent experience, but missing most core requirements.
5–6: Meets several core requirements with reasonable evidence, but has clear, specific gaps in others.
7–8: Meets most core requirements with specific, credible evidence from real projects or roles.
9–10: Strong, well-evidenced match across nearly all core requirements, with depth (not just breadth) demonstrated.

Respond with ONLY a valid JSON object, no other text before or after it, in exactly this shape:
{{
  "reasoning": "<2-4 sentences walking through which core requirements were met with evidence, which were reasonably inferred from related work, and which are genuinely missing>",
  "score": <integer 0-10>,
  "summary": "<one or two sentence overall verdict>",
  "strengths": ["<short strength 1>", "<short strength 2>"],
  "gaps": ["<short gap 1>", "<short gap 2>"]
}}

Do not include markdown code fences or any text outside the JSON object.
"""

HARSHNESS_CLAUSES = {
    "strict": (
        "Be unforgiving about gaps. Only explicit, project-tied evidence counts — do not "
        "stretch adjacent or transferable experience into a match. If the candidate is "
        "missing more than one must-have requirement, the score should not exceed 4, "
        "regardless of how strong their nice-to-have skills are."
    ),
    "standard": (
        "Be fair and evidence-based: give credit for genuine, specific, described work that "
        "reasonably demonstrates a requirement even if the exact tool/keyword differs, but do "
        "not credit vague or unrelated experience as a match."
    ),
    "lenient": (
        "Give the candidate reasonable benefit of the doubt on transferable and adjacent "
        "skills — strong fundamentals (e.g. programming, APIs, relevant tooling) that suggest "
        "the candidate could quickly pick up a missing requirement should still count "
        "meaningfully toward the score, even without an exact keyword match."
    ),
}

DEFAULT_MODE = "standard"


def build_prompt(mode):
    clause = HARSHNESS_CLAUSES.get(mode, HARSHNESS_CLAUSES[DEFAULT_MODE])
    return PROMPT_TEMPLATE.replace("<<HARSHNESS_CLAUSE>>", clause)


def call_groq_with_retry(messages, max_retries=2):
    """Calls Groq's chat completion with streaming, retrying if the reply comes back empty."""
    for attempt in range(1, max_retries + 2):
        reply = ""
        finish_reason = "unknown"

        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                temperature=0,
                max_tokens=2048,
                stream=True,
            )

            for chunk in completion:
                delta = chunk.choices[0].delta.content
                if delta:
                    reply += delta
                if chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason
        except Exception as e:
            log(f"[retry] API call raised on attempt {attempt}: {e}")
            if attempt <= max_retries:
                time.sleep(1)
            continue

        if reply.strip() and finish_reason != "length":
            if attempt > 1:
                log(f"[retry] Got a non-empty reply on attempt {attempt} (finish_reason: {finish_reason})")
            return reply.strip()

        if finish_reason == "length":
            log(f"[retry] Response got cut off by token limit on attempt {attempt} (finish_reason: length) — retrying won't help unless max_tokens changes")
        else:
            log(f"[retry] Empty reply on attempt {attempt} (finish_reason: {finish_reason})")

        if attempt <= max_retries:
            time.sleep(1)

    log(f"[retry] Still empty after {max_retries + 1} attempts, giving up.")
    return ""


def parse_score_json(reply):
    """
    Tries to parse the model's reply as the structured JSON we asked for.
    Falls back to the old regex-based number extraction if parsing fails,
    so a malformed response doesn't crash the whole request.
    """
    cleaned = reply.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        score = int(data.get("score", -1))
        reasoning = data.get("reasoning", "").strip()
        summary = data.get("summary", "").strip()
        strengths = data.get("strengths", [])
        gaps = data.get("gaps", [])

        reason_parts = []
        if summary:
            reason_parts.append(summary)
        if reasoning:
            reason_parts.append("Reasoning: " + reasoning)
        if strengths:
            reason_parts.append("Strengths: " + "; ".join(strengths))
        if gaps:
            reason_parts.append("Gaps: " + "; ".join(gaps))
        reason = "\n".join(reason_parts) if reason_parts else reply

        return score, reason

    except (json.JSONDecodeError, ValueError, AttributeError):
        log("[parse] JSON parsing failed, falling back to regex. Raw reply was:", reply)
        match = re.search(r"\b([0-9]|10)\b", reply)
        score = int(match.group(1)) if match else -1
        return score, reply


def get_resume_score(resume_bytes, jd_text, mode=DEFAULT_MODE, session_id="default"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(resume_bytes)
        temp_pdf_path = temp_pdf.name

    try:
        loader = PyMuPDFLoader(temp_pdf_path)
        documents = loader.load()
        resume_text = "\n".join(doc.page_content for doc in documents)
    finally:
        try:
            os.unlink(temp_pdf_path)
        except OSError as e:
            # Best-effort cleanup — a failed unlink (e.g. the loader still
            # holding the file open) should never mask the real error above.
            log(f"[cleanup] Could not delete temp file {temp_pdf_path}: {e}")

    log(f"[extract] Resume length: {len(resume_text)} chars")
    log(f"[extract] Preview: {resume_text[:200]}...")

    prompt_template = PromptTemplate.from_template(build_prompt(mode))
    formatted_prompt = prompt_template.format(jd=jd_text, resume=resume_text)

    messages = [{"role": "user", "content": formatted_prompt}]

    reply = call_groq_with_retry(messages)
    log("[model] Raw output:", reply if reply else "(empty)")

    score, reason = parse_score_json(reply)

    # Store the CLEANED, human-readable reason in history (not the raw JSON) —
    # this keeps the conversation looking like natural language, so follow-up
    # questions don't get answered in JSON just because the model's last turn was.
    messages.append({"role": "assistant", "content": reason})
    message_store[session_id] = messages

    return score, reason


def continue_conversation(user_input, session_id="default"):
    if session_id not in message_store:
        return "Session expired. Please re-upload your resume."

    messages = message_store[session_id]

    # Explicitly tell it to drop JSON mode for conversational follow-ups —
    # otherwise it can pattern-match off its own earlier structured reply.
    wrapped_input = (
        "Answer the following question in plain, conversational English. "
        "Do NOT respond in JSON, do not use any special formatting or field labels — "
        "just answer naturally like you're talking to the person.\n\n"
        f"Question: {user_input}"
    )
    messages.append({"role": "user", "content": wrapped_input})

    reply = call_groq_with_retry(messages)

    if not reply:
        reply = "Sorry, I couldn't generate a response. Please try asking again."

    messages.append({"role": "assistant", "content": reply})
    log("[chat] Raw output:", reply)
    return reply