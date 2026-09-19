from dotenv import load_dotenv
load_dotenv()


from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import PromptTemplate
from groq import Groq
import tempfile
import time
import json
import re

client = Groq()
message_store = {}

# Flip this to False once things are stable to silence all the debug prints
# without having to delete them one by one.
DEBUG = True


def log(*args):
    if DEBUG:
        print(*args)


def call_groq_with_retry(messages, max_retries=2):
    """Calls Groq's chat completion with streaming, retrying if the reply comes back empty."""
    for attempt in range(1, max_retries + 2):
        reply = ""
        finish_reason = "unknown"

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


def get_resume_score(resume_bytes, jd_text, prompt_text, session_id="default"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(resume_bytes)
        temp_pdf_path = temp_pdf.name

    loader = PyMuPDFLoader(temp_pdf_path)
    documents = loader.load()
    resume_text = "\n".join(doc.page_content for doc in documents)

    log(f"[extract] Resume length: {len(resume_text)} chars")
    log(f"[extract] Preview: {resume_text[:200]}...")

    prompt_template = PromptTemplate.from_template(prompt_text)
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