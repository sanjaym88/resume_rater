from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from rr2 import get_resume_score, continue_conversation

from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/evaluate")
async def evaluate(
    resume: UploadFile = File(...),
    jd: UploadFile = File(...),
    mode: str = Form("standard"),
    session_id: str = Form("default")
):
    resume_bytes = await resume.read()

    try:
        jd_text = (await jd.read()).decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400,
            content={"score": -1, "reason": "Job description file must be UTF-8 encoded plain text."}
        )

    try:
        score, reason = get_resume_score(resume_bytes, jd_text, mode, session_id)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"score": -1, "reason": f"Error processing this resume: {e}"}
        )

    return JSONResponse(content={"score": score, "reason": reason})

@app.post("/chat")
async def chat(user_input: str = Form(...), session_id: str = Form("default")):
    reply = continue_conversation(user_input, session_id)
    return JSONResponse(content={"response": reply})

@app.post("/batch-evaluate")
async def batch_evaluate(
    resumes: List[UploadFile] = File(...),
    jd: UploadFile = File(...),
    mode: str = Form("standard"),
):
    try:
        jd_text = (await jd.read()).decode("utf-8")
    except UnicodeDecodeError:
        return JSONResponse(
            status_code=400,
            content={"results": [], "error": "Job description file must be UTF-8 encoded plain text."}
        )

    results = []
    for resume in resumes:
        resume_bytes = await resume.read()
        session_id = f"batch_{resume.filename}"
        try:
            score, reason = get_resume_score(resume_bytes, jd_text, mode, session_id)
        except Exception as e:
            score, reason = -1, f"Error processing this resume: {str(e)}"
        results.append({
            "filename": resume.filename,
            "score": score,
            "reason": reason
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return JSONResponse(content={"results": results})


app.mount("/", StaticFiles(directory="../frontend", html=True), name="static")
