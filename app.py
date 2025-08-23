import os, time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma   


# Small model 
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Large model 
from huggingface_hub import InferenceClient

# Configurations 
load_dotenv()
HF_TOKEN   = os.getenv("HF_TOKEN", "").strip()
SMALL_ID   = os.getenv("SMALL_MODEL", "google/flan-t5-base")
LARGE_ID   = os.getenv("LARGE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
EMB_ID     = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
PROMPT_LEN_THRESHOLD = int(os.getenv("PROMPT_LEN_THRESHOLD", "220"))
app = FastAPI(title="RAG + LLM Switch (LangChain + HF)")

# Vector DB 
os.makedirs(CHROMA_DIR, exist_ok=True)
_embeddings = HuggingFaceEmbeddings(model_name=EMB_ID)
def get_vs() -> Chroma:
    return Chroma(collection_name="rag", embedding_function=_embeddings, persist_directory=CHROMA_DIR)

_split = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120, separators=["\n\n","\n","."," "])

# Small LLM (local FLAN)
_small_pipe = None
def small_generate(prompt: str, max_new_tokens: int = 220) -> str:
    global _small_pipe
    if _small_pipe is None:
        tok = AutoTokenizer.from_pretrained(SMALL_ID)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(SMALL_ID)
        _small_pipe = pipeline("text2text-generation", model=mdl, tokenizer=tok, device=-1)
    out = _small_pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4, temperature=0.2)
    return out[0]["generated_text"].strip()

#  Large LLM (HF remote)
_large_client: Optional[InferenceClient] = None
def large_generate(prompt: str, max_new_tokens: int = 300) -> str:
    global _large_client
    if not HF_TOKEN:
        raise HTTPException(400, "HF_TOKEN missing for remote model.")
    if _large_client is None:
        _large_client = InferenceClient(model=LARGE_ID, token=HF_TOKEN, timeout=60)
    # Prefer chat.completions; fallback to text_generation
    last_err = None
    for delay in (0, 2, 4):
        if delay: time.sleep(delay)
        try:
            resp = _large_client.chat.completions.create(
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_new_tokens,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e1:
            last_err = e1
            try:
                txt = _large_client.text_generation(prompt=prompt, max_new_tokens=max_new_tokens, temperature=0.2, return_full_text=False)
                return txt.strip()
            except Exception as e2:
                last_err = e2
    raise HTTPException(502, f"Remote model error: {type(last_err).__name__}: {last_err}")

# Prompts 
SMALL_PROMPT = (
    "Answer ONLY from the context below. If answer not in context, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)
LARGE_PROMPT = (
    "[INST] Answer using ONLY the context below. If not found, say 'I don't know'. [/INST]\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

# Router
def choose_mode(user_mode: str, question: str, context: str) -> str:
    # Respect manual override
    if user_mode in ("small", "large"):
        return user_mode

    q_len = len(question)
    c_len = len(context)

    # Case 1: Very long user question will done by large model
    if q_len > 60:
        return "large"

    # Case 2: Context retrieved is big (heavy doc chunk) by large model
    if c_len > 2500:
        return "large"

    # Case 3: No context found at all aslo by large model
    if context.strip() == "(no context)":
        return "large"

    # Default: small
    return "small"

# HTML UI 
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse("""
<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI AGENT for LLM Switching( RAG Based )</title>
<style>
:root{--bg:#0f172a;--panel:#111827;--text:#e5e7eb;--muted:#9ca3af;--btn:#3b82f6;--chip:#1f2937}
body{margin:0;background:var(--bg);color:var(--text);font:16px/1.5 system-ui,Segoe UI,Roboto}
.wrap{max-width:960px;margin:24px auto;padding:0 16px}
.card{background:var(--panel);border:1px solid #222;border-radius:14px;padding:18px}
h1{margin:0 0 8px}.muted{color:var(--muted)}
textarea{width:100%;min-height:130px;background:#0b1220;color:var(--text);border:1px solid #222;border-radius:12px;padding:12px}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:10px}
select,input[type=number]{background:#0b1220;color:var(--text);border:1px solid #222;border-radius:10px;padding:8px}
.btn{background:var(--btn);color:white;border:0;border-radius:10px;padding:10px 14px;cursor:pointer}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}
.chip{background:var(--chip);color:#cbd5e1;border-radius:20px;padding:6px 10px;font-size:13px}
pre{white-space:pre-wrap;background:#0b1220;border:1px solid #222;border-radius:12px;padding:12px}
.split{display:grid;grid-template-columns:1fr 320px;gap:16px}
@media (max-width:900px){.split{grid-template-columns:1fr}}
</style></head>
<body>
<div class="wrap">
  <div class="card"><h1>AI Agent for LLM Switching( RAG Based )</h1>
    <div class="muted">Upload a PDF, then ask a question. App chooses <b>small</b> (local FLAN) or <b>large</b> (remote Mistral) or chooses automatically.</div>
  </div>
  <div class="split" style="margin-top:16px">
    <div class="card">
      <h3>Your question</h3>
      <textarea id="q" placeholder="e.g., List 3 business areas mentioned in the PDF."></textarea>
      <div class="row">
        <label>Max tokens <input id="tok" type="number" value="220" min="32" max="600"/></label>
        <label>Mode
          <select id="mode">
            <option value="auto" selected>auto</option>
            <option value="small">small (FLAN-T5)</option>
            <option value="large">large (Mistral-7B)</option>
          </select>
        </label>
        <button class="btn" onclick="ask()">Ask</button>
      </div>
      <div class="chips" id="meta"></div>
      <pre id="out" style="margin-top:10px"></pre>
    </div>
    <div class="card">
      <h3>Ingest PDF</h3>
      <input id="pdf" type="file" accept="application/pdf"/>
      <div class="row">
        <button class="btn" onclick="ingest()">Ingest PDF</button>
        <span class="muted" id="ingRes"></span>
      </div>
      <h3 style="margin:16px 0 8px">Status</h3>
      <div class="chips"><span class="chip">ready</span></div>
    </div>
  </div>
</div>
<script>
async function ingest(){
  const f = document.getElementById('pdf').files[0];
  if(!f){ alert('Pick a PDF first'); return; }
  const fd = new FormData(); fd.append('file', f);
  const r = await fetch('/ingest',{method:'POST', body:fd});
  const j = await r.json();
  document.getElementById('ingRes').textContent = r.ok ? ('OK: '+j.chunks+' chunks') : ('Error: '+(j.detail||'ingest failed'));
}
function chip(t){ const s=document.createElement('span'); s.className='chip'; s.textContent=t; return s; }
async function ask(){
  const q = document.getElementById('q').value.trim();
  if(!q){ alert('Type a question'); return; }
  const tok = +document.getElementById('tok').value || 220;
  const mode = document.getElementById('mode').value;
  document.getElementById('out').textContent = 'Thinking...';
  const r = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question:q, max_tokens:tok, mode})});
  const box = document.getElementById('meta'); box.innerHTML='';
  const j = await r.json();
  if(!r.ok){ document.getElementById('out').textContent='Error: '+(j.detail||r.statusText); return; }
  box.appendChild(chip('model: '+j.model));
  box.appendChild(chip('latency: '+Math.round(j.latency_ms)+' ms'));
  box.appendChild(chip('reason: '+j.reason));
  box.appendChild(chip('context: '+(j.context_used?'yes':'no')));
  document.getElementById('out').textContent = j.text;
}
</script>
</body></html>
""")

# API 
class AskIn(BaseModel):
    question: str
    max_tokens: int = 220
    mode: str = "auto"   # small | large | auto

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a .pdf file.")
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())
    pages = PyPDFLoader(path).load()
    docs  = _split.split_documents(pages)
    vs    = get_vs()
    vs.add_documents(docs); vs.persist()
    return {"ok": True, "chunks": len(docs)}

@app.post("/ask")
async def ask(body: AskIn):
    vs = get_vs()
    hits = vs.similarity_search(body.question, k=4)
    context = "\n\n".join(d.page_content for d in hits) if hits else "(no context)"
    mode = choose_mode(body.mode, body.question, context)
    prompt = (SMALL_PROMPT if mode=="small" else LARGE_PROMPT).format(context=context, question=body.question)

    t0 = time.perf_counter()
    try:
        if mode == "small":
            text = small_generate(prompt, body.max_tokens)
            model = SMALL_ID
        else:
            text = large_generate(prompt, body.max_tokens)
            model = LARGE_ID
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    ms = (time.perf_counter()-t0)*1000.0

    return {"model": model, "latency_ms": ms, "reason": f"{mode}_ok", "context_used": bool(hits), "text": text}
