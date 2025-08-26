import os
import io
import re
import json
import random
import numpy as np
import streamlit as st
import pdfplumber

from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

# LangChain + FAISS + Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Hugging Face pipelines
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Evaluation
import evaluate  # rouge, bleu, etc.

# Datasets (for optional fine-tuning)
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments


# ------------------------------
# Streamlit Config
# ------------------------------
st.set_page_config(page_title="AI Study Assistant (LLM + FAISS)", page_icon="📚", layout="wide")
st.title("📚 AI Study Assistant — Summaries, Flashcards, and MCQs")
st.caption("LangChain · FAISS · Hugging Face · Streamlit")


# ------------------------------
# CACHED LOADERS (Models & Embeddings)
# ------------------------------
@st.cache_resource(show_spinner=True)
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource(show_spinner=True)
def load_generator_model(model_name="google/flan-t5-base"):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    gen = pipeline("text2text-generation", model=mdl, tokenizer=tok)
    return gen, tok, mdl

@st.cache_resource(show_spinner=True)
def load_qa_pipeline():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")


# ------------------------------
# PDF / TXT TEXT EXTRACTION
# ------------------------------
def extract_text(file) -> str:
    if file.name.lower().endswith(".pdf"):
        text = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text.append(txt)
        return "\n".join(text)
    else:
        # assume txt
        return file.read().decode("utf-8", errors="ignore")


# ------------------------------
# CHUNK + INDEX (FAISS)
# ------------------------------
def build_faiss_index(text: str, embeddings) -> Tuple[FAISS, List[Document]]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]
    vs = FAISS.from_documents(docs, embeddings)
    return vs, docs


# ------------------------------
# SUMMARIZATION
# ------------------------------
def summarize_text(text: str, summarizer, max_len=200) -> str:
    # handle long text by map-reduce
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    parts = splitter.split_text(text)
    partials = []
    for p in parts:
        out = summarizer(p, max_length=max_len, min_length=max(40, max_len//4), do_sample=False)[0]["summary_text"]
        partials.append(out)
    merged = " ".join(partials)
    # final condense pass if long
    if len(merged.split()) > 350:
        out = summarizer(merged, max_length=220, min_length=80, do_sample=False)[0]["summary_text"]
        return out
    return merged


# ------------------------------
# FLASHCARDS + MCQs with FLAN-T5 (PROMPTED)
# ------------------------------
FLASHCARD_SYSTEM_PROMPT = (
    "You are a helpful study assistant. Create {n} flashcards from the content below. "
    "Each flashcard must be on its own line in the exact format 'Q: <question> | A: <answer>'. "
    "Be concise and factual.\n\nCONTENT:\n{content}\n"
)

MCQ_SYSTEM_PROMPT = (
    "Generate {n} multiple-choice questions from the content below. For each question, provide exactly 4 options "
    "labeled A), B), C), D), and indicate the correct option at the end using 'Answer: <letter>'. "
    "Keep questions unambiguous and directly tied to the content.\n\nCONTENT:\n{content}\n"
)

def generate_flashcards(content: str, gen, n=8, max_new_tokens=512) -> List[Tuple[str, str]]:
    prompt = FLASHCARD_SYSTEM_PROMPT.format(n=n, content=content)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    cards = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # expect: Q: ... | A: ...
        m = re.match(r"Q:\s*(.*?)\s*\|\s*A:\s*(.*)", line)
        if m:
            q, a = m.group(1).strip(), m.group(2).strip()
            if q and a:
                cards.append((q, a))
    # fallback if parsing failed
    if not cards:
        # try simple split on "Q:" occurrences
        bits = [b.strip() for b in out.split("Q:") if b.strip()]
        for b in bits:
            if "| A:" in b:
                q, a = b.split("| A:", 1)
                cards.append((q.strip(), a.strip()))
    # truncate to n
    return cards[:n]

def generate_mcqs(content: str, gen, n=5, max_new_tokens=768) -> List[dict]:
    prompt = MCQ_SYSTEM_PROMPT.format(n=n, content=content)
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
    # parse MCQs
    questions = []
    blocks = re.split(r"\n\s*\n", out.strip())
    for blk in blocks:
        lines = [l.strip() for l in blk.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        q = lines[0]
        opts = []
        ans = None
        for ln in lines[1:]:
            if re.match(r"^[A-D]\)", ln):
                opts.append(ln)
            elif ln.lower().startswith("answer:"):
                ans = ln.split(":", 1)[1].strip().upper()[:1]
        if q and len(opts) == 4 and ans in ["A","B","C","D"]:
            questions.append({"question": q, "options": opts, "answer": ans})
    return questions[:n]


# ------------------------------
# EVALUATION
# ------------------------------
def lead_baseline_reference(text: str, max_words=220) -> str:
    # weak baseline: first N sentences (or words)
    words = text.split()
    return " ".join(words[:max_words])

def compute_rouge_scores(pred_summary: str, ref_summary: str):
    rouge = evaluate.load("rouge")
    scores = rouge.compute(predictions=[pred_summary], references=[ref_summary])
    return scores  # dict with rouge1, rouge2, rougeL, rougeLsum

def compression_ratio(orig: str, summ: str) -> float:
    o = max(1, len(orig.split()))
    s = max(1, len(summ.split()))
    return round(s / o, 4)

def semantic_similarity(orig: str, summ: str, embeddings) -> float:
    # cosine similarity between mean pooled embeddings
    emb = embeddings.embed_documents([orig, summ])
    vec_o = np.array(emb[0]).reshape(1, -1)
    vec_s = np.array(emb[1]).reshape(1, -1)
    return float(cosine_similarity(vec_o, vec_s)[0, 0])

def distinct_n(texts: List[str], n=1) -> float:
    # diversity metric: distinct-n (ratio of unique n-grams)
    tokens = []
    for t in texts:
        tokens += t.split()
    if len(tokens) < n:
        return 0.0
    ngrams = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return round(len(ngrams) / max(1, (len(tokens)-n+1)), 4)

def answerability_at_k(items: List[Tuple[str, str]],  # for flashcards: (question, answer)
                       retriever,
                       qa_pipe,
                       k=3,
                       score_thresh=0.2) -> float:
    """
    For each question, retrieve top-k chunks and run QA. Count proportion that yields a confident answer.
    Works for both flashcards (question only) and MCQs (we'll use the question stem).
    """
    ok = 0
    for (q, _) in items:
        docs = retriever.get_relevant_documents(q, k=k)
        context = "\n".join([d.page_content for d in docs])
        try:
            res = qa_pipe(question=q, context=context)
            if res and (res.get("score", 0) >= score_thresh) and res.get("answer"):
                ok += 1
        except Exception:
            pass
    return round(ok / max(1, len(items)), 4)

def answerability_mcq_at_k(mcqs: List[dict], retriever, qa_pipe, k=3, score_thresh=0.2) -> float:
    ok = 0
    for item in mcqs:
        q = item["question"]
        docs = retriever.get_relevant_documents(q, k=k)
        context = "\n".join([d.page_content for d in docs])
        try:
            res = qa_pipe(question=q, context=context)
            if res and (res.get("score", 0) >= score_thresh) and res.get("answer"):
                ok += 1
        except Exception:
            pass
    return round(ok / max(1, len(mcqs)), 4)


# ------------------------------
# OPTIONAL: QUICK FINE-TUNE (FLAN-T5) ON MINI SQuAD (QG STYLE)
# ------------------------------
def optional_quick_finetune_flan_t5(base_model_name="google/flan-t5-base", steps=200):
    """
    Tiny demo fine-tune: trains the model to map (context+answer) -> question on a small subset of SQuAD.
    WARNING: training on CPU will be slow; keep steps tiny. Skip if not needed.
    """
    dataset = load_dataset("squad")
    # build small training pairs
    def build_pairs(example):
        ctx = example["context"]
        q = example["question"]
        # choose first answer text if available
        ans = example["answers"]["text"][0] if example["answers"]["text"] else ""
        inp = f"Generate a question for the answer: {ans}\nContext: {ctx}"
        out = q
        return {"input": inp, "target": out}

    train_small = dataset["train"].select(range(800)).map(build_pairs)
    eval_small  = dataset["validation"].select(range(200)).map(build_pairs)

    tok = AutoTokenizer.from_pretrained(base_model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    def tokenize_fn(batch):
        model_inputs = tok(batch["input"], max_length=512, truncation=True)
        with tok.as_target_tokenizer():
            labels = tok(batch["target"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_tok = train_small.map(tokenize_fn, batched=True, remove_columns=["input", "target", "id", "title", "context", "question", "answers"])
    eval_tok  = eval_small.map(tokenize_fn, batched=True, remove_columns=["input", "target", "id", "title", "context", "question", "answers"])

    data_collator = DataCollatorForSeq2Seq(tok, model=mdl)
    args = TrainingArguments(
        output_dir="./flan_qg_mini",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-5,
        num_train_epochs=1,
        max_steps=steps,
        evaluation_strategy="steps",
        eval_steps=steps//4 if steps>=4 else 1,
        save_steps=steps,
        logging_steps=max(1, steps//10),
        report_to=[],
        fp16=False
    )

    trainer = Trainer(
        model=mdl,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer.train()
    return pipeline("text2text-generation", model=mdl, tokenizer=tok), tok, mdl


# ------------------------------
# SIDEBAR: SETTINGS
# ------------------------------
with st.sidebar:
    st.header("Settings")
    n_flash = st.slider("Number of flashcards", 4, 20, 8)
    n_mcq = st.slider("Number of MCQs", 3, 15, 6)
    enable_finetune = st.checkbox("Experimental: quick fine-tune FLAN-T5 on mini SQuAD (adds time)")
    steps = st.number_input("Fine-tune max steps", min_value=50, max_value=1000, value=200, step=50)
    st.caption("Tip: leave fine-tune off for faster runs; turn on only if you want to demonstrate training.")


# ------------------------------
# MAIN APP
# ------------------------------
# ------------------------------
# MAIN APP
# ------------------------------
uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded:
    with st.spinner("Extracting text..."):
        raw_text = extract_text(uploaded).strip()
    if not raw_text:
        st.error("No text found in the file. Try another document.")
        st.stop()

    st.subheader("📄 Extracted Text (preview)")
    st.text_area("Preview", raw_text[:3000], height=180)

    # Build embeddings & vector store
    embeddings = load_embeddings()
    with st.spinner("Chunking & indexing..."):
        vs, docs = build_faiss_index(raw_text, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    # Load models
    summarizer = load_summarizer()
    qa_pipe = load_qa_pipeline()
    if enable_finetune:
        with st.spinner("Fine-tuning FLAN-T5 (mini) ..."):
            gen, gen_tok, gen_model = optional_quick_finetune_flan_t5(steps=int(steps))
    else:
        gen, gen_tok, gen_model = load_generator_model()

    # Buttons
    col1, col2, col3 = st.columns(3)
    go_sum = col1.button("📝 Generate Summary")
    go_cards = col2.button("🃏 Generate Flashcards")
    go_mcq = col3.button("❓ Generate MCQs")

    summary = ""

    # --- SUMMARY ---
    if go_sum:
        with st.spinner("Summarizing..."):
            summary = summarize_text(raw_text, summarizer, max_len=220)
        st.subheader("📝 Summary")
        st.write(summary)

        # Evaluation metrics
        st.markdown("##### ✅ Evaluation (Summarization)")
        ref = lead_baseline_reference(raw_text, max_words=220)
        rouge_scores = compute_rouge_scores(summary, ref)
        comp = compression_ratio(raw_text, summary)
        sim = semantic_similarity(raw_text, summary, embeddings)
        c1, c2, c3 = st.columns(3)
        c1.metric("ROUGE-1", f"{rouge_scores.get('rouge1', 0):.4f}")
        c2.metric("ROUGE-2", f"{rouge_scores.get('rouge2', 0):.4f}")
        c3.metric("ROUGE-L", f"{rouge_scores.get('rougeL', 0):.4f}")
        c1.metric("Compression Ratio", f"{comp:.4f}")
        c2.metric("Semantic Similarity", f"{sim:.4f}")

    # --- FLASHCARDS ---
if go_cards:
    base = summary if summary else " ".join(raw_text.split()[:1500])
    flashcard_prompt = (
        f"Generate 5 concise flashcards in Q&A format. Each flashcard must be on its own line in the exact format 'Q: <question> | A: <answer>'.\n"
        f"Context:\n{base}"
    )
    with st.spinner("Generating Flashcards..."):
        flashcard_result = gen(flashcard_prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"].strip()

    st.subheader("📗 Generated Flashcards")
    lines = [line.strip() for line in flashcard_result.splitlines() if line.strip()]
    any_flashcards = False
    for line in lines:
        # try parsing Q and A
        match = re.match(r"Q[:\s]\s*(.*?)\s*\|\s*A[:\s]\s*(.*)", line, re.IGNORECASE)
        if match:
            q, a = match.group(1), match.group(2)
            any_flashcards = True
            with st.expander(f"Q: {q}"):
                st.write(a)
        else:
            # fallback: just display line as question
            any_flashcards = True
            with st.expander(f"Q: {line}"):
                st.write("Answer not detected, see raw text.")

    if not any_flashcards:
        st.info("No flashcards could be parsed. Model output:\n" + flashcard_result)



    # --- MCQs ---
if go_mcq:
    base = summary if summary else " ".join(raw_text.split()[:1500])
    mcq_texts = []

    # Split into chunks for reliable generation
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(base)

    # Generate MCQs per chunk
    for chunk in chunks:
        with st.spinner("Generating MCQs..."):
            out = gen(
                f"Generate 3 multiple-choice questions with 4 options each from the content below. Include the correct answer at the end using 'Answer: <letter>'.\n\nContent:\n{chunk}",
                max_new_tokens=512,
                do_sample=False
            )[0]["generated_text"].strip()
            mcq_texts.append(out)

    # Combine all chunks and split by "Q:" or line breaks
    all_mcq_lines = "\n".join(mcq_texts).splitlines()
    all_mcq_lines = [l for l in all_mcq_lines if l.strip()]

    # Take only first 3 questions
    mcqs_to_show = []
    current_q = ""
    options = []
    answer = ""
    q_count = 0

    for line in all_mcq_lines:
        if line.lower().startswith("q:") or line.lower().startswith("question"):
            if current_q:
                mcqs_to_show.append((current_q, options, answer))
                q_count += 1
                if q_count >= 3:
                    break
                options = []
                answer = ""
            current_q = line.split(":",1)[1].strip() if ":" in line else line
        elif re.match(r"^[A-D]\)", line):
            options.append(line)
        elif line.lower().startswith("answer:"):
            answer = line.split(":",1)[1].strip().upper()

    # Add the last question if less than 3
    if current_q and q_count < 3:
        mcqs_to_show.append((current_q, options, answer))

    # Display MCQs in expanders
    st.subheader("📘 Generated MCQs")
    for q, opts, ans in mcqs_to_show:
        with st.expander(f"Q: {q}"):
            for opt in opts:
                st.write(opt)
            st.write(f"Answer: {ans if ans else 'Answer not detected'}")
