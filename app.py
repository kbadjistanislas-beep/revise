import os
import json
import re
import datetime
from pathlib import Path

import streamlit as st
import img2pdf
from dotenv import load_dotenv

from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

# ==========================================
# 1. CONFIGURATION & VARIABLES GLOBALES
# ==========================================
load_dotenv()

PAGE_TITLE = "ReviseAI"
UPLOAD_FOLDER = "uploads"
DB_PATH = "./chroma_db"
META_FILE_PREFIX = "documents_metadata"

# Correction du nom du modèle ici :
MODEL_NAME = "gemini-1.5-flash"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = """
Tu es ReviseAI, un assistant de révision intelligent.
Réponds en français naturel, clairement et directement.
Utilise Markdown si nécessaire.

Si un contexte de document est fourni, utilise-le en priorité.
N'invente jamais une information absente du document.

Pour un cours :
- explique simplement
- donne un exemple si utile
- termine par les points essentiels

Pour un exercice :
- identifie ce qui est demandé
- donne la méthode
- fais les calculs
- donne clairement la réponse finale
"""

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

# ==========================================
# 2. SERVICES & FONCTIONS
# ==========================================

@st.cache_resource
def get_gemini_client():
    key = None
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

    key = key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError(
            "Erreur: GEMINI_API_KEY est absente. "
            "Ajoute-la dans Streamlit -> Settings -> Secrets."
        )

    return genai.Client(api_key=key)

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_user_meta_file(user_id):
    return f"{META_FILE_PREFIX}_{user_id}.json"

def load_metadata(user_id):
    meta_path = get_user_meta_file(user_id)
    if not os.path.exists(meta_path):
        return []
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_metadata(data, user_id):
    meta_path = get_user_meta_file(user_id)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def sanitize_collection_name(name, user_id):
    raw_name = f"{user_id}_{name}"
    clean_name = re.sub(r"[^a-zA-Z0-9._-]", "_", raw_name).strip("_")
    if not clean_name:
        clean_name = "document"
    if clean_name[0].isdigit():
        clean_name = "doc_" + clean_name
    return clean_name[:512]

def clean_text(text):
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text or "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def load_document(path):
    ext = Path(path).suffix.lower()

    if ext == ".pdf":
        try:
            import fitz
            from langchain_core.documents import Document

            pdf = fitz.open(path)
            docs = [
                Document(
                    page_content=clean_text(page.get_text()),
                    metadata={"page": i + 1}
                )
                for i, page in enumerate(pdf)
                if page.get_text().strip()
            ]
            pdf.close()
            return docs
        except Exception:
            return PyPDFLoader(path).load()

    if ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    if ext in [".docx", ".doc"]:
        return Docx2txtLoader(path).load()

    raise ValueError("Format non supporté")

def process_document(path, meta, user_id):
    docs = load_document(path)

    for d in docs:
        d.page_content = clean_text(d.page_content)
        d.metadata.update(meta)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    collection_name = sanitize_collection_name(
        f"{meta['subject']}_{meta['filename']}",
        user_id=user_id
    )

    Chroma.from_documents(
        chunks,
        get_embeddings(),
        persist_directory=DB_PATH,
        collection_name=collection_name
    )

    return len(chunks)

def search_documents(query, user_id, selected_doc=None, k=8):
    results = []
    user_docs = load_metadata(user_id)

    if selected_doc and selected_doc != "Tous les documents":
        user_docs = [d for d in user_docs if d["filename"] == selected_doc]

    for d in user_docs:
        try:
            store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=get_embeddings(),
                collection_name=sanitize_collection_name(
                    f"{d['subject']}_{d['filename']}",
                    user_id=user_id
                )
            )
            results += store.similarity_search(query, k=k)
        except Exception:
            pass

    return results[:12]

def format_context(chunks):
    return "\n\n".join(
        f"[{x.metadata.get('filename', 'Document')} "
        f"- {x.metadata.get('subject', 'Matière')} "
        f"- page {x.metadata.get('page', '')}]\n"
        f"{x.page_content}"
        for x in chunks
    )

def ask_gemini(prompt, context_str=""):
    full_prompt = f"""
CONTEXTE DU COURS :
{context_str}

QUESTION OU MESSAGE :
{prompt}
""" if context_str else prompt

    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config={
                "system_instruction": SYSTEM_PROMPT,
                "max_output_tokens": 4096,
            }
        )
        return response.text or "Je n'ai pas reçu de réponse."
    except Exception as e:
        return f"Erreur Gemini : {e}"

def generate_summary(subject, user_id):
    chunks = search_documents(
        f"cours {subject} définitions formules théorèmes méthodes",
        user_id=user_id,
        k=12
    )
    if not chunks:
        return "Aucun contenu trouvé pour cette matière."

    prompt = f"""
Résume le cours de {subject}.
Présente :
1. Les notions essentielles
2. Les définitions
3. Les formules/théorèmes
4. Les méthodes
5. Les erreurs à éviter
6. Un résumé final

Cours :
{format_context(chunks)}
"""
    return ask_gemini(prompt)

def generate_quiz(subject, n, qtype, user_id):
    chunks = search_documents(
        f"cours {subject} notions définitions exemples",
        user_id=user_id,
        k=15
    )

    if not chunks:
        return None, "Aucun contenu trouvé pour cette matière."

    format_json = """
{
 "question":"...",
 "options":["A. ...","B. ...","C. ...","D. ..."],
 "correct_answer":"A",
 "explanation":"..."
}
""" if qtype == "QCM" else """
{
 "question":"...",
 "options":[],
 "correct_answer":"...",
 "explanation":"..."
}
"""

    prompt = f"""
Génère {n} questions de révision sur {subject}.
Retourne UNIQUEMENT un JSON valide sous la forme d'un tableau d'objets :
[{format_json}]

Utilise uniquement le contenu du cours.

COURS :
{format_context(chunks)}
"""
    response = ask_gemini(prompt)

    try:
        match = re.search(r"\[.*\]", response, re.S)
        return json.loads(match.group() if match else response), None
    except Exception as e:
        return None, f"Quiz invalide : {e}"

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

for key, default in {"messages": [], "quiz_data": None, "quiz_answers": {}}.items():
    if key not in st.session_state:
        st.session_state[key] = default

if not st.session_state.user_id:
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.write("")
        st.write("")
        with st.container(border=True):
            st.title("Bienvenue sur ReviseAI")
            st.write("Entrez votre prénom ou un pseudo pour ouvrir votre espace de cours :")

            with st.form("login_form"):
                username = st.text_input("Votre identifiant / Pseudo :", placeholder="ex: Thomas")
                submit = st.form_submit_button("Accéder à mes cours", use_container_width=True)

                if submit and username.strip():
                    clean_id = "".join(c for c in username if c.isalnum()).lower()
                    st.session_state.user_id = clean_id
                    st.rerun()
    st.stop()

user_id = st.session_state.user_id

with st.sidebar:
    st.title("ReviseAI")
    st.caption(f"Espace personnel : {user_id.capitalize()}")
    st.divider()

    docs = load_metadata(user_id)
    st.metric(label="Documents indexés", value=len(docs))

    st.divider()
    if st.button("Changer d'utilisateur", use_container_width=True):
        st.session_state.user_id = None
        st.session_state.messages = []
        st.rerun()

    if st.button("Effacer la discussion", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

chat, documents, summary, quizzes = st.tabs([
    "Chat",
    "Documents",
    "Résumé",
    "Quiz"
])

with chat:
    st.subheader("Espace de Discussion")

    docs = load_metadata(user_id)
    doc_options = ["Tous les documents"] + [d["filename"] for d in docs]

    selected_doc = st.selectbox(
        "Cibler la recherche sur un document :",
        doc_options,
        key="chat_doc_filter"
    )

    st.divider()

    if not st.session_state.messages:
        st.info("Bonjour ! Posez une question sur vos cours enregistrés.")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Posez votre question ici..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                chunks = search_documents(prompt, user_id=user_id, selected_doc=selected_doc)
                context_str = format_context(chunks) if chunks else ""
                answer = ask_gemini(prompt, context_str)

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

with documents:
    st.subheader("Gestion de vos Documents")

    with st.container(border=True):
        st.write("### Ajouter un nouveau document")
        with st.form("upload_form", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                subject = st.text_input("Matière *", placeholder="Ex: Mathématiques")
                professor = st.text_input("Professeur", placeholder="Ex: M. Dupont")
            with col2:
                date = st.date_input("Date du cours", datetime.date.today())
                file = st.file_uploader("Fichier", type=["pdf", "docx", "doc", "txt", "jpg", "jpeg", "png"])

            submit = st.form_submit_button("Indexer le document", use_container_width=True)

    if submit:
        if not subject or not file:
            st.error("Veuillez remplir la matière et ajouter un fichier.")
        else:
            is_image = file.type.startswith("image/")
            user_upload_dir = os.path.join(UPLOAD_FOLDER, user_id)
            os.makedirs(user_upload_dir, exist_ok=True)

            if is_image:
                path = os.path.join(user_upload_dir, Path(file.name).stem + ".pdf")
                with open(path, "wb") as f:
                    f.write(img2pdf.convert(file.getvalue()))
                filename = Path(path).name
            else:
                filename = file.name
                path = os.path.join(user_upload_dir, filename)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

            meta = {
                "filename": filename,
                "subject": subject,
                "professor": professor or "Non spécifié",
                "date": str(date),
                "file_path": path,
                "is_image": is_image
            }

            try:
                with st.spinner("Réflexion..."):
                    count = process_document(path, meta, user_id=user_id)

                meta["chunks_count"] = count
                data = load_metadata(user_id)
                data.append(meta)
                save_metadata(data, user_id)

                st.success(f"Document {filename} ajouté avec succès.")
                st.rerun()

            except Exception as e:
                st.error(f"Erreur lors du traitement : {e}")

    st.write("---")
    st.write("### Liste de vos documents")

    docs = load_metadata(user_id)
    if not docs:
        st.caption("Aucun document n'a été importé pour le moment.")
    else:
        for i, d in enumerate(docs):
            with st.container(border=True):
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**{d['filename']}** | Matière : *{d['subject']}*")
                    st.caption(f"Professeur: {d.get('professor', '-')} | Date: {d.get('date', '-')} | Chunks: {d.get('chunks_count', 0)}")
                with col_btn:
                    if st.button("Supprimer", key=f"delete_{i}", use_container_width=True):
                        try:
                            if os.path.exists(d["file_path"]) and user_id in d["file_path"]:
                                os.remove(d["file_path"])
                            data = [x for x in load_metadata(user_id) if x["filename"] != d["filename"]]
                            save_metadata(data, user_id)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

with summary:
    st.subheader("Générateur de Fiche de Révision")
    docs = load_metadata(user_id)

    if not docs:
        st.warning("Veuillez importer au moins un document avant de générer un résumé.")
    else:
        subjects = sorted({d["subject"] for d in docs})
        
        with st.container(border=True):
            subject = st.selectbox("Sélectionnez la matière à résumer :", subjects, key="summary_subject")
            btn_generate = st.button("Générer la fiche", use_container_width=True)

        if btn_generate:
            with st.spinner("Réflexion..."):
                summary_text = generate_summary(subject, user_id=user_id)
                with st.container(border=True):
                    st.markdown(summary_text)

with quizzes:
    st.subheader("Entraînement et Quiz")
    docs = load_metadata(user_id)

    if not docs:
        st.warning("Veuillez importer au moins un document avant d'utiliser les quiz.")
    else:
        subjects = sorted({d["subject"] for d in docs})

        with st.container(border=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                subject = st.selectbox("Matière :", subjects, key="quiz_subject")
            with col2:
                number = st.number_input("Nombre de questions :", 1, 15, 5)
            with col3:
                qtype = st.selectbox("Type d'exercice :", ["QCM", "Réponse courte"])

            btn_quiz = st.button("Lancer le Quiz", use_container_width=True)

        if btn_quiz:
            with st.spinner("Réflexion..."):
                data, error = generate_quiz(subject, number, qtype, user_id=user_id)

            if error:
                st.error(error)
            else:
                st.session_state.quiz_data = data
                st.session_state.quiz_answers = {}
                st.rerun()

        if st.session_state.quiz_data:
            data = st.session_state.quiz_data
            st.write("---")

            with st.form("quiz_form"):
                answers = {}
                for i, q in enumerate(data):
                    with st.container(border=True):
                        st.markdown(f"**Question {i + 1}**")
                        st.write(q["question"])

                        if qtype == "QCM":
                            options = q.get("options", ["A", "B", "C", "D"])
                            answers[i] = st.radio("Sélectionnez une réponse :", options, key=f"quiz_{i}")
                        else:
                            answers[i] = st.text_input("Votre réponse :", key=f"quiz_{i}")

                check = st.form_submit_button("Valider l'ensemble des réponses", use_container_width=True)

            if check:
                score = 0
                st.write("---")
                st.subheader("Résultats")
                for i, q in enumerate(data):
                    user_ans = str(answers[i]).strip()
                    expected = str(q["correct_answer"]).strip()

                    if qtype == "QCM":
                        correct = user_ans[:1].upper() == expected[:1].upper()
                    else:
                        correct = user_ans.lower() == expected.lower()

                    with st.container(border=True):
                        if correct:
                            score += 1
                            st.success(f"Question {i + 1} : Correct")
                        else:
                            st.error(f"Question {i + 1} : Incorrect")

                        st.write(f"Réponse attendue : {expected}")
                        if q.get("explanation"):
                            st.info(f"Explication : {q['explanation']}")

                st.metric("Score final", f"{score} / {len(data)}")

st.divider()
st.caption("ReviseAI")
