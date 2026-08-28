import streamlit as st
import os, json, re, datetime
from pathlib import Path
import img2pdf
from google import genai
from dotenv import load_dotenv
from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

load_dotenv()

# ================= CONFIG =================

st.set_page_config(
    page_title="ReviseAI",
    page_icon="📚",
    layout="wide"
)

UPLOAD_FOLDER = "uploads"
DB_PATH = "./chroma_db"
META_FILE = "documents_metadata.json"
MODEL = "gemini-3.6-flash"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_PATH, exist_ok=True)

SYSTEM = """
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

# ================= GEMINI =================

@st.cache_resource
def gemini():
    key = None

    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except:
        pass

    key = key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key:
        raise RuntimeError(
            "❌ GEMINI_API_KEY est absente. "
            "Ajoute-la dans Streamlit → Settings → Secrets."
        )

    return genai.Client(api_key=key)


# ================= UTILS =================

def metadata():
    if not os.path.exists(META_FILE):
        return []
    try:
        with open(META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_meta(data):
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def collection(name):
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("_")
    if not name:
        name = "document"
    if name[0].isdigit():
        name = "doc_" + name
    return name[:512]


@st.cache_resource
def embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


@st.cache_resource
def tokenizer():
    return AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )


# ================= SALUTATIONS =================

def greeting(text):
    text = re.sub(r"[!?.,;:]", "", text.lower()).strip()

    greetings = [
        "bonjour", "salut", "hello", "hi",
        "hey", "coucou", "bonsoir", "bjr", "slt"
    ]

    return text in greetings or text in [
        f"{x} reviseai" for x in greetings
    ]


def greeting_answer(text):
    if "bonsoir" in text.lower():
        return "Bonsoir 👋 Je suis ReviseAI ! Prêt à réviser ? 📚"

    if "coucou" in text.lower():
        return "Coucou 👋 ! Que veux-tu réviser aujourd'hui ? 📚"

    return (
        "Bonjour 👋 ! Je suis ReviseAI.\n\n"
        "Je peux t'aider à :\n"
        "- 📚 comprendre tes cours\n"
        "- ✏️ résoudre des exercices\n"
        "- 📝 résumer tes documents\n"
        "- 🧠 créer des quiz"
    )


# ================= DOCUMENTS =================

def clean(text):
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
                    page_content=clean(page.get_text()),
                    metadata={"page": i + 1}
                )
                for i, page in enumerate(pdf)
                if page.get_text().strip()
            ]
            pdf.close()
            return docs
        except:
            return PyPDFLoader(path).load()

    if ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()

    if ext in [".docx", ".doc"]:
        return Docx2txtLoader(path).load()

    raise ValueError("Format non supporté")


def process(path, meta):
    docs = load_document(path)

    for d in docs:
        d.page_content = clean(d.page_content)
        d.metadata.update(meta)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    name = collection(
        f"{meta['subject']}_{meta['filename']}"
    )

    Chroma.from_documents(
        chunks,
        embeddings(),
        persist_directory=DB_PATH,
        collection_name=name
    )

    return len(chunks)


def search(query, k=8):
    results = []

    for d in metadata():
        try:
            store = Chroma(
                persist_directory=DB_PATH,
                embedding_function=embeddings(),
                collection_name=collection(
                    f"{d['subject']}_{d['filename']}"
                )
            )
            results += store.similarity_search(query, k=k)
        except:
            pass

    return results[:12]


def context(chunks):
    return "\n\n".join(
        f"[{x.metadata.get('filename', 'Document')} "
        f"- {x.metadata.get('subject', 'Matière')} "
        f"- page {x.metadata.get('page', '')}]\n"
        f"{x.page_content}"
        for x in chunks
    )


# ================= GEMINI CHAT =================

def ask(prompt, ctx=""):
    text = f"""
CONTEXTE DU COURS :
{ctx}

QUESTION :
{prompt}
""" if ctx else prompt

    try:
        previous = st.session_state.get("interaction_id")

        request = {
            "model": MODEL,
            "system_instruction": SYSTEM,
            "input": text,
            "generation_config": {
                "max_output_tokens": 4096
            }
        }

        if previous:
            request["previous_interaction_id"] = previous

        result = gemini().interactions.create(**request)

        st.session_state.interaction_id = result.id

        return result.output_text or "Je n'ai pas reçu de réponse."

    except Exception as e:
        return f"❌ Erreur Gemini : {e}"


# ================= RÉSUMÉ =================

def summarize(subject):
    chunks = search(
        f"cours {subject} définitions formules théorèmes méthodes",
        12
    )

    if not chunks:
        return "❌ Aucun contenu trouvé."

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
{context(chunks)}
"""

    return ask(prompt)


# ================= QUIZ =================

def quiz(subject, n, qtype):
    chunks = search(
        f"cours {subject} notions définitions exemples",
        15
    )

    if not chunks:
        return None, "❌ Aucun contenu trouvé."

    if qtype == "QCM":
        format_json = """
{
 "question":"...",
 "options":["A. ...","B. ...","C. ...","D. ..."],
 "correct_answer":"A",
 "explanation":"..."
}
"""
    else:
        format_json = """
{
 "question":"...",
 "options":[],
 "correct_answer":"...",
 "explanation":"..."
}
"""

    prompt = f"""
Génère {n} questions de révision sur {subject}.

Retourne UNIQUEMENT un JSON valide sous cette forme :
[{format_json}]

Utilise uniquement le contenu du cours.

COURS :
{context(chunks)}
"""

    response = ask(prompt)

    try:
        match = re.search(r"\[.*\]", response, re.S)
        return json.loads(match.group() if match else response), None
    except Exception as e:
        return None, f"❌ Quiz invalide : {e}"


# ================= SESSION =================

for key, default in {
    "messages": [],
    "interaction_id": None,
    "quiz_data": None,
    "quiz_answers": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ================= SIDEBAR =================

with st.sidebar:
    st.title("🤖 ReviseAI")

    docs = metadata()
    st.metric("📚 Documents", len(docs))

    st.toggle(
        "🔎 Recherche Web",
        value=False,
        key="web_enabled"
    )

    if st.button(
        "🆕 Nouvelle conversation",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.session_state.interaction_id = None
        st.rerun()

    st.caption("ReviseAI — assistant de révision")


# ================= TABS =================

chat, documents, summary, quizzes = st.tabs([
    "💬 Chat",
    "📁 Documents",
    "📝 Résumé",
    "🧠 Quiz"
])


# ================= CHAT =================

with chat:
    st.header("💬 ReviseAI")
    st.caption("Ton assistant intelligent de révision")

    if not st.session_state.messages:
        st.info(
            "👋 Bonjour ! Je peux t'aider à comprendre "
            "tes cours, résoudre des exercices, résumer "
            "tes documents et créer des quiz."
        )

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Pose ta question à ReviseAI...")

    if prompt:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):

                if greeting(prompt):
                    answer = greeting_answer(prompt)
                else:
                    chunks = search(prompt)
                    answer = ask(
                        prompt,
                        context(chunks)
                    )

            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


# ================= DOCUMENTS =================

with documents:
    st.header("📁 Mes documents")

    with st.form("upload"):
        subject = st.text_input(
            "📚 Matière *",
            placeholder="Mathématiques"
        )

        professor = st.text_input(
            "👨‍🏫 Professeur"
        )

        date = st.date_input(
            "📅 Date",
            datetime.date.today()
        )

        file = st.file_uploader(
            "📎 Document",
            type=[
                "pdf",
                "docx",
                "doc",
                "txt",
                "jpg",
                "jpeg",
                "png"
            ]
        )

        submit = st.form_submit_button(
            "➕ Ajouter",
            use_container_width=True
        )

    if submit:

        if not subject or not file:
            st.error("❌ Matière et fichier obligatoires.")

        else:
            is_image = file.type.startswith("image/")

            if is_image:
                path = os.path.join(
                    UPLOAD_FOLDER,
                    Path(file.name).stem + ".pdf"
                )

                with open(path, "wb") as f:
                    f.write(img2pdf.convert(file.getvalue()))

                filename = Path(path).name

            else:
                filename = file.name
                path = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

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
                with st.spinner("📚 Analyse du document..."):
                    count = process(path, meta)

                meta["chunks_count"] = count

                data = metadata()
                data.append(meta)
                save_meta(data)

                st.success(
                    f"✅ Document ajouté : {count} morceaux indexés."
                )

            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    st.divider()

    for i, d in enumerate(metadata()):

        with st.expander(
            f"📄 {d['filename']} — {d['subject']}"
        ):

            st.write(
                f"**Professeur :** {d.get('professor', '-')}"
            )

            st.write(
                f"**Date :** {d.get('date', '-')}"
            )

            st.write(
                f"**Chunks :** {d.get('chunks_count', 0)}"
            )

            if st.button(
                "🗑️ Supprimer",
                key=f"delete_{i}"
            ):

                try:
                    if os.path.exists(d["file_path"]):
                        os.remove(d["file_path"])

                    data = [
                        x for x in metadata()
                        if x["filename"] != d["filename"]
                    ]

                    save_meta(data)
                    st.rerun()

                except Exception as e:
                    st.error(str(e))


# ================= SUMMARY =================

with summary:
    st.header("📝 Résumé intelligent")

    docs = metadata()

    if not docs:
        st.info("📭 Ajoute d'abord un document.")

    else:
        subjects = sorted({
            d["subject"] for d in docs
        })

        subject = st.selectbox(
            "📚 Matière",
            subjects,
            key="summary_subject"
        )

        if st.button(
            "📝 Générer le résumé",
            use_container_width=True
        ):
            with st.spinner("Analyse du cours..."):
                st.markdown(
                    summarize(subject)
                )


# ================= QUIZ =================

with quizzes:
    st.header("🧠 Quiz personnalisé")

    docs = metadata()

    if not docs:
        st.info("📭 Ajoute d'abord un document.")

    else:

        subjects = sorted({
            d["subject"] for d in docs
        })

        c1, c2, c3 = st.columns(3)

        with c1:
            subject = st.selectbox(
                "📚 Matière",
                subjects,
                key="quiz_subject"
            )

        with c2:
            number = st.number_input(
                "🔢 Questions",
                1,
                15,
                5
            )

        with c3:
            qtype = st.selectbox(
                "📝 Type",
                ["QCM", "Réponse courte"]
            )

        if st.button(
            "🧠 Générer le quiz",
            use_container_width=True
        ):

            with st.spinner(
                "Gemini prépare ton quiz..."
            ):

                data, error = quiz(
                    subject,
                    number,
                    qtype
                )

            if error:
                st.error(error)
            else:
                st.session_state.quiz_data = data
                st.session_state.quiz_answers = {}
                st.rerun()

        if st.session_state.quiz_data:

            data = st.session_state.quiz_data

            st.divider()

            with st.form("quiz_form"):

                answers = {}

                for i, q in enumerate(data):

                    st.markdown(
                        f"### Question {i + 1}"
                    )

                    st.write(
                        q["question"]
                    )

                    if qtype == "QCM":

                        options = q.get(
                            "options",
                            ["A", "B", "C", "D"]
                        )

                        answers[i] = st.radio(
                            "Réponse",
                            options,
                            key=f"quiz_{i}"
                        )

                    else:

                        answers[i] = st.text_input(
                            "Ta réponse",
                            key=f"quiz_{i}"
                        )

                check = st.form_submit_button(
                    "✅ Vérifier"
                )

            if check:

                score = 0

                st.divider()

                for i, q in enumerate(data):

                    user = str(
                        answers[i]
                    ).strip()

                    expected = str(
                        q["correct_answer"]
                    ).strip()

                    if qtype == "QCM":
                        correct = (
                            user[:1].upper()
                            == expected[:1].upper()
                        )
                    else:
                        correct = (
                            user.lower()
                            == expected.lower()
                        )

                    if correct:
                        score += 1
                        st.success(
                            f"✅ Question {i + 1} : Correct"
                        )
                    else:
                        st.error(
                            f"❌ Question {i + 1} : Incorrect"
                        )

                    st.write(
                        f"**Réponse :** {expected}"
                    )

                    if q.get("explanation"):
                        st.info(
                            f"💡 {q['explanation']}"
                        )

                st.metric(
                    "🏆 Score",
                    f"{score}/{len(data)}"
                )

st.divider()
st.caption("ReviseAI — un avenir prometteur 🚀")
