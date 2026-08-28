import streamlit as st
import os
import json
import datetime
import re
import warnings
import logging
from pathlib import Path
from PIL import Image
import img2pdf
from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from transformers import AutoTokenizer
from dotenv import load_dotenv
load_dotenv()

# Configuration de la page avec icônes Font Awesome
st.set_page_config(
    page_title="ReviseAI",
    page_icon="📚",  # on garde un emoji pour l'onglet navigateur, mais on utilisera Font Awesome dans l'app
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajouter Font Awesome CDN
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

UPLOAD_FOLDER = "uploads"
METADATA_FILE = "documents_metadata.json"
CHROMA_DB_PATH = "./chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

GEMINI_MODEL = "gemini-3.6-flash"

@st.cache_resource(show_spinner=False)
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Clé Gemini absente. Ajoute GEMINI_API_KEY dans ton fichier .env."
        )
    return genai.Client(api_key=api_key)

SYSTEM_INSTRUCTION = """
Tu es ReviseAI, un assistant intelligent et naturel.
TON STYLE DOIT ÊTRE PROCHE D'UN ASSISTANT GEMINI MODERNE :
- Comprends le contexte des messages précédents.
- Réponds directement à la demande.
- Ne répète pas inutilement la question.
- Utilise un français naturel, clair et moderne.
- Adapte la longueur à la question.
- Pour une question simple, réponds simplement.
- Pour une question complexe, raisonne étape par étape mais ne montre pas tes pensées internes.
- Utilise Markdown quand cela améliore la lisibilité.
- Utilise des listes, tableaux et titres lorsque c'est utile.
- Pour le code, utilise toujours des blocs de code avec le langage.
- Pour les mathématiques et les sciences, utilise LaTeX proprement.
- Si une information est incertaine, indique-le au lieu d'inventer.
- Si l'utilisateur se trompe, corrige-le gentiment et explique pourquoi.
- Tu peux répondre aux questions générales même si aucun document n'est chargé.
MODE DOCUMENT :
Quand un CONTEXTE DE DOCUMENT est fourni :
- Utilise-le comme source prioritaire pour les questions qui concernent les documents de l'utilisateur.
- Ne prétends jamais qu'une information vient du document si elle n'y apparaît pas.
- Si le contexte est insuffisant, dis clairement que le document ne permet pas de répondre complètement.
- Tu peux compléter avec tes connaissances générales uniquement si la question le demande ou si cela est clairement utile.
- Ne recopie pas de gros passages du document : explique-les.
MODE PROFESSEUR :
Quand l'utilisateur demande une explication de cours :
1. Donne l'idée principale.
2. Explique simplement.
3. Donne un exemple si utile.
4. Termine par un petit résumé ou une méthode à retenir.
MODE EXERCICE :
Quand l'utilisateur donne un exercice :
- Identifie ce qui est demandé.
- Donne une méthode claire.
- Fais les calculs correctement.
- Donne la réponse finale clairement.
- N'invente aucune donnée absente de l'énoncé.
IMPORTANT :
Ne commence pas systématiquement par "Bien sûr !".
Ne termine pas systématiquement par une question.
Évite les phrases artificielles et répétitives.
"""

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\text\s*", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    cleaned = cleaned.strip("_")
    if cleaned and cleaned[0].isdigit():
        cleaned = "doc_" + cleaned
    if not cleaned:
        cleaned = "document"
    return cleaned[:512]

def load_metadata():
    if not os.path.exists(METADATA_FILE):
        return []
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_metadata(metadata):
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def convert_image_to_pdf(image_file):
    try:
        base_name = os.path.splitext(image_file.name)[0]
        pdf_filename = f"{base_name}.pdf"
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        with open(pdf_path, "wb") as f:
            f.write(img2pdf.convert(image_file.getvalue()))
        return pdf_path, pdf_filename
    except Exception as e:
        st.error(f"Erreur de conversion : {e}")
        return None, None

def load_pdf_with_fitz(file_path):
    try:
        import fitz
        doc = fitz.open(file_path)
        documents = []
        from langchain_core.documents import Document
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text())
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_path,
                            "page": page_number,
                        },
                    )
                )
        doc.close()
        return documents
    except Exception:
        return None

def process_document(file_path, metadata):
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        documents = load_pdf_with_fitz(file_path)
        if not documents:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
    elif ext == ".txt":
        documents = TextLoader(
            file_path,
            encoding="utf-8",
        ).load()
    elif ext in [".docx", ".doc"]:
        documents = Docx2txtLoader(file_path).load()
    else:
        raise ValueError(f"Format non supporté : {ext}")
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        doc.metadata.update(metadata)
    documents = [
        doc for doc in documents
        if doc.page_content.strip()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
        ],
    )
    chunks = splitter.split_documents(documents)
    tokenizer = get_tokenizer()
    tokenizer.encode("ReviseAI", add_special_tokens=False)
    collection_name = clean_collection_name(
        f"{metadata['subject']}_{metadata['filename']}"
    )
    embeddings = get_embeddings()
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name=collection_name,
    )
    return len(chunks)

def search_chunks(query, k=8):
    metadata = load_metadata()
    if not metadata:
        return []
    embeddings = get_embeddings()
    results = []
    for doc in metadata:
        try:
            collection_name = clean_collection_name(
                f"{doc['subject']}_{doc['filename']}"
            )
            vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            found = vector_store.similarity_search(
                query,
                k=k,
            )
            results.extend(found)
        except Exception:
            continue
    seen = set()
    unique = []
    for item in results:
        key = (
            item.page_content.strip(),
            item.metadata.get("filename", ""),
            item.metadata.get("page", ""),
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique[:12]

def build_context(chunks):
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("filename", "Document")
        subject = chunk.metadata.get("subject", "Matière")
        page = chunk.metadata.get("page")
        location = f"{source} — {subject}"
        if page:
            location += f" — page {page}"
        parts.append(
            f"[SOURCE {i}: {location}]\n"
            f"{chunk.page_content}"
        )
    return "\n\n".join(parts)

def ask_gemini(
    user_message,
    context="",
    use_web=False,
):
    client = get_gemini_client()
    if context:
        final_input = f"""
CONTEXTE DES DOCUMENTS DE L'UTILISATEUR
---------------------------------------
{context}
FIN DU CONTEXTE
---------------------------------------
QUESTION DE L'UTILISATEUR
{user_message}
Réponds naturellement. Si la question concerne les documents,
utilise le contexte ci-dessus en priorité.
"""
    else:
        final_input = user_message
    request = {
        "model": GEMINI_MODEL,
        "system_instruction": SYSTEM_INSTRUCTION,
        "input": final_input,
        "generation_config": {
            "max_output_tokens": 4096,
        },
    }
    if use_web:
        request["tools"] = [{"type": "google_search"}]
    previous_id = st.session_state.get("gemini_interaction_id")
    if previous_id:
        request["previous_interaction_id"] = previous_id
    try:
        interaction = client.interactions.create(**request)
        st.session_state.gemini_interaction_id = interaction.id
        answer = interaction.output_text
        if not answer:
            return "Je n'ai pas reçu de réponse exploitable de Gemini."
        return answer.strip()
    except Exception as e:
        error = str(e)
        if "API key" in error or "api_key" in error.lower():
            return (
                " La clé Gemini semble incorrecte ou absente. "
                "Vérifie GEMINI_API_KEY dans ton fichier .env."
            )
        return f" Gemini n'a pas pu répondre : {error}"

def new_conversation():
    st.session_state.messages = []
    st.session_state.gemini_interaction_id = None

def summarize_document(subject, use_web=False):
    chunks = search_chunks(
        f"cours complet {subject} définitions théorèmes formules méthodes exemples",
        k=10,
    )
    if not chunks:
        return " Aucun contenu trouvé pour cette matière."
    context = build_context(chunks)
    prompt = f"""
Résume ce cours de manière pédagogique.
Matière : {subject}
Je veux :
1. Les notions essentielles.
2. Les définitions importantes.
3. Les formules / théorèmes importants.
4. Les méthodes à retenir.
5. Les erreurs fréquentes à éviter.
6. Un mini résumé final.
Voici le contenu du cours :
{context}
"""
    return ask_gemini(
        prompt,
        context="",
        use_web=use_web,
    )

# --- NOUVELLE FONCTION POUR LE QUIZ ---
def generate_quiz(subject, num_questions=5, question_type="qcm"):
    chunks = search_chunks(f"cours {subject} notions définitions exemples", k=15)
    if not chunks:
        return None, "Aucun contenu trouvé pour cette matière."
    
    context = build_context(chunks)
    
    if question_type == "qcm":
        type_desc = "Questions à choix multiple avec 4 propositions (A, B, C, D)."
    else:
        type_desc = "Questions à réponse courte (une phrase ou un mot)."
    
    prompt = f"""
Génère {num_questions} questions de révision sur le cours de {subject}.
Type demandé : {type_desc}

Consignes :
- Les questions doivent porter sur les concepts clés, définitions, formules, méthodes.
- Pour chaque question, fournis la réponse correcte et une explication brève.
- Formate la sortie en JSON valide comme suit :
[
  {{
    "question": "texte de la question",
    "options": ["A. ...", "B. ...", "C. ...", "D. ..."]  // seulement si QCM
    "correct_answer": "A" ou "réponse texte",
    "explanation": "explication"
  }}
]
- Ne mets que le JSON, sans texte autour.

Contenu du cours :
{context}
"""
    response = ask_gemini(prompt, context="", use_web=False)
    try:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            json_str = match.group()
            quiz_data = json.loads(json_str)
            return quiz_data, None
        else:
            quiz_data = json.loads(response)
            return quiz_data, None
    except Exception as e:
        return None, f"Erreur de parsing : {e}\nRéponse brute : {response}"

# Initialisation de session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_interaction_id" not in st.session_state:
    st.session_state.gemini_interaction_id = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}

# Sidebar
with st.sidebar:
    st.markdown('<h1 style="display:inline;"><i class="fas fa-robot" style="margin-right:10px;"></i>ReviseAI</h1>', unsafe_allow_html=True)
    metadata = load_metadata()
    st.metric("Documents", len(metadata))  # on garde un emoji pour metric car Streamlit le gère bien, mais on peut aussi utiliser une icône HTML
    if metadata:
        subjects = sorted(
            set(d.get("subject", "Autre") for d in metadata)
        )
        st.metric("Matières", len(subjects))
    st.divider()
    web_enabled = st.toggle(
        " Recherche Web",
        value=False,
        help="Autorise Gemini à utiliser Google Search pour les informations actuelles.",
    )
    st.divider()
    if st.button(
        " Nouvelle conversation",
        use_container_width=True,
    ):
        new_conversation()
        st.rerun()
    st.divider()
    st.caption("ReviseAI — assistant de révision")

# Onglets
tab_chat, tab_docs, tab_summary, tab_quiz = st.tabs(
    [" Chat", "Documents", " Résumé", " Quiz"]
)

with tab_chat:
    st.markdown('<h2><i class="fas fa-comments"></i> ReviseAI</h2>', unsafe_allow_html=True)
    st.caption(
        "Un assistant de révision conversationnel"
    )
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px;">
                <i class="fas fa-hand-peace" style="font-size:24px; margin-right:10px;"></i>
                Bonjour !<br>
                Je peux t'aider à :<br>
                <i class="fas fa-graduation-cap"></i> comprendre tes cours ;<br>
                <i class="fas fa-pencil-alt"></i> résoudre des exercices ;<br>
                <i class="fas fa-file-alt"></i> résumer tes documents ;<br>
                <i class="fas fa-lightbulb"></i> expliquer une notion simplement ;<br>
                <i class="fas fa-globe"></i> rechercher des informations actuelles si la recherche Web est activée.
            </div>
            """,
            unsafe_allow_html=True
        )
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input(
        "Pose ta question à ReviseAI..."
    )
    if prompt:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Réflection..."):
                chunks = search_chunks(prompt, k=8)
                context = build_context(chunks)
                answer = ask_gemini(
                    prompt,
                    context=context,
                    use_web=web_enabled,
                )
            st.markdown(answer)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

with tab_docs:
    st.markdown('<h2><i class="fas fa-folder-open"></i> Mes documents</h2>', unsafe_allow_html=True)
    with st.form("upload_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            subject = st.text_input(
                " Matière *",
                placeholder="Mathématiques",
            )
            professor = st.text_input(
                " Professeur",
                placeholder="Nom du professeur",
            )
        with col2:
            date = st.date_input(
                " Date",
                datetime.date.today(),
            )
            uploaded_file = st.file_uploader(
                "📎 Fichier",
                type=[
                    "pdf",
                    "docx",
                    "doc",
                    "txt",
                    "jpg",
                    "jpeg",
                    "png",
                ],
            )
        submitted = st.form_submit_button(
            "Ajouter le document",
            use_container_width=True,
        )
    if submitted:
        if not subject or not uploaded_file:
            st.error(" La matière et le fichier sont obligatoires.")
        else:
            is_image = (
                uploaded_file.type.startswith("image/")
                or uploaded_file.name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            )
            if is_image:
                with st.spinner(" Conversion de l'image en PDF..."):
                    file_path, actual_filename = convert_image_to_pdf(
                        uploaded_file
                    )
                if not file_path:
                    st.stop()
            else:
                actual_filename = uploaded_file.name
                file_path = os.path.join(
                    UPLOAD_FOLDER,
                    actual_filename,
                )
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            metadata = {
                "filename": actual_filename,
                "subject": subject,
                "professor": professor or "Non spécifié",
                "date": date.strftime("%Y-%m-%d"),
                "upload_date": datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
                "file_path": file_path,
                "is_image": is_image,
                "original_name": (
                    uploaded_file.name
                    if is_image
                    else None
                ),
            }
            with st.spinner(
                " Analyse et indexation du document..."
            ):
                try:
                    chunks_count = process_document(
                        file_path,
                        metadata,
                    )
                    metadata["chunks_count"] = chunks_count
                    all_metadata = load_metadata()
                    all_metadata.append(metadata)
                    save_metadata(all_metadata)
                    st.success(
                        f" {actual_filename} ajouté — "
                        f"{chunks_count} morceaux indexés."
                    )
                except Exception as e:
                    st.error(
                        f"Impossible de traiter le document : {e}"
                    )
    st.divider()
    metadata = load_metadata()
    if not metadata:
        st.info(" Aucun document chargé.")
    else:
        for index, doc in enumerate(metadata):
            icon = "" if doc.get("is_image") else ""
            title = (
                f"{icon} {doc['filename']} "
                f"— {doc.get('subject', 'Autre')}"
            )
            with st.expander(title):
                st.write(
                    f" **Matière :** {doc.get('subject', '—')}"
                )
                st.write(
                    f" **Professeur :** "
                    f"{doc.get('professor', '—')}"
                )
                st.write(
                    f"**Date :** {doc.get('date', '—')}"
                )
                st.write(
                    f" **Chunks :** "
                    f"{doc.get('chunks_count', 0)}"
                )
                if st.button(
                    "Supprimer",
                    key=f"delete_{index}",
                ):
                    try:
                        if os.path.exists(
                            doc["file_path"]
                        ):
                            os.remove(doc["file_path"])
                        all_metadata = [
                            d for d in load_metadata()
                            if d["filename"]
                            != doc["filename"]
                        ]
                        save_metadata(all_metadata)
                        st.success(" Document supprimé.")
                        st.rerun()
                    except Exception as e:
                        st.error(
                            f" Erreur lors de la suppression : {e}"
                        )

with tab_summary:
    st.markdown('<h2><i class="fas fa-file-signature"></i> Résumé intelligent</h2>', unsafe_allow_html=True)
    metadata = load_metadata()
    if not metadata:
        st.info(
            " Ajoute d'abord un document pour générer un résumé."
        )
    else:
        subjects = sorted(
            set(
                d.get("subject", "Autre")
                for d in metadata
            )
        )
        selected_subject = st.selectbox(
            " Choisis une matière",
            subjects,
        )
        if st.button(
            "Générer le résumé",
            use_container_width=True,
        ):
            with st.spinner(
                "Analyse du cours..."
            ):
                summary = summarize_document(
                    selected_subject,
                    use_web=False,
                )
            st.markdown(summary)

# --- ONGLET QUIZ ---
with tab_quiz:
    st.markdown('<h2><i class="fas fa-brain"></i> Quiz personnalisé</h2>', unsafe_allow_html=True)
    metadata = load_metadata()
    if not metadata:
        st.info("📭 Ajoute d'abord des documents pour générer un quiz.")
    else:
        subjects = sorted(set(d.get("subject", "Autre") for d in metadata))
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            selected_subject = st.selectbox(" Matière", subjects, key="quiz_subject")
        with col2:
            num_q = st.number_input(" Nombre de questions", min_value=1, max_value=15, value=5, step=1, key="quiz_num")
        with col3:
            q_type = st.radio(" Type", ["QCM", "Réponse courte"], index=0, key="quiz_type")
        
        if st.button("Générer le quiz", use_container_width=True):
            with st.spinner(" Gemini prépare les questions..."):
                quiz_data, error = generate_quiz(selected_subject, num_q, q_type.lower())
                if error:
                    st.error(error)
                else:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_answers = {}
                    st.rerun()
        
        if st.session_state.get("quiz_data"):
            quiz_data = st.session_state.quiz_data
            st.divider()
            st.subheader(" Réponds aux questions")
            
            with st.form(key="quiz_form"):
                answers = {}
                for i, q in enumerate(quiz_data):
                    st.markdown(f"**Q{i+1}.** {q['question']}")
                    if q_type == "QCM":
                        options = q.get("options", [])
                        if not options:
                            options = ["A", "B", "C", "D"]
                        default_val = st.session_state.quiz_answers.get(i, None)
                        answer = st.radio(
                            "Choisis une option",
                            options,
                            index=options.index(default_val) if default_val in options else 0,
                            key=f"q_{i}",
                            label_visibility="collapsed"
                        )
                        answers[i] = answer
                    else:
                        default_text = st.session_state.quiz_answers.get(i, "")
                        answer = st.text_input("Ta réponse", value=default_text, key=f"q_{i}")
                        answers[i] = answer.strip()
                    st.divider()
                
                submitted = st.form_submit_button(" Vérifier mes réponses")
            
            if submitted:
                correct_count = 0
                st.subheader("Résultats")
                for i, q in enumerate(quiz_data):
                    user_ans = answers.get(i, "")
                    expected = q.get("correct_answer", "")
                    user_ans_clean = user_ans.strip()
                    expected_clean = expected.strip()
                    if q_type == "QCM":
                        is_correct = (user_ans_clean == expected_clean)
                    else:
                        is_correct = (user_ans_clean.lower() == expected_clean.lower())
                    
                    if is_correct:
                        correct_count += 1
                        st.success(f" Question {i+1} : Correct !")
                    else:
                        st.error(f" Question {i+1} : Incorrect.")
                    st.markdown(f"**Réponse attendue :** {expected_clean}")
                    if q.get("explanation"):
                        st.info(f" {q['explanation']}")
                    st.divider()
                
                st.metric("Score", f"{correct_count}/{len(quiz_data)}")
                if st.button(" Nouveau quiz", use_container_width=True):
                    st.session_state.quiz_data = None
                    st.rerun()

st.divider()
st.caption(
    
    "ReviseAI , un avenir prometeur"
)