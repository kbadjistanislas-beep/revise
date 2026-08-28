"""
Pipeline RAG Complet - 2026
Version moderne sans classes, sans décorateurs
Utilise uniquement les modules stables de LangChain v1 et ses intégrations dédiées
"""

import os
import hashlib
from pathlib import Path
from typing import List, Tuple

# === INSTALLATION DES DEPENDANCES MODERNES ===
# pip install langchain langchain-core langchain-classic
# pip install langchain-chroma langchain-huggingface langchain-ollama
# pip install sentence-transformers pypdf
# pip install ragas (optionnel pour l'évaluation)

# === IMPORTS MODERNES LANGCHAIN V1 ===

# Core components (nouveau package de base)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Text splitters (package dédié)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector stores (intégration dédiée)
from langchain_chroma import Chroma

# Embeddings (intégration dédiée)
from langchain_huggingface import HuggingFaceEmbeddings

# LLM (intégration dédiée)
from langchain_ollama import OllamaLLM

# Document loaders (intégrations dédiées)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

# Retrievers (via langchain-classic pour la compatibilité ascendante)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# ============================================
# 1. CONFIGURATION
# ============================================

CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_device": "cpu",
    "chunk_size": 1024,
    "chunk_overlap": 200,
    "chunk_separators": ["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    "hybrid_search_weights": (0.5, 0.5),
    "top_k_initial": 20,
    "top_k_final": 5,
    "llm_model": "llama3.2",
    "llm_temperature": 0.1,
    "llm_streaming": True,
    "docs_directory": "./documents/",
    "chroma_persist_directory": "./chroma_db/",
    "collection_name": "rag_collection",
    "enable_agentic_retrieval": True,
    "max_retrieval_iterations": 3,
}

# ============================================
# 2. FONCTIONS DE CHARGEMENT DES DOCUMENTS
# ============================================


def load_documents(directory=None):
    """Charge tous les documents d'un répertoire"""
    directory = directory or CONFIG["docs_directory"]
    documents = []

    if not Path(directory).exists():
        raise FileNotFoundError(f"Directory {directory} not found")

    supported_extensions = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
    }

    for file_path in Path(directory).rglob("*"):
        if file_path.suffix in supported_extensions:
            try:
                loader_class = supported_extensions[file_path.suffix]
                loader = loader_class(str(file_path))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["file_hash"] = hashlib.md5(
                        str(file_path).encode()
                    ).hexdigest()

                documents.extend(docs)
                print(f"✓ Loaded: {file_path.name}")
            except Exception as e:
                print(f"✗ Error loading {file_path.name}: {e}")

    print(f"\nTotal documents loaded: {len(documents)}")
    return documents


def chunk_documents(documents):
    """Découpe les documents en chunks avec RecursiveCharacterTextSplitter"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        separators=CONFIG["chunk_separators"],
        length_function=len,
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"chunk_{i:06d}"
        chunk.metadata["chunk_length"] = len(chunk.page_content)
        chunk.metadata["chunk_hash"] = hashlib.md5(
            chunk.page_content.encode()
        ).hexdigest()

    print(f"✓ Created {len(chunks)} chunks")
    return chunks


# ============================================
# 3. FONCTIONS VECTOR STORE (CHROMADB)
# ============================================


def get_embeddings():
    """Retourne le modèle d'embeddings"""
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": CONFIG["embedding_device"]},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks):
    """Crée un index vectoriel avec ChromaDB"""
    print("🔄 Generating embeddings and creating ChromaDB index...")

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CONFIG["chroma_persist_directory"],
        collection_name=CONFIG["collection_name"],
    )

    print(f"✓ Vector store saved to {CONFIG['chroma_persist_directory']}")
    return vector_store


def load_vector_store():
    """Charge un index vectoriel existant depuis ChromaDB"""
    if Path(CONFIG["chroma_persist_directory"]).exists():
        embeddings = get_embeddings()
        vector_store = Chroma(
            persist_directory=CONFIG["chroma_persist_directory"],
            embedding_function=embeddings,
            collection_name=CONFIG["collection_name"],
        )
        print(f"✓ Vector store loaded from {CONFIG['chroma_persist_directory']}")
        return vector_store
    return None


def get_retriever_from_vector_store(vector_store, search_kwargs=None):
    if search_kwargs is None:
        search_kwargs = {"k": CONFIG["top_k_initial"]}
    return vector_store.as_retriever(search_kwargs=search_kwargs)


# ============================================
# 4. FONCTIONS DE RECHERCHE HYBRIDE
# ============================================


def create_hybrid_retriever(chunks, vector_store):
    """Crée un retriever hybride (BM25 + Vector)"""
    try:
        bm25_retriever = BM25Retriever.from_documents(chunks)
        bm25_retriever.k = CONFIG["top_k_initial"]

        vector_retriever = get_retriever_from_vector_store(vector_store)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=CONFIG["hybrid_search_weights"],
        )

        return ensemble_retriever

    except Exception as e:
        print(f"⚠️ Hybrid retriever not available: {e}")
        print("🔄 Falling back to vector-only retriever")
        return get_retriever_from_vector_store(vector_store)


def hybrid_retrieve(retriever, query):
    return retriever.invoke(query)


# ============================================
# 5. FONCTIONS LLM ET AGENTIQUES
# ============================================


def create_llm():
    """Crée le LLM avec Ollama"""
    return OllamaLLM(
        model=CONFIG["llm_model"],
        temperature=CONFIG["llm_temperature"],
        streaming=CONFIG["llm_streaming"],
        callbacks=[StreamingStdOutCallbackHandler()] if CONFIG["llm_streaming"] else [],
    )


def create_generation_prompt():
    """Crée le prompt pour la génération en utilisant ChatPromptTemplate"""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Vous êtes un assistant IA expert qui répond aux questions en utilisant 
        UNIQUEMENT le contexte fourni.
        
        Règles:
        1. Répondez de manière précise et détaillée
        2. Si le contexte ne contient pas l'information, dites que vous ne savez pas
        3. Citez les sources quand c'est pertinent
        4. Organisez votre réponse de façon claire""",
            ),
            # Utilisation de MessagesPlaceholder pour un historique optionnel
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "Contexte: \n{context}\n\nQuestion: {question}"),
        ]
    )


def evaluate_context_sufficiency(llm, query, documents):
    """Évalue si le contexte est suffisant pour répondre"""
    if not documents:
        return False

    context = "\n\n".join([doc.page_content[:500] for doc in documents[:3]])

    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Vous êtes un agent évaluateur. Déterminez si le contexte fourni est SUFFISANT 
        pour répondre à la question de l'utilisateur.
        
        Répondez UNIQUEMENT par:
        - "SUFFICIENT" si le contexte contient toutes les informations nécessaires
        - "INSUFFICIENT" si des informations cruciales manquent
        - "NEED_CLARIFICATION" si la question est ambiguë""",
            ),
            ("human", "Contexte: {context}\n\nQuestion: {question}"),
        ]
    )

    try:
        chain = evaluation_prompt | llm
        response = chain.invoke(
            {
                "context": context,
                "question": query,
                "chat_history": [],  # Pas d'historique pour l'évaluation
            }
        )

        response_text = response.strip().upper()
        return "SUFFICIENT" in response_text

    except Exception as e:
        print(f"⚠️ Evaluation error: {e}")
        return len(documents) >= CONFIG["top_k_final"]


def refine_query(llm, original_query, retrieved_docs):
    """Reformule la requête pour une meilleure recherche"""
    if not retrieved_docs:
        return original_query

    refinement_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Vous êtes un expert en recherche d'information.
        Reformulez la question suivante pour améliorer la recherche de documents.
        Ajoutez des synonymes et des concepts connexes.""",
            ),
            (
                "human",
                "Question originale: {query}\n\nDocuments déjà trouvés: {context}\n\nNouvelle question:",
            ),
        ]
    )

    try:
        context = "\n".join([doc.page_content[:200] for doc in retrieved_docs[:3]])
        chain = refinement_prompt | llm
        response = chain.invoke(
            {"query": original_query, "context": context, "chat_history": []}
        )
        return response.strip()
    except:
        return original_query


def agentic_retrieve(llm, hybrid_retriever, query):
    """Récupération itérative avec vérification de suffisance du contexte"""
    iteration = 0
    all_documents = []

    while iteration < CONFIG["max_retrieval_iterations"]:
        iteration += 1
        print(f"🔍 Retrieval iteration {iteration}")

        docs = hybrid_retrieve(hybrid_retriever, query)
        all_documents.extend(docs)

        is_sufficient = evaluate_context_sufficiency(
            llm, query, all_documents[: CONFIG["top_k_final"]]
        )

        if is_sufficient:
            print(f"✅ Context sufficient after {iteration} iterations")
            return all_documents[: CONFIG["top_k_final"]], True

        if iteration >= CONFIG["max_retrieval_iterations"]:
            print(f"⚠️ Max iterations reached ({CONFIG['max_retrieval_iterations']})")
            return all_documents[: CONFIG["top_k_final"]], False

        query = refine_query(llm, query, all_documents)
        print(f"🔄 Refined query: {query}")

    return all_documents[: CONFIG["top_k_final"]], False


# ============================================
# 6. FONCTIONS DE GÉNÉRATION
# ============================================


def generate_response(llm, prompt, question, documents):
    """Génère une réponse basée sur les documents"""
    context = "\n\n---\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
            for doc in documents
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": [],  # Pas d'historique pour les requêtes simples
        }
    )

    return response


# ============================================
# 7. FONCTIONS PRINCIPALES DU PIPELINE
# ============================================


def initialize_pipeline():
    """Initialise tous les composants du pipeline"""
    print("\n" + "=" * 60)
    print("🚀 INITIALIZING RAG PIPELINE (LangChain v1)")
    print("=" * 60)

    llm = create_llm()
    print("✓ LLM initialized")

    prompt = create_generation_prompt()
    print("✓ Prompt template created")

    return {
        "llm": llm,
        "prompt": prompt,
        "hybrid_retriever": None,
        "chunks": None,
        "vector_store": None,
        "is_indexed": False,
    }


def index_documents(pipeline_state):
    """Indexe tous les documents"""
    print("\n" + "=" * 60)
    print("🔄 STARTING INDEXATION")
    print("=" * 60)

    docs = load_documents()
    if not docs:
        raise ValueError("No documents loaded")

    chunks = chunk_documents(docs)
    pipeline_state["chunks"] = chunks

    vector_store = create_vector_store(chunks)
    pipeline_state["vector_store"] = vector_store

    hybrid_retriever = create_hybrid_retriever(chunks, vector_store)
    pipeline_state["hybrid_retriever"] = hybrid_retriever

    pipeline_state["is_indexed"] = True

    print("\n✅ Indexation complete!")
    print(f"   - {len(docs)} documents")
    print(f"   - {len(chunks)} chunks")
    print("=" * 60)

    return pipeline_state


def query_pipeline(pipeline_state, question):
    """Interroge le pipeline RAG"""
    if not pipeline_state["is_indexed"]:
        raise ValueError("Pipeline not indexed. Call index_documents() first.")

    print("\n" + "=" * 60)
    print(f"❓ Question: {question}")
    print("=" * 60)

    llm = pipeline_state["llm"]
    hybrid_retriever = pipeline_state["hybrid_retriever"]
    prompt = pipeline_state["prompt"]

    if CONFIG["enable_agentic_retrieval"]:
        documents, context_sufficient = agentic_retrieve(
            llm, hybrid_retriever, question
        )
        print(f"📊 Context sufficient: {context_sufficient}")
    else:
        documents = hybrid_retrieve(hybrid_retriever, question)
        print(f"📊 Retrieved {len(documents)} documents")

    print("\n🤖 Generating answer...")
    answer = generate_response(llm, prompt, question, documents)

    print("\n" + "=" * 60)
    print("✅ Answer generated")
    print("=" * 60)

    return answer, documents


def simple_query(pipeline_state, question):
    answer, _ = query_pipeline(pipeline_state, question)
    return answer


# ============================================
# 8. FONCTIONS UTILITAIRES ET MAIN
# ============================================


def create_sample_documents():
    """Crée des documents d'exemple"""
    os.makedirs(CONFIG["docs_directory"], exist_ok=True)

    sample_docs = [
        {
            "filename": "machine_learning.txt",
            "content": """
            Machine Learning is a subset of artificial intelligence that enables systems to learn from data.
            It uses algorithms to find patterns and make decisions with minimal human intervention.
            
            Key types of machine learning:
            1. Supervised Learning: Trained on labeled data
            2. Unsupervised Learning: Finds hidden patterns in unlabeled data
            3. Reinforcement Learning: Learns through trial and error
            
            Popular algorithms include neural networks, decision trees, and support vector machines.
            """,
        },
        {
            "filename": "rag_pipeline.txt",
            "content": """
            RAG (Retrieval-Augmented Generation) is a framework that combines retrieval systems with LLMs.
            
            The RAG pipeline consists of two main phases:
            1. Indexing: Documents are processed, chunked, and vectorized
            2. Retrieval & Generation: Query is processed, documents are retrieved, and answer is generated
            
            Key components of modern RAG in 2026:
            - Hybrid search (BM25 + Vector) for better recall
            - Agentic retrieval with iterative search
            - ChromaDB for vector storage
            """,
        },
    ]

    for doc in sample_docs:
        filepath = os.path.join(CONFIG["docs_directory"], doc["filename"])
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["content"])
        print(f"✓ Created sample: {doc['filename']}")

    print(f"\n📁 Sample documents created in {CONFIG['docs_directory']}")


def main():
    print("\n" + "█" * 60)
    print("█      RAG PIPELINE 2026 - LANGCHAIN V1             █")
    print("█" * 60)

    create_sample_documents()
    pipeline = initialize_pipeline()
    index_documents(pipeline)

    questions = [
        "Qu'est-ce que le RAG et comment fonctionne-t-il?",
        "Explique les différents types de machine learning",
    ]

    for question in questions:
        answer = simple_query(pipeline, question)
        print(f"\nFINAL ANSWER:\n{answer}\n")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Vérification des prérequis
    try:
        import langchain_core
        import langchain_chroma
        import langchain_ollama

        print("✓ Modern LangChain v1 dependencies installed")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install langchain langchain-core langchain-classic")
        print("pip install langchain-chroma langchain-huggingface langchain-ollama")
        print("pip install sentence-transformers pypdf")
        exit(1)

    # Vérification d'Ollama
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama service running")
        else:
            print("⚠️ Ollama not running. Please start with: ollama serve")
    except:
        print("⚠️ Could not connect to Ollama. Make sure it's running:")
        print("   ollama serve")
        print("   ollama pull llama3.2")

    main()
