# app.py
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente antes de qualquer inicialização
load_dotenv()

from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from langfuse import observe

# =====================================================
# Configurações
# =====================================================

CONFIDENCE_THRESHOLD = 0.80

FORBIDDEN_PATTERNS = [
    "ignore regras",
    "ignore instruções",
    "responda fora",
    "conhecimento geral",
    "como você sabe",
    "bypass",
]

# =====================================================
# Prompt loader
# =====================================================

def load_prompt_template() -> str:
    prompt_path = os.getenv("PROMPT_FILE")

    if not prompt_path:
        raise EnvironmentError("PROMPT_FILE não definido")

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt não encontrado: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

PROMPT_TEMPLATE = load_prompt_template()

# =====================================================
# Inicialização LLM, Embeddings e Pinecone
# =====================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("sisu")

# =====================================================
# Estado do LangGraph
# =====================================================

class ChatState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    confidence: float
    answer: str

# =====================================================
# Nós do Grafo
# =====================================================

@observe()
def guardrails_input(state: ChatState, config=None):
    q = state["question"].lower()

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in q:
            state["answer"] = (
                "Não posso atender a essa solicitação. "
                "Este assistente responde apenas com base em informações oficiais do SISU."
            )
            return state

    return state


@observe()
def retrieve_from_pinecone(state: ChatState, config=None):
    query_embedding = embeddings.embed_query(state["question"])

    result = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
    )

    docs, scores = [], []

    for match in result["matches"]:
        docs.append(
            Document(
                page_content=match["metadata"]["text"],
                metadata=match["metadata"]
            )
        )
        scores.append(match["score"])

    state["retrieved_docs"] = docs
    state["confidence"] = max(scores) if scores else 0.0

    return state


@observe()
def confidence_gate(state: ChatState, config=None):
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        state["answer"] = (
            "Não encontrei informações suficientes na base oficial do SISU "
            "para responder com segurança.\n\n"
            "Consulte o site oficial:\n"
            "https://www.gov.br/mec/pt-br/acesso-a-informacao/perguntas-frequentes/sisu"
        )
    
    return state


@observe()
def generate_answer(state: ChatState, config=None):
    if state.get("answer"):
        return state

    context = "\n\n".join(
        [doc.page_content for doc in state["retrieved_docs"]]
    )

    prompt = (
        PROMPT_TEMPLATE
        .replace("{{context}}", context)
        .replace("{{question}}", state["question"])
    )

    response = llm.invoke(prompt)
    state["answer"] = response.content

    return state

# =====================================================
# Construção do LangGraph
# =====================================================

graph = StateGraph(ChatState)

graph.add_node("guardrails", guardrails_input)
graph.add_node("retrieve", retrieve_from_pinecone)
graph.add_node("confidence", confidence_gate)
graph.add_node("generate", generate_answer)

graph.set_entry_point("guardrails")

graph.add_edge("guardrails", "retrieve")
graph.add_edge("retrieve", "confidence")
graph.add_edge("confidence", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# =====================================================
# Entry point (API / Webhook / WhatsApp)
# =====================================================

@observe()
def ask_sisu(question: str):
    result = app.invoke({"question": question})

    return {
        "answer": result["answer"],
        "confidence": result.get("confidence", 0)
    }

# =====================================================
# Execução local (teste)
# =====================================================

if __name__ == "__main__":
    print("--- Assistente SISU Iniciado ---")
    print("(Digite 'sair' para encerrar)")
    
    while True:
        try:
            user_input = input("\nPergunta: ")
            if user_input.lower() in ["sair", "exit", "quit"]:
                break
            
            if not user_input.strip():
                continue

            response = ask_sisu(user_input)
            print(f"\nResposta: {response['answer']}")
            print(f"Confiança: {response['confidence']:.2f}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erro: {e}")
