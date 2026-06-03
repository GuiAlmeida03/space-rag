import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.config import GROQ_MODEL

OUT_OF_SCOPE_MSG = (
    "Não encontrei documentos relevantes na base de conhecimento sobre esse tema. "
    "Minha especialidade é a nova economia espacial: satélites, clima, agricultura "
    "geoespacial, monitoramento ambiental, desastres naturais e exploração espacial."
)

_SYSTEM_PROMPT = """Você é um assistente especializado na nova economia espacial.
Use APENAS o contexto abaixo para responder à pergunta do usuário.
Se o contexto for insuficiente para uma resposta completa, diga explicitamente.
Cite as fontes mencionadas no contexto.

Contexto recuperado:
{context}"""


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[{i}] Fonte: {os.path.basename(chunk['source'])} (score: {chunk['score']})\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def _build_chain():
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.2,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return prompt | llm | StrOutputParser()


def answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return {"response": OUT_OF_SCOPE_MSG, "chunks": []}

    context = _build_context(chunks)
    chain = _build_chain()
    response = chain.invoke({"context": context, "question": question})
    return {"response": response, "chunks": chunks}
