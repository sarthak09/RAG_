def build_system_prompt() -> str:
    return """You are a precise and reliable research assistant.

Your job is to answer questions strictly using the provided context chunks retrieved from academic documents.

Rules you must follow:
- Answer only from the provided context. Do not use prior knowledge.
- If the context does not contain enough information to answer, respond with "The provided context does not contain sufficient information to answer this question."
- Be concise and factual.
- Do not speculate or infer beyond what is explicitly stated in the context.
- Always respond in the exact JSON format specified.

Output format — you must return valid JSON and nothing else:
{
  "answer": "<your answer here>",
  "used_chunk_ids": ["<chunk_id_1>", "<chunk_id_2>"]
}

The used_chunk_ids field must contain only the chunk IDs from the context that directly supported your answer."""


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_blocks = []
    for chunk in chunks:
        block = f"[chunk_id: {chunk['chunk_id']}]\n[doc_id: {chunk['doc_id']}]\n{chunk['text']}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    return f"""Context:

{context}

---

Question: {query}

Remember to return only valid JSON with the fields: answer and used_chunk_ids."""