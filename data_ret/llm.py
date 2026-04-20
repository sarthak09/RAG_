import re
import json
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda
from data_ret.models import RAGResponse, FailedRAGResponse, Citation
from data_ret.prompts import build_system_prompt, build_user_prompt

class LLMGenerator:
    def __init__(self, config: dict, ollama_base_url: str):
        cfg = config["llm"]
        self.model_name  = cfg["model_name"]
        self.temperature = cfg["temperature"]
        self.max_tokens  = cfg["max_tokens"]
        self.num_ctx     = cfg.get("num_ctx", 2048)
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=ollama_base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
            num_ctx=self.num_ctx,
            reasoning=False,      
            format="json",        
            keep_alive="60m")
        self._chain = self._build_chain()

    def _build_chain(self):
        preprocess = RunnableParallel({
            "system_prompt": RunnableLambda(lambda x: build_system_prompt()),
            "user_prompt": RunnableLambda(lambda x: build_user_prompt(x["query"], x["chunks"])),
            "chunk_map": RunnableLambda(lambda x: {c["chunk_id"]: c for c in x["chunks"]}),
            "query":  RunnableLambda(lambda x: x["query"])})
        def assemble(x):
            return {
                "messages":  [
                    SystemMessage(content=x["system_prompt"]),
                    HumanMessage(content=x["user_prompt"]),
                ],
                "chunk_map": x["chunk_map"],
                "query": x["query"]}
        def invoke_llm(x):
            response = self.llm.invoke(x["messages"])
            return {
                "raw": response.content,
                "chunk_map": x["chunk_map"],
                "query": x["query"]}
        return preprocess | RunnableLambda(assemble) | RunnableLambda(invoke_llm)

    def _extract_json(self, raw: str) -> dict:
        if not raw:
            raise ValueError("LLM returned an empty response")
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in LLM response")
        return json.loads(match.group())

    def generate(self, query: str, chunks: list[dict]) -> tuple[RAGResponse | FailedRAGResponse, float]:
        if not chunks:
            return FailedRAGResponse(query=query, model=self.model_name, error="No context chunks were retrieved for this query.",), 0.0
        try:
            t0 = time.perf_counter()
            result = self._chain.invoke({"query": query, "chunks": chunks})
            generation_ms = round((time.perf_counter() - t0) * 1000, 2)
            parsed = self._extract_json(result["raw"].strip())
            answer = parsed.get("answer", "")
            used_chunk_ids = parsed.get("used_chunk_ids", [])
            chunk_map = result["chunk_map"]
            citations = [
                Citation(
                    chunk_id=cid,
                    doc_id=chunk_map[cid]["doc_id"],
                    arxiv_id=chunk_map[cid].get("arxiv_id"),
                    source_file=chunk_map[cid].get("source_file"),
                    page_number=chunk_map[cid].get("page_number"),
                    section_title=chunk_map[cid].get("section_title"),
                    chunk_index=chunk_map[cid].get("chunk_index"),
                    snippet=chunk_map[cid]["text"][:200],)
                for cid in used_chunk_ids if cid in chunk_map]

            return RAGResponse(query=query, answer=answer, citations=citations, model=self.model_name, total_chunks_used=len(chunks),), generation_ms
        except json.JSONDecodeError as e:
            return FailedRAGResponse(query=query, model=self.model_name, error=f"LLM returned invalid JSON: {e}",), 0.0
        except Exception as e:
            return FailedRAGResponse(query=query, model=self.model_name, error=str(e)), 0.0
