import re
import json
from langchain_ollama import ChatOllama
from data_ret.tracer import PipelineTimer
from langchain_core.messages import SystemMessage, HumanMessage
from data_ret.models import RAGResponse, FailedRAGResponse, Citation
from data_ret.prompts import build_system_prompt, build_user_prompt

class LLMGenerator:
    def __init__(self, config: dict, ollama_base_url: str):
        cfg = config["llm"]
        self.model_name = cfg["model_name"]
        self.temperature = cfg["temperature"]
        self.max_tokens = cfg["max_tokens"]
        self.llm = ChatOllama(model=self.model_name, base_url=ollama_base_url, temperature=self.temperature, num_predict=self.max_tokens)

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

    def generate(self, query: str, chunks: list[dict]) -> RAGResponse | FailedRAGResponse:
        if not chunks:
            return FailedRAGResponse(query=query, model=self.model_name, error="No context chunks were retrieved for this query."), 0.0
        timer = PipelineTimer()
        try:
            messages = [
                SystemMessage(content=build_system_prompt()),
                HumanMessage(content=build_user_prompt(query, chunks))
            ]
            with timer.measure("generation"):
                response = self.llm.invoke(messages)
            raw = response.content.strip()
            parsed = self._extract_json(raw)
            answer = parsed.get("answer", "")
            used_chunk_ids = parsed.get("used_chunk_ids", [])
            chunk_map = {c["chunk_id"]: c for c in chunks}
            citations = [
                Citation(
                    chunk_id=cid,
                    doc_id=chunk_map[cid]["doc_id"],
                    arxiv_id=chunk_map[cid].get("arxiv_id"),
                    source_file=chunk_map[cid].get("source_file"),
                    page_number=chunk_map[cid].get("page_number"),
                    section_title=chunk_map[cid].get("section_title"),
                    chunk_index=chunk_map[cid].get("chunk_index"),
                    snippet=chunk_map[cid]["text"][:200]
                )
                for cid in used_chunk_ids
                if cid in chunk_map
            ]
            return RAGResponse(
                query=query,
                answer=answer,
                citations=citations,
                model=self.model_name,
                total_chunks_used=len(chunks)
            ), timer.get("generation")
        except json.JSONDecodeError as e:
            return FailedRAGResponse(query=query,model=self.model_name, error=f"LLM returned invalid JSON: {e}"), timer.get("generation")
        except Exception as e:
            return FailedRAGResponse(query=query, model=self.model_name, error=str(e)), timer.get("generation")