import re
import json
import time
import logging
import ollama
from typing import Any, Optional
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.graph import StateGraph, START, END
from data_ret.models import RAGResponse, FailedRAGResponse, Citation
from data_ret.prompts import build_system_prompt, build_user_prompt
from data_ret.tracer import PipelineTimer
from agentic_ai.agent_tools import VectorRetrieverTool, BM25Tool, RerankerTool

logger = logging.getLogger("rag.agent.workflow")

_HYDE_SYSTEM = (
    "You are a research assistant. Given a question, write a single dense paragraph "
    "that looks like it was extracted from a scientific paper and directly answers the question. "
    "Output only the paragraph text, no preamble, no explanation."
)

_REWRITE_SYSTEM = (
    "You are a query rewriting assistant. Given a search query, output exactly {n} "
    "alternative phrasings that capture the same intent using different vocabulary. "
    "Output only a JSON array of strings and nothing else. "
    'Example: ["phrasing 1", "phrasing 2"]'
)

class AgentState(TypedDict):
    query: str
    hyde_doc: str
    rewritten_queries: list[str]
    merged_chunks: list[dict]
    reranked_chunks: list[dict]
    response: Optional[Any]
    latency: dict
    error: str

def _merge_chunks(all_chunk_lists: list[list[dict]]) -> list[dict]:
    seen: dict[str, dict] = {}
    for chunk_list in all_chunk_lists:
        for c in chunk_list:
            cid = c.get("chunk_id")
            if cid is None:
                continue
            if cid not in seen or c["score"] > seen[cid]["score"]:
                seen[cid] = c
    return list(seen.values())

def _extract_json(raw: str) -> dict:
    if not raw:
        raise ValueError("LLM returned empty response")
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(match.group())

class AgenticRAGWorkflow:
    def __init__(self, config: dict, ollama_base_url: str, store, bm25_retriever=None):
        self.config = config
        self.ollama_base_url = ollama_base_url.rstrip("/")
        acfg = config["agentic"]
        llm_cfg = config["llm"]
        self.model_name = llm_cfg["model_name"]
        self.temperature = llm_cfg["temperature"]
        self.max_tokens = llm_cfg["max_tokens"]
        self.use_hyde = acfg.get("use_hyde", True)
        self.use_rewrite = acfg.get("use_query_rewriting", True)
        self.num_rewrites = acfg.get("num_rewrites", 2)
        self.max_workers = acfg.get("max_workers", 4)
        self.client = ollama.Client(host=self.ollama_base_url)
        self.vector_tool = VectorRetrieverTool.from_config(config, store)
        self.bm25_tool = BM25Tool.from_config(config, bm25_retriever) if bm25_retriever else None
        self.reranker_tool = RerankerTool.from_config(config)
        self.graph = self._build_graph()
        logger.info(
            f"AgenticRAGWorkflow ready | hyde={self.use_hyde} "
            f"| rewrite={self.use_rewrite} | bm25={self.bm25_tool is not None} "
            f"| max_workers={self.max_workers}")

    def _hyde_generate(self, query: str) -> str:
        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _HYDE_SYSTEM},
                    {"role": "user",   "content": query},
                ],
                think=False,
                options={"temperature": 0.3, "num_predict": 256},)
            return resp["message"]["content"].strip()
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}")
            return ""

    def _rewrite_query(self, query: str) -> list[str]:
        try:
            resp = self.client.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM.format(n=self.num_rewrites)},
                    {"role": "user",   "content": query},
                ],
                think=False,
                options={"temperature": 0.4, "num_predict": 256})
            raw = resp["message"]["content"].strip()
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            return json.loads(match.group())
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}")
            return []

    def _prep_node(self, state: AgentState) -> dict:
        timer = PipelineTimer()
        hyde_doc = ""
        rewritten_queries = []
        with timer.measure("prep_llm"):
            with ThreadPoolExecutor(max_workers=2) as ex:
                futures = {}
                if self.use_hyde:
                    futures["hyde"] = ex.submit(self._hyde_generate, state["query"])
                if self.use_rewrite:
                    futures["rewrite"] = ex.submit(self._rewrite_query, state["query"])
            if "hyde" in futures:
                hyde_doc = futures["hyde"].result() or ""
            if "rewrite" in futures:
                rewritten_queries = futures["rewrite"].result() or []
        logger.info(f"[prep] hyde_chars={len(hyde_doc)} | rewrites={len(rewritten_queries)}")
        latency = dict(state.get("latency") or {})
        latency["prep_llm_ms"] = timer.get("prep_llm")
        return {
            "hyde_doc": hyde_doc,
            "rewritten_queries": rewritten_queries,
            "latency": latency}

    def _retrieve_node(self, state: AgentState) -> dict:
        timer = PipelineTimer()
        query = state["query"]
        retrieval_queries = [query] + (state.get("rewritten_queries") or [])
        if state.get("hyde_doc"):
            retrieval_queries.append(state["hyde_doc"])
        all_chunk_lists: list[list[dict]] = []
        with timer.measure("retrieval"):
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                retrieve_futures = {
                    ex.submit(self.vector_tool._run, q): f"dense:{q[:30]}"
                    for q in retrieval_queries}
                if self.bm25_tool:
                    retrieve_futures[ex.submit(self.bm25_tool._run, query)] = "bm25"
            for fut in as_completed(retrieve_futures):
                label = retrieve_futures[fut]
                try:
                    chunks, _ = fut.result()
                    if chunks:
                        all_chunk_lists.append(chunks)
                except Exception as e:
                    logger.warning(f"Retrieval future failed [{label}]: {e}")
        merged = _merge_chunks(all_chunk_lists)
        logger.info(f"[retrieve] queries={len(retrieval_queries)} | merged={len(merged)}")
        latency = dict(state.get("latency") or {})
        latency["retrieval_ms"] = timer.get("retrieval")
        return {"merged_chunks": merged, "latency": latency}

    def _rerank_node(self, state: AgentState) -> dict:
        timer = PipelineTimer()
        with timer.measure("reranking"):
            reranked = self.reranker_tool._run(
                query=state["query"],
                chunks=state.get("merged_chunks") or [])
        logger.info(f"[rerank] input={len(state.get('merged_chunks') or [])} | output={len(reranked)}")
        latency = dict(state.get("latency") or {})
        latency["reranking_ms"] = timer.get("reranking")
        return {"reranked_chunks": reranked, "latency": latency}

    def _generate_node(self, state: AgentState) -> dict:
        query   = state["query"]
        chunks  = state.get("reranked_chunks") or []
        latency = dict(state.get("latency") or {})
        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": build_user_prompt(query, chunks)},
        ]
        try:
            t0 = time.perf_counter()
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                think=False,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},)
            generation_ms = round((time.perf_counter() - t0) * 1000, 1)
            raw = resp["message"]["content"].strip()
            parsed = _extract_json(raw)
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
                    snippet=chunk_map[cid]["text"][:200],
                )
                for cid in used_chunk_ids
                if cid in chunk_map
            ]
            response = RAGResponse(
                query=query,
                answer=answer,
                citations=citations,
                model=self.model_name,
                total_chunks_used=len(chunks))
            latency["generation_ms"] = generation_ms
            latency["total_ms"] = round(sum(
                latency.get(k, 0.0)
                for k in ["prep_llm_ms", "retrieval_ms", "reranking_ms", "generation_ms"]), 1)
            latency["hyde_used"]     = bool(state.get("hyde_doc"))
            latency["rewrites_used"] = len(state.get("rewritten_queries") or [])
            latency["chunks_merged"] = len(state.get("merged_chunks") or [])
            latency["chunks_final"]  = len(chunks)
        except Exception as e:
            response = FailedRAGResponse(query=query, model=self.model_name, error=str(e))
        return {"response": response, "latency": latency}

    def _should_generate(self, state: AgentState) -> str:
        if not state.get("reranked_chunks"):
            logger.warning("[route] No chunks after reranking — skipping generation")
            return "no_chunks"
        return "generate"

    def _no_chunks_node(self, state: AgentState) -> dict:
        return {
            "response": FailedRAGResponse(
                query=state["query"],
                model=self.model_name,
                error="No relevant chunks found after retrieval and reranking")
        }

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("prep", self._prep_node)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("rerank", self._rerank_node)
        builder.add_node("generate", self._generate_node)
        builder.add_node("no_chunks", self._no_chunks_node)
        builder.add_edge(START, "prep")
        builder.add_edge("prep", "retrieve")
        builder.add_edge("retrieve", "rerank")
        builder.add_conditional_edges("rerank", self._should_generate, {"generate": "generate", "no_chunks": "no_chunks"})
        builder.add_edge("generate",  END)
        builder.add_edge("no_chunks", END)
        return builder.compile()

    def run(self, query: str) -> tuple[RAGResponse | FailedRAGResponse, dict]:
        if not query or not query.strip():
            return FailedRAGResponse(query=query, model=self.model_name, error="Empty query"), {}
        initial_state: AgentState = {
            "query": query,
            "hyde_doc": "",
            "rewritten_queries": [],
            "merged_chunks": [],
            "reranked_chunks": [],
            "response": None,
            "latency": {},
            "error": "",
        }
        final_state = self.graph.invoke(initial_state)
        response = final_state.get("response") or FailedRAGResponse(query=query, model=self.model_name, error="Graph produced no response")
        return response, final_state.get("latency", {})