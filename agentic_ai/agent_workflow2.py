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

logger = logging.getLogger("rag.agent.workflow2")

_GRADE_SYSTEM = (
    "You are a relevance grader. Given a QUESTION and retrieved CONTEXT chunks, "
    "decide if the context contains enough information to answer the question. "
    "Reply with exactly one word: YES or NO. Nothing else."
)

_HYDE_SYSTEM = (
    "You are a research assistant. Given a question, write a single dense paragraph "
    "that looks like it was extracted from a scientific paper and directly answers the question. "
    "Output only the paragraph text, no preamble, no explanation."
)


class AgentState(TypedDict):
    query: str
    current_query: str
    merged_chunks: list[dict]
    reranked_chunks: list[dict]
    grade: str
    retry_count: int
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


class CorrectiveRAGWorkflow:
    def __init__(self, config: dict, ollama_base_url: str, store, bm25_retriever=None):
        self.config      = config
        self.base_url    = ollama_base_url.rstrip("/")
        llm_cfg          = config["llm"]
        acfg             = config["agentic"]

        self.model_name  = llm_cfg["model_name"]
        self.temperature = llm_cfg["temperature"]
        self.max_tokens  = llm_cfg["max_tokens"]
        self.max_workers = acfg.get("max_workers", 4)
        self.max_retries = acfg.get("max_retries", 2)

        self.client       = ollama.Client(host=self.base_url)
        self.vector_tool  = VectorRetrieverTool.from_config(config, store)
        self.bm25_tool    = BM25Tool.from_config(config, bm25_retriever) if bm25_retriever else None
        self.reranker_tool = RerankerTool.from_config(config)

        self.graph = self._build_graph()

        logger.info(
            f"CorrectiveRAGWorkflow ready | bm25={self.bm25_tool is not None} "
            f"| max_retries={self.max_retries} | max_workers={self.max_workers}"
        )

    def _retrieve_node(self, state: AgentState) -> dict:
        timer   = PipelineTimer()
        query   = state["current_query"]
        latency = dict(state.get("latency") or {})

        all_chunk_lists: list[list[dict]] = []
        with timer.measure("retrieval"):
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures: dict = {ex.submit(self.vector_tool._run, query): "dense"}
                if self.bm25_tool:
                    futures[ex.submit(self.bm25_tool._run, query)] = "bm25"
            for fut in as_completed(futures):
                try:
                    chunks, _ = fut.result()
                    if chunks:
                        all_chunk_lists.append(chunks)
                except Exception as e:
                    logger.warning(f"Retrieval future failed [{futures[fut]}]: {e}")

        merged = _merge_chunks(all_chunk_lists)
        logger.info(f"[retrieve] retry={state['retry_count']} | merged={len(merged)}")
        latency["retrieval_ms"] = latency.get("retrieval_ms", 0.0) + timer.get("retrieval")
        return {"merged_chunks": merged, "latency": latency}

    def _rerank_node(self, state: AgentState) -> dict:
        timer   = PipelineTimer()
        latency = dict(state.get("latency") or {})
        with timer.measure("reranking"):
            reranked = self.reranker_tool._run(
                query=state["query"],
                chunks=state.get("merged_chunks") or [],
            )
        logger.info(f"[rerank] input={len(state.get('merged_chunks') or [])} | output={len(reranked)}")
        latency["reranking_ms"] = latency.get("reranking_ms", 0.0) + timer.get("reranking")
        return {"reranked_chunks": reranked, "latency": latency}

    def _grade_node(self, state: AgentState) -> dict:
        timer   = PipelineTimer()
        latency = dict(state.get("latency") or {})
        chunks  = state.get("reranked_chunks") or []

        if not chunks:
            logger.info("[grade] no chunks — marking bad")
            return {"grade": "bad", "latency": latency}

        context = "\n\n".join(c["text"] for c in chunks)
        with timer.measure("grading"):
            try:
                resp = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": _GRADE_SYSTEM},
                        {"role": "user",   "content": f"QUESTION: {state['query']}\n\nCONTEXT:\n{context}"},
                    ],
                    think=False,
                    options={"temperature": 0.0, "num_predict": 8},
                )
                raw   = resp["message"]["content"].strip().upper()
                grade = "good" if "YES" in raw else "bad"
            except Exception as e:
                logger.warning(f"[grade] LLM call failed: {e} — defaulting to good")
                grade = "good"

        logger.info(f"[grade] result={grade} | retry={state['retry_count']}")
        latency["grading_ms"] = latency.get("grading_ms", 0.0) + timer.get("grading")
        return {"grade": grade, "latency": latency}

    def _hyde_rewrite_node(self, state: AgentState) -> dict:
        timer   = PipelineTimer()
        latency = dict(state.get("latency") or {})
        with timer.measure("hyde_rewrite"):
            try:
                resp = self.client.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {"role": "user",   "content": state["query"]},
                    ],
                    think=False,
                    options={"temperature": 0.3, "num_predict": 256},
                )
                new_query = resp["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"[hyde_rewrite] failed: {e} — keeping original query")
                new_query = state["query"]

        retry = state.get("retry_count", 0) + 1
        logger.info(f"[hyde_rewrite] retry={retry} | new_query_chars={len(new_query)}")
        latency["hyde_ms"] = latency.get("hyde_ms", 0.0) + timer.get("hyde_rewrite")
        return {"current_query": new_query, "retry_count": retry, "latency": latency}

    def _generate_node(self, state: AgentState) -> dict:
        query   = state["query"]
        chunks  = state.get("reranked_chunks") or []
        latency = dict(state.get("latency") or {})

        if not chunks:
            return {
                "response": FailedRAGResponse(query=query, model=self.model_name, error="No chunks available for generation"),
                "latency": latency,
            }

        messages = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user",   "content": build_user_prompt(query, chunks)},
        ]
        try:
            t0   = time.perf_counter()
            resp = self.client.chat(
                model=self.model_name,
                messages=messages,
                think=False,
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
            )
            generation_ms  = round((time.perf_counter() - t0) * 1000, 1)
            raw            = resp["message"]["content"].strip()
            parsed         = _extract_json(raw)
            answer         = parsed.get("answer", "")
            used_chunk_ids = parsed.get("used_chunk_ids", [])
            chunk_map      = {c["chunk_id"]: c for c in chunks}
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
                for cid in used_chunk_ids if cid in chunk_map
            ]
            response = RAGResponse(
                query=query, answer=answer, citations=citations,
                model=self.model_name, total_chunks_used=len(chunks),
            )
            latency["generation_ms"] = generation_ms
        except Exception as e:
            response = FailedRAGResponse(query=query, model=self.model_name, error=str(e))

        latency["total_ms"]      = round(sum(
            latency.get(k, 0.0)
            for k in ["retrieval_ms", "reranking_ms", "grading_ms", "hyde_ms", "generation_ms"]
        ), 1)
        latency["retries"]       = state.get("retry_count", 0)
        latency["chunks_final"]  = len(chunks)
        return {"response": response, "latency": latency}

    def _after_grade(self, state: AgentState) -> str:
        if state["grade"] == "good":
            return "generate"
        if state.get("retry_count", 0) >= self.max_retries:
            logger.warning(f"[route] max_retries={self.max_retries} reached — generating anyway")
            return "generate"
        return "hyde_rewrite"

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)

        builder.add_node("retrieve",     self._retrieve_node)
        builder.add_node("rerank",       self._rerank_node)
        builder.add_node("grade",        self._grade_node)
        builder.add_node("hyde_rewrite", self._hyde_rewrite_node)
        builder.add_node("generate",     self._generate_node)

        builder.add_edge(START,          "retrieve")
        builder.add_edge("retrieve",     "rerank")
        builder.add_edge("rerank",       "grade")
        builder.add_conditional_edges(
            "grade",
            self._after_grade,
            {"generate": "generate", "hyde_rewrite": "hyde_rewrite"},
        )
        builder.add_edge("hyde_rewrite", "retrieve")
        builder.add_edge("generate",     END)

        return builder.compile()

    def run(self, query: str) -> tuple[RAGResponse | FailedRAGResponse, dict]:
        if not query or not query.strip():
            return FailedRAGResponse(query=query, model=self.model_name, error="Empty query"), {}

        initial_state: AgentState = {
            "query":           query,
            "current_query":   query,
            "merged_chunks":   [],
            "reranked_chunks": [],
            "grade":           "",
            "retry_count":     0,
            "response":        None,
            "latency":         {},
            "error":           "",
        }

        final_state = self.graph.invoke(initial_state)
        response = final_state.get("response") or FailedRAGResponse(
            query=query, model=self.model_name, error="Graph produced no response"
        )
        return response, final_state.get("latency", {})