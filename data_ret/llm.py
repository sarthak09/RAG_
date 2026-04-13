import re
import json
import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from data_ret.models import RAGResponse, FailedRAGResponse, Citation
from data_ret.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class LLMGenerator:
    def __init__(self, config: dict, ollama_base_url: str):
        cfg = config["llm"]

        self.model_name  = cfg["model_name"]
        self.temperature = cfg["temperature"]
        self.max_tokens  = cfg["max_tokens"]

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=ollama_base_url,
            temperature=self.temperature,
            num_predict=self.max_tokens,
        )

        logger.info("LLM initialised — model: %s", self.model_name)

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
            logger.warning("No chunks provided for generation")
            return FailedRAGResponse(
                query=query,
                model=self.model_name,
                error="No context chunks were retrieved for this query."
            )

        try:
            messages = [
                SystemMessage(content=build_system_prompt()),
                HumanMessage(content=build_user_prompt(query, chunks))
            ]

            logger.info("Calling LLM — model: %s | chunks: %d", self.model_name, len(chunks))
            response = self.llm.invoke(messages)
            raw      = response.content.strip()

            logger.info("LLM response received — parsing JSON")
            logger.debug("Raw LLM output: %s", raw[:500])

            parsed         = self._extract_json(raw)
            answer         = parsed.get("answer", "")
            used_chunk_ids = parsed.get("used_chunk_ids", [])

            chunk_map = {c["chunk_id"]: c for c in chunks}
            citations = [
                Citation(
                    chunk_id=cid,
                    doc_id=chunk_map[cid]["doc_id"],
                    source=chunk_map[cid].get("source"),
                    snippet=chunk_map[cid]["text"][:200]
                )
                for cid in used_chunk_ids
                if cid in chunk_map
            ]

            logger.info("Generation complete — citations: %d", len(citations))

            return RAGResponse(
                query=query,
                answer=answer,
                citations=citations,
                model=self.model_name,
                total_chunks_used=len(chunks)
            )

        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON response: %s", e)
            return FailedRAGResponse(
                query=query,
                model=self.model_name,
                error=f"LLM returned invalid JSON: {e}"
            )

        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return FailedRAGResponse(
                query=query,
                model=self.model_name,
                error=str(e)
            )