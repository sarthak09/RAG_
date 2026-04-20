import json
import csv
from typing import List, Dict, Any
from pathlib import Path
from langchain_ollama import ChatOllama
from typing_extensions import Annotated, TypedDict

class CorrectnessGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    correct: Annotated[bool, ..., "True if the answer is correct, False otherwise."]

class RelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "Provide the score on whether the answer addresses the question"]

class GroundedGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    grounded: Annotated[bool, ..., "Provide the score on if the answer hallucinates from the documents"]

class RetrievalRelevanceGrade(TypedDict):
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if the retrieved documents are relevant to the question, False otherwise"]


correctness_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION, the GROUND TRUTH (correct) ANSWER, and the STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Grade the student answers based ONLY on their factual accuracy relative to the ground truth answer. 
(2) Ensure that the student answer does not contain any conflicting statements.
(3) It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the ground truth answer.

Correctness:
A correctness value of True means that the student's answer meets all of the criteria.
A correctness value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""


relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is concise and relevant to the QUESTION
(2) Ensure the STUDENT ANSWER helps to answer the QUESTION

Relevance:
A relevance value of True means that the student's answer meets all of the criteria.
A relevance value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""


grounded_instructions = """You are a teacher grading a quiz. 

You will be given FACTS and a STUDENT ANSWER. 

Here is the grade criteria to follow:
(1) Ensure the STUDENT ANSWER is grounded in the FACTS. 
(2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

Grounded:
A grounded value of True means that the student's answer meets all of the criteria.
A grounded value of False means that the student's answer does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""


retrieval_relevance_instructions = """You are a teacher grading a quiz. 

You will be given a QUESTION and a set of FACTS provided by the student. 

Here is the grade criteria to follow:
(1) You goal is to identify FACTS that are completely unrelated to the QUESTION
(2) If the facts contain ANY keywords or semantic meaning related to the question, consider them relevant
(3) It is OK if the facts have SOME information that is unrelated to the question as long as (2) is met

Relevance:
A relevance value of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance value of False means that the FACTS are completely unrelated to the QUESTION.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 
Avoid simply stating the correct answer at the outset."""


def load_config(config_path: Path) -> List[Dict[str, Any]]:
    print(f"Loading config from: {config_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)

def main(filename: str):
    data_file = load_config(Path(filename))
    parent_dir = Path(__file__).resolve().parent.parent
    config_path = parent_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    grader_llm = ChatOllama(model=config["llm"]["model_name"]).with_structured_output(CorrectnessGrade, method="json_schema", strict=True)
    relevance_llm = ChatOllama(model=config["llm"]["model_name"]).with_structured_output(RelevanceGrade, method="json_schema", strict=True)
    grounded_llm = ChatOllama(model=config["llm"]["model_name"]).with_structured_output(GroundedGrade, method="json_schema", strict=True)
    retrieval_relevance_llm = ChatOllama(model=config["llm"]["model_name"]).with_structured_output(RetrievalRelevanceGrade, method="json_schema", strict=True)
    csv_path = Path(__file__).resolve().parent / "llm_judge_results.csv"
    csv_columns = [
        "query_id", "query", "ground_truth_answer", "llm_answer",
        "correctness", "correctness_explanation",
        "answer_relevance", "answer_relevance_explanation",
        "groundedness", "groundedness_explanation",
        "retrieval_relevance", "retrieval_relevance_explanation",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, item in enumerate(data_file[:5]):
            print(f"\n=== Item {i+1}: {item['query'][:80]} ===")
            answers = f"QUESTION: {item['query']}\nGROUND TRUTH ANSWER: {item['ground_truth_answer']}\nSTUDENT ANSWER: {item['llm_answer']}"
            correctness_grade = grader_llm.invoke([{"role": "system", "content": correctness_instructions}, {"role": "user", "content": answers}])
            print(f"Correctness:          {correctness_grade['correct']}")
            answer = f"QUESTION: {item['query']}\nSTUDENT ANSWER: {item['llm_answer']}"
            relevance_grade = relevance_llm.invoke([{"role": "system", "content": relevance_instructions}, {"role": "user", "content": answer}])
            print(f"Answer relevance:     {relevance_grade['relevant']}")
            doc_string = "\n\n".join(doc["text_preview"] for doc in item["retrieved_context"])
            answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {item['llm_answer']}"
            grounded_grade = grounded_llm.invoke([{"role": "system", "content": grounded_instructions}, {"role": "user", "content": answer}])
            print(f"Groundedness:         {grounded_grade['grounded']}")
            answer = f"FACTS: {doc_string}\nQUESTION: {item['query']}"
            retrieval_grade = retrieval_relevance_llm.invoke([{"role": "system", "content": retrieval_relevance_instructions}, {"role": "user", "content": answer}])
            print(f"Retrieval relevance:  {retrieval_grade['relevant']}")
            writer.writerow({
                "query_id":                        item.get("query_id", ""),
                "query":                           item["query"],
                "ground_truth_answer":             item["ground_truth_answer"],
                "llm_answer":                      item["llm_answer"],
                "correctness":                     int(correctness_grade["correct"]),
                "correctness_explanation":         correctness_grade["explanation"],
                "answer_relevance":                int(relevance_grade["relevant"]),
                "answer_relevance_explanation":    relevance_grade["explanation"],
                "groundedness":                    int(grounded_grade["grounded"]),
                "groundedness_explanation":        grounded_grade["explanation"],
                "retrieval_relevance":             int(retrieval_grade["relevant"]),
                "retrieval_relevance_explanation": retrieval_grade["explanation"],
            })
            csvfile.flush()
    print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    filename = "/home/sarthak/workspace/gen_ai/RAG/project/projects/professional_projects/project1/backend/logs/eval/detailed_results_20260420_014246.json"
    main(filename)