"""
rag_pipeline.py
Orchestrates: document loading → embedding → retrieval → LLM answer → hallucination check → correction.
"""

from typing import List, Dict
from src.document_loader import load_and_chunk
from src.vector_store import VectorStore
from src.llm_client import HFInferenceClient
from src.hallucination_checker import HallucinationChecker
from config import TOP_K


class HallucinationAwareRAG:
    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = HFInferenceClient()
        self.checker = HallucinationChecker()
        self.sources_loaded: List[str] = []

    def add_source(self, source: str) -> int:
        """Load a file or URL, chunk it, embed and index it. Returns chunk count."""
        chunks = load_and_chunk(source)
        self.vector_store.add_chunks(chunks)
        self.sources_loaded.append(source)
        return len(chunks)

    def clear_sources(self):
        """Wipe all loaded sources from memory."""
        self.vector_store.clear()
        self.sources_loaded = []

    def query(self, question: str, top_k: int = TOP_K) -> Dict:
        """
        Full pipeline for one question:
          1. Retrieve relevant chunks from vector store
          2. LLM generates answer from context
          3. LLM breaks answer into individual claims
          4. NLI model verifies each claim against sources
          5. If hallucinations found → LLM generates corrected answer

        Returns a result dict with everything needed for the UI.
        """
        if self.vector_store.total_chunks == 0:
            return {"error": "No sources loaded. Please upload at least one document first."}

        # Step 1 – Retrieve
        hits = self.vector_store.search(question, top_k=top_k)
        if not hits:
            return {"error": "No relevant chunks found for this question."}

        context = "\n\n---\n\n".join([chunk["text"] for chunk, _ in hits])
        sources_used = list(set(chunk["source"] for chunk, _ in hits))

        # Step 2 – Generate answer
        answer = self.llm.answer_question(question, context)

        # Step 3 – Extract claims
        claims_text = self.llm.extract_claims(answer)

        # Step 4 – Verify claims
        check = self.checker.check(claims_text, context)

        # Step 5 – Correct if needed
        corrected_answer = None
        if check["hallucination_score"] > 0:
            corrected_answer = self.llm.correct_answer(
                question=question,
                original_answer=answer,
                context=context,
                issues=check["issues"],
            )

        return {
            "question": question,
            "answer": answer,
            "corrected_answer": corrected_answer,
            "needs_correction": check["hallucination_score"] > 0,
            "hallucination_score": check["hallucination_score"],
            "check_summary": check["summary"],
            "claims": check["claims"],
            "context_used": context,
            "sources_referenced": sources_used,
        }