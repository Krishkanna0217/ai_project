"""
llm_client.py
Uses huggingface_hub InferenceClient SDK — the correct, stable way to call
HuggingFace models. Automatically handles provider routing.
"""

from typing import Optional
from huggingface_hub import InferenceClient
from config import HF_API_KEY, LLM_MODEL


class HFInferenceClient:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or HF_API_KEY
        if not key or "xxx" in key:
            raise ValueError("HuggingFace API key not set! Open config.py and add your key.")
        # provider="auto" works correctly here — SDK handles routing automatically
        self.client = InferenceClient(
            provider="auto",
            api_key=key,
        )
        self.model = LLM_MODEL

    def _call(self, system_prompt: str, user_prompt: str,
              max_tokens: int = 512, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def answer_question(self, question: str, context: str) -> str:
        system = (
            "You are a helpful assistant. Answer questions using ONLY the provided context. "
            "Be concise and factual. If the context lacks enough info, say so clearly."
        )
        user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        return self._call(system, user, max_tokens=400, temperature=0.2)

    def extract_claims(self, answer: str) -> str:
        system = (
            "You are a claim extraction assistant. "
            "Break answers into individual atomic verifiable claims. "
            "Output ONLY a numbered list of single-sentence claims. No extra text."
        )
        user = f"Answer:\n{answer}\n\nClaims:"
        return self._call(system, user, max_tokens=300, temperature=0.1)

    def correct_answer(self, question: str, original_answer: str,
                       context: str, issues: str) -> str:
        system = (
            "You are a fact-checking assistant. "
            "Rewrite answers using ONLY information clearly supported by the provided context. "
            "Do NOT add any information not present in the context."
        )
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Original Answer (contains hallucinations):\n{original_answer}\n\n"
            f"Problems found:\n{issues}\n\n"
            f"Write a corrected, fully source-grounded answer:"
        )
        return self._call(system, user, max_tokens=400, temperature=0.1)