"""
hallucination_checker.py
Uses cross-encoder/nli-MiniLM2-L6-H768 — lightweight, fast, CPU-friendly,
and works reliably on Windows without tokenizer conflicts.
"""

import re
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class HallucinationChecker:
    def __init__(self):
        model_name = "cross-encoder/nli-MiniLM2-L6-H768"
        print(f"Loading NLI model: {model_name} (downloads ~90MB on first run)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # cross-encoder NLI models output: [contradiction, entailment, neutral]
        self.label_map = {0: "contradicted", 1: "supported", 2: "unsupported"}
        print("NLI model ready!")

    def _parse_claims(self, claims_text: str) -> List[str]:
        """Parse numbered list of claims from LLM output."""
        claims = []
        for line in claims_text.strip().split("\n"):
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            line = re.sub(r"^\s*[-•]\s*", "", line).strip()
            if len(line) > 20:
                claims.append(line)
        return claims

    def _verify_claim(self, claim: str, context: str) -> Dict:
        """Check if context supports, contradicts, or doesn't mention the claim."""
        context = context[:800]

        inputs = self.tokenizer(
            context,
            claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

        # probs order: [contradiction, entailment, neutral]
        contradiction = round(probs[0] * 100, 1)
        entailment    = round(probs[1] * 100, 1)
        neutral       = round(probs[2] * 100, 1)

        max_idx = probs.index(max(probs))
        verdict = self.label_map[max_idx]
        confidence = round(max(probs) * 100, 1)

        return {
            "claim": claim,
            "verdict": verdict,
            "confidence": confidence,
            "scores": {
                "supported": entailment,
                "unsupported": neutral,
                "contradicted": contradiction,
            }
        }

    def check(self, claims_text: str, context: str) -> Dict:
        """Full hallucination check — returns per-claim verdicts + overall score."""
        claims = self._parse_claims(claims_text)

        if not claims:
            return {
                "claims": [],
                "hallucination_score": 0.0,
                "summary": "Could not extract any claims to verify.",
                "issues": "None",
                "total": 0, "supported": 0, "unsupported": 0, "contradicted": 0,
            }

        results = []
        for claim in claims:
            results.append(self._verify_claim(claim, context))

        total        = len(results)
        supported    = sum(1 for r in results if r["verdict"] == "supported")
        unsupported  = sum(1 for r in results if r["verdict"] == "unsupported")
        contradicted = sum(1 for r in results if r["verdict"] == "contradicted")
        bad          = unsupported + contradicted
        hallucination_score = round((bad / total) * 100, 1) if total else 0.0

        issues_lines = [
            f'- "{r["claim"]}" → {r["verdict"].upper()} ({r["confidence"]}% confidence)'
            for r in results if r["verdict"] in ("unsupported", "contradicted")
        ]
        issues = "\n".join(issues_lines) if issues_lines else "None"

        summary = (
            f"{total} claims checked — "
            f"{supported} supported ✅  "
            f"{unsupported} unsupported ⚠️  "
            f"{contradicted} contradicted ❌"
        )

        return {
            "claims": results,
            "hallucination_score": hallucination_score,
            "summary": summary,
            "issues": issues,
            "total": total,
            "supported": supported,
            "unsupported": unsupported,
            "contradicted": contradicted,
        }