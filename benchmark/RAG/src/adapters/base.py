from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Optional
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from core.logger import get_logger


_REFUSAL_PHRASES = (
    "insufficient information",
    "not mentioned",
    "no information",
)

FINAL_ANSWER_RULE = """Output format:
- Start with exactly one line: Final answer: <short final answer>
- Answer exactly what the question asks. Do not add unrelated facts, guesses, or a general summary.
- For yes/no questions, put Yes or No first and include only the essential qualifier needed by the question.
- For date questions, output the exact date or time period supported by the evidence.
- For list questions, include every distinct item required by the question, separated by commas or semicolons.
- For numeric questions, put the final number and unit in the final answer line.
- If a calculation is needed, add at most one short formula line and then the final value.
- If the evidence supports an answer, do not say "Insufficient information" or "Not mentioned".
- Do not wrap the final answer in Markdown formatting."""


def _clean_answer_line(line: str) -> str:
    line = re.sub(r"[`*_]", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^[>\-\*\u2022\uf0b7]+\s*", "", line)
    line = re.sub(r"^\d+[.)]\s*", "", line)
    return line


def normalize_answer_text(raw_answer: str) -> str:
    """
    Normalize a raw model answer into a concise final answer.

    The function prefers explicit final-answer markers, preserves refusal/
    missing-information phrases, and otherwise falls back to the last
    meaningful line.
    """
    if raw_answer is None:
        return ""

    text = str(raw_answer).replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    text = text.replace("```", "")
    text = re.sub(r"(?im)^\s*```[a-z0-9_-]*\s*$", "", text)

    cleaned_lines = []
    for line in text.split("\n"):
        cleaned = _clean_answer_line(line)
        if cleaned:
            cleaned_lines.append(cleaned)

    if not cleaned_lines:
        return ""

    label_pattern = re.compile(
        r"^(?:"
        r"final answer|answer|conclusion|result"
        r")\s*(?:[:：\-—]|is)\s*(.+)$",
        re.IGNORECASE,
    )

    for line in reversed(cleaned_lines):
        low = line.lower()

        for phrase in _REFUSAL_PHRASES:
            if phrase in low:
                if "insufficient information" in low:
                    return "Insufficient information"
                if "not mentioned" in low or "no information" in low:
                    return "Not mentioned"

        match = label_pattern.match(line)
        if match:
            tail = match.group(1).strip()
            if tail:
                return tail

    if len(cleaned_lines) == 1:
        return cleaned_lines[0]

    return cleaned_lines[-1]


@dataclass
class StandardQA:
    """Standardized single question-answer pair"""
    question: str
    gold_answers: List[str]
    evidence: List[str] = field(default_factory=list)
    category: Optional[Union[int, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardSample:
    """Standardized sample containing document content and corresponding QA list"""
    sample_id: str
    qa_pairs: List[StandardQA]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class StandardDoc:
    """Standardized sampleid to doc_path mapping structure"""
    sample_id:str
    doc_path:str


class BaseAdapter(ABC):
    """Base class for all dataset adapters"""
    
    def __init__(self, raw_file_path: str):
        self.raw_file_path = raw_file_path
        self.logger = get_logger()

    @abstractmethod
    def data_prepare(self, doc_dir:str) -> List[StandardDoc]:
        """
        Data preparation.
        1. Convert dataset format to OpenViking-friendly format
        2. Return converted (or unconverted) file paths
        
        Returns:
            List[StandardDoc]: Array of file paths expected to be input to OpenViking
        """
        pass

    @abstractmethod
    def load_and_transform(self) -> List[StandardSample]:
        """
        Read raw files and convert to standard format list.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def build_prompt(self, qa: StandardQA, context_blocks: List[str]) -> tuple[str, Dict[str, Any]]:
        """
        Build final prompt to send to LLM based on retrieved context and QA pair.
        
        Returns:
            - full_prompt (str): Complete prompt string
            - meta (Dict): Metadata to pass to post-processing function (e.g., option mapping for multiple choice)
        """
        pass

    def post_process_answer(self, qa: StandardQA, raw_answer: str, meta: Dict[str, Any]) -> str:
        """
        Post-process raw LLM output using a shared final-answer normalizer.
        """
        return normalize_answer_text(raw_answer)
