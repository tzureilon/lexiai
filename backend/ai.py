"""Core machine learning utilities powering LexiAI."""
from __future__ import annotations

import json
import logging
import math
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .llm import ClaudeClient, LLMGenerationError

logger = logging.getLogger(__name__)

HEBREW_STOP_WORDS: set[str] = {
    "של",
    "עם",
    "גם",
    "על",
    "שהם",
    "שהוא",
    "שהיא",
    "זה",
    "זו",
    "הם",
    "הן",
    "הוא",
    "היא",
    "ה",
    "או",
    "אם",
    "כי",
    "כדי",
    "אך",
    "אבל",
    "ל",
    "אל",
    "כל",
    "יותר",
    "פחות",
    "לא",
    "כן",
    "יש",
    "אין",
    "היה",
    "היו",
    "תוך",
    "עד",
}

ENGLISH_STOP_WORDS: set[str] = {
    "the",
    "is",
    "are",
    "to",
    "of",
    "and",
    "a",
    "in",
    "for",
    "on",
    "by",
    "an",
    "be",
    "or",
    "as",
    "with",
    "that",
}

STOP_WORDS = HEBREW_STOP_WORDS | ENGLISH_STOP_WORDS
TOKEN_PATTERN = re.compile(r"[A-Za-zא-ת0-9']+")


def tokenize(text: str) -> List[str]:
    tokens = [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
    return [token for token in tokens if token not in STOP_WORDS]


@dataclass
class KnowledgeEntry:
    title: str
    question: str
    answer: str
    citations: list[str]
    tags: list[str]
    insight: str

    def corpus_text(self) -> str:
        return " ".join([self.title, self.question, self.answer, " ".join(self.tags), self.insight])


@dataclass
class VectorEntry:
    text: str
    payload: object


class TfidfVectorStore:
    def __init__(self, entries: Sequence[VectorEntry]) -> None:
        self.entries = list(entries)
        self.document_frequency: Counter[str] = Counter()
        self.vectors: list[tuple[Dict[str, float], float]] = []
        self._build()

    def _build(self) -> None:
        if not self.entries:
            self.vectors = []
            return
        term_counters: list[Counter[str]] = []
        for entry in self.entries:
            tokens = tokenize(entry.text)
            counter = Counter(tokens)
            term_counters.append(counter)
            self.document_frequency.update(counter.keys())
        total_documents = len(self.entries)
        for counter in term_counters:
            vector: Dict[str, float] = {}
            total_terms = sum(counter.values()) or 1
            for token, freq in counter.items():
                idf = math.log((1 + total_documents) / (1 + self.document_frequency[token])) + 1
                vector[token] = (freq / total_terms) * idf
            norm = math.sqrt(sum(weight * weight for weight in vector.values())) or 1.0
            self.vectors.append((vector, norm))

    def search(self, query: str, limit: int = 5) -> list[tuple[object, float]]:
        if not query.strip() or not self.entries:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        query_counter = Counter(tokens)
        total_terms = sum(query_counter.values()) or 1
        query_vector: Dict[str, float] = {}
        total_documents = len(self.entries)
        for token, freq in query_counter.items():
            idf = math.log((1 + total_documents) / (1 + self.document_frequency.get(token, 0))) + 1
            query_vector[token] = (freq / total_terms) * idf
        query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0
        results: list[tuple[int, float]] = []
        for idx, (vector, norm) in enumerate(self.vectors):
            score = 0.0
            for token, weight in query_vector.items():
                if token in vector:
                    score += weight * vector[token]
            if score > 0:
                results.append((idx, score / (norm * query_norm)))
        results.sort(key=lambda item: item[1], reverse=True)
        return [(self.entries[idx].payload, score) for idx, score in results[:limit]]


class LegalKnowledgeBase:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.entries: list[KnowledgeEntry] = [KnowledgeEntry(**raw) for raw in data]
        self._vector_store = TfidfVectorStore(
            [VectorEntry(text=entry.corpus_text(), payload=entry) for entry in self.entries]
        )

    def search(self, query: str, limit: int = 3) -> list[tuple[KnowledgeEntry, float]]:
        return self._vector_store.search(query, limit)


@dataclass
class DocumentSegment:
    document_id: int
    filename: str
    text: str
    order: int


@dataclass
class DocumentMatch:
    segment: DocumentSegment
    score: float
    explanation: str | None = None


class DocumentVectorStore:
    def __init__(self) -> None:
        self._segments: list[DocumentSegment] = []
        self._vector_store = TfidfVectorStore([])

    def rebuild(self, segments: Sequence[DocumentSegment]) -> None:
        self._segments = list(segments)
        entries = [VectorEntry(text=segment.text, payload=segment) for segment in self._segments]
        self._vector_store = TfidfVectorStore(entries)

    def search(self, query: str, limit: int = 5) -> List[DocumentMatch]:
        results = self._vector_store.search(query, limit)
        return [DocumentMatch(segment=segment, score=score) for segment, score in results]


@dataclass
class RagAnswer:
    text: str
    confidence: float
    knowledge_references: list[str]
    guardrails: list[str]
    document_highlights: list[str]


class LegalRagEngine:
    def __init__(
        self,
        knowledge_base: LegalKnowledgeBase,
        llm_client: ClaudeClient | None = None,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._llm = llm_client

    def build_answer(self, query: str, matches: Sequence[DocumentMatch]) -> RagAnswer:
        knowledge_hits = self._knowledge_base.search(query, limit=3)
        knowledge_lines: list[str] = []
        knowledge_refs: list[str] = []
        for entry, score in knowledge_hits:
            pct = int(min(score, 1.0) * 100)
            knowledge_lines.append(
                f"• {entry.insight} (מבוסס על {entry.title}) — ביטחון {pct}%"
            )
            knowledge_refs.append(f"{entry.title} ({', '.join(entry.citations)})")

        document_lines: list[str] = []
        document_highlights: list[str] = []
        top_doc_score = 0.0
        for match in matches:
            top_doc_score = max(top_doc_score, match.score)
            snippet = textwrap.shorten(match.segment.text.replace("\n", " "), width=220, placeholder="...")
            note = match.explanation or "קטע רלוונטי מהמסמך"
            document_lines.append(f"• {note} ({match.segment.filename}) — {snippet}")
            document_highlights.append(snippet)

        raw_confidence = 0.25 + 0.45 * (knowledge_hits[0][1] if knowledge_hits else 0.0) + 0.3 * top_doc_score
        confidence = max(0.15, min(0.95, raw_confidence))

        guardrails: list[str] = []
        if confidence < 0.4:
            guardrails.append("זוהתה התאמה חלקית בלבד – מומלץ לאשרר עם גורם משפטי אנושי.")
        if not document_lines:
            guardrails.append("לא נמצאו מסמכים תומכים בבסיס הנתונים – כדאי להעלות מסמכים רלוונטיים.")

        fallback_text = self._build_fallback_answer(knowledge_lines, document_lines, confidence, guardrails)
        llm_text = self._llm_answer(
            query=query,
            knowledge_lines=knowledge_lines,
            document_lines=document_lines,
            confidence=confidence,
            guardrails=guardrails,
        )

        return RagAnswer(
            text=llm_text or fallback_text,
            confidence=confidence,
            knowledge_references=knowledge_refs,
            guardrails=guardrails,
            document_highlights=document_highlights,
        )

    def _build_fallback_answer(
        self,
        knowledge_lines: Sequence[str],
        document_lines: Sequence[str],
        confidence: float,
        guardrails: Sequence[str],
    ) -> str:
        sections = [
            "להלן ניתוח משולב של מאגר הידע המשפטי והמסמכים שהועלו:",
        ]
        if knowledge_lines:
            sections.append("פסיקה ומדיניות רלוונטיות:\n" + "\n".join(knowledge_lines))
        if document_lines:
            sections.append("עדויות מתוך המסמכים שלך:\n" + "\n".join(document_lines))
        sections.append(
            f"ציון הביטחון של המודל: {confidence:.0%}. זהו כלי מסייע בלבד ואינו מהווה ייעוץ משפטי מחייב."
        )
        if guardrails:
            sections.append("בקרות איכות:\n" + "\n".join(f"- {item}" for item in guardrails))
        return "\n\n".join(sections)

    def _llm_answer(
        self,
        *,
        query: str,
        knowledge_lines: Sequence[str],
        document_lines: Sequence[str],
        confidence: float,
        guardrails: Sequence[str],
    ) -> str | None:
        if not self._llm or not self._llm.is_configured:
            return None

        payload = {
            "user_query": query,
            "knowledge_findings": list(knowledge_lines),
            "document_findings": list(document_lines),
            "model_confidence": confidence,
            "quality_guardrails": list(guardrails),
        }
        system_prompt = (
            "אתה עוזר משפטי בכיר. הפק תשובה בעברית עם מבנה ברור הכוללת הקשר קצר, "
            "התייחסות לפסיקה, אזכורים למסמכים שסופקו, והוראות אזהרה במידת הצורך. "
            "שלב נקודות תבליט ברורות והדגש המלצות אופרטיביות."
        )
        try:
            response = self._llm.generate(
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ],
                max_tokens=900,
            )
            text = response.strip()
            if guardrails:
                guardrail_block = "\n\nבקרות איכות:\n" + "\n".join(f"- {item}" for item in guardrails)
                if guardrail_block not in text:
                    text += guardrail_block
            return text
        except LLMGenerationError as exc:  # pragma: no cover - network path
            logger.warning("Claude RAG generation failed, using fallback: %s", exc)
            return None


class NaiveBayesBinary:
    def __init__(self, positive_texts: Sequence[str], negative_texts: Sequence[str]) -> None:
        self.positive_docs = len(positive_texts)
        self.negative_docs = len(negative_texts)
        self.prior_positive = (self.positive_docs + 1) / (self.positive_docs + self.negative_docs + 2)
        self.prior_negative = 1.0 - self.prior_positive
        self.positive_counts = Counter()
        self.negative_counts = Counter()
        for text in positive_texts:
            self.positive_counts.update(tokenize(text))
        for text in negative_texts:
            self.negative_counts.update(tokenize(text))
        self.positive_total = sum(self.positive_counts.values())
        self.negative_total = sum(self.negative_counts.values())
        self.vocabulary = set(self.positive_counts) | set(self.negative_counts)
        self.vocabulary_size = len(self.vocabulary) or 1

    def score(self, text: str) -> float:
        tokens = tokenize(text)
        if not tokens:
            return self.prior_positive
        log_pos = math.log(self.prior_positive)
        log_neg = math.log(self.prior_negative)
        for token in tokens:
            pos_freq = self.positive_counts.get(token, 0)
            neg_freq = self.negative_counts.get(token, 0)
            log_pos += math.log((pos_freq + 1) / (self.positive_total + self.vocabulary_size))
            log_neg += math.log((neg_freq + 1) / (self.negative_total + self.vocabulary_size))
        max_log = max(log_pos, log_neg)
        pos_exp = math.exp(log_pos - max_log)
        neg_exp = math.exp(log_neg - max_log)
        return pos_exp / (pos_exp + neg_exp)

    def token_influence(self, text: str) -> Dict[str, float]:
        tokens = tokenize(text)
        influences: Dict[str, float] = defaultdict(float)
        for token in tokens:
            pos_prob = (self.positive_counts.get(token, 0) + 1) / (self.positive_total + self.vocabulary_size)
            neg_prob = (self.negative_counts.get(token, 0) + 1) / (self.negative_total + self.vocabulary_size)
            influences[token] += math.log(pos_prob / neg_prob)
        return influences

    def top_tokens(self, n: int = 3) -> List[str]:
        scores: Dict[str, float] = {}
        for token in self.vocabulary:
            pos_prob = (self.positive_counts.get(token, 0) + 1) / (self.positive_total + self.vocabulary_size)
            neg_prob = (self.negative_counts.get(token, 0) + 1) / (self.negative_total + self.vocabulary_size)
            scores[token] = math.log(pos_prob / neg_prob)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [f"{token} ({score:.2f})" for token, score in ranked[:n] if score > 0]


class LegalInsightClassifier:
    def __init__(self, path: Path) -> None:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        labels = sorted({label for item in dataset for label in item["labels"]})
        self._classifiers: dict[str, NaiveBayesBinary] = {}
        texts = [item["text"] for item in dataset]
        for label in labels:
            positive_texts = [item["text"] for item in dataset if label in item["labels"]]
            negative_texts = [text for text, item in zip(texts, dataset) if label not in item["labels"]]
            if positive_texts and negative_texts:
                self._classifiers[label] = NaiveBayesBinary(positive_texts, negative_texts)

    def predict_scores(self, sentences: Sequence[str]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for sentence in sentences:
            label_scores: dict[str, float] = {}
            for label, classifier in self._classifiers.items():
                label_scores[label] = classifier.score(sentence)
            results.append(label_scores)
        return results

    def explain(self, sentence: str, label: str, top_k: int = 3) -> list[str]:
        if label not in self._classifiers:
            return []
        classifier = self._classifiers[label]
        influences = classifier.token_influence(sentence)
        ranked = sorted(influences.items(), key=lambda item: item[1], reverse=True)
        return [f"{token} ({weight:.2f})" for token, weight in ranked[:top_k] if weight > 0]


@dataclass
class CasePrediction:
    probability: float
    rationale: str
    recommended_actions: list[str]
    signals: list[tuple[str, float]]
    negative_signals: list[tuple[str, float]]
    quality_warnings: list[str]
    signal_explanations: dict[str, str] = field(default_factory=dict)


class CaseOutcomeClassifier:
    def __init__(self, path: Path, llm_client: ClaudeClient | None = None) -> None:
        dataset = json.loads(path.read_text(encoding="utf-8"))
        positive_texts = [item["text"] for item in dataset if int(item["label"]) == 1]
        negative_texts = [item["text"] for item in dataset if int(item["label"]) == 0]
        self._classifier = NaiveBayesBinary(positive_texts, negative_texts)
        self._dataset = dataset
        self._llm = llm_client

    def predict(self, case_details: str) -> CasePrediction:
        probability = self._classifier.score(case_details)
        influences = self._classifier.token_influence(case_details)
        positive_signals = [(token, weight) for token, weight in influences.items() if weight > 0]
        negative_signals = [(token, weight) for token, weight in influences.items() if weight < 0]
        positive_signals.sort(key=lambda item: item[1], reverse=True)
        negative_signals.sort(key=lambda item: item[1])
        positive_signals = positive_signals[:5]
        negative_signals = negative_signals[:5]

        rationale_lines = [
            f"המודל מצא אינדיקציות חיוביות מרכזיות: {', '.join(token for token, _ in positive_signals) or 'לא זוהו'}.",
            f"אינדיקציות מחלישות: {', '.join(token for token, _ in negative_signals) or 'לא זוהו'}.",
        ]
        rationale = "\n".join(rationale_lines)
        recommended_actions = self._suggest_actions(probability, positive_signals, negative_signals)
        quality_warnings = self._quality_controls(probability, positive_signals, negative_signals)
        base_prediction = CasePrediction(
            probability=probability,
            rationale=rationale,
            recommended_actions=recommended_actions,
            signals=positive_signals,
            negative_signals=negative_signals,
            quality_warnings=quality_warnings,
        )
        return self._refine_prediction(case_details, base_prediction)

    def _refine_prediction(self, case_details: str, prediction: CasePrediction) -> CasePrediction:
        if not self._llm or not self._llm.is_configured:
            return prediction

        payload = {
            "case_details": case_details,
            "initial_probability": prediction.probability,
            "positive_signals": prediction.signals[:5],
            "negative_signals": prediction.negative_signals[:5],
            "recommended_actions": prediction.recommended_actions,
            "quality_warnings": prediction.quality_warnings,
        }
        system_prompt = (
            "אתה משמש כאנליסט Litigation. קבל ניתוח הסתברות ראשוני ומאפייני טקסט והחזר JSON "
            "עם השדות הבאים: probability_adjustment (מספר בין -0.25 ל-0.25), rationale (מחרוזת או רשימה), "
            "recommended_actions (רשימה), quality_warnings (רשימה), signal_evidence (אובייקט עם מפת אות->הסבר)."
        )
        try:
            raw = self._llm.generate(
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ],
                max_tokens=600,
            )
            data = json.loads(raw)
        except LLMGenerationError as exc:  # pragma: no cover - network path
            logger.warning("Claude prediction refinement failed: %s", exc)
            return prediction
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Claude prediction refinement returned non-JSON response", exc_info=exc)
            return prediction

        adjustment = data.get("probability_adjustment")
        probability = prediction.probability
        if isinstance(adjustment, (int, float)):
            probability = max(0.0, min(1.0, probability + float(adjustment)))

        rationale_value = data.get("rationale")
        if isinstance(rationale_value, list):
            rationale = "\n".join(str(item).strip() for item in rationale_value if str(item).strip()) or prediction.rationale
        elif isinstance(rationale_value, str) and rationale_value.strip():
            rationale = rationale_value.strip()
        else:
            rationale = prediction.rationale

        recommended_actions = prediction.recommended_actions
        if isinstance(data.get("recommended_actions"), list):
            refined_actions = [str(item).strip() for item in data["recommended_actions"] if str(item).strip()]
            if refined_actions:
                recommended_actions = refined_actions

        quality_warnings = prediction.quality_warnings
        if isinstance(data.get("quality_warnings"), list):
            refined_warnings = [str(item).strip() for item in data["quality_warnings"] if str(item).strip()]
            if refined_warnings:
                quality_warnings = refined_warnings

        signal_explanations = dict(prediction.signal_explanations)
        evidence_map = data.get("signal_evidence")
        if isinstance(evidence_map, dict):
            for token, explanation in evidence_map.items():
                if isinstance(explanation, str) and explanation.strip():
                    signal_explanations[token] = explanation.strip()

        return replace(
            prediction,
            probability=probability,
            rationale=rationale,
            recommended_actions=recommended_actions,
            quality_warnings=quality_warnings,
            signal_explanations=signal_explanations,
        )

    def _suggest_actions(
        self,
        probability: float,
        positive_pairs: Sequence[tuple[str, float]],
        negative_pairs: Sequence[tuple[str, float]],
    ) -> list[str]:
        actions: list[str] = []
        if probability < 0.6:
            actions.append("לחדד את הראיות התומכות ולבחון חיזוק באמצעות חוות דעת מומחה.")
        if negative_pairs:
            actions.append("לטפל בנקודות החולשה שזוהו על ידי המודל ולספק אסמכתאות מאזנות.")
        if not actions:
            actions.append('להיערך להצגת הראיות במתכונת סדורה ולבחון סגירת הסכסוך במו"מ.')
        return actions

    def _quality_controls(
        self,
        probability: float,
        positive_pairs: Sequence[tuple[str, float]],
        negative_pairs: Sequence[tuple[str, float]],
    ) -> list[str]:
        warnings: list[str] = []
        if 0.35 < probability < 0.65 and positive_pairs and negative_pairs:
            warnings.append("זוהתה תמונה מאוזנת – מומלץ להצליב נתונים נוספים לפני קבלת החלטה.")
        if probability < 0.3 and not positive_pairs:
            warnings.append("המודל לא מצא חיזוקים מרכזיים – יש לאסוף ראיות נוספות.")
        return warnings


@dataclass
class WitnessPlanPayload:
    strategy: str
    focus_areas: list[str]
    question_sets: list[dict[str, list[str]]]
    risk_controls: list[str]
    quality_notes: list[str]


class WitnessStrategyGenerator:
    def __init__(self, path: Path, llm_client: ClaudeClient | None = None) -> None:
        self._templates = json.loads(path.read_text(encoding="utf-8"))
        self._llm = llm_client

    def generate(
        self,
        witness_role: str,
        case_summary: str,
        matches: Sequence[DocumentMatch],
        knowledge_refs: Sequence[str],
    ) -> WitnessPlanPayload:
        template = self._select_template(witness_role, case_summary)
        focus_areas = list(dict.fromkeys(template.get("themes", [])))
        dynamic_focus = self._derive_focus_from_documents(matches)
        focus_areas.extend(item for item in dynamic_focus if item not in focus_areas)

        question_sets = list(template.get("question_sets", []))
        follow_up_questions = self._build_followups(matches)
        if follow_up_questions:
            question_sets.append({"stage": "חיזוק באמצעות המסמכים שלך", "questions": follow_up_questions})

        risk_controls = list(template.get("risk_controls", []))
        if knowledge_refs:
            risk_controls.append("לעגן את החקירה בסימוכין: " + "; ".join(knowledge_refs))

        quality_notes: list[str] = []
        if not matches:
            quality_notes.append("לא נמצאו מסמכים תומכים – מומלץ להשלים חומרים לפני הדיון.")
        if len(focus_areas) > 5:
            quality_notes.append("ריבוי נושאים – עדיף לבחור 3-4 קווי חקירה מרכזיים כדי למקד את הדיון.")

        payload = WitnessPlanPayload(
            strategy=template.get("strategy", ""),
            focus_areas=focus_areas,
            question_sets=question_sets,
            risk_controls=risk_controls,
            quality_notes=quality_notes,
        )
        return self._refine_plan_with_llm(
            witness_role,
            case_summary,
            matches,
            payload,
        )

    def _refine_plan_with_llm(
        self,
        witness_role: str,
        case_summary: str,
        matches: Sequence[DocumentMatch],
        payload: WitnessPlanPayload,
    ) -> WitnessPlanPayload:
        if not self._llm or not self._llm.is_configured:
            return payload

        context_matches = [
            {
                "filename": match.segment.filename,
                "snippet": textwrap.shorten(match.segment.text.replace("\n", " "), width=200, placeholder="..."),
                "score": round(match.score, 3),
            }
            for match in matches[:5]
        ]
        request_payload = {
            "witness_role": witness_role,
            "case_summary": case_summary,
            "current_plan": {
                "strategy": payload.strategy,
                "focus_areas": payload.focus_areas,
                "question_sets": payload.question_sets,
                "risk_controls": payload.risk_controls,
                "quality_notes": payload.quality_notes,
            },
            "document_matches": context_matches,
        }
        system_prompt = (
            "אתה בונה חקירה נגדית מתקדמת. החזר JSON עם השדות strategy, focus_areas, question_sets, "
            "risk_controls, quality_notes. שמור את המיקוד בעברית והקפד שקבוצות השאלות יכילו מפתח stage ורשימת שאלות."
        )

        try:
            raw = self._llm.generate(
                system_prompt,
                [
                    {
                        "role": "user",
                        "content": json.dumps(request_payload, ensure_ascii=False, indent=2),
                    }
                ],
                max_tokens=900,
            )
            data = json.loads(raw)
        except LLMGenerationError as exc:  # pragma: no cover - network path
            logger.warning("Claude witness planning failed, keeping template plan: %s", exc)
            return payload
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.warning("Claude witness planning returned non-JSON response", exc_info=exc)
            return payload

        strategy = data.get("strategy") if isinstance(data.get("strategy"), str) else payload.strategy

        def _normalise_list(name: str, fallback: list[str]) -> list[str]:
            value = data.get(name)
            if isinstance(value, list):
                cleaned = [str(item).strip() for item in value if str(item).strip()]
                if cleaned:
                    return cleaned
            return fallback

        focus_areas = _normalise_list("focus_areas", payload.focus_areas)
        risk_controls = _normalise_list("risk_controls", payload.risk_controls)
        quality_notes = _normalise_list("quality_notes", payload.quality_notes)

        question_sets_data = data.get("question_sets")
        question_sets: list[dict[str, list[str]]] = []
        if isinstance(question_sets_data, list):
            for item in question_sets_data:
                if not isinstance(item, dict):
                    continue
                stage = str(item.get("stage", "שלב"))
                questions_value = item.get("questions", [])
                if isinstance(questions_value, list):
                    questions = [str(q).strip() for q in questions_value if str(q).strip()]
                elif isinstance(questions_value, str):
                    questions = [questions_value.strip()]
                else:
                    questions = []
                if questions:
                    question_sets.append({"stage": stage, "questions": questions})
        if not question_sets:
            question_sets = payload.question_sets

        return WitnessPlanPayload(
            strategy=strategy.strip() if isinstance(strategy, str) and strategy.strip() else payload.strategy,
            focus_areas=focus_areas,
            question_sets=question_sets,
            risk_controls=risk_controls,
            quality_notes=quality_notes,
        )

    def _select_template(self, witness_role: str, case_summary: str) -> dict:
        role_lower = witness_role.lower()
        summary_lower = case_summary.lower()
        best_score = -1
        best_template = self._templates[0] if self._templates else {}
        for template in self._templates:
            score = 0
            if template["witness_role"].lower() == role_lower:
                score += 3
            for trigger in template.get("triggers", []):
                if trigger.lower() in summary_lower or trigger.lower() in role_lower:
                    score += 1
            if score > best_score:
                best_score = score
                best_template = template
        return best_template

    def _derive_focus_from_documents(self, matches: Sequence[DocumentMatch]) -> list[str]:
        focus: list[str] = []
        for match in matches[:4]:
            text = match.segment.text
            if "קנס" in text or "סיכון" in text:
                focus.append("בקרת סיכונים והתחייבויות")
            if "מועד" in text or "ימים" in text:
                focus.append("עמידה בלוחות זמנים")
            if "חוזה" in text or "הסכם" in text:
                focus.append("פרשנות סעיפי חוזה")
        return focus

    def _build_followups(self, matches: Sequence[DocumentMatch]) -> list[str]:
        followups: list[str] = []
        for match in matches[:3]:
            snippet = textwrap.shorten(match.segment.text.replace("\n", " "), width=160, placeholder="...")
            followups.append(
                f"בהתייחס למסמך {match.segment.filename}: כיצד אתה מסביר את הקטע הבא – '{snippet}'?"
            )
        return followups


__all__ = [
    "CaseOutcomeClassifier",
    "CasePrediction",
    "DocumentMatch",
    "DocumentSegment",
    "DocumentVectorStore",
    "HEBREW_STOP_WORDS",
    "LegalInsightClassifier",
    "LegalKnowledgeBase",
    "LegalRagEngine",
    "RagAnswer",
    "STOP_WORDS",
    "WitnessPlanPayload",
    "WitnessStrategyGenerator",
]
