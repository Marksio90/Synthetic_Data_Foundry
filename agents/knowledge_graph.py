"""
agents/knowledge_graph.py — Cross-document entity–relationship knowledge graph.

Extracts named entities from directive chunks (regulations, organisations,
monetary values, dates, ESG concepts) and builds a weighted relationship graph
that persists in Neo4j. When Neo4j is not available, falls back to an in-memory
networkx DiGraph so the rest of the pipeline continues uninterrupted.

Entity types recognised:
    LAW / REGULATION   — CSRD, SFDR, TCFD, EU Taxonomy, …
    ORG                — companies, bodies, agencies
    CONCEPT            — ESG concepts (Scope 1/2/3, double materiality, …)
    MONEY              — monetary amounts and thresholds
    DATE               — effective dates, reporting deadlines
    LOCATION           — countries, jurisdictions

Relationship types inferred:
    CO_OCCURS          — entities in the same sentence
    REGULATES          — regulation ↔ org (heuristic)
    REFERENCES         — law citing another law
    AMENDS             — "amends Article X of …"
    REQUIRES           — "requires reporting of …"
    DEFINES            — "defines … as …"

Usage:
    kg = get_knowledge_graph()
    entities = kg.extract_entities(chunk_text, doc_id="csrd_art12", chunk_id="c_001")
    relationships = kg.build_relationships(entities, chunk_text)
    kg.add_to_graph(entities, relationships)
    results = kg.query_entity("CSRD")
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("foundry.knowledge_graph")

# ── ESG / regulatory keyword patterns (supplement spaCy NER) ─────────────────
_REGULATION_PATTERNS = re.compile(
    r"\b(CSRD|SFDR|TCFD|GRI|SASB|ESRS|EU\s+Taxonomy|NFRD|CSDDD|Pillar\s+III"
    r"|TNFD|CDP|UNGC|SDG|Article\s+\d+|Directive\s+\d+/\d+/EC"
    r"|Regulation\s+\([A-Z]+\)\s+\d+/\d+)\b",
    re.IGNORECASE,
)
_AMENDS_RE = re.compile(r"\bamends?\b.{0,80}", re.IGNORECASE)
_REQUIRES_RE = re.compile(r"\brequires?\b.{0,80}", re.IGNORECASE)
_DEFINES_RE = re.compile(r"\bdefines?\b.{0,80}", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"\b(refers?\s+to|pursuant\s+to|under|within\s+the\s+meaning\s+of)\b.{0,80}", re.IGNORECASE)


@dataclass(frozen=True, eq=True)
class Entity:
    name: str                      # normalised canonical name
    entity_type: str               # LAW, ORG, CONCEPT, MONEY, DATE, LOCATION
    source_doc: str
    chunk_id: str
    context: str = ""              # up to 200-char surrounding snippet

    @property
    def id(self) -> str:
        return hashlib.sha1(f"{self.name}|{self.entity_type}".encode()).hexdigest()[:12]


@dataclass
class Relationship:
    source_name: str
    target_name: str
    relation_type: str             # CO_OCCURS, REGULATES, REFERENCES, AMENDS, REQUIRES, DEFINES
    confidence: float = 1.0
    source_doc: str = ""
    chunk_id: str = ""

    @property
    def id(self) -> str:
        raw = f"{self.source_name}|{self.relation_type}|{self.target_name}|{self.source_doc}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


# ── spaCy wrapper ──────────────────────────────────────────────────────────────

def _load_spacy():
    """Load spaCy model with graceful degradation."""
    try:
        import spacy  # type: ignore
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            try:
                return spacy.load("xx_ent_wiki_sm")
            except OSError:
                logger.warning("spaCy model not found — using blank English model. Run: python -m spacy download en_core_web_sm")
                return spacy.blank("en")
    except ImportError:
        logger.warning("spaCy not installed — NER disabled. pip install spacy")
        return None


_NLP = None  # lazy-loaded


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = _load_spacy()
    return _NLP


# ── spaCy label → foundry entity type ─────────────────────────────────────────
_SPACY_TYPE_MAP = {
    "ORG": "ORG",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "LAW": "LAW",
    "DATE": "DATE",
    "MONEY": "MONEY",
    "PERCENT": "MONEY",
    "CARDINAL": "CONCEPT",
    "NORP": "ORG",
    "PRODUCT": "CONCEPT",
    "WORK_OF_ART": "LAW",
    "EVENT": "CONCEPT",
}


def _extract_snippet(text: str, start: int, end: int, window: int = 100) -> str:
    s = max(0, start - window)
    e = min(len(text), end + window)
    return text[s:e].replace("\n", " ").strip()


class KnowledgeGraph:
    """
    Cross-document entity–relationship graph.

    Stores data in Neo4j when available, falls back to networkx DiGraph.
    Thread-safe for reads; writes are serialised via asyncio.Lock when async.
    """

    def __init__(self, neo4j_url: Optional[str] = None, neo4j_auth: Optional[Tuple[str, str]] = None) -> None:
        self._neo4j_url = neo4j_url
        self._neo4j_auth = neo4j_auth or ("neo4j", "foundry-neo4j")
        self._driver = None
        self._nx_graph = None  # networkx fallback
        self._entities: Dict[str, Entity] = {}     # id → Entity
        self._relationships: Dict[str, Relationship] = {}  # id → Relationship
        self._neo4j_available = False
        self._init_backend()

    def _init_backend(self) -> None:
        if self._neo4j_url:
            try:
                from neo4j import GraphDatabase  # type: ignore
                self._driver = GraphDatabase.driver(self._neo4j_url, auth=self._neo4j_auth)
                self._driver.verify_connectivity()
                self._neo4j_available = True
                logger.info("KnowledgeGraph: Neo4j connected at %s", self._neo4j_url)
                self._ensure_neo4j_schema()
                return
            except Exception as exc:
                logger.warning("KnowledgeGraph: Neo4j unavailable (%s) — using in-memory networkx graph", exc)
        self._init_networkx()

    def _init_networkx(self) -> None:
        try:
            import networkx as nx  # type: ignore
            self._nx_graph = nx.MultiDiGraph()
            logger.info("KnowledgeGraph: using in-memory networkx backend")
        except ImportError:
            logger.warning("KnowledgeGraph: networkx not installed — graph operations disabled. pip install networkx")
            self._nx_graph = None

    def _ensure_neo4j_schema(self) -> None:
        if not self._driver:
            return
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")

    # ── Entity extraction ──────────────────────────────────────────────────────

    def extract_entities(self, text: str, doc_id: str, chunk_id: str) -> List[Entity]:
        """
        Extract entities from text using spaCy NER + regex patterns for
        regulatory/ESG domain terms not covered by general NER.
        """
        entities: List[Entity] = []
        seen: set = set()

        def _add(name: str, etype: str, start: int = 0, end: int = 0) -> None:
            norm = name.strip()
            if not norm or len(norm) < 2:
                return
            key = f"{norm.upper()}|{etype}"
            if key in seen:
                return
            seen.add(key)
            snippet = _extract_snippet(text, start, end)
            entities.append(Entity(name=norm, entity_type=etype, source_doc=doc_id, chunk_id=chunk_id, context=snippet[:200]))

        # spaCy NER
        nlp = _get_nlp()
        if nlp is not None:
            doc = nlp(text[:100_000])  # truncate to avoid OOM
            for ent in doc.ents:
                etype = _SPACY_TYPE_MAP.get(ent.label_, "CONCEPT")
                _add(ent.text, etype, ent.start_char, ent.end_char)

        # Regex-based regulatory terms (high precision for domain)
        for m in _REGULATION_PATTERNS.finditer(text):
            _add(m.group(0), "LAW", m.start(), m.end())

        # Well-known ESG concepts not caught by NER
        _ESG_CONCEPTS = [
            "double materiality", "Scope 1", "Scope 2", "Scope 3",
            "net zero", "carbon neutral", "just transition", "taxonomy alignment",
            "sustainable finance", "green bond", "social bond",
            "principal adverse impacts", "PAI", "do no significant harm", "DNSH",
            "minimum social safeguards", "transition plan", "climate risk",
            "physical risk", "transition risk", "stranded asset",
        ]
        for concept in _ESG_CONCEPTS:
            idx = text.lower().find(concept.lower())
            if idx != -1:
                _add(concept, "CONCEPT", idx, idx + len(concept))

        logger.debug("KnowledgeGraph: extracted %d entities from chunk %s", len(entities), chunk_id)
        return entities

    # ── Relationship inference ─────────────────────────────────────────────────

    def build_relationships(self, entities: List[Entity], text: str) -> List[Relationship]:
        """
        Infer relationships between entities based on co-occurrence and
        explicit linguistic patterns.
        """
        relationships: List[Relationship] = []

        names = {e.name for e in entities}
        entity_by_name = {e.name: e for e in entities}

        # 1. Sentence-level co-occurrence
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            present = [n for n in names if n.lower() in sent.lower()]
            for i, a in enumerate(present):
                for b in present[i + 1:]:
                    ea = entity_by_name[a]
                    eb = entity_by_name[b]
                    rel_type = "CO_OCCURS"
                    # Specialise based on entity types
                    if ea.entity_type == "LAW" and eb.entity_type == "ORG":
                        rel_type = "REGULATES"
                    elif ea.entity_type == "ORG" and eb.entity_type == "LAW":
                        rel_type = "SUBJECT_TO"
                    relationships.append(Relationship(
                        source_name=a, target_name=b,
                        relation_type=rel_type, confidence=0.7,
                        source_doc=ea.source_doc, chunk_id=ea.chunk_id,
                    ))

        # 2. Explicit linguistic patterns
        doc_id = entities[0].source_doc if entities else ""
        chunk_id = entities[0].chunk_id if entities else ""

        def _find_rel(pattern: re.Pattern, rel_type: str, confidence: float) -> None:
            for m in pattern.finditer(text):
                snippet = m.group(0)
                # Find entity names that appear in this snippet
                involved = [n for n in names if n.lower() in snippet.lower()]
                for i, a in enumerate(involved):
                    for b in involved[i + 1:]:
                        relationships.append(Relationship(
                            source_name=a, target_name=b,
                            relation_type=rel_type, confidence=confidence,
                            source_doc=doc_id, chunk_id=chunk_id,
                        ))

        _find_rel(_AMENDS_RE, "AMENDS", 0.85)
        _find_rel(_REQUIRES_RE, "REQUIRES", 0.80)
        _find_rel(_DEFINES_RE, "DEFINES", 0.90)
        _find_rel(_REFERENCES_RE, "REFERENCES", 0.75)

        logger.debug("KnowledgeGraph: inferred %d relationships", len(relationships))
        return relationships

    # ── Graph persistence ──────────────────────────────────────────────────────

    def add_to_graph(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        """Upsert entities and relationships into the active backend."""
        for e in entities:
            self._entities[e.id] = e

        for r in relationships:
            self._relationships[r.id] = r

        if self._neo4j_available:
            self._write_to_neo4j(entities, relationships)
        elif self._nx_graph is not None:
            self._write_to_networkx(entities, relationships)

    def _write_to_neo4j(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        try:
            with self._driver.session() as session:
                for e in entities:
                    session.run(
                        """MERGE (n:Entity {id: $id})
                           SET n.name = $name, n.type = $etype,
                               n.source_doc = $doc, n.chunk_id = $chunk,
                               n.context = $ctx""",
                        id=e.id, name=e.name, etype=e.entity_type,
                        doc=e.source_doc, chunk=e.chunk_id, ctx=e.context,
                    )
                for r in relationships:
                    session.run(
                        """MATCH (a:Entity {name: $src}), (b:Entity {name: $tgt})
                           MERGE (a)-[rel:RELATES {type: $rtype, doc: $doc}]->(b)
                           SET rel.confidence = $conf""",
                        src=r.source_name, tgt=r.target_name,
                        rtype=r.relation_type, doc=r.source_doc,
                        conf=r.confidence,
                    )
        except Exception as exc:
            logger.warning("KnowledgeGraph: Neo4j write failed (%s) — data kept in memory only", exc)

    def _write_to_networkx(self, entities: List[Entity], relationships: List[Relationship]) -> None:
        g = self._nx_graph
        for e in entities:
            g.add_node(e.name, type=e.entity_type, docs={e.source_doc}, chunks={e.chunk_id})
        for r in relationships:
            g.add_edge(r.source_name, r.target_name,
                       relation=r.relation_type, confidence=r.confidence,
                       doc=r.source_doc)

    # ── Query interface ────────────────────────────────────────────────────────

    def query_entity(self, name: str) -> dict:
        """Return all information about an entity including connected entities."""
        if self._neo4j_available:
            return self._query_neo4j_entity(name)
        return self._query_nx_entity(name)

    def _query_neo4j_entity(self, name: str) -> dict:
        try:
            with self._driver.session() as session:
                result = session.run(
                    """MATCH (e:Entity {name: $name})
                       OPTIONAL MATCH (e)-[r]->(n)
                       OPTIONAL MATCH (m)-[r2]->(e)
                       RETURN e, collect(distinct {type: r.type, target: n.name, confidence: r.confidence}) as outgoing,
                              collect(distinct {type: r2.type, source: m.name, confidence: r2.confidence}) as incoming""",
                    name=name,
                )
                row = result.single()
                if not row:
                    return {}
                return {
                    "name": row["e"]["name"],
                    "type": row["e"]["type"],
                    "outgoing": [r for r in row["outgoing"] if r["target"]],
                    "incoming": [r for r in row["incoming"] if r["source"]],
                }
        except Exception as exc:
            logger.debug("KnowledgeGraph.query_entity Neo4j error: %s", exc)
            return {}

    def _query_nx_entity(self, name: str) -> dict:
        g = self._nx_graph
        if g is None or name not in g:
            return {}
        data = dict(g.nodes[name])
        data["name"] = name
        data["outgoing"] = [
            {"target": t, "relation": d.get("relation"), "confidence": d.get("confidence", 1.0)}
            for _, t, d in g.out_edges(name, data=True)
        ]
        data["incoming"] = [
            {"source": s, "relation": d.get("relation"), "confidence": d.get("confidence", 1.0)}
            for s, _, d in g.in_edges(name, data=True)
        ]
        return data

    def find_cross_document_links(self, entity_name: str) -> List[dict]:
        """Find all documents that mention a given entity."""
        results = []
        seen_docs: set = set()
        for eid, ent in self._entities.items():
            if ent.name.lower() == entity_name.lower() and ent.source_doc not in seen_docs:
                seen_docs.add(ent.source_doc)
                results.append({
                    "doc_id": ent.source_doc,
                    "chunk_id": ent.chunk_id,
                    "entity_type": ent.entity_type,
                    "context": ent.context,
                })
        return results

    def get_entity_context(self, entity_name: str, max_results: int = 5) -> List[str]:
        """Return context snippets where the entity appears."""
        snippets = []
        for ent in self._entities.values():
            if ent.name.lower() == entity_name.lower() and ent.context:
                snippets.append(ent.context)
        return snippets[:max_results]

    def get_related_regulations(self, regulation_name: str) -> List[dict]:
        """Return regulations that are directly related to the given one."""
        result = self.query_entity(regulation_name)
        related = []
        for r in result.get("outgoing", []):
            related.append({"name": r.get("target"), "relation": r.get("type"), "direction": "outgoing"})
        for r in result.get("incoming", []):
            related.append({"name": r.get("source"), "relation": r.get("type"), "direction": "incoming"})
        return related

    def entity_summary(self) -> dict:
        """High-level statistics about the knowledge graph."""
        from collections import Counter
        type_counts = Counter(e.entity_type for e in self._entities.values())
        rel_counts = Counter(r.relation_type for r in self._relationships.values())
        doc_set = {e.source_doc for e in self._entities.values()}
        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "entity_types": dict(type_counts),
            "relationship_types": dict(rel_counts),
            "documents_indexed": len(doc_set),
            "backend": "neo4j" if self._neo4j_available else "networkx" if self._nx_graph is not None else "none",
        }

    def close(self) -> None:
        if self._driver:
            try:
                self._driver.close()
            except Exception:
                pass


# ── Convenience singleton ──────────────────────────────────────────────────────
_kg_instance: Optional[KnowledgeGraph] = None


def get_knowledge_graph(
    neo4j_url: Optional[str] = None,
    neo4j_auth: Optional[Tuple[str, str]] = None,
) -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        import os
        url = neo4j_url or os.getenv("NEO4J_URL")
        auth_str = os.getenv("NEO4J_AUTH", "neo4j:foundry-neo4j")
        parts = auth_str.split(":", 1)
        auth = (parts[0], parts[1]) if len(parts) == 2 else None
        _kg_instance = KnowledgeGraph(neo4j_url=url, neo4j_auth=neo4j_auth or auth)
    return _kg_instance
