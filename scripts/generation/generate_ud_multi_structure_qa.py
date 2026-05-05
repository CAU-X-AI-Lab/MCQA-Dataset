import argparse
import csv
import html
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
except ImportError:
    pass

import networkx as nx
from tqdm import tqdm


DEFAULT_CONFIG = {
    # Command line --api-key and OPENAI_API_KEY override this value.
    "api_key": "sk-QWtRU87Af1ILa6kyJhsWzhxx1r5by3ijpbuPGTUVgvrW3LP8",
    "input_root": r"D:\MCQA\UD\ultradomain",
    "output_root": r"D:\MCQA\UD\UD-MultiStructure-QA",
    "checkpoint_file": r"D:\MCQA\UD\ud_multi_structure_qa.checkpoint.json",
    "error_log_file": r"D:\MCQA\UD\ud_multi_structure_qa.errors.jsonl",
    "base_url": "https://ai.nengyongai.cn/v1",
    "model": "gpt-4o-mini",
    "qps_limit": 5,
    "max_workers": 10,
    "num_per_type": 1000,
    "candidate_multiplier": 10,
    "seed": 42,
}


STRUCTURE_TYPES = ["single_edge", "path_4", "star_1hop", "star_2hop", "cycle"]
GENERATION_VERSION = "ud_multi_structure_api_v5"


SYSTEM_PROMPT = (
    "You create English knowledge-graph QA queries from domain graph evidence. "
    "The query must sound like a normal human question, have exactly one entity answer, "
    "not reveal the answer, and not add facts beyond the evidence. "
    "Use the edge descriptions as concrete constraints instead of vague association wording."
)


QUERY_PROMPT_TEMPLATE = """Generate one English QA query from the domain graph evidence.

Domain: {domain}
Structure type: {structure_type}
Gold answer: {answer}
Entities that must appear verbatim in the query: {required_entities}
Structure-specific requirements:
{structure_guidance}

Readable evidence constraints:
{constraints}

Hard rules:
1. Output an English query only inside JSON.
2. The query must have exactly one entity answer, which is the gold answer.
3. The query must not contain the gold answer.
4. Every required entity must appear verbatim in the query.
5. Do not ask for a list, multiple answers, advice, comparison, or explanation.
6. Do not mention graph terms such as path, star, center node, two-hop, cycle, loop, graph, triple, evidence, or entity.
7. Avoid vague wording such as "associated with", "related to", "connected to what", "what does this indicate", or "the answer".
8. Prefer one clear human question over stylistic variety.
9. Use the relationship descriptions as meaningful constraints. If a description is awkward, paraphrase it conservatively without adding new facts.
10. Copy every required entity exactly as shown. Do not replace a required entity with a pronoun, synonym, or shorter phrase.
11. The query must ask for an entity. Start with "Which", "What", or "Who". Do not use "how" or "why" anywhere.

Return only JSON:
{{
  "query": "...",
  "answer": "{answer}",
  "used_evidence_indices": [1, 2, 3]
}}
"""


REVIEW_PROMPT_TEMPLATE = """Review whether this English knowledge-graph query is acceptable.

Domain: {domain}
Structure type: {structure_type}
Gold answer: {answer}
Required entities: {required_entities}
Structure-specific requirements:
{structure_guidance}

Query:
{query}

Readable evidence constraints:
{constraints}

Review criteria:
1. It sounds like a plausible human question.
2. It asks exactly one question.
3. It asks for exactly one entity answer, which is the gold answer.
4. It does not ask for a list or multiple answers.
5. It includes every required entity verbatim and does not reveal the gold answer.
6. It does not add facts outside the evidence constraints.
7. It uses concrete relation meanings rather than vague words such as "associated with", "related to", or "connected to".
8. For star structures, every branch narrows down the same missing answer.
9. For cycle structures, the relations among the ring nodes are expressed, and if a shared answer exists, the query asks for that shared answer.
10. It starts with "Which", "What", or "Who" and does not use "how" or "why" anywhere.

Return only JSON:
{{
  "accepted": true,
  "score": 1,
  "single_answer_entity": true,
  "asks_multiple_questions": false,
  "asks_for_list": false,
  "relation_coverage_ok": true,
  "cycle_relations_expressed": true,
  "reason": "brief reason"
}}

accepted can be true only when score >= 4 and all boolean checks are satisfied.
"""


def clean_text(value):
    value = "" if value is None else str(value)
    value = html.unescape(value)
    value = value.replace("<SEP>", "; ").replace("&lt;SEP&gt;", "; ").replace("|>", "; ")
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("'\"").strip()
    return "" if value.lower() == "nan" else value


def clean_entity(value):
    value = clean_text(value)
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return clean_text(value)


def is_good_entity(value):
    value = clean_entity(value)
    if not value or len(value) < 2 or len(value) > 70:
        return False
    bad = {"entity", "category", "event", "organization", "person", "location", "concept"}
    if value.lower() in bad or value.lower().startswith("chunk-"):
        return False
    if value.count(" ") > 10:
        return False
    if any(token in value for token in ["鈥", "峒", "埼", "曃", "溛", "�", "{", "}", '("']):
        return False
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", value):
        return False
    return True


def edge_description(attrs):
    description = clean_text(attrs.get("description", ""))
    keywords = clean_text(attrs.get("keywords", ""))
    text = description or keywords
    text = re.sub(r"\(\"?entity\"?.*", "", text).strip()
    return clean_text(text)


def truncate_sentence(text, limit=220):
    text = clean_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(" ,.;") + "."


def make_edge(domain, u, v, attrs):
    return {
        "domain": domain,
        "subject": clean_entity(u),
        "subject_id": str(u),
        "relation": truncate_sentence(edge_description(attrs), 220) or "is connected in the source text with",
        "object": clean_entity(v),
        "object_id": str(v),
        "source_id": clean_text(attrs.get("source_id", "")),
        "keywords": clean_text(attrs.get("keywords", "")),
    }


def edge_key(edge):
    return (edge["subject"], edge["relation"], edge["object"])


def reverse_edge(edge):
    rev = dict(edge)
    rev["subject"], rev["object"] = edge["object"], edge["subject"]
    rev["subject_id"], rev["object_id"] = edge["object_id"], edge["subject_id"]
    return rev


def sample_entity_nodes(sample):
    nodes = []
    seen = set()
    for edge in sample["triples"]:
        for key in ("subject", "object"):
            node = clean_entity(edge[key])
            if node not in seen:
                nodes.append(node)
                seen.add(node)
    return nodes


def required_query_entities(sample):
    answer = clean_entity(sample["answer"])
    return [node for node in sample_entity_nodes(sample) if clean_entity(node) != answer]


def missing_required_entities(sample, query):
    query_text = clean_text(query)
    return [entity for entity in required_query_entities(sample) if clean_text(entity) not in query_text]


def answer_is_exposed(sample, query):
    answer = clean_entity(sample["answer"])
    query_text = clean_text(query)
    for entity in required_query_entities(sample):
        query_text = query_text.replace(clean_text(entity), "")
    return answer and answer in query_text


MULTI_ANSWER_TERMS = [
    "which entities",
    "which ones",
    "what are all",
    "list",
    "respectively",
    "how are they different",
    "what should",
    "what would you recommend",
]


UNNATURAL_TERMS = [
    "the answer",
    "the entity",
    "associated with",
    "connected to",
    "related to",
    "connection to",
    "connections to",
    "connected to what",
    "related to what",
    "what does this indicate",
    "how does",
    "how did",
    "how is",
    "how are",
    "why does",
    "why did",
    "taken together, what",
    "has a relationship with",
    "share a connection",
]


def has_multiple_question_risk(query):
    query_lower = clean_text(query).lower()
    if query_lower.count("?") > 1:
        return True
    return any(re.search(r"\b" + re.escape(term) + r"\b", query_lower) for term in MULTI_ANSWER_TERMS)


def validate_query(sample, query):
    query = clean_text(query)
    if not query:
        raise ValueError("empty query")
    banned_terms = [" path", " star", "center node", "two-hop", "cycle", "loop", " graph", " triple", " evidence"]
    if any(term in query.lower() for term in banned_terms):
        raise ValueError(f"query contains graph/data term: {query}")
    if any(term in query.lower() for term in UNNATURAL_TERMS):
        raise ValueError(f"query contains unnatural wording: {query}")
    if has_multiple_question_risk(query):
        raise ValueError("query has multiple-answer or multiple-question risk")
    if re.search(r"\b(how|why)\b", query, re.I):
        raise ValueError("query asks for an explanation instead of an entity")
    if not re.match(r"^(which|what|who)\b", query.strip(), re.I):
        raise ValueError("query does not start with an entity-seeking question word")
    if answer_is_exposed(sample, query):
        raise ValueError("query exposes the answer")
    missing = missing_required_entities(sample, query)
    if missing:
        raise ValueError(f"query misses required entities: {missing}")


def relation_constraints_text(sample):
    lines = []
    for index, edge in enumerate(sample["triples"], start=1):
        lines.append(f"{index}. {edge['subject']} - {edge['relation']} - {edge['object']}")
    return "\n".join(lines)


def structure_guidance(sample):
    structure_type = sample["structure_type"]
    if structure_type == "single_edge":
        return (
            "Single-edge query. Use the one relationship as a precise constraint. "
            "Ask for the gold answer only, not for an open-ended set of related items."
        )
    if structure_type == "path_4":
        return (
            "Four-node path query. Ask for the selected internal answer node, not the first or last node. "
            "The query must mention the non-answer nodes and express the consecutive constraints naturally."
        )
    if structure_type == "star_1hop":
        return (
            "One-hop star query. The gold answer is the shared center. "
            "Use every neighboring entity as a constraint that identifies the same missing answer."
        )
    if structure_type == "star_2hop":
        return (
            "Two-hop star query. The gold answer is the shared center. "
            "Use every branch as a constraint, and ask one question for the shared missing answer."
        )
    if structure_type == "cycle":
        if sample.get("common_edges"):
            return (
                "Cycle query with a shared answer. Express the relations among the ring nodes, then ask for the entity "
                "that is also connected to all ring nodes. Do not merely list the ring nodes."
            )
        return (
            "Cycle query. Express the relations among the ring nodes and ask for one selected ring entity. "
            "Do not use words such as cycle, loop, or graph."
        )
    return "Ask one natural question for the gold answer."


class QpsLimiter:
    def __init__(self, qps):
        self.qps = max(int(qps), 1)
        self.lock = threading.Lock()
        self.call_times = []

    def wait(self):
        with self.lock:
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < 1.0]
            if len(self.call_times) >= self.qps:
                sleep_for = max(1.0 - (now - self.call_times[0]), 0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self.call_times.append(time.time())


def extract_json(text):
    text = clean_text(text)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_query_prompt(sample):
    return QUERY_PROMPT_TEMPLATE.format(
        domain=sample["domain"],
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=", ".join(required_query_entities(sample)),
        structure_guidance=structure_guidance(sample),
        constraints=relation_constraints_text(sample),
    )


def build_review_prompt(sample, query):
    return REVIEW_PROMPT_TEMPLATE.format(
        domain=sample["domain"],
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=", ".join(required_query_entities(sample)),
        structure_guidance=structure_guidance(sample),
        query=query,
        constraints=relation_constraints_text(sample),
    )


def call_chat_json(client, limiter, model, messages, temperature):
    limiter.wait()
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=False)
    return extract_json(response.choices[0].message.content)


def review_query(client, limiter, model, sample, query):
    data = call_chat_json(
        client,
        limiter,
        model,
        [
            {"role": "system", "content": "You are a strict English QA quality reviewer. Return only JSON."},
            {"role": "user", "content": build_review_prompt(sample, query)},
        ],
        0.1,
    )
    score = int(data.get("score", 0) or 0)
    checks = {
        "accepted": bool(data.get("accepted", False)),
        "score": score,
        "single_answer_entity": bool(data.get("single_answer_entity", False)),
        "asks_multiple_questions": bool(data.get("asks_multiple_questions", True)),
        "asks_for_list": bool(data.get("asks_for_list", True)),
        "relation_coverage_ok": bool(data.get("relation_coverage_ok", False)),
        "cycle_relations_expressed": bool(data.get("cycle_relations_expressed", sample["structure_type"] != "cycle")),
        "reason": clean_text(data.get("reason", "")),
        "source": "api",
    }
    if re.search(r"\b(how|why)\b", query, re.I):
        raise ValueError("query rejected by reviewer: explanation-style question")
    if (
        score < 3
        or not checks["single_answer_entity"]
        or checks["asks_multiple_questions"]
        or checks["asks_for_list"]
        or (sample["structure_type"] == "cycle" and not checks["cycle_relations_expressed"])
    ):
        raise ValueError(f"query rejected by reviewer: score={score}, reason={checks['reason']}")
    return checks


def generate_query(client, limiter, model, sample, retries, enable_review=True):
    base_prompt = build_query_prompt(sample)
    last_error = None
    for attempt in range(retries + 1):
        try:
            prompt = base_prompt
            if last_error:
                prompt += f"\n\nPrevious attempt failed: {last_error}\nRegenerate and fix that specific problem."
            data = call_chat_json(
                client,
                limiter,
                model,
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                0.45,
            )
            query = clean_text(data.get("query", ""))
            validate_query(sample, query)
            review = {"accepted": True, "score": 5, "reason": "review disabled", "source": "api"}
            if enable_review:
                review = review_query(client, limiter, model, sample, query)
            return query, review
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API query generation failed: {last_error}")


def graphml_files(input_root):
    return sorted(Path(input_root).glob("*/graph_chunk_entity_relation.graphml"))


def load_domain_graph(path):
    domain = path.parent.name
    graph = nx.read_graphml(path)
    simple = nx.Graph()
    edge_lookup = {}
    for node, attrs in graph.nodes(data=True):
        label = clean_entity(node)
        if not is_good_entity(label):
            continue
        simple.add_node(str(node), label=label, **{k: clean_text(v) for k, v in attrs.items()})
    for u, v, attrs in graph.edges(data=True):
        if str(u) not in simple or str(v) not in simple or str(u) == str(v):
            continue
        edge = make_edge(domain, u, v, attrs)
        if not (is_good_entity(edge["subject"]) and is_good_entity(edge["object"])):
            continue
        simple.add_edge(str(u), str(v), **edge)
        edge_lookup[frozenset((str(u), str(v)))] = edge
    return domain, simple, edge_lookup


def get_edge(edge_lookup, u, v, subject=None):
    edge = dict(edge_lookup[frozenset((str(u), str(v)))])
    if subject is not None and edge["subject_id"] != str(subject):
        edge = reverse_edge(edge)
    elif edge["subject_id"] != str(u):
        edge = reverse_edge(edge)
    return edge


def add_unique_sample(selected, used_answers, sample, count):
    answer_key = clean_entity(sample["answer"]).lower()
    if answer_key in used_answers:
        return False
    for entity in required_query_entities(sample):
        entity_key = clean_entity(entity).lower()
        if len(answer_key) >= 4 and answer_key in entity_key:
            return False
        if len(entity_key) >= 4 and entity_key in answer_key:
            return False
    used_answers.add(answer_key)
    selected.append(sample)
    return len(selected) >= count


def sample_single_edges(domain_graphs, count):
    selected = []
    used_answers = set()
    candidates = []
    for domain, graph, edge_lookup in domain_graphs:
        for u, v in graph.edges():
            edge = get_edge(edge_lookup, u, v)
            candidates.append((domain, edge))
    random.shuffle(candidates)
    for domain, edge in tqdm(candidates, desc="Sampling UD single_edge", unit="edge"):
        answer_edge = edge if random.random() < 0.5 else reverse_edge(edge)
        sample = {
            "domain": domain,
            "structure_type": "single_edge",
            "answer": answer_edge["object"],
            "triples": [answer_edge],
        }
        if add_unique_sample(selected, used_answers, sample, count):
            break
    return selected


def sample_paths(domain_graphs, count):
    selected = []
    used_answers = set()
    domains = list(domain_graphs)
    random.shuffle(domains)
    for domain, graph, edge_lookup in tqdm(domains, desc="Sampling UD path_4 domains", unit="domain"):
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        for a in nodes:
            neigh_b = list(graph.neighbors(a))
            random.shuffle(neigh_b)
            for b in neigh_b[:50]:
                neigh_c = list(graph.neighbors(b))
                random.shuffle(neigh_c)
                for c in neigh_c[:50]:
                    if c in {a, b}:
                        continue
                    neigh_d = list(graph.neighbors(c))
                    random.shuffle(neigh_d)
                    for d in neigh_d[:50]:
                        if d in {a, b, c}:
                            continue
                        triples = [
                            get_edge(edge_lookup, a, b, subject=a),
                            get_edge(edge_lookup, b, c, subject=b),
                            get_edge(edge_lookup, c, d, subject=c),
                        ]
                        sample = {
                            "domain": domain,
                            "structure_type": "path_4",
                            "answer": triples[1]["subject"],
                            "triples": triples,
                        }
                        if add_unique_sample(selected, used_answers, sample, count):
                            return selected
                        break
                    if len(selected) >= count:
                        return selected
    return selected


def sample_star_1hop(domain_graphs, count, min_neighbors=3, max_neighbors=4):
    selected = []
    used_answers = set()
    candidates = []
    for domain, graph, edge_lookup in domain_graphs:
        for node, degree in graph.degree():
            if degree >= min_neighbors:
                candidates.append((domain, graph, edge_lookup, node, degree))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[4], reverse=True)
    for domain, graph, edge_lookup, center, _ in tqdm(candidates, desc="Sampling UD star_1hop", unit="center"):
        neighbors = list(graph.neighbors(center))
        random.shuffle(neighbors)
        neighbors = neighbors[: min(max_neighbors, len(neighbors))]
        if len(neighbors) < min_neighbors:
            continue
        triples = [get_edge(edge_lookup, center, n, subject=center) for n in neighbors]
        sample = {
            "domain": domain,
            "structure_type": "star_1hop",
            "answer": triples[0]["subject"],
            "triples": triples,
        }
        if add_unique_sample(selected, used_answers, sample, count):
            break
    return selected


def sample_star_2hop(domain_graphs, count, branches=2):
    selected = []
    used_answers = set()
    candidates = []
    for domain, graph, edge_lookup in domain_graphs:
        for node, degree in graph.degree():
            if degree >= branches:
                candidates.append((domain, graph, edge_lookup, node, degree))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[4], reverse=True)
    for domain, graph, edge_lookup, center, _ in tqdm(candidates, desc="Sampling UD star_2hop", unit="center"):
        used_nodes = {center}
        pairs = []
        mids = list(graph.neighbors(center))
        random.shuffle(mids)
        for mid in mids:
            if mid in used_nodes:
                continue
            seconds = [node for node in graph.neighbors(mid) if node not in used_nodes and node != center]
            random.shuffle(seconds)
            if not seconds:
                continue
            leaf = seconds[0]
            pairs.append((mid, leaf))
            used_nodes.update({mid, leaf})
            if len(pairs) >= branches:
                break
        if len(pairs) < branches:
            continue
        triples = []
        for mid, leaf in pairs:
            triples.append(get_edge(edge_lookup, center, mid, subject=center))
            triples.append(get_edge(edge_lookup, mid, leaf, subject=mid))
        sample = {
            "domain": domain,
            "structure_type": "star_2hop",
            "answer": triples[0]["subject"],
            "triples": triples,
        }
        if add_unique_sample(selected, used_answers, sample, count):
            break
    return selected


def sample_cycles(domain_graphs, count, max_start_nodes_per_domain=3000):
    selected = []
    used_answers = set()
    domains = list(domain_graphs)
    random.shuffle(domains)
    for domain, graph, edge_lookup in tqdm(domains, desc="Sampling UD cycle domains", unit="domain"):
        nodes = list(graph.nodes())
        random.shuffle(nodes)
        seen_rings = set()

        def try_ring(ring):
            if len(set(ring)) != len(ring):
                return False
            canonical = min(tuple(ring[i:] + ring[:i]) for i in range(len(ring)))
            if canonical in seen_rings:
                return False
            seen_rings.add(canonical)
            ring_set = set(ring)
            common = None
            for node in ring:
                targets = set(graph.neighbors(node)) - ring_set
                common = targets if common is None else common & targets
                if not common:
                    return False
            common_nodes = list(common)
            random.shuffle(common_nodes)
            for answer_node in common_nodes:
                if not is_good_entity(graph.nodes[answer_node].get("label", answer_node)):
                    continue
                cycle_edges = []
                for index, source in enumerate(ring):
                    target = ring[(index + 1) % len(ring)]
                    cycle_edges.append(get_edge(edge_lookup, source, target, subject=source))
                common_edges = [get_edge(edge_lookup, node, answer_node, subject=node) for node in ring]
                sample = {
                    "domain": domain,
                    "structure_type": "cycle",
                    "answer": clean_entity(graph.nodes[answer_node].get("label", answer_node)),
                    "triples": cycle_edges + common_edges,
                    "cycle_edges": cycle_edges,
                    "common_edges": common_edges,
                    "cycle_length": len(cycle_edges),
                }
                return add_unique_sample(selected, used_answers, sample, count)
            return False

        for a in nodes[:max_start_nodes_per_domain]:
            bs = list(graph.neighbors(a))
            random.shuffle(bs)
            for b in bs[:80]:
                cs = list(graph.neighbors(b))
                random.shuffle(cs)
                for c in cs[:80]:
                    if c in {a, b}:
                        continue
                    if graph.has_edge(c, a) and try_ring([a, b, c]):
                        if len(selected) >= count:
                            return selected
                    ds = list(graph.neighbors(c))
                    random.shuffle(ds)
                    for d in ds[:40]:
                        if d in {a, b, c}:
                            continue
                        if graph.has_edge(d, a) and try_ring([a, b, c, d]):
                            if len(selected) >= count:
                                return selected
    return selected


def collect_samples(domain_graphs, count, selected_types, max_cycle_start_nodes):
    samples = {}
    if "single_edge" in selected_types:
        samples["single_edge"] = sample_single_edges(domain_graphs, count)
    if "path_4" in selected_types:
        samples["path_4"] = sample_paths(domain_graphs, count)
    if "star_1hop" in selected_types:
        samples["star_1hop"] = sample_star_1hop(domain_graphs, count)
    if "star_2hop" in selected_types:
        samples["star_2hop"] = sample_star_2hop(domain_graphs, count)
    if "cycle" in selected_types:
        samples["cycle"] = sample_cycles(domain_graphs, count, max_cycle_start_nodes)
    return samples


def generate_evidence_text(sample):
    sentences = []
    for edge in sample["triples"]:
        sentences.append(f"{edge['subject']} {edge['relation']} {edge['object']}.")
    return " ".join(sentences)


def add_structure_to_graphml(sample, output_path):
    graphml = ET.Element(
        "graphml",
        {
            "xmlns": "http://graphml.graphdrawing.org/xmlns",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
        },
    )
    ET.SubElement(graphml, "key", {"id": "d3", "for": "edge", "attr.name": "source_id", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d2", "for": "edge", "attr.name": "description", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d1", "for": "node", "attr.name": "label", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d0", "for": "node", "attr.name": "id", "attr.type": "string"})
    graph = ET.SubElement(graphml, "graph", {"edgedefault": "undirected"})

    node_ids = {}
    for edge in sample["triples"]:
        for label in (edge["subject"], edge["object"]):
            if label not in node_ids:
                node_ids[label] = f"n{len(node_ids)}"
                node = ET.SubElement(graph, "node", {"id": node_ids[label]})
                ET.SubElement(node, "data", {"key": "d0"}).text = node_ids[label]
                ET.SubElement(node, "data", {"key": "d1"}).text = label

    for index, edge in enumerate(sample["triples"]):
        elem = ET.SubElement(
            graph,
            "edge",
            {"id": f"e{index}", "source": node_ids[edge["subject"]], "target": node_ids[edge["object"]]},
        )
        ET.SubElement(elem, "data", {"key": "d2"}).text = edge["relation"]
        ET.SubElement(elem, "data", {"key": "d3"}).text = edge.get("source_id", "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(graphml).write(output_path, encoding="utf-8", xml_declaration=True)


def clear_structure_output(output_root, structure_type):
    structure_root = output_root / structure_type
    evidence_root = structure_root / "evidence"
    if evidence_root.exists():
        for item in evidence_root.rglob("*"):
            if item.is_file() and item.suffix.lower() in {".graphml", ".txt"}:
                item.unlink()
    query_path = structure_root / "query" / "output_queries_part_1.csv"
    if query_path.exists():
        query_path.unlink()


def write_query_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "query", "answer", "structure_type", "domain"])
        writer.writerows(rows)


def load_checkpoint(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, checkpoint):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def append_error(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def sample_signature(sample):
    triples = ["|".join(edge_key(edge)) for edge in sample["triples"]]
    return f"{sample['domain']}::{sample['structure_type']}::{'##'.join(triples)}::{sample['answer']}"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate UD multi-structure QA data from domain GraphML files.")
    parser.add_argument("--input-root", default=DEFAULT_CONFIG["input_root"])
    parser.add_argument("--output-root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--checkpoint-file", default=DEFAULT_CONFIG["checkpoint_file"])
    parser.add_argument("--error-log-file", default=DEFAULT_CONFIG["error_log_file"])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_CONFIG["api_key"]))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_CONFIG["base_url"]))
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--qps-limit", type=int, default=DEFAULT_CONFIG["qps_limit"])
    parser.add_argument("--max-workers", type=int, default=DEFAULT_CONFIG["max_workers"])
    parser.add_argument("--num-per-type", type=int, default=DEFAULT_CONFIG["num_per_type"])
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CONFIG["candidate_multiplier"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--disable-human-review", action="store_true")
    parser.add_argument("--structure-types", default=",".join(STRUCTURE_TYPES))
    parser.add_argument("--max-cycle-start-nodes", type=int, default=3000)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    selected_types = [item.strip() for item in args.structure_types.split(",") if item.strip()]
    invalid = [item for item in selected_types if item not in STRUCTURE_TYPES]
    if invalid:
        raise ValueError(f"Unsupported structure types: {invalid}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from exc
    if not args.api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY, pass --api-key, or fill DEFAULT_CONFIG['api_key'].")

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_file)
    error_log_path = Path(args.error_log_file)
    candidate_count = args.num_per_type * args.candidate_multiplier

    files = graphml_files(input_root)
    if not files:
        raise FileNotFoundError(f"No graph_chunk_entity_relation.graphml files found under {input_root}")

    domain_graphs = []
    for path in tqdm(files, desc="Loading UD GraphML domains", unit="domain"):
        domain_graphs.append(load_domain_graph(path))

    print(f"Output root: {output_root}")
    print(f"Domains: {len(domain_graphs)}")
    print(f"Selected types: {', '.join(selected_types)}")
    print(f"Requested per type: {args.num_per_type}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print("Quality rule: answer diversity first; API/review failures are skipped, not forced.")

    samples_by_type = collect_samples(domain_graphs, candidate_count, selected_types, args.max_cycle_start_nodes)
    for structure_type, samples in samples_by_type.items():
        unique_answers = len({clean_entity(sample["answer"]).lower() for sample in samples})
        print(f"{structure_type}: collected {len(samples)} candidates, unique answers={unique_answers}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    limiter = QpsLimiter(args.qps_limit)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_lock = threading.Lock()
    results_by_type = defaultdict(list)
    accepted_answers = defaultdict(set)
    stats = Counter()

    tasks = []
    for structure_type, samples in samples_by_type.items():
        for index, sample in enumerate(samples, start=1):
            sample["candidate_index"] = index
            tasks.append(sample)

    def worker(sample):
        key = f"{GENERATION_VERSION}::{sample_signature(sample)}"
        if key in checkpoint:
            item = checkpoint[key]
            return sample, item["query"], item["evidence_text"], item.get("review", {}), "checkpoint"
        query, review = generate_query(
            client,
            limiter,
            args.model,
            sample,
            args.retries,
            enable_review=not args.disable_human_review,
        )
        evidence_text = generate_evidence_text(sample)
        with checkpoint_lock:
            checkpoint[key] = {
                "query": query,
                "evidence_text": evidence_text,
                "answer": sample["answer"],
                "structure_type": sample["structure_type"],
                "domain": sample["domain"],
                "signature": sample_signature(sample),
                "review": review,
            }
            if len(checkpoint) % 50 == 0:
                save_checkpoint(checkpoint_path, checkpoint)
        return sample, query, evidence_text, review, "api"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, sample): sample for sample in tasks}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating UD queries", unit="item"):
            sample = future_map[future]
            structure_type = sample["structure_type"]
            try:
                done_sample, query, evidence_text, review, source = future.result()
                answer_key = clean_entity(done_sample["answer"]).lower()
                if answer_key in accepted_answers[structure_type]:
                    stats[f"{structure_type}_duplicate_answer_skipped"] += 1
                    continue
                if len(results_by_type[structure_type]) >= args.num_per_type:
                    continue
                accepted_answers[structure_type].add(answer_key)
                results_by_type[structure_type].append((done_sample, query, evidence_text, review))
                stats[f"{source}_accepted"] += 1
            except Exception as exc:
                stats[f"{structure_type}_rejected"] += 1
                append_error(
                    error_log_path,
                    {
                        "structure_type": sample["structure_type"],
                        "domain": sample.get("domain", ""),
                        "candidate_index": sample.get("candidate_index", ""),
                        "answer": sample["answer"],
                        "error": str(exc),
                    },
                )

    for structure_type in selected_types:
        clear_structure_output(output_root, structure_type)
        accepted = results_by_type.get(structure_type, [])
        accepted.sort(key=lambda item: item[0].get("candidate_index", 0))
        rows = []
        for idx, (sample, query, evidence_text, review) in enumerate(accepted[: args.num_per_type], start=1):
            filename = f"{structure_type}_{idx:05d}.graphml"
            rel_file = f"batch_000/{filename}"
            evidence_path = output_root / structure_type / "evidence" / rel_file
            txt_path = evidence_path.with_suffix(".txt")
            add_structure_to_graphml(sample, evidence_path)
            txt_path.write_text(evidence_text + "\n", encoding="utf-8")
            rows.append([rel_file, query, sample["answer"], sample["structure_type"], sample["domain"]])
        query_path = output_root / structure_type / "query" / "output_queries_part_1.csv"
        write_query_csv(query_path, rows)
        unique_answers = len({row[2].lower() for row in rows})
        print(f"{structure_type}: wrote {len(rows)} / {args.num_per_type}, unique answers={unique_answers}")

    save_checkpoint(checkpoint_path, checkpoint)
    print("Stats:", dict(stats))
    print("Done.")


if __name__ == "__main__":
    main()
