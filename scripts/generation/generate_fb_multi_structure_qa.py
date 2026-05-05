import argparse
import csv
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


DEFAULT_CONFIG = {
    # Fill this if you want to keep the API key in this script.
    # Command line --api-key and OPENAI_API_KEY will override this value.
    "api_key": "sk-QWtRU87Af1ILa6kyJhsWzhxx1r5by3ijpbuPGTUVgvrW3LP8",
    "nodes_csv": r"D:\MCQA\FB\nodes.csv",
    "edges_csv": r"D:\MCQA\FB\edges_rewritten.csv",
    "output_root": r"D:\MCQA\FB\FB-MultiStructure-QA",
    "checkpoint_file": r"D:\MCQA\FB\fb_multi_structure_qa.checkpoint.json",
    "error_log_file": r"D:\MCQA\FB\fb_multi_structure_qa.errors.jsonl",
    "base_url": "https://ai.nengyongai.cn/v1",
    "model": "gpt-4o-mini",
    "qps_limit": 5,
    "max_workers": 10,
    "num_per_type": 1000,
    "candidate_multiplier": 6,
    "max_input_edges": 300000,
    "seed": 42,
}


STRUCTURE_TYPES = ["single_edge", "path_4", "star_1hop", "star_2hop", "cycle"]
GENERATION_VERSION = "fb_precise_relation_prompt_v11"


SYSTEM_PROMPT = (
    "You create English knowledge-graph QA queries from Freebase-style evidence. "
    "The query must sound like a normal human question, have exactly one entity answer, "
    "not reveal the answer, and not add facts beyond the evidence. "
    "Use the meaning of each relation as a concrete constraint; do not hide relations behind vague words."
)


QUERY_PROMPT_TEMPLATE = """Generate one English QA query from the Freebase-style evidence.

Structure type: {structure_type}
Gold answer: {answer}
Entities that must appear verbatim in the query: {required_entities}
Structure-specific requirements:
{structure_guidance}

Readable relation constraints that should be expressed:
{relation_constraints}

Hard rules:
1. Output an English query only inside JSON.
2. The query must have exactly one entity answer, which is the gold answer.
3. The query must not contain the gold answer.
4. Every required entity must appear verbatim in the query.
5. Do not ask for a list, multiple answers, advice, comparison, or explanation.
6. Do not use wording such as "which entities", "what are all", "list", "respectively", "how are they different", or "what should be considered".
7. Do not mention graph terms such as path, star, center node, two-hop, cycle, loop, graph, triple, evidence, or entity.
8. For star structures, all branches must function as constraints that identify the gold answer; do not ask about a branch target.
9. For cycle structures, use a stable chain style: explicitly state each relation among the cycle nodes in order, then ask for the shared answer entity if one exists.
10. Prefer a clear, plain question over stylistic variety.
11. Do not use placeholder wording such as "connected to what", "related to what", "that item", "that thing", "the answer", or "the entity".
12. Avoid vague relation substitutes such as "associated with", "related to", "connected to", "has a relationship with", or "is linked to"; replace them with the concrete relation meaning from the constraints above.
13. The final question should read like a person asking about a real fact, for example "Which county is X in?", "Which time zone is X in?", "Who was nominated for X and also ...?", or "Which place contains X?", not like a database query.

Triples:
{triples_text}

Return only JSON:
{{
  "query": "...",
  "answer": "{answer}",
  "used_evidence_indices": [1, 2, 3]
}}
"""


REVIEW_PROMPT_TEMPLATE = """Review whether this English knowledge-graph query is acceptable.

Structure type: {structure_type}
Gold answer: {answer}
Required entities: {required_entities}
Structure-specific requirements:
{structure_guidance}

Readable relation constraints:
{relation_constraints}

Query:
{query}

Triples:
{triples_text}

Review criteria:
1. It sounds like a plausible human question.
2. It asks exactly one question.
3. It asks for exactly one entity answer.
4. It does not ask for a list or multiple answers.
5. It includes every required entity verbatim and does not reveal the gold answer.
6. It does not add facts outside the triples.
7. For cycle structures, it expresses every cycle-node relation in chain order rather than merely listing the nodes.
8. For star structures, it uses every branch as a constraint for the gold answer and does not ask about branch targets.
9. It expresses the actual relation meanings, not vague wording such as "associated with", "related to", "connected to", "linked to", or "has a relationship with".
10. It is answer-directed: the constraints make the gold answer more specific, rather than asking an open-ended association question.

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


EVIDENCE_TEXT_PROMPT_TEMPLATE = """Convert the full set of Freebase-style triples into one English evidence paragraph.

Rules:
1. Cover every numbered triple exactly once or by clear grouping.
2. Do not add external knowledge.
3. Preserve all entity names and readable relation descriptions.
4. Do not output raw slash-delimited Freebase relation paths.
5. Output one paragraph, not a list.

Triples:
{triples_text}

Return only JSON:
{{
  "paragraph": "...",
  "used_triple_indices": [1, 2, 3]
}}
"""


class QpsLimiter:
    def __init__(self, qps):
        self.qps = max(1, int(qps))
        self.call_times = []
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            self.call_times = [t for t in self.call_times if now - t < 1.0]
            if len(self.call_times) >= self.qps:
                sleep_for = max(1.0 - (now - self.call_times[0]), 0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self.call_times.append(time.time())


def clean_text(value):
    value = "" if value is None else str(value)
    value = value.replace("&lt;SEP&gt;", "; ").replace("<SEP>", "; ").replace("|>", "; ")
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip("'\"").strip()
    return "" if value.lower() == "nan" else value


def readable_relation(value):
    value = clean_text(value)
    value = re.sub(r"/[A-Za-z0-9_./]+", "", value)
    value = value.replace("|", ";")
    value = re.sub(r"\s+", " ", value).strip(" ;.")
    return value or "specified relation"


def is_good_entity(value):
    value = clean_text(value)
    if not value or len(value) < 2 or len(value) > 60:
        return False
    if value.startswith("/") or value.lower() in {"nan", "none"}:
        return False
    return True


def is_good_query_entity(value):
    value = clean_text(value)
    if not is_good_entity(value) or len(value) > 48:
        return False
    noisy = ["http://", "https://", "<", ">", "|", "\t"]
    return not any(token in value for token in noisy)


def load_fb_graph(nodes_csv, edges_csv, max_input_edges):
    node_labels = {}
    with Path(nodes_csv).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label = clean_text(row.get("label"))
            node_id = clean_text(row.get("id"))
            if node_id and is_good_entity(label):
                node_labels[node_id] = label

    out_adj = defaultdict(list)
    edges = []
    by_pair = {}
    with Path(edges_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading FB triples", unit="edge"):
            start_id = clean_text(row.get("start"))
            end_id = clean_text(row.get("end"))
            if start_id not in node_labels or end_id not in node_labels or start_id == end_id:
                continue
            subject = node_labels[start_id]
            obj = node_labels[end_id]
            relation = readable_relation(row.get("description") or row.get("id"))
            if not (is_good_query_entity(subject) and is_good_query_entity(obj) and relation):
                continue
            edge = {
                "subject": subject,
                "subject_id": start_id,
                "relation": relation,
                "object": obj,
                "object_id": end_id,
            }
            edges.append(edge)
            out_adj[start_id].append(edge)
            by_pair.setdefault((start_id, end_id), edge)
            if max_input_edges and len(edges) >= max_input_edges:
                break
    return node_labels, out_adj, by_pair, edges


def edge_key(edge):
    return (edge["subject"], edge["relation"], edge["object"])


def sample_signature(sample):
    triples = ["|".join(edge_key(edge)) for edge in sample["triples"]]
    return f"{sample['structure_type']}::{'##'.join(triples)}::{sample['answer']}"


def sample_entity_nodes(sample):
    nodes = []
    seen = set()
    for edge in sample["triples"]:
        for key in ("subject", "object"):
            node = edge[key]
            if node not in seen:
                nodes.append(node)
                seen.add(node)
    return nodes


def required_query_entities(sample):
    answer = clean_text(sample["answer"])
    return [node for node in sample_entity_nodes(sample) if clean_text(node) != answer]


def missing_required_entities(sample, query):
    query_text = clean_text(query)
    return [entity for entity in required_query_entities(sample) if clean_text(entity) not in query_text]


def format_edge(edge):
    return f"{edge['subject']} --{edge['relation']}--> {edge['object']}"


def natural_relation_phrase(relation):
    relation_lower = relation.lower()
    if "time zones" in relation_lower:
        return "is in the time zone"
    if "contains" in relation_lower or "partially contains" in relation_lower:
        return "contains"
    if "capital" in relation_lower and "administrative" in relation_lower:
        return "belongs to the county or administrative division"
    if "webpage" in relation_lower or "official website" in relation_lower:
        return "has the webpage category"
    if "country related" in relation_lower or "nationality" in relation_lower:
        return "has the country"
    if "military conflicts" in relation_lower:
        return "has a military conflict with"
    if "medals" in relation_lower:
        return "won medals at"
    if "players" in relation_lower:
        return "has players from"
    if "roster" in relation_lower or "position" in relation_lower:
        return "lists the position"
    if "students" in relation_lower or "graduates" in relation_lower:
        return "has a student or graduate"
    if "school type" in relation_lower:
        return "has the school type"
    if "leaders" in relation_lower:
        return "has as a leader"
    if "headquarters" in relation_lower:
        return "is headquartered in"
    if "instrumentalists" in relation_lower:
        return "is played by"
    if "regular performances" in relation_lower:
        return "is regularly performed with"
    if "award nominations" in relation_lower:
        return "received a nomination for"
    if "awards presented" in relation_lower:
        return "presented an award to"
    if "winners" in relation_lower:
        return "has winner"
    if "profession" in relation_lower:
        return "has the profession"
    if "colors" in relation_lower:
        return "uses the color"
    if "release date" in relation_lower:
        return "has a release-related record for"
    if "language" in relation_lower:
        return "uses the language"
    if "genre" in relation_lower:
        return "has the genre"
    if "film" in relation_lower:
        return "appears in or is credited on the film"
    if "government" in relation_lower and "members" in relation_lower:
        return "has government-member records involving"
    return "has the specified relation"


def natural_edge_clause(edge):
    subject = edge["subject"]
    obj = edge["object"]
    relation = edge["relation"]
    relation_lower = relation.lower()
    if "time zones" in relation_lower:
        return f"{subject} is in {obj}"
    if "contains" in relation_lower or "partially contains" in relation_lower:
        return f"{subject} contains {obj}"
    if "capital" in relation_lower and "administrative" in relation_lower:
        return f"{subject} belongs to the administrative division {obj}"
    if "webpage" in relation_lower or "official website" in relation_lower:
        return f"{subject} has {obj} as its webpage category"
    if "country related" in relation_lower or "nationality" in relation_lower:
        return f"{subject} has the country {obj}"
    if "military conflicts" in relation_lower:
        return f"{subject} had a military conflict with {obj}"
    if "medals" in relation_lower:
        return f"{subject} won medals at {obj}"
    if "players" in relation_lower:
        return f"{subject} includes players from {obj}"
    if "roster" in relation_lower or "position" in relation_lower:
        return f"{subject} lists {obj} as a roster position"
    if "students" in relation_lower or "graduates" in relation_lower:
        return f"{subject} has {obj} among its students or graduates"
    if "school type" in relation_lower:
        return f"{subject} is classified as a {obj}"
    if "leaders" in relation_lower:
        return f"{subject} has {obj} as a leader"
    if "headquarters" in relation_lower:
        return f"{subject} is headquartered in {obj}"
    if "instrumentalists" in relation_lower:
        return f"{subject} is played by {obj}"
    if "regular performances" in relation_lower:
        return f"{subject} is regularly performed with {obj}"
    if "award nominations" in relation_lower:
        return f"{subject} received a nomination for {obj}"
    if "awards presented" in relation_lower:
        return f"{subject} presented an award to {obj}"
    if "winners" in relation_lower:
        return f"{subject} has {obj} as a winner"
    if "profession" in relation_lower:
        return f"{subject} has the profession {obj}"
    if "colors" in relation_lower:
        return f"{subject} uses {obj} as a color"
    if "release date" in relation_lower:
        return f"{subject} has a release record for {obj}"
    if "language" in relation_lower:
        return f"{subject} uses {obj} as a language"
    if "genre" in relation_lower:
        return f"{subject} has the genre {obj}"
    if "film" in relation_lower:
        return f"{subject} appears in or is credited on the film {obj}"
    if "government" in relation_lower and "members" in relation_lower:
        return f"{subject} includes {obj} in its government membership records"
    return f"{subject} has the relation '{relation}' with {obj}"


def natural_edge_clause_with_subject(edge, subject_alias):
    aliased = dict(edge)
    aliased["subject"] = subject_alias
    return natural_edge_clause(aliased)


def relation_clause(edge):
    return natural_edge_clause(edge)


def relation_noun_phrase(relation):
    return natural_relation_phrase(relation)


def answer_question_phrase(relation):
    relation_lower = relation.lower()
    if "award nominations" in relation_lower:
        return "which nominated work"
    if "government" in relation_lower and "members" in relation_lower:
        return "which Congress"
    if "time zones" in relation_lower:
        return "which time zone"
    if "contains" in relation_lower:
        return "which place"
    if "country" in relation_lower or "nationality" in relation_lower:
        return "which country"
    if "players" in relation_lower or "roster" in relation_lower or "position" in relation_lower:
        return "which playing position"
    if "award" in relation_lower or "winner" in relation_lower or "nomination" in relation_lower:
        return "which award or person"
    if "profession" in relation_lower:
        return "which profession"
    if "colors" in relation_lower:
        return "which color"
    if "language" in relation_lower:
        return "which language"
    if "genre" in relation_lower:
        return "which genre"
    if "webpage" in relation_lower:
        return "which webpage category"
    return "what"


def answer_alias_phrase(relation):
    relation_lower = relation.lower()
    if "award nominations" in relation_lower:
        return "that work"
    if "government" in relation_lower and "members" in relation_lower:
        return "that Congress"
    if "time zones" in relation_lower:
        return "that time zone"
    if "contains" in relation_lower:
        return "that place"
    if "country" in relation_lower or "nationality" in relation_lower:
        return "that country"
    if "players" in relation_lower or "roster" in relation_lower or "position" in relation_lower:
        return "that playing position"
    if "award" in relation_lower or "winner" in relation_lower or "nomination" in relation_lower:
        return "that award or person"
    if "profession" in relation_lower:
        return "that profession"
    if "colors" in relation_lower:
        return "that color"
    if "language" in relation_lower:
        return "that language"
    if "genre" in relation_lower:
        return "that genre"
    if "webpage" in relation_lower:
        return "that webpage category"
    if "film" in relation_lower:
        return "that film"
    return "that item"


def join_items(items):
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def single_edge_query(edge):
    relation_lower = edge["relation"].lower()
    subject = edge["subject"]
    obj = edge["object"]
    answer_side = edge.get("answer_side", "object")
    if answer_side == "subject":
        if "contains" in relation_lower or "partially contains" in relation_lower:
            return f"Which place contains {obj}?"
        return f"What is directly related to {obj} here?"
    if "time zones" in relation_lower:
        return f"Which time zone is {subject} in?"
    if "contains" in relation_lower:
        return f"Which place does {subject} contain?"
    if "county" in relation_lower:
        return f"Which county is {subject} in?"
    if "capital" in relation_lower and "administrative" in relation_lower:
        return f"Which county or administrative division is {subject} in?"
    if "place of birth" in relation_lower:
        return f"Where was {subject} born?"
    if "nationality" in relation_lower:
        return f"Which country is {subject} a national of?"
    if "country related" in relation_lower:
        return f"Which country is {subject} tied to?"
    if "webpage" in relation_lower:
        return f"Which webpage category is associated with {subject}?"
    if "players" in relation_lower:
        return f"Which team is associated with players in {subject}?"
    if "students" in relation_lower or "graduates" in relation_lower:
        return f"Which person is listed as a student or graduate of {subject}?"
    if "award nominations" in relation_lower:
        return f"Which award or work is {subject} nominated for?"
    if "profession" in relation_lower:
        return f"Which profession is associated with {subject}?"
    if "film" in relation_lower:
        return f"Which film is associated with {subject}?"
    return f"What is {subject} associated with?"


BAD_SINGLE_EDGE_ANSWERS = {
    "Official Website",
    "United States Department of Housing and Urban Development",
}


def single_edge_answer_spec(edge):
    relation_lower = edge["relation"].lower()
    answer = edge["object"]
    answer_side = "object"
    if answer in BAD_SINGLE_EDGE_ANSWERS:
        return None
    if "webpage" in relation_lower or "estimated number of mortgages" in relation_lower or "biblioness" in relation_lower:
        return None
    if "contains" in relation_lower or "partially contains" in relation_lower:
        answer = edge["subject"]
        answer_side = "subject"
    elif "time zones" in relation_lower:
        answer = edge["object"]
    elif "county" in relation_lower or ("capital" in relation_lower and "administrative" in relation_lower):
        answer = edge["object"]
    elif "place of birth" in relation_lower:
        answer = edge["object"]
    elif "nationality" in relation_lower or "country related" in relation_lower:
        answer = edge["object"]
    else:
        return None
    if not is_good_query_entity(answer):
        return None
    return answer, answer_side


def path_query(first, second, third):
    first_relation = first["relation"].lower()
    alias = answer_alias_phrase(first["relation"])
    second_clause = natural_edge_clause_with_subject(second, alias)
    third_clause = natural_edge_clause(third)
    if "award nominations" in first_relation:
        return (
            f"Which nominated work did {first['subject']} receive recognition for, "
            f"given that {second_clause} and {third_clause}?"
        )
    if "contains" in first_relation:
        return (
            f"Which place does {first['subject']} contain, given that "
            f"{second_clause}, and {third_clause}?"
        )
    if "students" in first_relation or "graduates" in first_relation:
        return (
            f"Which student or graduate of {first['subject']} also fits this context: "
            f"{second_clause}, and {third_clause}?"
        )
    if "players" in first_relation:
        return (
            f"{first['subject']} has players in which playing position, given that "
            f"{second_clause}, and {third_clause}?"
        )
    if "country" in first_relation or "nationality" in first_relation:
        return (
            f"{first['subject']} is tied to which country, given that "
            f"{second_clause}, and {third_clause}?"
        )
    if "film" in first_relation:
        return (
            f"Which film is {first['subject']} connected to, given that "
            f"{second_clause}, and {third_clause}?"
        )
    if "release date" in first_relation:
        return (
            f"{first['subject']} has a release record for which place or market, given that "
            f"{second_clause}, and {third_clause}?"
        )
    return (
        f"{first['subject']} is connected to {answer_question_phrase(first['relation'])}, given that "
        f"{second_clause}, and {third_clause}?"
    )


def generated_template_query(sample):
    structure_type = sample["structure_type"]
    triples = sample["triples"]
    if structure_type == "single_edge":
        edge = triples[0]
        return single_edge_query(edge)
    if structure_type == "path_4":
        first, second, third = triples
        return path_query(first, second, third)
    if structure_type == "star_1hop":
        relation = triples[0]["relation"]
        targets = [edge["object"] for edge in triples]
        target_text = join_items(targets)
        q_phrase = answer_question_phrase(relation)
        if "contains" in relation.lower():
            return f"{target_text} are all contained in {q_phrase}?"
        if "players" in relation.lower():
            return f"{target_text} are all teams associated with players in {q_phrase}?"
        if "instrumentalists" in relation.lower():
            return f"{target_text} are all performers associated with {q_phrase}?"
        if "award" in relation.lower():
            return f"{target_text} are all tied to nominations or wins for {q_phrase}?"
        if "genre" in relation.lower():
            return f"{target_text} are all works in which genre?"
        if "profession" in relation.lower():
            return f"{target_text} are all people with which profession?"
        if "country" in relation.lower() or "nationality" in relation.lower():
            return f"{target_text} are all tied to which country?"
        return f"{target_text} are all described by which category?"
    if structure_type == "star_2hop":
        first_objects = []
        branch_clauses = []
        for index in range(0, len(triples), 2):
            first = triples[index]
            second = triples[index + 1]
            first_objects.append(first["object"])
            branch_clauses.append(natural_edge_clause(second))
        text = ", and ".join(branch_clauses)
        q_phrase = answer_question_phrase(triples[0]["relation"])
        first_text = join_items(first_objects)
        if "players" in triples[0]["relation"].lower():
            return f"{text}; {q_phrase} has players from {first_text}?"
        if "contains" in triples[0]["relation"].lower():
            return f"{text}; {q_phrase} contains {first_text}?"
        return f"{text}; taken together, {q_phrase} do these details indicate?"
    if structure_type == "cycle":
        cycle_edges = sample.get("cycle_edges") or triples[: sample.get("cycle_length", 0)]
        common_edges = sample.get("common_edges", [])
        chain = ", ".join(natural_edge_clause(edge) for edge in cycle_edges)
        q_phrase = answer_question_phrase(common_edges[0]["relation"] if common_edges else "")
        return f"{chain}. Considering these connected records, {q_phrase} also applies across the same set?"
    return ""


def question_phrase(relation):
    relation = relation.lower()
    if "contains" in relation or "location" in relation or "country" in relation:
        return "which place or country satisfies the shared condition"
    if "award" in relation or "winner" in relation or "nomination" in relation:
        return "which person or award satisfies the shared condition"
    if "student" in relation or "graduate" in relation or "education" in relation:
        return "which school or education-related item satisfies the shared condition"
    if "film" in relation or "tv" in relation or "music" in relation or "artist" in relation:
        return "which work, artist, or media-related item satisfies the shared condition"
    return "which item satisfies the shared condition"


def structure_guidance(sample):
    structure_type = sample["structure_type"]
    if structure_type == "cycle":
        cycle_edges = sample.get("cycle_edges") or sample["triples"][: sample.get("cycle_length", 0)]
        common_edges = sample.get("common_edges", [])
        chain = ", ".join(relation_clause(edge) for edge in cycle_edges)
        q_phrase = question_phrase(common_edges[0]["relation"] if common_edges else "")
        return (
            "Cycle query style must be stable and chain-like. First express every cycle relation in order, "
            "then ask one question for the shared answer entity. Do not merely list the cycle nodes.\n"
            f"Recommended chain draft: {chain}; in this setting, {q_phrase}?\n"
            "Cycle relations:\n"
            + "\n".join(f"- {format_edge(edge)}" for edge in cycle_edges)
            + "\nShared answer relations:\n"
            + ("\n".join(f"- {format_edge(edge)}" for edge in common_edges) if common_edges else "None")
        )
    if structure_type == "star_2hop":
        branches = []
        triples = sample["triples"]
        for index in range(0, len(triples), 2):
            if index + 1 < len(triples):
                branches.append(f"- {natural_edge_clause(triples[index])}; {natural_edge_clause(triples[index + 1])}")
        return (
            "Two-hop star query. Use every branch as a constraint for the gold answer. "
            "Ask only for the gold answer, not for a branch target. "
            "Do not make separate questions for separate branches. "
            "Write one sentence where all branch facts narrow down the same missing answer.\n"
            + "\n".join(branches)
        )
    if structure_type == "star_1hop":
        return (
            "One-hop star query. Use all required entities as constraints and ask only for the shared gold answer. "
            "Use the repeated relation meaning directly, such as contains, has players from, uses language, has genre, "
            "has profession, or is in a country. Avoid saying the answer is merely related to the listed entities."
        )
    if structure_type == "path_4":
        first, second, third = sample["triples"]
        return (
            "Four-node path query. Ask only for the middle answer node. "
            "Write it as a normal fact-seeking question, not as a graph description. "
            "The question must mention the start node and the two downstream non-answer nodes, "
            "and it must express the two consecutive constraints after the answer in natural English. "
            "Do not use vague wording such as 'connected to what', 'related to what', 'that item', "
            "'that entity', or 'the answer'.\n"
            "Relation flow to express:\n"
            f"- The answer is reached from {first['subject']} by: {first['relation']}\n"
            f"- The answer then relates to {second['object']} by: {second['relation']}\n"
            f"- {second['object']} then relates to {third['object']} by: {third['relation']}"
        )
    return (
        "Single-edge query. Ask only for the gold answer. "
        "Use the single relation as a precise constraint, and write a normal human question. "
        "Do not ask an open-ended question that could have many valid answers. "
        "For containment/location-style relations, ask for the containing place or the location, "
        "not for a broad list of things contained by a place. "
        "For time zone, county, birthplace, nationality, or country relations, ask directly for that property."
    )


def triples_to_text(triples):
    return "\n".join(
        f"{index}. ({edge['subject']}, {edge['relation']}, {edge['object']})"
        for index, edge in enumerate(triples, start=1)
    )


def relation_constraints_text(sample):
    lines = []
    for index, edge in enumerate(sample["triples"], start=1):
        lines.append(f"{index}. {natural_edge_clause(edge)}")
    return "\n".join(lines)


def sample_single_edges(edges, count):
    selected = []
    seen = set()
    preferred_order = [
        "contains",
        "time zones",
        "county",
        "capital",
        "place of birth",
        "nationality",
        "country related",
    ]

    def priority(edge):
        rel = edge["relation"].lower()
        for index, token in enumerate(preferred_order):
            if token in rel:
                return index
        return 99

    sorted_edges = sorted(edges, key=priority)
    for edge in sorted_edges:
        key = edge_key(edge)
        if key in seen:
            continue
        answer_spec = single_edge_answer_spec(edge)
        if not answer_spec:
            continue
        answer, answer_side = answer_spec
        sample_edge = dict(edge)
        sample_edge["answer_side"] = answer_side
        selected.append(
            {
                "structure_type": "single_edge",
                "answer": answer,
                "triples": [sample_edge],
                "answer_side": answer_side,
            }
        )
        seen.add(key)
        if len(selected) >= count:
            break
    return selected


def sample_directed_paths(out_adj, count, path_edges=3):
    selected = []
    seen = set()
    starts = list(out_adj.keys())
    random.shuffle(starts)

    def dfs(node_id, triples, visited):
        if len(triples) == path_edges:
            ids = [triples[0]["subject_id"]] + [edge["object_id"] for edge in triples]
            if len(set(ids)) != len(ids):
                return None
            answer = triples[len(triples) // 2]["subject"]
            path_id = tuple(ids)
            if path_id in seen:
                return None
            seen.add(path_id)
            return {"structure_type": "path_4", "answer": answer, "triples": triples}
        candidates = list(out_adj.get(node_id, []))
        random.shuffle(candidates)
        for edge in candidates[:40]:
            nxt = edge["object_id"]
            if nxt in visited:
                continue
            result = dfs(nxt, triples + [edge], visited | {nxt})
            if result:
                return result
        return None

    for start in starts:
        result = dfs(start, [], {start})
        if result:
            selected.append(result)
        if len(selected) >= count:
            break
    return selected


def sample_one_hop_stars(out_adj, count, min_neighbors=3, max_neighbors=4):
    selected = []
    centers = sorted(out_adj.keys(), key=lambda key: len(out_adj[key]), reverse=True)
    used = set()
    for center in centers:
        if center in used:
            continue
        by_relation = defaultdict(dict)
        for edge in out_adj[center]:
            if edge["object_id"] != center and is_good_query_entity(edge["object"]):
                by_relation[edge["relation"]].setdefault(edge["object_id"], edge)
        best = sorted(by_relation.values(), key=len, reverse=True)
        if not best or len(best[0]) < min_neighbors:
            continue
        candidates = list(best[0].values())
        random.shuffle(candidates)
        triples = candidates[: random.randint(min_neighbors, min(max_neighbors, len(candidates)))]
        selected.append({"structure_type": "star_1hop", "answer": triples[0]["subject"], "triples": triples})
        used.add(center)
        if len(selected) >= count:
            break
    return selected


def sample_two_hop_stars(out_adj, count, branches=2):
    selected = []
    centers = sorted(out_adj.keys(), key=lambda key: len(out_adj[key]), reverse=True)
    used_centers = set()
    for center in centers:
        if center in used_centers:
            continue
        by_relation = defaultdict(list)
        for edge in out_adj[center]:
            if is_good_query_entity(edge["object"]):
                by_relation[edge["relation"]].append(edge)
        triples = None
        for first_edges in sorted(by_relation.values(), key=len, reverse=True):
            used_nodes = {center}
            pairs = []
            random.shuffle(first_edges)
            for first in first_edges:
                mid = first["object_id"]
                if mid in used_nodes:
                    continue
                second_candidates = [
                    edge
                    for edge in out_adj.get(mid, [])
                    if edge["object_id"] not in used_nodes
                    and edge["object_id"] != center
                    and is_good_query_entity(edge["object"])
                ]
                random.shuffle(second_candidates)
                if not second_candidates:
                    continue
                second = second_candidates[0]
                pairs.append((first, second))
                used_nodes.update({mid, second["object_id"]})
                if len(pairs) >= branches:
                    triples = []
                    for pair in pairs:
                        triples.extend(pair)
                    break
            if triples:
                break
        if triples and len(triples) == branches * 2:
            selected.append({"structure_type": "star_2hop", "answer": triples[0]["subject"], "triples": triples})
            used_centers.add(center)
        if len(selected) >= count:
            break
    return selected


def sample_cycles(out_adj, by_pair, count, min_len=3, max_len=4):
    selected = []
    seen = set()
    nodes = list(out_adj.keys())
    random.shuffle(nodes)
    target_sets = {node: {edge["object_id"] for edge in edges} for node, edges in out_adj.items()}

    def canonical(ring):
        rotations = [tuple(ring[i:] + ring[:i]) for i in range(len(ring))]
        return min(rotations)

    def add_cycle(ring):
        if len(set(ring)) != len(ring):
            return False
        cid = canonical(ring)
        if cid in seen:
            return False
        common = None
        for node in ring:
            targets = target_sets.get(node, set()) - set(ring)
            common = targets if common is None else common & targets
            if not common:
                return False
        answer_id = sorted(common)[0]
        cycle_edges = []
        for index, source in enumerate(ring):
            target = ring[(index + 1) % len(ring)]
            edge = by_pair.get((source, target))
            if not edge:
                return False
            cycle_edges.append(edge)
        common_edges = []
        for node in ring:
            edge = by_pair.get((node, answer_id))
            if not edge:
                return False
            common_edges.append(edge)
        seen.add(cid)
        selected.append(
            {
                "structure_type": "cycle",
                "answer": common_edges[0]["object"],
                "triples": cycle_edges + common_edges,
                "cycle_edges": cycle_edges,
                "common_edges": common_edges,
                "cycle_length": len(cycle_edges),
            }
        )
        return True

    for a in nodes:
        for eab in out_adj.get(a, [])[:80]:
            b = eab["object_id"]
            if b == a:
                continue
            for ebc in out_adj.get(b, [])[:80]:
                c = ebc["object_id"]
                if c in {a, b}:
                    continue
                if (c, a) in by_pair and add_cycle([a, b, c]):
                    if len(selected) >= count:
                        return selected
                if max_len >= 4:
                    for ecd in out_adj.get(c, [])[:40]:
                        d = ecd["object_id"]
                        if d in {a, b, c}:
                            continue
                        if (d, a) in by_pair and add_cycle([a, b, c, d]):
                            if len(selected) >= count:
                                return selected
    return selected


def collect_samples(out_adj, by_pair, edges, count, selected_types):
    samples = {}
    if "single_edge" in selected_types:
        samples["single_edge"] = sample_single_edges(edges, count)
    if "path_4" in selected_types:
        samples["path_4"] = sample_directed_paths(out_adj, count)
    if "star_1hop" in selected_types:
        samples["star_1hop"] = sample_one_hop_stars(out_adj, count)
    if "star_2hop" in selected_types:
        samples["star_2hop"] = sample_two_hop_stars(out_adj, count)
    if "cycle" in selected_types:
        samples["cycle"] = sample_cycles(out_adj, by_pair, count)
    return samples


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


def answer_is_exposed(sample, query):
    answer = clean_text(sample["answer"])
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


def has_multiple_question_risk(query):
    query_lower = clean_text(query).lower()
    if query_lower.count("?") > 1:
        return True
    for term in MULTI_ANSWER_TERMS:
        if term == "list":
            if re.search(r"\blist\b", query_lower):
                return True
            continue
        if term in query_lower:
            return True
    return False


def validate_query(sample, query, allow_template_fallback=False):
    query = clean_text(query)
    if not query:
        raise ValueError("empty query")
    banned_terms = [" path", " star", "center node", "two-hop", "cycle", "loop", " graph", " triple", " evidence"]
    if any(term in query.lower() for term in banned_terms):
        raise ValueError(f"query contains banned graph/data term: {query}")
    unnatural_terms = [
        "the answer",
        "the entity",
        "is associated with",
        "associated with",
        "connected to what",
        "related to what",
        "connected to which",
        "connected to",
        "related to",
        "is linked to",
        "linked to",
        "relationship with",
        "relationships with",
        "that item",
        "that thing",
        "that entity",
        "has a recorded relationship",
        "all point to",
        "in this chain",
        "linked to each of them",
        "shared by these relationships",
        "shared item",
        "follow-up facts",
    ]
    if not allow_template_fallback and any(term in query.lower() for term in unnatural_terms):
        raise ValueError(f"query contains unnatural template wording: {query}")
    if has_multiple_question_risk(query):
        raise ValueError("query has multiple-answer or multiple-question risk")
    if answer_is_exposed(sample, query):
        raise ValueError("query exposes the answer")
    missing = missing_required_entities(sample, query)
    if missing:
        raise ValueError(f"query misses required entities: {missing}")


def template_query_allowed(sample):
    return False


def build_query_prompt(sample):
    return QUERY_PROMPT_TEMPLATE.format(
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=", ".join(required_query_entities(sample)),
        structure_guidance=structure_guidance(sample),
        relation_constraints=relation_constraints_text(sample),
        triples_text=triples_to_text(sample["triples"]),
    )


def build_review_prompt(sample, query):
    return REVIEW_PROMPT_TEMPLATE.format(
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=", ".join(required_query_entities(sample)),
        structure_guidance=structure_guidance(sample),
        relation_constraints=relation_constraints_text(sample),
        query=query,
        triples_text=triples_to_text(sample["triples"]),
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
    accepted = bool(data.get("accepted", False))
    checks = {
        "single_answer_entity": bool(data.get("single_answer_entity", False)),
        "asks_multiple_questions": bool(data.get("asks_multiple_questions", True)),
        "asks_for_list": bool(data.get("asks_for_list", True)),
        "relation_coverage_ok": bool(data.get("relation_coverage_ok", False)),
        "cycle_relations_expressed": bool(data.get("cycle_relations_expressed", sample["structure_type"] != "cycle")),
    }
    if (
        not accepted
        or score < 4
        or not checks["single_answer_entity"]
        or checks["asks_multiple_questions"]
        or checks["asks_for_list"]
        or not checks["relation_coverage_ok"]
        or (sample["structure_type"] == "cycle" and not checks["cycle_relations_expressed"])
    ):
        raise ValueError(f"query rejected by reviewer: score={score}, reason={clean_text(data.get('reason'))}")
    checks.update({"accepted": accepted, "score": score, "reason": clean_text(data.get("reason"))})
    return checks


def generate_query(client, limiter, model, sample, retries, enable_review=True):
    if template_query_allowed(sample):
        query = generated_template_query(sample)
        validate_query(sample, query)
        return query, {"accepted": True, "score": 5, "reason": "accepted deterministic template", "source": "template"}

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
                0.5,
            )
            query = clean_text(data.get("query", ""))
            validate_query(sample, query)
            review = {"accepted": True, "score": 5, "reason": "review disabled"}
            if enable_review:
                review = review_query(client, limiter, model, sample, query)
            review["source"] = "api"
            return query, review
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    query = generated_template_query(sample)
    validate_query(sample, query, allow_template_fallback=True)
    return query, {
        "accepted": True,
        "score": 3,
        "reason": f"template fallback after LLM/review failure: {last_error}",
        "source": "template_fallback",
    }


def generate_evidence_text(sample):
    grouped = defaultdict(list)
    ordered_keys = []
    for edge in sample["triples"]:
        key = (edge["subject"], edge["relation"])
        if key not in grouped:
            ordered_keys.append(key)
        grouped[key].append(edge["object"])

    sentences = []
    for subject, relation in ordered_keys:
        objects = grouped[(subject, relation)]
        if len(objects) == 1:
            object_text = objects[0]
        elif len(objects) == 2:
            object_text = f"{objects[0]} and {objects[1]}"
        else:
            object_text = ", ".join(objects[:-1]) + f", and {objects[-1]}"
        if len(objects) == 1:
            edge = {"subject": subject, "relation": relation, "object": objects[0]}
            sentences.append(natural_edge_clause(edge) + ".")
        else:
            sentences.append(f"{subject} {natural_relation_phrase(relation)} {object_text}.")
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
    ET.SubElement(graphml, "key", {"id": "d2", "for": "edge", "attr.name": "description", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d1", "for": "node", "attr.name": "id", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d0", "for": "node", "attr.name": "label", "attr.type": "string"})
    graph = ET.SubElement(graphml, "graph", {"edgedefault": "directed"})

    nodes = {}
    for edge in sample["triples"]:
        nodes.setdefault(edge["subject"], edge["subject_id"])
        nodes.setdefault(edge["object"], edge["object_id"])
    for label, node_id in nodes.items():
        node_el = ET.SubElement(graph, "node", {"id": label})
        ET.SubElement(node_el, "data", {"key": "d0"}).text = label
        ET.SubElement(node_el, "data", {"key": "d1"}).text = node_id
    for edge in sample["triples"]:
        edge_el = ET.SubElement(graph, "edge", {"source": edge["subject"], "target": edge["object"]})
        ET.SubElement(edge_el, "data", {"key": "d2"}).text = edge["relation"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(graphml)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


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
    os.replace(tmp_path, path)


def append_error(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
        writer.writerow(["file", "query", "answer", "structure_type"])
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate English FB multi-structure QA data with GraphML and TXT evidence.")
    parser.add_argument("--nodes-csv", default=DEFAULT_CONFIG["nodes_csv"])
    parser.add_argument("--edges-csv", default=DEFAULT_CONFIG["edges_csv"])
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
    parser.add_argument("--max-input-edges", type=int, default=DEFAULT_CONFIG["max_input_edges"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--disable-human-review", action="store_true")
    parser.add_argument("--structure-types", default=",".join(STRUCTURE_TYPES))
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from exc
    if not args.api_key:
        raise RuntimeError("Missing API key. Fill DEFAULT_CONFIG['api_key'], set OPENAI_API_KEY, or pass --api-key.")

    selected_types = [item.strip() for item in args.structure_types.split(",") if item.strip()]
    unknown = set(selected_types) - set(STRUCTURE_TYPES)
    if unknown:
        raise ValueError(f"Unknown structure types: {sorted(unknown)}")

    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_file)
    error_log_path = Path(args.error_log_file)
    candidate_count = max(args.num_per_type, args.num_per_type * max(1, args.candidate_multiplier))

    _, out_adj, by_pair, edges = load_fb_graph(args.nodes_csv, args.edges_csv, args.max_input_edges)
    samples_by_type = collect_samples(out_adj, by_pair, edges, candidate_count, selected_types)

    print(f"Output root: {output_root}")
    print(f"Requested per type: {args.num_per_type}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print(f"LLM human review: {'disabled' if args.disable_human_review else 'enabled'}")
    for structure_type, samples in samples_by_type.items():
        print(f"{structure_type}: collected {len(samples)} candidates")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    limiter = QpsLimiter(args.qps_limit)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_lock = threading.Lock()
    stats = defaultdict(int)
    stats_lock = threading.Lock()
    tasks = []
    for structure_type, samples in samples_by_type.items():
        for candidate_index, sample in enumerate(samples, start=1):
            sample["candidate_index"] = candidate_index
            tasks.append(sample)

    results_by_type = defaultdict(list)

    def worker(sample):
        key = f"{GENERATION_VERSION}::{sample_signature(sample)}"
        if key in checkpoint:
            item = checkpoint[key]
            with stats_lock:
                stats["checkpoint_reused"] += 1
            return sample, item["query"], item["evidence_text"], item.get("review", {})
        query, review = generate_query(
            client,
            limiter,
            args.model,
            sample,
            args.retries,
            enable_review=not args.disable_human_review,
        )
        evidence_text = generate_evidence_text(sample)
        with stats_lock:
            stats[f"{review.get('source', 'unknown')}_generated"] += 1
        with checkpoint_lock:
            checkpoint[key] = {
                "query": query,
                "evidence_text": evidence_text,
                "answer": sample["answer"],
                "structure_type": sample["structure_type"],
                "signature": sample_signature(sample),
                "review": review,
            }
            if len(checkpoint) % 50 == 0:
                save_checkpoint(checkpoint_path, checkpoint)
        return sample, query, evidence_text, review

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, sample): sample for sample in tasks}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating FB queries/evidence", unit="item"):
            sample = future_map[future]
            try:
                done_sample, query, evidence_text, review = future.result()
                structure_type = done_sample["structure_type"]
                if len(results_by_type[structure_type]) < args.num_per_type:
                    results_by_type[structure_type].append((done_sample, query, evidence_text, review))
            except Exception as exc:
                append_error(
                    error_log_path,
                    {
                        "structure_type": sample["structure_type"],
                        "candidate_index": sample.get("candidate_index", ""),
                        "answer": sample["answer"],
                        "error": str(exc),
                    },
                )

    for structure_type in selected_types:
        clear_structure_output(output_root, structure_type)
        rows = []
        accepted = results_by_type.get(structure_type, [])
        accepted.sort(key=lambda item: item[0].get("candidate_index", 0))
        for idx, (sample, query, evidence_text, review) in enumerate(accepted[: args.num_per_type], start=1):
            filename = f"{structure_type}_{idx:05d}.graphml"
            rel_file = f"batch_000/{filename}"
            evidence_path = output_root / structure_type / "evidence" / rel_file
            txt_path = evidence_path.with_suffix(".txt")
            add_structure_to_graphml(sample, evidence_path)
            txt_path.write_text(evidence_text + "\n", encoding="utf-8")
            rows.append([rel_file, query, sample["answer"], sample["structure_type"]])
        query_path = output_root / structure_type / "query" / "output_queries_part_1.csv"
        write_query_csv(query_path, rows)
        if len(rows) < args.num_per_type:
            print(f"Warning: {structure_type} accepted {len(rows)} / {args.num_per_type}. Increase --candidate-multiplier if needed.")

    save_checkpoint(checkpoint_path, checkpoint)
    print(
        "Generation sources: "
        f"API-generated={stats.get('api_generated', 0)}, "
        f"template-generated={stats.get('template_generated', 0)}, "
        f"template-fallback={stats.get('template_fallback_generated', 0)}, "
        f"checkpoint-reused={stats.get('checkpoint_reused', 0)}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
