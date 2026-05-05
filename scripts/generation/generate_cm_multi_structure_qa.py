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

import networkx as nx
from tqdm import tqdm


DEFAULT_CONFIG = {
    # Fill this if you want to keep the API key in this script.
    # Command line --api-key and OPENAI_API_KEY will override this value.
    "api_key": "sk-QWtRU87Af1ILa6kyJhsWzhxx1r5by3ijpbuPGTUVgvrW3LP8",
    "input_csv": r"D:\MCQA\CM\structured_triples.csv",
    "output_root": r"D:\MCQA\CM\CM-MultiStructure-QA",
    "checkpoint_file": r"D:\MCQA\CM\cm_multi_structure_qa.checkpoint.json",
    "error_log_file": r"D:\MCQA\CM\cm_multi_structure_qa.errors.jsonl",
    "base_url": "https://ai.nengyongai.cn/v1",
    "model": "gpt-4o-mini",
    "qps_limit": 5,
    "max_workers": 10,
    "num_per_type": 1000,
    "candidate_multiplier": 4,
    "max_input_edges": 1500000,
    "seed": 42,
}


STRUCTURE_TYPES = ["single_edge", "path_4", "star_1hop", "star_2hop", "cycle"]
QUERY_GENERATION_VERSION = "cycle_chain_style_v6"

RELATION_WHITELIST = {
    "影像学检查",
    "实验室检查",
    "临床表现",
    "病因",
    "病理分型",
    "药物治疗",
    "手术治疗",
    "辅助治疗",
    "内窥镜检查",
}

SYSTEM_PROMPT = (
    "你是一个中文医学问答数据构造助手。你的任务不是翻译知识图谱三元组，而是把 evidence "
    "改写成真实场景里可能被医生、学生、患者家属或科研人员提出的问题。问题必须自然、明确、"
    "只有一个答案，且不能泄露标准答案或增加 evidence 外的信息。"
)

USER_PROMPT_TEMPLATE = """请基于下面的医学知识图谱 evidence 生成 1 个中文问答 query。

结构类型：{structure_type}
标准答案：{answer}
query 中必须显式出现的实体：{required_entities}
结构专用要求：
{structure_guidance}

硬性要求：
1. query 必须符合真实人类提问习惯，不能像在描述图结构或数据标注任务。
2. query 不能出现“路径”“星型”“中心节点”“二跳”“环”“闭环”“图结构”等结构术语。
3. query 不能直接包含标准答案。
4. query 必须完整利用 evidence 中与答案相关的关键线索，不能遗漏主要约束。
5. “query 中必须显式出现的实体”里的每一个实体都必须原样出现在 query 中，不能改写、泛化、替换或遗漏。
6. 标准答案不能出现在 query 中；答案只能由这些实体线索推断出来。
7. 不允许新增 evidence 中没有的疾病、检查、治疗、症状、病因或医学背景。
8. 提问方式要自然，但不要为了多样化牺牲唯一答案；宁可朴素，也不要奇怪。
9. 可以使用临床医生、规培医生、医学生、患者家属、科研检索或病历讨论语气，但必须只问一个实体答案。
10. 推荐句式是“更符合哪种疾病/诊断/检查/治疗/表现？”“通常会考虑哪项检查/哪种治疗/哪种情况？”这类单答案问法。
11. 即使结构复杂，也要把所有非答案实体自然放入一个问题中；不要只选择其中一部分线索。
12. 不要使用“某种疾病”“某类问题”“该实体”“共同指向”“线索”“根据这些信息”等明显的数据集构造表达。
13. 不要机械使用“其中A又关联B，同时C又关联D”这种模板；可以写成病例讨论、检查选择、教学提问或临床检索需求。
14. query 只能包含一个明确问题，不能连续问两个问题，不能写成“A是什么？B又如何？”。
15. query 不能问开放列表或建议，禁止使用“哪些、哪几种、哪几项、有哪些、分别、各自、有什么不同、如何选择、给我建议、是否需要、是否可以、能否”等问法。
16. query 的答案必须是标准答案这个单一实体，不能是多个检查、多个治疗、多个表现或一个判断结论。
17. 如果是路径结构，答案必须是路径中的中间节点，不要把首尾节点作为答案。
18. 如果是一跳或二跳星状结构，答案必须是多个条件共同定位出的实体，不能问分支上的检查、治疗、症状或处理建议。
19. 如果是环状结构，必须采用链式表达：按环上关系顺序写清楚 A 与 B、B 与 C、C 与 D、D 与 A 的关系，最后只问共同连接实体；不要只罗列实体，不要自由改成其他问法。

Evidence 三元组：
{triples_text}

只输出 JSON，不要输出 Markdown 或解释。格式：
{{
  "query": "...",
  "answer": "{answer}",
  "used_evidence_indices": [1, 2, 3]
}}
"""

REVIEW_PROMPT_TEMPLATE = """请评审下面这个中文医学问答 query 是否像真实人类在常规场景中会提出的问题。

结构类型：{structure_type}
标准答案：{answer}
query 中必须显式出现的实体：{required_entities}
结构专用要求：
{structure_guidance}
query：
{query}

Evidence 三元组：
{triples_text}

评审标准：
1. query 是否自然，是否像医生、学生、患者家属、科研人员或病历讨论中可能出现的问题。
2. query 是否只有一个明确问题，不能包含两个或多个并列问题。
3. query 是否避免了“某种疾病”“某类问题”“共同指向”“线索”“实体”等数据集构造语言。
4. query 是否完整保留必须出现的实体，且没有直接泄露标准答案。
5. query 是否没有新增 evidence 外的医学事实。
6. 如果结构较复杂，query 是否自然表达了节点之间的关系，而不是只把实体并列罗列。
7. 如果是环状结构，是否采用链式表达逐条写出了环上每一条关系，而不是只说这些节点同时出现或简单并列。
8. 如果是一跳或二跳星状结构，是否把所有分支作为定位标准答案的条件，而不是询问分支上的检查、治疗、症状或建议。

只输出 JSON，不要输出 Markdown 或解释。格式：
{{
  "accepted": true,
  "score": 1到5的整数,
  "single_answer_entity": true,
  "asks_multiple_questions": false,
  "asks_for_list": false,
  "relation_coverage_ok": true,
  "cycle_relations_expressed": true,
  "reason": "简短说明",
  "suggested_query": ""
}}

只有当 score >= 4，且 single_answer_entity=true、asks_multiple_questions=false、asks_for_list=false、relation_coverage_ok=true 时，accepted 才能为 true。
如果是环状结构，cycle_relations_expressed 也必须为 true；只有 query 按顺序表达了环上每一条边关系，才能给 true。
如果不自然、像三元组复述、像数据集构造语句、问题很奇怪、包含多个问题、询问开放列表或没有唯一实体答案，accepted 必须为 false。
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
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_good_entity(value):
    value = clean_text(value)
    if not value or len(value) < 2 or len(value) > 40:
        return False
    bad_values = {"s", "n", "nan", "None", "对照组", "观察组", "治疗组", "实验组"}
    if value in bad_values:
        return False
    if re.fullmatch(r"[A-Za-z]?", value):
        return False
    return True


def is_good_query_entity(value):
    value = clean_text(value)
    if not is_good_entity(value):
        return False
    if len(value) > 24:
        return False
    noisy_tokens = ["对照组", "观察组", "正常组", "治疗组", "实验组", "%", "，", ",", "；", ";"]
    if any(token in value for token in noisy_tokens):
        return False
    return True


def is_good_relation(value):
    return clean_text(value) in RELATION_WHITELIST


def load_cm_graph(input_csv, max_input_edges):
    graph = nx.MultiDiGraph()
    out_adj = defaultdict(list)
    undirected_adj = defaultdict(list)
    edges = []

    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Loading CM triples", unit="edge"):
            head = clean_text(row.get("head"))
            tail = clean_text(row.get("tail"))
            relation = clean_text(row.get("relation"))
            head_type = clean_text(row.get("head_type"))
            tail_type = clean_text(row.get("tail_type"))

            if head == tail:
                continue
            if not (is_good_entity(head) and is_good_entity(tail) and is_good_relation(relation)):
                continue

            graph.add_node(head, label=head, type=head_type)
            graph.add_node(tail, label=tail, type=tail_type)
            graph.add_edge(head, tail, description=relation)

            edge = {
                "subject": head,
                "subject_type": head_type,
                "relation": relation,
                "object": tail,
                "object_type": tail_type,
            }
            edges.append(edge)
            out_adj[head].append(edge)
            undirected_adj[head].append(edge)
            undirected_adj[tail].append(
                {
                    "subject": tail,
                    "subject_type": tail_type,
                    "relation": relation,
                    "object": head,
                    "object_type": head_type,
                    "reversed": True,
                }
            )

            if max_input_edges and len(edges) >= max_input_edges:
                break

    return graph, out_adj, undirected_adj, edges


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


def format_edge_sentence(edge):
    return f"{edge['subject']} --{edge['relation']}--> {edge['object']}"


def relation_to_natural_clause(edge):
    subject = edge["subject"]
    relation = edge["relation"]
    obj = edge["object"]
    templates = {
        "病因": f"{subject}可能由{obj}引起",
        "临床表现": f"{subject}的一个临床表现是{obj}",
        "病理分型": f"{subject}可分型为{obj}",
        "药物治疗": f"{subject}可使用{obj}进行药物治疗",
        "手术治疗": f"{subject}可采用{obj}进行手术治疗",
        "辅助治疗": f"{subject}可采用{obj}进行辅助治疗",
        "影像学检查": f"{subject}常借助{obj}进行影像学检查",
        "实验室检查": f"{subject}常借助{obj}进行实验室检查",
        "内窥镜检查": f"{subject}常借助{obj}进行内窥镜检查",
    }
    return templates.get(relation, f"{subject}与{obj}存在{relation}关系")


def relation_to_question_phrase(relation):
    phrases = {
        "病因": "通常要考虑哪种病因",
        "临床表现": "通常会出现哪种临床表现",
        "病理分型": "通常属于哪种病理分型",
        "药物治疗": "通常会考虑哪种药物治疗",
        "手术治疗": "通常会考虑哪种手术治疗",
        "辅助治疗": "通常会考虑哪种辅助治疗",
        "影像学检查": "通常会进行哪项影像学检查",
        "实验室检查": "通常会进行哪项实验室检查",
        "内窥镜检查": "通常会进行哪项内窥镜检查",
    }
    return phrases.get(relation, f"通常会考虑哪种{relation}")


def structure_guidance(sample):
    structure_type = sample["structure_type"]
    if structure_type == "cycle":
        cycle_edges = sample.get("cycle_edges") or sample["triples"][: sample.get("cycle_length", 0)]
        common_edges = sample.get("common_edges", [])
        cycle_text = "\n".join(f"- {format_edge_sentence(edge)}" for edge in cycle_edges)
        common_text = "\n".join(f"- {format_edge_sentence(edge)}" for edge in common_edges) if common_edges else "无"
        clause_text = "，而".join(relation_to_natural_clause(edge) for edge in cycle_edges)
        answer_relation = common_edges[0]["relation"] if common_edges else "相关处理"
        question_phrase = relation_to_question_phrase(answer_relation)
        return (
            "这是环状关系样本。所有 cycle query 必须稳定使用链式表达，不要自由发挥。\n"
            "query 必须先按照环上关系的顺序逐条表达每一条边，形成类似“慢性脑供血不足可能由高血脂症引起，而高血脂症又与甲状腺功能减退相关，甲状腺功能减退的一个临床表现是认知功能障碍，这种情况下应该进行什么样的辅助治疗？”的句式。\n"
            "query 不能只把环上实体并列成症状或疾病清单，必须体现 A-B、B-C、C-D、D-A 的关系链。\n"
            "如果存在共同答案关系，query 的最后一个问题必须询问标准答案这个共同连接实体。\n"
            f"建议问题结尾：这种情况下{question_phrase}？\n"
            "推荐链式表达草稿，生成时可以润色但不能改变实体和关系：\n"
            f"{clause_text}，这种情况下{question_phrase}？\n"
            "环上关系：\n"
            f"{cycle_text}\n"
            "共同答案关系：\n"
            f"{common_text}"
        )
    if structure_type == "star_2hop":
        branches = []
        triples = sample["triples"]
        for index in range(0, len(triples), 2):
            if index + 1 < len(triples):
                branches.append(f"- {format_edge_sentence(triples[index])}; {format_edge_sentence(triples[index + 1])}")
        return (
            "这是二跳星状样本。query 必须把每个分支都作为定位标准答案的条件，只能询问标准答案这个实体。\n"
            "禁止询问分支上的检查、治疗、表现或处理建议，禁止写成两个问题。\n"
            "分支关系：\n"
            + "\n".join(branches)
        )
    if structure_type == "star_1hop":
        return (
            "这是一跳星状样本。query 必须把所有非答案实体作为共同限定条件，唯一问题只能询问标准答案这个实体。\n"
            "禁止问“有哪些检查/治疗/表现”，禁止让答案变成列表。"
        )
    if structure_type == "path_4":
        return "这是四节点路径样本。query 必须自然表达路径上的连续关系，唯一问题只能询问标准答案这个中间节点。"
    return "这是单边样本。query 必须只询问标准答案这个实体。"


def sample_single_edges(edges, count):
    selected = []
    seen = set()
    preferred = ["影像学检查", "实验室检查", "临床表现", "病因", "药物治疗", "手术治疗"]
    sorted_edges = sorted(edges, key=lambda e: preferred.index(e["relation"]) if e["relation"] in preferred else 99)
    for edge in sorted_edges:
        key = edge_key(edge)
        if key in seen:
            continue
        answer = edge["object"]
        selected.append(
            {
                "structure_type": "single_edge",
                "answer": answer,
                "triples": [edge],
            }
        )
        seen.add(key)
        if len(selected) >= count:
            break
    return selected


def sample_directed_paths(out_adj, count, path_edges=3):
    selected = []
    seen_paths = set()
    starts = list(out_adj.keys())
    random.shuffle(starts)

    def dfs(node, triples, visited):
        if len(triples) == path_edges:
            nodes = [triples[0]["subject"]] + [e["object"] for e in triples]
            if len(set(nodes)) != len(nodes):
                return None
            answer = nodes[len(nodes) // 2]
            if answer in (nodes[0], nodes[-1]):
                return None
            path_id = tuple(nodes)
            if path_id in seen_paths:
                return None
            seen_paths.add(path_id)
            return {
                "structure_type": "path_4",
                "answer": answer,
                "triples": triples,
            }

        candidates = list(out_adj.get(node, []))
        random.shuffle(candidates)
        for edge in candidates[:30]:
            nxt = edge["object"]
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
    centers = sorted(out_adj.keys(), key=lambda n: len(out_adj[n]), reverse=True)
    random.shuffle(centers[:1000])
    used_centers = set()
    preferred_relations = ["临床表现", "实验室检查", "药物治疗", "影像学检查", "辅助治疗", "手术治疗", "病因"]

    for center in centers:
        if center in used_centers:
            continue
        by_relation = defaultdict(dict)
        for edge in out_adj[center]:
            if edge.get("subject_type") != "疾病":
                continue
            if edge["object"] != center and is_good_query_entity(edge["object"]):
                by_relation[edge["relation"]].setdefault(edge["object"], edge)
        chosen_edges = None
        for relation in preferred_relations:
            candidates = list(by_relation.get(relation, {}).values())
            if len(candidates) >= min_neighbors:
                random.shuffle(candidates)
                chosen_edges = candidates[: random.randint(min_neighbors, min(max_neighbors, len(candidates)))]
                break
        if not chosen_edges:
            continue
        selected.append(
            {
                "structure_type": "star_1hop",
                "answer": center,
                "triples": chosen_edges,
            }
        )
        used_centers.add(center)
        if len(selected) >= count:
            break
    return selected


def sample_two_hop_stars(out_adj, count, branches=2):
    selected = []
    centers = sorted(out_adj.keys(), key=lambda n: len(out_adj[n]), reverse=True)
    used_centers = set()
    preferred_relations = ["临床表现", "实验室检查", "药物治疗", "影像学检查", "辅助治疗", "手术治疗", "病因"]

    for center in centers:
        if center in used_centers:
            continue
        if not out_adj[center] or out_adj[center][0].get("subject_type") != "疾病":
            continue
        triples = None
        for relation in preferred_relations:
            used_nodes = {center}
            branch_pairs = []
            first_edges = [
                edge
                for edge in out_adj[center]
                if edge["relation"] == relation and is_good_query_entity(edge["object"])
            ]
            random.shuffle(first_edges)
            for first in first_edges:
                mid = first["object"]
                if mid in used_nodes:
                    continue
                second_candidates = [
                    edge
                    for edge in out_adj.get(mid, [])
                    if edge["object"] not in used_nodes
                    and edge["object"] != center
                    and is_good_query_entity(edge["object"])
                ]
                random.shuffle(second_candidates)
                if not second_candidates:
                    continue
                second = second_candidates[0]
                branch_pairs.append((first, second))
                used_nodes.update({mid, second["object"]})
                if len(branch_pairs) >= branches:
                    triples = []
                    for pair in branch_pairs:
                        triples.extend(pair)
                    break
            if triples:
                break
        if triples and len(triples) == branches * 2:
            selected.append(
                {
                    "structure_type": "star_2hop",
                    "answer": center,
                    "triples": triples,
                }
            )
            used_centers.add(center)
        if len(selected) >= count:
            break
    return selected


def sample_cycles(out_adj, count, min_len=3, max_len=4):
    selected = []
    seen_cycles = set()
    nodes = list(out_adj.keys())
    random.shuffle(nodes)
    by_pair = {}
    target_sets = {}
    for source, edge_list in out_adj.items():
        targets = set()
        for edge in edge_list:
            target = edge["object"]
            by_pair.setdefault((source, target), edge)
            if is_good_entity(target):
                targets.add(target)
        target_sets[source] = targets

    def canonical_cycle(nodes):
        rotations = [tuple(nodes[i:] + nodes[:i]) for i in range(len(nodes))]
        return min(rotations)

    def pick_common_answer(ring_nodes):
        common_targets = None
        for node in ring_nodes:
            targets = target_sets.get(node, set()) - set(ring_nodes)
            common_targets = targets if common_targets is None else common_targets & targets
            if not common_targets:
                return None
        return sorted(common_targets, key=lambda value: (len(value), value))[0]

    def add_cycle(ring_nodes):
        if len(ring_nodes) < min_len or len(ring_nodes) > max_len or len(set(ring_nodes)) != len(ring_nodes):
            return False
        cycle_id = canonical_cycle(ring_nodes)
        if cycle_id in seen_cycles:
            return False
        answer = pick_common_answer(ring_nodes)
        if not answer:
            return False
        cycle_edges = []
        for index, source in enumerate(ring_nodes):
            target = ring_nodes[(index + 1) % len(ring_nodes)]
            edge = by_pair.get((source, target))
            if not edge:
                return False
            cycle_edges.append(edge)
        common_edges = []
        for ring_node in ring_nodes:
            edge = by_pair.get((ring_node, answer))
            if not edge:
                return False
            common_edges.append(edge)
        seen_cycles.add(cycle_id)
        selected.append(
            {
                "structure_type": "cycle",
                "answer": answer,
                "triples": cycle_edges + common_edges,
                "cycle_edges": cycle_edges,
                "common_edges": common_edges,
                "cycle_length": len(cycle_edges),
            }
        )
        return True

    for a in nodes:
        for edge_ab in out_adj.get(a, [])[:80]:
            b = edge_ab["object"]
            if b == a:
                continue
            for edge_bc in out_adj.get(b, [])[:80]:
                c = edge_bc["object"]
                if c in {a, b}:
                    continue
                if (c, a) in by_pair and add_cycle([a, b, c]):
                    if len(selected) >= count:
                        return selected
                if max_len >= 4:
                    for edge_cd in out_adj.get(c, [])[:40]:
                        d = edge_cd["object"]
                        if d in {a, b, c}:
                            continue
                        if (d, a) in by_pair and add_cycle([a, b, c, d]):
                            if len(selected) >= count:
                                return selected
    return selected


def triples_to_text(triples):
    lines = []
    for index, edge in enumerate(triples, start=1):
        lines.append(
            f"{index}. ({edge['subject']}[{edge.get('subject_type', '')}], "
            f"{edge['relation']}, {edge['object']}[{edge.get('object_type', '')}])"
        )
    return "\n".join(lines)


def add_structure_to_graphml(sample, output_path):
    graphml = ET.Element(
        "graphml",
        {
            "xmlns": "http://graphml.graphdrawing.org/xmlns",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": (
                "http://graphml.graphdrawing.org/xmlns "
                "http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"
            ),
        },
    )
    ET.SubElement(graphml, "key", {"id": "d2", "for": "edge", "attr.name": "description", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d1", "for": "node", "attr.name": "type", "attr.type": "string"})
    ET.SubElement(graphml, "key", {"id": "d0", "for": "node", "attr.name": "label", "attr.type": "string"})
    graph = ET.SubElement(graphml, "graph", {"edgedefault": "directed"})

    nodes = {}
    for edge in sample["triples"]:
        nodes.setdefault(edge["subject"], edge.get("subject_type", ""))
        nodes.setdefault(edge["object"], edge.get("object_type", ""))

    for node_id, node_type in nodes.items():
        node_el = ET.SubElement(graph, "node", {"id": node_id})
        label_el = ET.SubElement(node_el, "data", {"key": "d0"})
        label_el.text = node_id
        type_el = ET.SubElement(node_el, "data", {"key": "d1"})
        type_el.text = node_type

    for edge in sample["triples"]:
        edge_el = ET.SubElement(graph, "edge", {"source": edge["subject"], "target": edge["object"]})
        desc_el = ET.SubElement(edge_el, "data", {"key": "d2"})
        desc_el.text = edge["relation"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(graphml)
    ET.indent(tree, space="  ", level=0)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def build_prompt(sample):
    required_entities = "、".join(required_query_entities(sample))
    return USER_PROMPT_TEMPLATE.format(
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=required_entities,
        structure_guidance=structure_guidance(sample),
        triples_text=triples_to_text(sample["triples"]),
    )


def build_review_prompt(sample, query):
    required_entities = "、".join(required_query_entities(sample))
    return REVIEW_PROMPT_TEMPLATE.format(
        structure_type=sample["structure_type"],
        answer=sample["answer"],
        required_entities=required_entities,
        structure_guidance=structure_guidance(sample),
        query=query,
        triples_text=triples_to_text(sample["triples"]),
    )


def answer_is_exposed(sample, query):
    answer = clean_text(sample["answer"])
    if not answer:
        return False
    query_text = clean_text(query)
    for entity in required_query_entities(sample):
        query_text = query_text.replace(clean_text(entity), "")
    return answer in query_text


MULTI_ANSWER_TERMS = [
    "哪些",
    "哪几",
    "哪几种",
    "哪几项",
    "有哪些",
    "分别",
    "各自",
    "有什么不同",
    "如何选择",
    "怎么选择",
    "给我一些建议",
    "给我建议",
]

YES_NO_TERMS = [
    "是否需要",
    "是否可以",
    "是否能",
    "能否",
    "可不可以",
    "有没有必要",
]


def has_multiple_question_risk(query):
    query = clean_text(query)
    if query.count("？") + query.count("?") > 1:
        return True
    if any(term in query for term in MULTI_ANSWER_TERMS + YES_NO_TERMS):
        return True
    risky_patterns = [
        r"[？?]\s*(同时|另外|还|并且)",
        r"吗[？?]?\s*(同时|另外|还|并且)",
        r"(检查|治疗|表现|症状|药物).*(和|以及|或).*(检查|治疗|表现|症状|药物).*[？?]",
        r"(应该|需要).*(检查|治疗).*(并|同时|以及).*(治疗|检查)",
    ]
    return any(re.search(pattern, query) for pattern in risky_patterns)


def validate_query(sample, query):
    if not query:
        raise ValueError("empty query")
    banned_terms = ["路径", "星型", "中心节点", "二跳", "环", "闭环", "图结构", "实体", "共同指向", "线索"]
    if any(term in query for term in banned_terms):
        raise ValueError(f"query contains banned term: {query}")
    if answer_is_exposed(sample, query):
        raise ValueError("query exposes the answer")
    missing_entities = missing_required_entities(sample, query)
    if missing_entities:
        raise ValueError(f"query misses required entities: {missing_entities}")
    question_marks = query.count("？") + query.count("?")
    if question_marks > 1:
        raise ValueError("query appears to contain multiple questions")
    if has_multiple_question_risk(query):
        raise ValueError("query has multiple-answer or yes/no question risk")


def review_query_with_llm(client, limiter, model, sample, query):
    prompt = build_review_prompt(sample, query)
    limiter.wait()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个严格的中文医学问答质量评审员，只输出 JSON。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=False,
    )
    data = extract_json(response.choices[0].message.content)
    accepted = bool(data.get("accepted", False))
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        score = 0
    reason = clean_text(data.get("reason", ""))
    single_answer_entity = bool(data.get("single_answer_entity", False))
    asks_multiple_questions = bool(data.get("asks_multiple_questions", True))
    asks_for_list = bool(data.get("asks_for_list", True))
    relation_coverage_ok = bool(data.get("relation_coverage_ok", False))
    cycle_relations_expressed = bool(data.get("cycle_relations_expressed", sample["structure_type"] != "cycle"))
    if (
        not accepted
        or score < 4
        or not single_answer_entity
        or asks_multiple_questions
        or asks_for_list
        or not relation_coverage_ok
        or (sample["structure_type"] == "cycle" and not cycle_relations_expressed)
    ):
        raise ValueError(f"query rejected by LLM reviewer: score={score}, reason={reason}")
    return {
        "accepted": accepted,
        "score": score,
        "single_answer_entity": single_answer_entity,
        "asks_multiple_questions": asks_multiple_questions,
        "asks_for_list": asks_for_list,
        "relation_coverage_ok": relation_coverage_ok,
        "cycle_relations_expressed": cycle_relations_expressed,
        "reason": reason,
    }


def call_llm_api(client, limiter, model, sample, retries, enable_review=True):
    base_prompt = build_prompt(sample)
    last_error = None
    for attempt in range(retries + 1):
        try:
            prompt = base_prompt
            if last_error:
                prompt += (
                    "\n\n上一次生成未通过，失败原因如下：\n"
                    f"{last_error}\n"
                    "请针对这个失败原因重新生成。必须保留所有必须出现的实体，只问一个单一实体答案，"
                    "不要问列表、建议或是否判断。"
                )
            limiter.wait()
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                stream=False,
            )
            data = extract_json(response.choices[0].message.content)
            query = clean_text(data.get("query", ""))
            validate_query(sample, query)
            review = {"accepted": True, "score": 5, "reason": "review disabled"}
            if enable_review:
                review = review_query_with_llm(client, limiter, model, sample, query)
            return query, review
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM query generation failed: {last_error}")


def load_checkpoint(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def append_error(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_query_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "query", "answer", "structure_type"])
        writer.writerows(rows)


def clear_structure_output(output_root, structure_type):
    structure_root = output_root / structure_type
    evidence_root = structure_root / "evidence"
    if evidence_root.exists():
        for graphml_path in evidence_root.rglob("*.graphml"):
            graphml_path.unlink()
    query_path = structure_root / "query" / "output_queries_part_1.csv"
    if query_path.exists():
        query_path.unlink()


def collect_samples(out_adj, edges, num_per_type, selected_types):
    samples = {}
    if "single_edge" in selected_types:
        samples["single_edge"] = sample_single_edges(edges, num_per_type)
    if "path_4" in selected_types:
        samples["path_4"] = sample_directed_paths(out_adj, num_per_type)
    if "star_1hop" in selected_types:
        samples["star_1hop"] = sample_one_hop_stars(out_adj, num_per_type)
    if "star_2hop" in selected_types:
        samples["star_2hop"] = sample_two_hop_stars(out_adj, num_per_type)
    if "cycle" in selected_types:
        samples["cycle"] = sample_cycles(out_adj, num_per_type)
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CM-MCQA trial QA data from multiple graph structures with rule-based structure extraction and LLM query generation."
    )
    parser.add_argument("--input-csv", default=DEFAULT_CONFIG["input_csv"])
    parser.add_argument("--output-root", default=DEFAULT_CONFIG["output_root"])
    parser.add_argument("--checkpoint-file", default=DEFAULT_CONFIG["checkpoint_file"])
    parser.add_argument("--error-log-file", default=DEFAULT_CONFIG["error_log_file"])
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", DEFAULT_CONFIG["api_key"]))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_CONFIG["base_url"]))
    parser.add_argument("--model", default=DEFAULT_CONFIG["model"])
    parser.add_argument("--qps-limit", type=int, default=DEFAULT_CONFIG["qps_limit"])
    parser.add_argument("--max-workers", type=int, default=DEFAULT_CONFIG["max_workers"])
    parser.add_argument("--num-per-type", type=int, default=DEFAULT_CONFIG["num_per_type"])
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=DEFAULT_CONFIG["candidate_multiplier"],
        help="Collect this many candidate structures per requested item so rejected queries can be replaced.",
    )
    parser.add_argument("--max-input-edges", type=int, default=DEFAULT_CONFIG["max_input_edges"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--disable-human-review",
        action="store_true",
        help="Disable the second LLM review pass for human-likeness.",
    )
    parser.add_argument(
        "--structure-types",
        default=",".join(STRUCTURE_TYPES),
        help="Comma-separated subset: single_edge,path_4,star_1hop,star_2hop,cycle",
    )
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
    _, out_adj, _, edges = load_cm_graph(args.input_csv, args.max_input_edges)
    samples_by_type = collect_samples(out_adj, edges, candidate_count, selected_types)

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

    tasks = []
    for structure_type, samples in samples_by_type.items():
        for candidate_index, sample in enumerate(samples, start=1):
            sample["candidate_index"] = candidate_index
            tasks.append(sample)

    results_by_type = defaultdict(list)

    def worker(sample):
        key = f"{QUERY_GENERATION_VERSION}::{sample_signature(sample)}"
        if key in checkpoint:
            item = checkpoint[key]
            return sample, item["query"], item.get("review", {})
        query, review = call_llm_api(
            client,
            limiter,
            args.model,
            sample,
            args.retries,
            enable_review=not args.disable_human_review,
        )
        with checkpoint_lock:
            checkpoint[key] = {
                "query": query,
                "answer": sample["answer"],
                "structure_type": sample["structure_type"],
                "signature": sample_signature(sample),
                "review": review,
            }
            if len(checkpoint) % 50 == 0:
                save_checkpoint(checkpoint_path, checkpoint)
        return sample, query, review

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, sample): sample for sample in tasks}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Generating queries", unit="query"):
            sample = future_map[future]
            try:
                done_sample, query, review = future.result()
                structure_type = done_sample["structure_type"]
                if len(results_by_type[structure_type]) < args.num_per_type:
                    results_by_type[structure_type].append((done_sample, query, review))
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
        for idx, (sample, query, review) in enumerate(accepted[: args.num_per_type], start=1):
            filename = f"{structure_type}_{idx:05d}.graphml"
            rel_file = f"batch_000/{filename}"
            evidence_path = output_root / structure_type / "evidence" / rel_file
            add_structure_to_graphml(sample, evidence_path)
            rows.append([rel_file, query, sample["answer"], sample["structure_type"]])
        query_path = output_root / structure_type / "query" / "output_queries_part_1.csv"
        write_query_csv(query_path, rows)
        if len(rows) < args.num_per_type:
            print(f"Warning: {structure_type} accepted {len(rows)} / {args.num_per_type}. Increase --candidate-multiplier if needed.")

    save_checkpoint(checkpoint_path, checkpoint)
    print("Done.")


if __name__ == "__main__":
    main()
