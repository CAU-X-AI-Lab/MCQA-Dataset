import argparse
import csv
import importlib.util
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


BASE_SCRIPT = Path(__file__).with_name("generate_fb_multi_structure_qa.py")
GENERATION_VERSION = "fb_single_cycle_diverse_api_v1"
STRUCTURE_TYPES = ["single_edge", "cycle"]


def load_base_module():
    spec = importlib.util.spec_from_file_location("fb_multi_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def sample_single_edges_diverse(edges, count):
    selected = []
    used_answers = set()
    used_pairs = set()
    relation_counts = Counter()

    preferred_order = [
        "contains",
        "time zones",
        "county",
        "capital",
        "place of birth",
        "nationality",
        "country related",
    ]

    shuffled = list(edges)
    random.shuffle(shuffled)

    def priority(edge):
        rel = edge["relation"].lower()
        for index, token in enumerate(preferred_order):
            if token in rel:
                return index
        return len(preferred_order)

    sorted_edges = sorted(shuffled, key=priority)
    for edge in tqdm(sorted_edges, desc="Sampling diverse single_edge", unit="edge"):
        answer_spec = base.single_edge_answer_spec(edge)
        if not answer_spec:
            continue
        answer, answer_side = answer_spec
        answer_key = base.clean_text(answer).lower()
        if answer_key in used_answers:
            continue
        relation_key = base.readable_relation(edge["relation"]).lower()
        if relation_counts[relation_key] >= max(3, count // 8):
            continue
        pair_key = (edge["subject_id"], edge["relation"], edge["object_id"])
        if pair_key in used_pairs:
            continue

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
        used_answers.add(answer_key)
        used_pairs.add(pair_key)
        relation_counts[relation_key] += 1
        if len(selected) >= count:
            break
    return selected


def sample_cycles_diverse(
    out_adj,
    by_pair,
    count,
    min_len=3,
    max_len=4,
    max_start_nodes=8000,
    first_edge_limit=100,
    second_edge_limit=100,
    third_edge_limit=60,
):
    selected = []
    seen_cycles = set()
    used_answers = set()
    nodes = list(out_adj.keys())
    random.shuffle(nodes)
    target_sets = {node: {edge["object_id"] for edge in edges} for node, edges in out_adj.items()}

    def canonical(ring):
        rotations = [tuple(ring[i:] + ring[:i]) for i in range(len(ring))]
        return min(rotations)

    def build_sample(ring):
        if len(set(ring)) != len(ring):
            return None
        cid = canonical(ring)
        if cid in seen_cycles:
            return None

        common = None
        ring_set = set(ring)
        for node in ring:
            targets = target_sets.get(node, set()) - ring_set
            common = targets if common is None else common & targets
            if not common:
                return None

        answer_candidates = list(common)
        random.shuffle(answer_candidates)
        for answer_id in answer_candidates:
            common_edges = []
            for node in ring:
                edge = by_pair.get((node, answer_id))
                if not edge:
                    common_edges = []
                    break
                common_edges.append(edge)
            if not common_edges:
                continue
            answer = base.clean_text(common_edges[0]["object"])
            answer_key = answer.lower()
            if answer_key in used_answers or not base.is_good_query_entity(answer):
                continue

            cycle_edges = []
            for index, source in enumerate(ring):
                target = ring[(index + 1) % len(ring)]
                edge = by_pair.get((source, target))
                if not edge:
                    cycle_edges = []
                    break
                cycle_edges.append(edge)
            if not cycle_edges:
                continue

            seen_cycles.add(cid)
            used_answers.add(answer_key)
            return {
                "structure_type": "cycle",
                "answer": answer,
                "triples": cycle_edges + common_edges,
                "cycle_edges": cycle_edges,
                "common_edges": common_edges,
                "cycle_length": len(cycle_edges),
            }
        return None

    start_nodes = nodes[:max_start_nodes]
    for a in tqdm(start_nodes, desc="Sampling diverse cycle", unit="node"):
        first_edges = list(out_adj.get(a, []))
        random.shuffle(first_edges)
        for eab in first_edges[:first_edge_limit]:
            b = eab["object_id"]
            if b == a:
                continue
            second_edges = list(out_adj.get(b, []))
            random.shuffle(second_edges)
            for ebc in second_edges[:second_edge_limit]:
                c = ebc["object_id"]
                if c in {a, b}:
                    continue
                if (c, a) in by_pair:
                    sample = build_sample([a, b, c])
                    if sample:
                        selected.append(sample)
                        if len(selected) >= count:
                            return selected
                if max_len >= 4:
                    third_edges = list(out_adj.get(c, []))
                    random.shuffle(third_edges)
                    for ecd in third_edges[:third_edge_limit]:
                        d = ecd["object_id"]
                        if d in {a, b, c}:
                            continue
                        if (d, a) in by_pair:
                            sample = build_sample([a, b, c, d])
                            if sample:
                                selected.append(sample)
                                if len(selected) >= count:
                                    return selected
    return selected


def sample_signature(sample):
    triples = sample["triples"]
    triple_text = "||".join(
        f"{edge['subject']}|{edge['relation']}|{edge['object']}" for edge in triples
    )
    return f"{sample['structure_type']}::{triple_text}::{sample['answer']}"


def generate_query_api_only(client, limiter, model, sample, retries, enable_review=True):
    prompt = base.build_query_prompt(sample)
    last_error = None
    for attempt in range(retries + 1):
        try:
            current_prompt = prompt
            if last_error:
                current_prompt += (
                    f"\n\nPrevious attempt failed: {last_error}\n"
                    "Regenerate one better query. Keep the same gold answer and all required entities."
                )
            data = base.call_chat_json(
                client,
                limiter,
                model,
                [
                    {"role": "system", "content": base.SYSTEM_PROMPT},
                    {"role": "user", "content": current_prompt},
                ],
                0.45,
            )
            query = base.clean_text(data.get("query", ""))
            base.validate_query(sample, query)
            review = {"accepted": True, "score": 5, "reason": "review disabled", "source": "api"}
            if enable_review:
                review = base.review_query(client, limiter, model, sample, query)
            review["source"] = "api"
            return query, review
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"API query generation/review failed: {last_error}")


def write_structure_outputs(output_root, structure_type, accepted):
    base.clear_structure_output(output_root, structure_type)
    rows = []
    for idx, (sample, query, evidence_text, review) in enumerate(accepted, start=1):
        filename = f"{structure_type}_{idx:05d}.graphml"
        rel_file = f"batch_000/{filename}"
        evidence_path = output_root / structure_type / "evidence" / rel_file
        txt_path = evidence_path.with_suffix(".txt")
        base.add_structure_to_graphml(sample, evidence_path)
        txt_path.write_text(evidence_text + "\n", encoding="utf-8")
        rows.append([rel_file, query, sample["answer"], sample["structure_type"]])
    query_path = output_root / structure_type / "query" / "output_queries_part_1.csv"
    base.write_query_csv(query_path, rows)


def load_checkpoint(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate only FB single_edge and cycle QA with answer diversity."
    )
    parser.add_argument("--nodes-csv", default=base.DEFAULT_CONFIG["nodes_csv"])
    parser.add_argument("--edges-csv", default=base.DEFAULT_CONFIG["edges_csv"])
    parser.add_argument("--output-root", default=base.DEFAULT_CONFIG["output_root"])
    parser.add_argument(
        "--checkpoint-file",
        default=r"D:\MCQA\FB\fb_single_cycle_diverse_qa.checkpoint.json",
    )
    parser.add_argument(
        "--error-log-file",
        default=r"D:\MCQA\FB\fb_single_cycle_diverse_qa.errors.jsonl",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", base.DEFAULT_CONFIG["api_key"]))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", base.DEFAULT_CONFIG["base_url"]))
    parser.add_argument("--model", default=base.DEFAULT_CONFIG["model"])
    parser.add_argument("--qps-limit", type=int, default=base.DEFAULT_CONFIG["qps_limit"])
    parser.add_argument("--max-workers", type=int, default=base.DEFAULT_CONFIG["max_workers"])
    parser.add_argument("--num-per-type", type=int, default=base.DEFAULT_CONFIG["num_per_type"])
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--max-input-edges", type=int, default=base.DEFAULT_CONFIG["max_input_edges"])
    parser.add_argument("--seed", type=int, default=base.DEFAULT_CONFIG["seed"])
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--disable-human-review", action="store_true")
    parser.add_argument("--structure-types", default=",".join(STRUCTURE_TYPES))
    parser.add_argument("--max-cycle-start-nodes", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    selected_types = [item.strip() for item in args.structure_types.split(",") if item.strip()]
    invalid = [item for item in selected_types if item not in STRUCTURE_TYPES]
    if invalid:
        raise ValueError(f"Unsupported structure types for this script: {invalid}")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: openai. Install with: pip install openai") from exc
    if not args.api_key:
        raise RuntimeError("Missing API key. Fill DEFAULT_CONFIG['api_key'], set OPENAI_API_KEY, or pass --api-key.")

    output_root = Path(args.output_root)
    checkpoint_path = Path(args.checkpoint_file)
    error_log_path = Path(args.error_log_file)
    candidate_count = args.num_per_type * args.candidate_multiplier

    _, out_adj, by_pair, edges = base.load_fb_graph(args.nodes_csv, args.edges_csv, args.max_input_edges)
    print(f"Output root: {output_root}")
    print(f"Selected types: {', '.join(selected_types)}")
    print(f"Requested per type: {args.num_per_type}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print("Diversity rule: one accepted query per answer; no template fallback.")

    samples_by_type = {}
    if "single_edge" in selected_types:
        samples_by_type["single_edge"] = sample_single_edges_diverse(edges, candidate_count)
    if "cycle" in selected_types:
        samples_by_type["cycle"] = sample_cycles_diverse(
            out_adj,
            by_pair,
            candidate_count,
            max_start_nodes=args.max_cycle_start_nodes,
        )

    for structure_type, samples in samples_by_type.items():
        answer_count = len({base.clean_text(sample["answer"]).lower() for sample in samples})
        print(f"{structure_type}: collected {len(samples)} candidates, unique answers={answer_count}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    limiter = base.QpsLimiter(args.qps_limit)
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_lock = threading.Lock()
    stats = Counter()
    accepted_by_type = defaultdict(list)
    accepted_answers = defaultdict(set)

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
        query, review = generate_query_api_only(
            client,
            limiter,
            args.model,
            sample,
            args.retries,
            enable_review=not args.disable_human_review,
        )
        evidence_text = base.generate_evidence_text(sample)
        with checkpoint_lock:
            checkpoint[key] = {
                "query": query,
                "evidence_text": evidence_text,
                "answer": sample["answer"],
                "structure_type": sample["structure_type"],
                "signature": sample_signature(sample),
                "review": review,
            }
            if len(checkpoint) % 25 == 0:
                save_checkpoint(checkpoint_path, checkpoint)
        return sample, query, evidence_text, review, "api"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, sample): sample for sample in tasks}
        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Regenerating diverse FB QA", unit="item"):
            sample = future_map[future]
            structure_type = sample["structure_type"]
            try:
                done_sample, query, evidence_text, review, source = future.result()
                answer_key = base.clean_text(done_sample["answer"]).lower()
                if answer_key in accepted_answers[structure_type]:
                    stats[f"{structure_type}_duplicate_answer_skipped"] += 1
                    continue
                if len(accepted_by_type[structure_type]) >= args.num_per_type:
                    continue
                accepted_answers[structure_type].add(answer_key)
                accepted_by_type[structure_type].append((done_sample, query, evidence_text, review))
                stats[f"{source}_accepted"] += 1
            except Exception as exc:
                stats[f"{structure_type}_rejected"] += 1
                base.append_error(
                    error_log_path,
                    {
                        "structure_type": sample["structure_type"],
                        "candidate_index": sample.get("candidate_index", ""),
                        "answer": sample["answer"],
                        "error": str(exc),
                    },
                )

    for structure_type in selected_types:
        accepted = accepted_by_type.get(structure_type, [])
        accepted.sort(key=lambda item: item[0].get("candidate_index", 0))
        write_structure_outputs(output_root, structure_type, accepted)
        print(
            f"{structure_type}: wrote {len(accepted)} / {args.num_per_type}, "
            f"unique answers={len({base.clean_text(item[0]['answer']).lower() for item in accepted})}"
        )

    save_checkpoint(checkpoint_path, checkpoint)
    print("Stats:", dict(stats))
    print("Done.")


if __name__ == "__main__":
    main()
