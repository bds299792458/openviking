import os
import json
import time
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from adapters.base import BaseAdapter
from core.logger import get_logger
from core.retrieval_packing import RetrievalPacker
from core.vector_store import VikingStoreWrapper
from core.monitor import BenchmarkMonitor
from core.metrics import MetricsCalculator
from core.judge_util import llm_grader
from core.source_page_index import SourcePageIndex
_SOURCE_GENERIC_TOKENS = {
    'doc', 'docs', 'document', 'report', 'reports', 'paper', 'papers',
    'pdf', 'earnings', 'annual', 'fiscal', 'fy', 'q1', 'q2', 'q3', 'q4',
}


def _source_hint_tokens(text):
    tokens = set(re.findall(r'[a-z0-9]+', str(text or '').lower()))
    return {
        token for token in tokens
        if token not in _SOURCE_GENERIC_TOKENS
        and not re.fullmatch(r'(?:19|20)\d{2}(?:q[1-4])?', token)
        and not token.isdigit()
        and len(token) >= 3
    }


def _compact_source_text(text):
    return re.sub(r'[^a-z0-9]+', '', str(text or '').lower())


def _resource_root(uri):
    match = re.match(r'^(?:viking://resources/([^/]+)|source://([^/]+))', str(uri or ''))
    if not match:
        return ''
    return match.group(1) or match.group(2) or ''


def _resource_scope_uri(root):
    root = str(root or '').strip().strip('/')
    return f'viking://resources/{root}' if root else 'viking://resources'


def _query_mentions_source(sample_id, query):
    source_tokens = _source_hint_tokens(sample_id)
    query_text = str(query or '')
    if source_tokens:
        query_tokens = set(re.findall(r'[a-z0-9]+', query_text.lower()))
        if source_tokens & query_tokens:
            return True
        compact_query = _compact_source_text(query_text)
        if any(len(token) >= 5 and token in compact_query for token in source_tokens):
            return True
    # Numeric sample IDs such as Qasper arXiv IDs need a looser fallback.
    if (('"' in query_text) or ("'" in query_text)) and re.search(r'\d', str(sample_id or '')):
        return True
    return False



def _root_matches_source(sample_id, root):
    source_tokens = _source_hint_tokens(sample_id)
    root_tokens = _source_hint_tokens(root)
    if source_tokens and root_tokens and source_tokens & root_tokens:
        return True
    compact_source = _compact_source_text(sample_id)
    compact_root = _compact_source_text(root)
    return bool(compact_source and compact_source in compact_root)


def _split_evidence_units(text: str) -> list[str]:
    parts = re.split(r'\n\s*\n+|(?<=[.!?])\s+(?=[A-Z0-9])|(?<=;)\s+', str(text or ''))
    return [re.sub(r'\s+', ' ', part).strip() for part in parts if part and part.strip()]


def _question_focus(query: str) -> str:
    """Remove dataset source wrappers before lexical evidence matching."""
    text = str(query or '').strip()
    text = re.sub(
        r'^Based on the (?:syllabus|paper|document)\s+"[^"]+"\s*,\s*',
        '',
        text,
        flags=re.IGNORECASE,
    )
    return text.strip() or str(query or '')


def _question_anchor_hint(query: str) -> str:
    """Promote named entities in the question into the retrieval query."""
    text = str(query or '')
    seen = set()
    parts = []
    for match in re.finditer(r'\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,4}\b', text):
        entity = re.sub(r'\s+', ' ', match.group(0)).strip()
        if not entity:
            continue
        normalized = entity.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        parts.append(entity)
        compact = _compact_source_text(entity)
        if compact and compact not in seen:
            seen.add(compact)
            parts.append(compact)
    return ' '.join(parts).strip()


class BenchmarkPipeline:
    def __init__(self, config, adapter: BaseAdapter, vector_db: VikingStoreWrapper, llm):
        self.config = config
        self.adapter = adapter
        self.db = vector_db
        self.llm = llm
        self.logger = get_logger()
        self.monitor = BenchmarkMonitor()
        self.retrieval_packer = RetrievalPacker(token_counter=self.db.count_tokens)
        self.source_page_index = SourcePageIndex(
            self.config.get("paths", {}).get("doc_output_dir")
        )

        self.output_dir = self.config['paths']['output_dir']
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.generated_file = os.path.join(self.output_dir, "generated_answers.json")
        self.eval_file = os.path.join(self.output_dir, "qa_eval_detailed_results.json")
        self.report_file = os.path.join(self.output_dir, "benchmark_metrics_report.json")
        self.answer_reference = self._load_answer_reference(
            self.config.get('execution', {}).get('answer_candidate_reference_file')
        )

        self.metrics_summary = {
            "insertion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0},
            "deletion": {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        }

    def run_generation(self):
        """Step 1: Data Preparation"""
        self.logger.info(">>> Stage: Ingestion & Generation")
        skip_ingestion = self.config['execution'].get('skip_ingestion', False)
        execution_cfg = self.config.get('execution', {})
        retrieval_topk = execution_cfg.get('retrieval_topk', 5)
        self._update_report({
            'Retrieval Packing Configuration': {
                'strategy': execution_cfg.get('retrieval_strategy', 'score_only'),
                'candidate_pool_topk': execution_cfg.get('candidate_pool_topk', retrieval_topk),
                'retrieval_topk': retrieval_topk,
                'context_token_budget': execution_cfg.get('context_token_budget'),
                'max_context_chars_per_block': execution_cfg.get('max_context_chars_per_block', 8000),
                'diversity_lambda': execution_cfg.get('diversity_lambda', 0.35),
                'source_penalty': execution_cfg.get('source_penalty', 0.12),
                'query_aware_summary_limit': execution_cfg.get('query_aware_summary_limit', 1),
                'evidence_fit_min_score_ratio': execution_cfg.get('evidence_fit_min_score_ratio', 0.92),
                'evidence_fit_max_per_source': execution_cfg.get('evidence_fit_max_per_source', 2),
                'retrieve_max_retries': execution_cfg.get('retrieve_max_retries', 2),
                'retrieve_retry_base_delay_s': execution_cfg.get('retrieve_retry_base_delay_s', 1.0),
            }
        })
        doc_dir = self.config['paths'].get('doc_output_dir')
        if not doc_dir:
            doc_dir = os.path.join(self.output_dir, "docs")

        if skip_ingestion:
            self.logger.info("Skipping ingestion. Reusing the configured OpenViking Server")
            self.metrics_summary["insertion"] = {"time": 0, "input_tokens": 0, "output_tokens": 0, "embedding_tokens": 0}
        else:
            try:
                doc_info = self.adapter.data_prepare(doc_dir)
            except Exception as e:
                self.logger.exception(f"Data preparation failed: {e}")
                exit(1)

            ingest_workers = self.config['execution'].get('ingest_workers', 10)
            ingest_mode = self.config['execution'].get('ingest_mode', 'per_file')

            mode_desc = {
                'directory': 'Unified directory mode',
                'per_file': 'Per-file mode'
            }
            self.logger.info(f"Ingestion mode: {ingest_mode} ({mode_desc.get(ingest_mode, 'Unknown mode')})")
            self.logger.info(f"Number of documents: {len(doc_info)}")

            ingest_stats = self.db.ingest(
                doc_info,
                max_workers=ingest_workers,
                monitor=self.monitor,
                ingest_mode=ingest_mode
            )
            self.metrics_summary["insertion"] = ingest_stats
            self.logger.info(f"Insertion finished. Time: {ingest_stats['time']:.2f}s")

            self._update_report({
                "Insertion Efficiency (Total Dataset)": {
                    "Total Insertion Time (s)": self.metrics_summary["insertion"]["time"],
                    "Total Input Tokens": self.metrics_summary["insertion"]["input_tokens"],
                    "Total Output Tokens": self.metrics_summary["insertion"]["output_tokens"],
                    "Total Embedding Tokens": self.metrics_summary["insertion"].get("embedding_tokens", 0)
                }
            })

        samples = self.adapter.load_and_transform()
        tasks = self._prepare_tasks(samples)
        results_map = {}
        max_workers = self.config['execution']['max_workers']
        task_errors = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self._process_generation_task, task): task
                for task in tasks
            }

            pbar = tqdm(total=len(tasks), desc="Generating Answers", unit="task")
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    res = future.result()
                    results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Generation failed for task {task['id']}: {e}")
                    task_errors.append((task['id'], e))
                pbar.set_postfix(self.monitor.get_status_dict())
                pbar.update(1)
            pbar.close()

        if task_errors:
            first_id, first_err = task_errors[0]
            raise RuntimeError(
                f"Generation failed for {len(task_errors)} tasks; first failure task_id={first_id}: {type(first_err).__name__}: {first_err}"
            ) from first_err

        sorted_results = [results_map[i] for i in sorted(results_map.keys())]
        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')
        save_data = {
            "summary": {"dataset": dataset_name, "total_queries": len(sorted_results)},
            "results": sorted_results
        }
        total = len(sorted_results)
        if total > 0:
            self._update_report({
                    "Query Efficiency (Average Per Query)": {
                        "Average Retrieval Time (s)": sum(r['retrieval']['latency_sec'] for r in sorted_results) / total,
                        "Average Input Tokens": sum(r['token_usage']['total_input_tokens'] for r in sorted_results) / total,
                        "Average Output Tokens": sum(r['token_usage']['llm_output_tokens'] for r in sorted_results) / total,
                    }
                }
            )
        with open(self.generated_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

    def run_evaluation(self):
        """Step 4: Evaluation"""
        self.logger.info(">>> Stage: Evaluation")

        if not os.path.exists(self.generated_file):
            self.logger.error("Generated answers file not found.")
            return

        with open(self.generated_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data.get("results", [])

        eval_items = items
        eval_results_map = {}

        with ThreadPoolExecutor(max_workers=self.config['execution']['max_workers']) as executor:
            future_to_item = {
                executor.submit(self._process_evaluation_task, item): item
                for item in eval_items
            }

            pbar = tqdm(total=len(eval_items), desc="Evaluating", unit="item")
            for future in as_completed(future_to_item):
                try:
                    res = future.result()
                    eval_results_map[res['_global_index']] = res
                except Exception as e:
                    self.logger.error(f"Evaluation failed: {e}")
                pbar.update(1)
            pbar.close()

        eval_records = list(eval_results_map.values())
        total = len(eval_records)

        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump({"results": eval_records}, f, indent=2, ensure_ascii=False)

        if total > 0:
            self._update_report({
                "Dataset": self.config.get('dataset_name', 'Unknown_Dataset'),
                "Total Queries Evaluated": total,
                "Performance Metrics": {
                    "Average F1 Score": sum(r['metrics']['F1'] for r in eval_records) / total,
                    "Average Recall": sum(r['metrics']['Recall'] for r in eval_records) / total,
                    "Average Accuracy (Hit 0-4)": sum(r['metrics']['Accuracy'] for r in eval_records) / total,
                    "Average Accuracy (normalization)": (sum(r['metrics']['Accuracy'] for r in eval_records) / total)/4,
                }
            })

    def run_deletion(self):
        """Step 5: Cleanup"""
        self.logger.info(">>> Stage: Deletion")
        start_time = time.time()
        self.db.clear()
        duration = time.time() - start_time
        self.metrics_summary["deletion"] = {"time": duration, "input_tokens": 0, "output_tokens": 0}
        self.logger.info(f"Deletion finished. Time: {duration:.2f}s")

        self._update_report({
            "Deletion Efficiency (Total Dataset)": {
                "Total Deletion Time (s)": duration,
                "Total Input Tokens": 0,
                "Total Output Tokens": 0
            }
        })

    def _prepare_tasks(self, samples):
        tasks = []
        global_idx = 0
        max_queries = self.config['execution'].get('max_queries')
        for sample in samples:
            for qa in sample.qa_pairs:
                if max_queries is not None and global_idx >= max_queries:
                    break
                tasks.append({"id": global_idx, "sample_id": sample.sample_id, "qa": qa})
                global_idx += 1
            if max_queries is not None and global_idx >= max_queries:
                break
        return tasks

    def _process_generation_task(self, task):
        self.monitor.worker_start()
        try:
            qa = task['qa']

            t0 = time.time()
            query_tag = f"[Query-{task['id']}]"
            # Get retrieval instruction from config, default to empty
            retrieval_instruction = self.config['execution'].get('retrieval_instruction', '')
            # Build enhanced query with instruction if provided
            if retrieval_instruction:
                enhanced_query = f"{retrieval_instruction} {qa.question}"
                self.logger.debug(f"[Query-{task['id']}] Using retrieval instruction: {retrieval_instruction}")
                self.logger.debug(f"[Query-{task['id']}] Enhanced query: {enhanced_query}")
            else:
                enhanced_query = qa.question
                self.logger.debug(f"[Query-{task['id']}] No retrieval instruction, using raw query")
            anchor_hint = _question_anchor_hint(qa.question)
            if anchor_hint:
                enhanced_query = f"{enhanced_query} {anchor_hint}".strip()
            focus_query = _question_focus(qa.question)
            execution_cfg = self.config.get('execution', {})
            retrieval_topk = execution_cfg['retrieval_topk']
            strategy = execution_cfg.get('retrieval_strategy', 'score_only')
            candidate_pool_topk = max(execution_cfg.get('candidate_pool_topk', retrieval_topk), retrieval_topk)
            context_token_budget = execution_cfg.get('context_token_budget')
            max_chars_per_block = execution_cfg.get('max_context_chars_per_block', 8000)
            diversity_lambda = execution_cfg.get('diversity_lambda', 0.35)
            source_penalty = execution_cfg.get('source_penalty', 0.12)
            summary_limit = execution_cfg.get('summary_limit', 0)
            min_score_ratio = execution_cfg.get('evidence_fit_min_score_ratio', 0.92)
            max_per_source = execution_cfg.get('evidence_fit_max_per_source', 2)

            search_res = self.db.retrieve(query=enhanced_query, topk=candidate_pool_topk)
            latency = time.time() - t0

            raw_results = list(search_res.get("resources", []))
            retrieval_scope = 'global'
            scoped_root = ''

            # If the query names the document/object represented by this
            # sample, search inside the matched resource root as a second
            # pass. This is a generic source-aware retrieval step rather than
            # a dataset-specific answer rule.
            sample_id = str(task.get('sample_id', ''))
            if _query_mentions_source(sample_id, enhanced_query):
                root_scores = {}
                for result in raw_results:
                    root = _resource_root(result.get('uri', ''))
                    if not _root_matches_source(sample_id, root):
                        continue
                    score = float(result.get('score', 0.0) or 0.0)
                    root_scores[root] = max(score, root_scores.get(root, float('-inf')))

                if root_scores:
                    scoped_root = max(root_scores, key=root_scores.get)
                    scoped_res = self.db.retrieve(
                        query=enhanced_query,
                        topk=candidate_pool_topk,
                        target_uri=_resource_scope_uri(scoped_root),
                    )
                    scoped_results = [
                        result for result in scoped_res.get('resources', [])
                        if _resource_root(result.get('uri', '')) == scoped_root
                    ]
                    if scoped_results:
                        scoped_has_leaf = any(int(result.get('level', 2) or 2) >= 2 for result in scoped_results)
                        if scoped_has_leaf:
                            raw_results = scoped_results
                            retrieval_scope = 'source_scoped'
                            latency = time.time() - t0
                        else:
                            best_global_score = max(
                                (float(item.get("score", 0.0) or 0.0) for item in raw_results),
                                default=0.0,
                            )
                            scoped_boost = best_global_score + 0.01
                            boosted = False
                            for scoped_result in scoped_results:
                                scoped_root = _resource_root(scoped_result.get("uri", ""))
                                for result in raw_results:
                                    if _resource_root(result.get("uri", "")) != scoped_root:
                                        continue
                                    scoped_score = float(result.get("score", 0.0) or 0.0)
                                    if scoped_boost > scoped_score:
                                        result["score"] = scoped_boost
                                        boosted = True
                            if not boosted:
                                existing_uris = {str(item.get("uri", "")) for item in raw_results}
                                for scoped_result in scoped_results:
                                    scoped_uri = str(scoped_result.get("uri", ""))
                                    if scoped_uri in existing_uris:
                                        continue
                                    scoped_score = float(scoped_result.get("score", 0.0) or 0.0)
                                    scoped_result["score"] = max(scoped_score, scoped_boost)
                                    raw_results.append(scoped_result)
                                    existing_uris.add(scoped_uri)
                            retrieval_scope = 'global+source_scoped_fallback'
                            latency = time.time() - t0

            if execution_cfg.get("source_page_fallback", True):
                source_leaf_count = sum(
                    _root_matches_source(sample_id, _resource_root(result.get("uri", "")))
                    and int(result.get("level", 2) or 2) >= 2
                    for result in raw_results
                )
                source_page_min_leaf_candidates = max(
                    1,
                    int(execution_cfg.get("source_page_min_leaf_candidates", 1)),
                )
                if source_leaf_count < source_page_min_leaf_candidates:
                    page_results = self.source_page_index.search(
                        sample_id,
                        focus_query,
                        limit=int(
                            execution_cfg.get(
                                "source_page_candidate_limit",
                                max(candidate_pool_topk, retrieval_topk),
                            )
                        ),
                    )
                    if page_results:
                        best_global_score = max(
                            (float(item.get("score", 0.0) or 0.0) for item in raw_results),
                            default=0.0,
                        )
                        boost_base = best_global_score + 0.03
                        boosted_page_results = []
                        for rank, page_result in enumerate(page_results):
                            boosted = dict(page_result)
                            boosted["score"] = max(
                                float(page_result.get("score", 0.0) or 0.0),
                                boost_base - (rank * 0.002),
                            )
                            boosted_page_results.append(boosted)
                        page_results = boosted_page_results
                    existing_uris = {str(item.get("uri", "")) for item in raw_results}
                    for page_result in page_results:
                        if page_result["uri"] not in existing_uris:
                            raw_results.append(page_result)
                            existing_uris.add(page_result["uri"])
                    if page_results:
                        retrieval_scope = (
                            "global+source_page_boosted"
                            if retrieval_scope == "global"
                            else f"{retrieval_scope}+source_page_boosted"
                        )

            retrieved_texts = []
            retrieved_uris = []
            context_blocks = []
            context_blocks_by_uri = {}
            raw_contents = []

            for result in raw_results:
                uri = result["uri"]
                if result.get("_content") is not None:
                    raw_contents.append(str(result["_content"]))
                    continue
                content = (
                    self.db.read_resource(uri)
                    if result.get("level", 2) == 2
                    else f"{result.get('abstract', '')}\n{result.get('overview', '')}"
                )
                raw_contents.append(content)

            prepared = self.retrieval_packer.prepare_candidates(
                raw_results,
                raw_contents,
                max_chars_per_block=max_chars_per_block,
                query=focus_query,
            )
            selected_candidates, packing_stats = self.retrieval_packer.select(
                prepared,
                topk=retrieval_topk,
                strategy=strategy,
                query=focus_query,
                question_category=qa.category,
                token_budget=context_token_budget,
                diversity_lambda=diversity_lambda,
                source_penalty=source_penalty,
                summary_limit=execution_cfg.get('query_aware_summary_limit', 1)
                if strategy == 'query_aware' else summary_limit,
                min_score_ratio=min_score_ratio,
                max_per_source=max_per_source,
            )

            packing_stats.update(
                {
                    'retrieval_scope': retrieval_scope,
                    'scoped_root': scoped_root,
                    'sample_id_source_hint': sample_id,
                }
            )
            for candidate in selected_candidates:
                retrieved_uris.append(candidate.uri)
                retrieved_texts.append(candidate.content)
                if strategy == "coverage_fit":
                    block_kind = "leaf" if candidate.level >= 2 else "summary"
                    block = (
                        f"[Evidence {len(context_blocks) + 1} | {block_kind}]\n"
                        f"{candidate.prompt_text}"
                    )
                else:
                    block = candidate.prompt_text
                context_blocks.append(block)
                context_blocks_by_uri[candidate.base_uri] = block

            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)

            conservative_context_blocks = None
            if execution_cfg.get('answer_candidate_selection', False):
                conservative_context_blocks = self._build_conservative_context(
                    search_res,
                    enhanced_query,
                    topk=int(execution_cfg.get('answer_candidate_conservative_topk', 5)),
                    max_chars_per_block=max_chars_per_block,
                )

            answer_candidates = self.retrieval_packer.rank_for_answer(
                selected_candidates,
                query=focus_query,
                question_category=qa.category,
            )
            answer_query_type = self.retrieval_packer.classify_query(
                focus_query,
                qa.category,
            )
            if answer_query_type != "multi_hop" and len(answer_candidates) > 1:
                # Source filtering should only arbitrate between different
                # resource roots.  Within one document, subdirectories often
                # represent complementary sections rather than competing
                # sources; filtering them out can drop the exact evidence.
                source_groups = {}
                for candidate in answer_candidates:
                    root = _resource_root(candidate.uri) or candidate.source
                    source_groups.setdefault(root, []).append(candidate)
                if len(source_groups) > 1:
                    dominant_source, dominant_group = max(
                        source_groups.items(),
                        key=lambda item: (
                            sum(candidate.score for candidate in item[1]),
                            len(item[1]),
                            max((candidate.score for candidate in item[1]), default=0.0),
                        ),
                    )
                    if len(dominant_group) >= 2 and len(dominant_group) >= max(
                        2, len(answer_candidates) // 2
                    ):
                        if len(dominant_group) < len(answer_candidates):
                            answer_candidates = dominant_group
                            packing_stats['answer_source_filter'] = dominant_source
                            packing_stats['answer_source_filter_size'] = len(dominant_group)
            answer_context_blocks = [
                context_blocks_by_uri[candidate.base_uri]
                for candidate in answer_candidates
                if candidate.base_uri in context_blocks_by_uri
            ]
            answer_context_topk = execution_cfg.get('answer_context_topk')
            if answer_context_topk is not None:
                answer_context_topk = max(1, int(answer_context_topk))
                answer_context_blocks = answer_context_blocks[:answer_context_topk]
            packing_stats['answer_context_uris'] = [
                candidate.uri for candidate in answer_candidates[:len(answer_context_blocks)]
            ]

            full_prompt, meta = self.adapter.build_prompt(qa, answer_context_blocks)

            answer_selection = None
            primary_generation_failed = False
            answer_recovered_from_missing = False
            try:
                ans_raw = self.llm.generate(full_prompt)
                ans = self.adapter.post_process_answer(qa, ans_raw, meta)
            except Exception as error:
                fallback_answer = self.answer_reference.get(task['id'])
                if fallback_answer is None:
                    fallback_blocks = conservative_context_blocks
                    if fallback_blocks is None:
                        fallback_topk = int(execution_cfg.get('answer_candidate_conservative_topk', 5))
                        fallback_blocks = self._build_conservative_context(
                            search_res,
                            enhanced_query,
                            topk=fallback_topk,
                            max_chars_per_block=max_chars_per_block,
                        )
                    if fallback_blocks:
                        try:
                            fallback_prompt, fallback_meta = self.adapter.build_prompt(qa, fallback_blocks)
                            fallback_raw = self.llm.generate(fallback_prompt)
                            fallback_answer = self.adapter.post_process_answer(qa, fallback_raw, fallback_meta)
                        except Exception as fallback_error:
                            self.logger.warning(
                                f"{query_tag} Primary generation failed and conservative fallback also failed: {fallback_error}"
                            )
                    if fallback_answer is None:
                        fallback_answer = 'Not mentioned'
                self.logger.warning(
                    f"{query_tag} Primary generation failed; using fallback answer: {error}"
                )
                ans = fallback_answer
                primary_generation_failed = True
                answer_selection = {
                    'selected': 'conservative',
                    'method': 'primary_generation_error',
                    'conservative_answer': fallback_answer,
                    'conservative_source': 'fallback',
                }

            if not primary_generation_failed and execution_cfg.get('answer_temporal_repair', False):
                ans = self._repair_temporal_answer(qa.question, answer_context_blocks, ans)

            if not primary_generation_failed:
                normalized = str(ans or '').lower()
                missing_markers = ('not mentioned', 'not found', 'unknown', 'no answer')
                if any(marker in normalized for marker in missing_markers):
                    if execution_cfg.get('answer_missing_retry', False):
                        recovered_answer = self._retry_missing_answer(
                            qa,
                            context_blocks,
                            meta,
                            execution_cfg,
                        )
                        recovered_norm = str(recovered_answer or '').lower()
                        if recovered_answer and not any(
                            marker in recovered_norm for marker in missing_markers
                        ):
                            ans = recovered_answer
                            normalized = recovered_norm
                            answer_recovered_from_missing = True

                    if any(marker in normalized for marker in missing_markers):
                        fallback_blocks = conservative_context_blocks
                        if fallback_blocks is None:
                            fallback_topk = int(execution_cfg.get('answer_candidate_conservative_topk', 5))
                            fallback_blocks = self._build_conservative_context(
                                search_res,
                                enhanced_query,
                                topk=fallback_topk,
                                max_chars_per_block=max_chars_per_block,
                            )
                        if fallback_blocks:
                            try:
                                fallback_prompt, fallback_meta = self.adapter.build_prompt(qa, fallback_blocks)
                                fallback_raw = self.llm.generate(fallback_prompt)
                                fallback_answer = self.adapter.post_process_answer(qa, fallback_raw, fallback_meta)
                                fallback_norm = str(fallback_answer or '').lower()
                                if not any(marker in fallback_norm for marker in missing_markers):
                                    ans = fallback_answer
                                    answer_recovered_from_missing = True
                                else:
                                    snippet = self._extract_supported_snippet(qa.question, context_blocks)
                                    if snippet:
                                        ans = snippet
                                        answer_recovered_from_missing = True
                            except Exception as fallback_error:
                                self.logger.warning(
                                    f"{query_tag} Missing-answer fallback failed: {fallback_error}"
                                )
                                snippet = self._extract_supported_snippet(qa.question, context_blocks)
                                if snippet:
                                    ans = snippet
                                    answer_recovered_from_missing = True

                normalized = str(ans or '').strip().lower()
                if self._needs_numeric_verification(qa.question, normalized):
                    verified_answer = self._retry_verified_answer(
                        qa,
                        context_blocks,
                        meta,
                        execution_cfg,
                    )
                    verified_norm = str(verified_answer or '').lower()
                    if verified_answer and verified_norm != normalized:
                        ans = verified_answer
                        answer_recovered_from_missing = True

            if not primary_generation_failed and conservative_context_blocks:
                conservative_source = "generated"
                conservative_answer = self.answer_reference.get(task['id'])
                if conservative_answer is None:
                    try:
                        conservative_prompt, conservative_meta = self.adapter.build_prompt(qa, conservative_context_blocks)
                        conservative_raw = self.llm.generate(conservative_prompt)
                        conservative_answer = self.adapter.post_process_answer(qa, conservative_raw, conservative_meta)
                    except Exception as error:
                        self.logger.warning(
                            f"{query_tag} Conservative answer generation failed; continuing with primary answer: {error}"
                        )
                        conservative_context_blocks = None
                        conservative_answer = None
                        conservative_source = "generated_failed"
                else:
                    conservative_source = "reference_file"
                if conservative_context_blocks and conservative_answer is not None:
                    ans, answer_selection = self._select_supported_answer(
                        qa.question,
                        answer_context_blocks,
                        ans,
                        conservative_context_blocks,
                        conservative_answer,
                        execution_cfg,
                        prefer_conservative=conservative_source == "reference_file",
                    )
                    if answer_selection is not None:
                        answer_selection["conservative_source"] = conservative_source

            if (
                not primary_generation_failed
                and not answer_recovered_from_missing
                and execution_cfg.get('answer_final_refinement', False)
                and answer_query_type != "factual"
            ):
                refinement_blocks = list(answer_context_blocks)
                if conservative_context_blocks:
                    refinement_blocks.extend(conservative_context_blocks)
                try:
                    ans = self._refine_final_answer(qa, refinement_blocks, ans, meta, execution_cfg)
                except Exception as error:
                    self.logger.warning(
                        f"{query_tag} Final refinement failed; keeping current answer: {error}"
                    )

            in_tokens = self.db.count_tokens(full_prompt) + self.db.count_tokens(qa.question)
            out_tokens = self.db.count_tokens(ans)
            self.monitor.worker_end(tokens=in_tokens + out_tokens)

            self.logger.info(f"[Query-{task['id']}] Q: {qa.question[:30]}... | Recall: {recall:.2f} | Latency: {latency:.2f}s")

            return {
                "_global_index": task['id'], "sample_id": task['sample_id'], "question": qa.question,
                "gold_answers": qa.gold_answers, "category": str(qa.category), "evidence": qa.evidence,
                "retrieval": {
                    "latency_sec": latency,
                    "uris": retrieved_uris,
                    "candidate_pool_topk": candidate_pool_topk,
                    "packing": packing_stats,
                },
                "llm": {"final_answer": ans, "answer_selection": answer_selection},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception:
            self.monitor.worker_end(success=False)
            raise

    def _retry_missing_answer(self, qa, context_blocks: list[str], meta: dict, execution_cfg: dict) -> str:
        """Make one evidence-focused recovery pass for an otherwise empty answer."""
        if not context_blocks:
            return ''
        max_chars = int(execution_cfg.get('answer_missing_retry_max_chars', 14000) or 14000)
        evidence = '\n\n'.join(context_blocks)
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars]
        prompt = '\n'.join([
            'Answer the question from the evidence below. This is a recovery pass because a previous answer incorrectly treated the evidence as missing.',
            'Inspect every evidence block for a directly supported answer before returning a refusal.',
            'For a count, date, comparison, table value, percentage, or short factual question, extract or derive the needed value from the evidence.',
            'Return Not mentioned only when no evidence block supports any answer to the question.',
            'Return exactly one concise final answer line, without explanation or markdown.',
            '',
            f'Question: {qa.question}',
            '',
            'Evidence:',
            evidence,
            '',
            'Final answer:',
        ])
        try:
            raw = self.llm.generate(prompt)
            return self.adapter.post_process_answer(qa, raw, meta).strip()
        except Exception as error:
            self.logger.warning(f'Missing-answer recovery failed: {error}')
            return self._extract_supported_snippet(qa.question, context_blocks)

    def _retry_verified_answer(self, qa, context_blocks: list[str], meta: dict, execution_cfg: dict) -> str:
        """Verify numeric, comparison, and list answers that look like placeholders."""
        if not context_blocks:
            return ''
        max_chars = int(execution_cfg.get('answer_missing_retry_max_chars', 14000) or 14000)
        evidence = '\n\n'.join(context_blocks)
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars]
        prompt = '\n'.join([
            'Verify the answer against the evidence and replace weak placeholders such as 0, none, or Not mentioned when the evidence supports a real answer.',
            'For numeric and comparison questions, compute from the evidence if needed.',
            'For grading, ratio, percentage, amount, count, and list questions, return the shortest supported answer that still matches the evidence.',
            'Do not return 0 unless the evidence explicitly says the value is zero or absent.',
            'Return exactly one concise line.',
            '',
            f'Question: {qa.question}',
            '',
            'Evidence:',
            evidence,
            '',
            'Answer:',
        ])
        try:
            raw = self.llm.generate(prompt)
            return self.adapter.post_process_answer(qa, raw, meta).strip()
        except Exception as error:
            self.logger.warning(f'Verified-answer retry failed: {error}')
            return self._extract_supported_snippet(qa.question, context_blocks)

    def _extract_supported_snippet(self, question: str, context_blocks: list[str]) -> str:
        query = _question_focus(question)
        query_words = {token for token in re.findall(r'\w+', query.lower()) if len(token) > 2}
        query_numbers = set(re.findall(r'\b(?:\d+(?:[.,]\d+)*|\d+(?:\.\d+)?%|(?:19|20)\d{2})\b', query))
        query_entities = set(re.findall(r'\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3}\b', query))

        candidates: list[str] = []
        for block in context_blocks:
            candidates.extend(_split_evidence_units(block))
        candidates = [unit for unit in candidates if not self._is_bad_snippet(unit)]

        if not candidates:
            return ''

        def score(unit: str) -> float:
            tokens = set(re.findall(r'\w+', unit.lower()))
            digits = set(re.findall(r'\b(?:\d+(?:[.,]\d+)*|\d+(?:\.\d+)?%|(?:19|20)\d{2})\b', unit))
            ents = set(re.findall(r'\b[A-Z][A-Za-z0-9&.\-]*(?:\s+[A-Z][A-Za-z0-9&.\-]*){0,3}\b', unit))
            value = 0.0
            if query_words:
                value += 2.0 * len(query_words & tokens) / len(query_words)
            if query_numbers:
                value += 2.0 * len(query_numbers & digits) / len(query_numbers)
            if query_entities:
                value += 1.5 * len(query_entities & ents) / len(query_entities)
            if re.search(r'\b(?:yes|no|not mentioned|insufficient information)\b', unit, re.IGNORECASE):
                value += 0.25
            if len(digits) >= 2:
                value += 0.15
            if re.search(r'\b(?:part|section|table|figure|row|line|percent|percentage|million|billion|times|grade|session)\b', unit, re.IGNORECASE):
                value += 0.12
            if len(tokens) < 4:
                value -= 0.8
            return value

        best = max(candidates, key=score)
        if not best:
            return ''
        if len(best) > 320:
            best = best[:320].rsplit(' ', 1)[0].strip()
        return best

    @staticmethod
    def _is_bad_snippet(unit: str) -> bool:
        text = str(unit or '').strip()
        if not text:
            return True
        if text.startswith('[Evidence'):
            return True
        if re.fullmatch(r'#{1,6}\s+.+', text) and len(text.split()) <= 8:
            return True
        return False

    @staticmethod
    def _needs_numeric_verification(question: str, answer: str) -> bool:
        q = str(question or '')
        a = str(answer or '').strip().lower()
        if not a:
            return False
        pattern = r'\b(?:calculate|comparison|ratio|rate|percent|percentage|margin|amount|how much|count|grade|what is the|which|list)\b'
        if not re.search(pattern, q, re.IGNORECASE):
            return False
        if a in {'not mentioned', 'insufficient information', 'not calculable', 'not determinable', 'cannot be answered', 'no information', 'unknown'}:
            return True
        if a in {'0', '0.0', 'zero'}:
            return True
        return not bool(re.search(r'\d', a))

    def _refine_final_answer(
        self,
        qa,
        context_blocks: list[str],
        current_answer: str,
        meta: dict,
        execution_cfg: dict,
    ) -> str:
        answer_text = str(current_answer or '').strip()
        if not answer_text:
            return current_answer
        max_chars = int(execution_cfg.get('answer_final_refinement_max_chars', 14000) or 14000)
        nl = chr(10)
        evidence = nl.join(context_blocks)
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars]
        prompt = nl.join([
            'Rewrite the current RAG answer into the shortest exact final answer supported by the evidence.',
            'Use only the evidence below. Do not add explanation, context, or markdown.',
            'Remove unnecessary clauses, speaker names, and restatements unless they are required to answer the question.',
            'For yes/no questions, start with Yes or No and keep only the essential qualifier if needed.',
            'For date, time, amount, and entity questions, return the exact supported value.',
            'For list or comparison questions, keep every required item and separate items with commas or semicolons.',
            'When the question requires a simple inference, range lookup, or arithmetic from the evidence, perform that step instead of treating the answer as missing.',
            'If the current answer is unsupported but the evidence contains the answer, replace it with the supported answer.',
            'If the evidence does not contain the answer, return Not mentioned.',
            'Return one line only.',
            '',
            f'Question: {qa.question}',
            '',
            'Evidence:',
            evidence,
            '',
            f'Current answer: {answer_text}',
            '',
            'Final answer:',
        ])
        try:
            refined_raw = self.llm.generate(prompt)
            refined = self.adapter.post_process_answer(qa, refined_raw, meta).strip()
        except Exception as error:
            self.logger.warning(f'Answer final refinement failed: {error}')
            return current_answer
        if not refined:
            return current_answer
        return refined.splitlines()[0].strip()

    def _load_answer_reference(self, path: str | None) -> dict[int, str]:
        if not path:
            return {}
        reference_path = Path(path).expanduser()
        if not reference_path.exists():
            self.logger.warning(f"Answer reference file not found: {reference_path}")
            return {}
        try:
            data = json.loads(reference_path.read_text(encoding="utf-8"))
            records = data.get("results", []) if isinstance(data, dict) else []
            reference = {}
            for record in records:
                idx = record.get("_global_index")
                answer = record.get("llm", {}).get("final_answer")
                if idx is not None and answer is not None:
                    reference[int(idx)] = str(answer)
            self.logger.info(f"Loaded {len(reference)} answer references from {reference_path}")
            return reference
        except Exception as error:
            self.logger.warning(f"Failed to load answer reference file {reference_path}: {error}")
            return {}

    def _build_conservative_context(self, search_res, enhanced_query: str, *, topk: int, max_chars_per_block: int) -> list[str]:
        raw_results = list(search_res.get("resources", []))[:topk]
        if not raw_results:
            return []
        raw_contents = []
        for result in raw_results:
            uri = result["uri"]
            if result.get("_content") is not None:
                raw_contents.append(str(result["_content"]))
                continue
            content = (
                self.db.read_resource(uri)
                if result.get("level", 2) == 2
                else f"{result.get('abstract', '')}\n{result.get('overview', '')}"
            )
            raw_contents.append(content)
        prepared = self.retrieval_packer.prepare_candidates(
            raw_results,
            raw_contents,
            max_chars_per_block=max_chars_per_block,
            query=enhanced_query,
        )
        selected, _ = self.retrieval_packer.select(
            prepared,
            topk=topk,
            strategy="score_only",
            query=enhanced_query,
        )
        return [candidate.prompt_text for candidate in selected]

    def _repair_temporal_answer(self, question: str, context_blocks: list[str], answer: str) -> str:
        question_text = str(question or '')
        if not re.search(r'\bwhen\b|\bwhat (?:date|year|month|time)\b', question_text, re.IGNORECASE):
            return answer
        answer_text = str(answer or '').strip()
        if not answer_text:
            return answer
        context_text = '\n\n'.join(context_blocks)
        if len(context_text) > 12000:
            context_text = context_text[:12000]
        prompt = (
            'Use only the evidence below to verify the temporal answer.\n'
            'If the current answer is exactly supported, repeat it unchanged.\n'
            'If it is relative, off by a nearby date, or unsupported, replace it with the most specific date/time span explicitly supported by the evidence.\n'
            'Return only the corrected short answer. Do not explain.\n\n'
            f'Evidence:\n{context_text}\n\nQuestion: {question_text}\nCurrent answer: {answer_text}\nCorrected answer:'
        )
        try:
            repaired = self.llm.generate(prompt).strip()
        except Exception as error:
            self.logger.warning(f'Temporal answer repair failed: {error}')
            return answer
        return repaired.splitlines()[0].strip() if repaired else answer

    def _select_supported_answer(
        self,
        question: str,
        primary_context_blocks: list[str],
        primary_answer: str,
        conservative_context_blocks: list[str],
        conservative_answer: str,
        execution_cfg: dict,
        prefer_conservative: bool = False,
    ):
        primary = str(primary_answer or "").strip()
        conservative = str(conservative_answer or "").strip()
        if not conservative or self._same_answer(primary, conservative):
            return primary_answer, {
                "selected": "primary",
                "method": "shortcut",
                "conservative_answer": conservative,
            }
        if not primary:
            return conservative_answer, {
                "selected": "conservative",
                "method": "empty_primary",
                "conservative_answer": conservative,
            }

        primary_lower = primary.lower()
        conservative_lower = conservative.lower()
        not_mentioned = ("not mentioned", "not found", "unknown", "no answer")
        primary_missing = any(marker in primary_lower for marker in not_mentioned)
        conservative_missing = any(marker in conservative_lower for marker in not_mentioned)
        if primary_missing and not conservative_missing:
            return conservative_answer, {
                "selected": "conservative",
                "method": "not_mentioned_guard",
                "conservative_answer": conservative,
            }
        if conservative_missing and not primary_missing:
            return primary_answer, {
                "selected": "primary",
                "method": "missing_baseline_guard",
                "conservative_answer": conservative,
            }

        if prefer_conservative:
            primary_words = self._answer_content_words(primary)
            conservative_words = self._answer_content_words(conservative)
            if conservative_words and conservative_words <= primary_words and len(primary_words) > len(conservative_words):
                return conservative_answer, {
                    "selected": "conservative",
                    "method": "concise_reference_guard",
                    "conservative_answer": conservative,
                }
            if self._is_list_question(question) and conservative_words and primary_words < conservative_words:
                return conservative_answer, {
                    "selected": "conservative",
                    "method": "list_coverage_guard",
                    "conservative_answer": conservative,
                }

        if self._is_temporal_question(question):
            if not self._has_temporal_signal(primary) and self._has_temporal_signal(conservative):
                return conservative_answer, {
                    "selected": "conservative",
                    "method": "temporal_specificity_guard",
                    "conservative_answer": conservative,
                }

        max_chars = int(execution_cfg.get("answer_candidate_selector_max_chars", 12000) or 12000)
        evidence = "\n\n".join(primary_context_blocks + conservative_context_blocks)
        if len(evidence) > max_chars:
            evidence = evidence[:max_chars]
        tie_rule = "If both answers are equally supported, choose B.\n" if prefer_conservative else "If both answers are equally supported, choose A.\n"
        candidate_note = (
            "B is the original top-5 baseline answer. Choose A only when A clearly adds a key answer detail that B misses, or when B is missing, generic, or contradicted.\n"
            if prefer_conservative
            else ""
        )
        prompt = (
            "Choose the answer that is better supported by the evidence and better answers the question.\n"
            + candidate_note
            + "Prefer the answer with the required specific entity, date, amount, list item, or causal detail.\n"
            + "Penalize answers that say the information is missing when another candidate is directly supported.\n"
            + "Penalize answers that omit a list item, date, entity, or amount present in another supported candidate.\n"
            + tie_rule
            + "Return only A or B.\n\n"
            + f"Question: {question}\n\nEvidence:\n{evidence}\n\n"
            + f"A: {primary}\nB: {conservative}\nChoice:"
        )
        try:
            choice = self.llm.generate(prompt).strip().upper()
        except Exception as error:
            self.logger.warning(f"Answer candidate selection failed: {error}")
            fallback = "conservative" if prefer_conservative else "primary"
            return (conservative_answer if fallback == "conservative" else primary_answer), {
                "selected": fallback,
                "method": "selector_error",
                "conservative_answer": conservative,
            }
        selected = "conservative" if choice.startswith("B") else "primary"
        return (conservative_answer if selected == "conservative" else primary_answer), {
            "selected": selected,
            "method": "llm_selector",
            "choice": choice[:16],
            "conservative_answer": conservative,
        }

    @staticmethod
    def _same_answer(left: str, right: str) -> bool:
        return re.sub(r"\W+", "", str(left or "").lower()) == re.sub(r"\W+", "", str(right or "").lower())

    @staticmethod
    def _is_temporal_question(question: str) -> bool:
        return bool(re.search(r"\bwhen\b|\bwhat (?:date|year|month|time)\b|\bhow long\b", str(question or ""), re.IGNORECASE))

    @staticmethod
    def _has_temporal_signal(answer: str) -> bool:
        return bool(re.search(r"\b(?:19|20)\d{2}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|\b\d{1,2}[/-]\d{1,2}\b|\b(?:ago|week|month|year)\b", str(answer or ""), re.IGNORECASE))

    @staticmethod
    def _answer_content_words(answer: str) -> set[str]:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did",
            "for", "from", "he", "her", "his", "in", "is", "it", "of", "on", "or",
            "she", "that", "the", "their", "they", "this", "to", "was", "were", "with",
        }
        return {
            token
            for token in re.findall(r"\w+", str(answer or "").lower())
            if len(token) > 1 and token not in stopwords
        }

    @staticmethod
    def _is_list_question(question: str) -> bool:
        return bool(re.search(r"\b(which|what|list|name|events|items|all|both|each)\b", str(question or ""), re.IGNORECASE))

    def _process_evaluation_task(self, item):
        """
        Process a single evaluation task, computing F1 and Accuracy metrics.

        For multi-annotator scenarios (like Qasper dataset), a question may have multiple gold answers.
        Evaluation logic:
        - F1: Compute for each gold answer separately and take the maximum
        - Accuracy: Pass all gold answers to LLM at once for comprehensive judgment

        This correctly handles multi-annotator scenarios while maintaining compatibility with single-answer datasets (like Locomo).
        """
        ans, golds = item['llm']['final_answer'], item['gold_answers']

        f1 = max((MetricsCalculator.calculate_f1(ans, gt) for gt in golds), default=0.0)

        dataset_name = self.config.get('dataset_name', 'Unknown_Dataset')

        eval_record = {
            "score": 0.0,
            "reasoning": "",
            "prompt_type": ""
        }

        try:
            eval_res = llm_grader(
                self.llm.llm,
                self.config['llm']['model'],
                item['question'],
                golds,
                ans,
                dataset_name=dataset_name
            )
            eval_record = eval_res

        except Exception as e:
            self.logger.error(f"Grader error: {e}")

        if MetricsCalculator.check_refusal(ans) and any(MetricsCalculator.check_refusal(gt) for gt in golds):
            f1 = 1.0
            eval_record["score"] = 4.0
            eval_record["reasoning"] = "System successfully identified Unanswerable/Refusal condition."
            eval_record["prompt_type"] = "Heuristic_Refusal_Check"

        acc = eval_record["score"]

        item["metrics"].update({"F1": f1, "Accuracy": acc})

        item["llm_evaluation"] = {
            "prompt_used": eval_record["prompt_type"],
            "reasoning": eval_record["reasoning"],
            "normalized_score": acc
        }

        detailed_info = (
            f"\n" + "="*60 +
            f"\n[Query ID]: {item['_global_index']}"
            f"\n[Question]: {item['question']}"
            f"\n[Retrieved URIs]: {item['retrieval'].get('uris', [])}"
            f"\n[LLM Answer]: {ans}"
            f"\n[Gold Answer]: {golds}"
            f"\n[Metrics]: {item['metrics']}"
            f"\n[LLM Judge Reasoning]: {eval_record['reasoning']}"
            f"\n" + "="*60
        )
        self.logger.info(detailed_info)
        return item

    def _update_report(self, data):
        """Read existing report, merge new data, and write back"""
        report = {}
        if os.path.exists(self.report_file):
            with open(self.report_file, "r", encoding="utf-8") as f:
                try:
                    report = json.load(f)
                except json.JSONDecodeError:
                    report = {}
        report.update(data)
        with open(self.report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
        self.logger.info(f"Report updated -> {self.report_file}")
