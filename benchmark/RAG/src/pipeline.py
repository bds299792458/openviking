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
    match = re.match(r'^(viking://resources/[^/]+)', str(uri or ''))
    return match.group(1) if match else ''


def _query_mentions_source(sample_id, query):
    source_tokens = _source_hint_tokens(sample_id)
    if not source_tokens:
        return False
    query_tokens = set(re.findall(r'[a-z0-9]+', str(query or '').lower()))
    if source_tokens & query_tokens:
        return True
    compact_query = _compact_source_text(query)
    return any(len(token) >= 5 and token in compact_query for token in source_tokens)


def _root_matches_source(sample_id, root):
    source_tokens = _source_hint_tokens(sample_id)
    root_tokens = _source_hint_tokens(root)
    return bool(source_tokens and root_tokens and source_tokens & root_tokens)


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
                        target_uri=scoped_root,
                    )
                    scoped_results = list(scoped_res.get('resources', []))
                    if scoped_results:
                        raw_results = scoped_results
                        retrieval_scope = 'source_scoped'
                        latency = time.time() - t0

            if execution_cfg.get("source_page_fallback", True):
                page_results = self.source_page_index.search(
                    sample_id,
                    enhanced_query,
                    limit=int(
                        execution_cfg.get(
                            "source_page_candidate_limit",
                            max(candidate_pool_topk, retrieval_topk),
                        )
                    ),
                )
                existing_uris = {str(item.get("uri", "")) for item in raw_results}
                for page_result in page_results:
                    if page_result["uri"] not in existing_uris:
                        raw_results.append(page_result)
                        existing_uris.add(page_result["uri"])
                if page_results:
                    retrieval_scope = (
                        "global+source_page"
                        if retrieval_scope == "global"
                        else f"{retrieval_scope}+source_page"
                    )

            retrieved_texts = []
            retrieved_uris = []
            context_blocks = []
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
            selected_candidates, packing_stats = self.retrieval_packer.select(
                prepared,
                topk=retrieval_topk,
                strategy=strategy,
                query=qa.question,
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
                    context_blocks.append(
                        f"[Evidence {len(context_blocks) + 1} | {block_kind}]\n"
                        f"{candidate.prompt_text}"
                    )
                else:
                    context_blocks.append(candidate.prompt_text)
            
            recall = MetricsCalculator.check_recall(retrieved_texts, qa.evidence)
            
            full_prompt, meta = self.adapter.build_prompt(qa, context_blocks)
            
            ans_raw = self.llm.generate(full_prompt)

            ans = self.adapter.post_process_answer(qa, ans_raw, meta)

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
                "llm": {"final_answer": ans},
                "metrics": {"Recall": recall}, "token_usage": {"total_input_tokens": in_tokens, "llm_output_tokens": out_tokens}
            }
        except Exception:
            self.monitor.worker_end(success=False)
            raise

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
