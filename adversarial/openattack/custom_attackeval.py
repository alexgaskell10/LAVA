import sys
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
import logging
from tqdm import tqdm
from OpenAttack import AttackEval
from OpenAttack.utils import visualizer, result_visualizer, get_language, language_by_name
from OpenAttack.attack_eval.utils import worker_process, worker_init, attack_process
from OpenAttack.tags import *
from OpenAttack.text_process.tokenizer import Tokenizer, get_default_tokenizer
from OpenAttack.victim.base import Victim
from OpenAttack.attackers.base import Attacker
from OpenAttack.metric import AttackMetric, MetricSelector

import multiprocessing as mp

logger = logging.getLogger(__name__)

class CustomAttackEval(AttackEval):
    # def __init__(self, tokenizer, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.tokenizer = tokenizer

    def __measure(self, data, adversarial_sample):
        ret = {}
        for it in self.metrics:
            value = it.after_attack(data, adversarial_sample)
            if value is not None:
                ret[it.name] = value
        return ret

    def __iter_dataset(self, dataset):
        for data in dataset:
            v = data
            for it in self.metrics:
                ret = it.before_attack(v)
                if ret is not None:
                    v = ret
            yield v
    
    def __iter_metrics(self, iterable_result):
        for data, result in iterable_result:
            output, attack_time, invoke_times = result
            if output is None:
                adversarial_sample, counter = None, None
            else:
                adversarial_sample, counter = output
            ret = {
                "data": data,
                "success": adversarial_sample is not None,
                "result": adversarial_sample,
                "counter": counter,
                "metrics": {
                    "Running Time": attack_time,
                    "Query Exceeded": self.invoke_limit is not None and invoke_times > self.invoke_limit,
                    "Victim Model Queries": invoke_times,
                    ** self.__measure(data, adversarial_sample)
                }
            }
            yield ret
        
    def ieval(self, dataset : Iterable[Dict[str, Any]], num_workers : int = 0, chunk_size : Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Iterable evaluation function of `AttackEval` returns an Iterator of result.

        Args:
            dataset: An iterable dataset.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Yields:
            A dict contains the result of each input samples.

        """
        if num_workers > 0:
            ctx = mp.get_context("spawn")
            if chunk_size is None:
                chunk_size = num_workers
            with ctx.Pool(num_workers, initializer=worker_init, initargs=(self.attacker, self.victim, self.invoke_limit)) as pool:
                for ret in self.__iter_metrics(zip(dataset, pool.imap(worker_process, self.__iter_dataset(dataset), chunksize=chunk_size))):
                    yield ret
                   
        else:
            def result_iter():
                for data in self.__iter_dataset(dataset):
                    yield attack_process(self.attacker, self.victim, data, self.invoke_limit)
            for ret in self.__iter_metrics(zip(dataset, result_iter())):
                yield ret

    def eval(self, dataset: Iterable[Dict[str, Any]], total_len : Optional[int] = None, visualize : bool = False, progress_bar : bool = False, num_workers : int = 0, chunk_size : Optional[int] = None):
        """
        Evaluation function of `AttackEval`.

        Args:
            dataset: An iterable dataset.
            total_len: Total length of dataset (will be used if dataset doesn't has a `__len__` attribute).
            visualize: Display a pretty result for each data in the dataset.
            progress_bar: Display a progress bar if `True`.
            num_worers: The number of processes running the attack algorithm. Default: 0 (running on the main process).
            chunk_size: Processing pool trunks size.
        
        Returns:
            A dict of attack evaluation summaries.

        """


        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if progress_bar:
            result_iterator = tqdm(self.ieval(dataset, num_workers, chunk_size), total=total_len)
        else:
            result_iterator = self.ieval(dataset, num_workers, chunk_size)

        total_result = {}
        total_result_cnt = {}
        total_inst = 0
        success_inst = 0
        results = []

        # Begin for
        for i, res in enumerate(result_iterator):
            total_inst += 1
            success_inst += int(res["success"])
            results.append(res)

            if visualize and (TAG_Classification in self.victim.TAGS):
                x_orig = res["data"]["x"]
                if res["success"]:
                    x_adv = res["result"]
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([
                                [x_orig['question_text'], x_orig['context']], 
                                [x_orig['question_text'], x_adv]])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                        y_adv = probs[1]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([
                                [x_orig['question_text'], x_orig['context']], 
                                [x_orig['question_text'], x_adv]])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                        y_adv = int(preds[1])
                    else:
                        raise RuntimeError("Invalid victim model")
                else:
                    y_adv = None
                    x_adv = None
                    if Tag("get_prob", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            probs = self.victim.get_prob([[x_orig['question_text'], x_orig['context']]])
                        finally:
                            self.victim.clear_context()
                        y_orig = probs[0]
                    elif Tag("get_pred", "victim") in self.victim.TAGS:
                        self.victim.set_context(res["data"], None)
                        try:
                            preds = self.victim.get_pred([[x_orig['question_text'], x_orig['context']]])
                        finally:
                            self.victim.clear_context()
                        y_orig = int(preds[0])
                    else:
                        raise RuntimeError("Invalid victim model")
                info = res["metrics"]
                info["Succeed"] = res["success"]
                if progress_bar:
                    visualizer(i + 1, x_orig['context'], y_orig, x_adv, y_adv, info, tqdm_writer, self.tokenizer)
                else:
                    visualizer(i + 1, x_orig['context'], y_orig, x_adv, y_adv, info, sys.stdout.write, self.tokenizer)
            for kw, val in res["metrics"].items():
                if val is None:
                    continue

                if kw not in total_result_cnt:
                    total_result_cnt[kw] = 0
                    total_result[kw] = 0
                total_result_cnt[kw] += 1
                total_result[kw] += float(val)
        # End for

        summary = {}
        summary["Total Attacked Instances"] = total_inst
        summary["Successful Instances"] = success_inst
        summary["Attack Success Rate"] = success_inst / total_inst
        for kw in total_result_cnt.keys():
            if kw in ["Succeed"]:
                continue
            if kw in ["Query Exceeded"]:
                summary["Total " + kw] = total_result[kw]
            else:
                summary["Avg. " + kw] = total_result[kw] / total_result_cnt[kw]
        
        if visualize:
            result_visualizer(summary, sys.stdout.write)
        return summary, results
    
    ## TODO generate adversarial samples
    