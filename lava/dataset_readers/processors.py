from __future__ import absolute_import, division, print_function

import csv
import logging
import os
import sys
from io import open
from tqdm import tqdm

import json
from nltk.tokenize import sent_tokenize
import numpy as np

from .proof_utils import get_proof_graph, get_proof_graph_with_fail, RRInputExample


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, *args):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_jsonl(cls, input_file):
        """Reads a tab separated value file."""
        records = []
        with open(input_file, "r", encoding="utf-8-sig") as f:
            for line in f:
                records.append(json.loads(line))
            return records


class RRProcessor(DataProcessor):
    def get_examples(self, data_dir, dset, get_proof=True, debug_num=-1, one_proof=False):
        return self._create_examples(
            self._read_jsonl(os.path.join(data_dir, dset+".jsonl")),
            self._read_jsonl(os.path.join(data_dir, "meta-"+dset+".jsonl")),
            get_proof=get_proof,
            debug_num=debug_num,
            one_proof=one_proof,
        )

    def get_labels(self):
        return [True, False]

    # Unconstrained training, use this for ablation
    def _get_node_edge_label_unconstrained(self, proofs, sentence_scramble, nfact, nrule):
        proof = proofs.split("OR")[0]
        #print(proof)
        node_label = [0] * (nfact + nrule + 1)
        edge_label = np.zeros((nfact+nrule+1, nfact+nrule+1), dtype=int)

        if "FAIL" in proof:
            nodes, edges = get_proof_graph_with_fail(proof)
        else:
            nodes, edges = get_proof_graph(proof)
        #print(nodes)
        #print(edges)

        component_index_map = {}
        for (i, index) in enumerate(sentence_scramble):
            if index <= nfact:
                component = "triple" + str(index)
            else:
                component = "rule" + str(index-nfact)
            component_index_map[component] = i

        for node in nodes:
            if node != "NAF":
                index = component_index_map[node]
            else:
                index = nfact+nrule
            node_label[index] = 1

        edges = list(set(edges))
        for edge in edges:
            if edge[0] != "NAF":
                start_index = component_index_map[edge[0]]
            else:
                start_index = nfact+nrule
            if edge[1] != "NAF":
                end_index = component_index_map[edge[1]]
            else:
                end_index = nfact+nrule

            edge_label[start_index][end_index] = 1

        return node_label, list(edge_label.flatten())

    def _get_node_edge_label_constrained(self, proofs, sentence_scramble, nfact, nrule):
        node_labels = []
        edge_labels = []
        proof_idx = -1
        max_len = float("inf")
        for n, proof in enumerate(proofs.split("OR")):
            # proof = proofs.split("OR")[0]
            node_label = [0] * (nfact + nrule + 1)
            edge_label = np.zeros((nfact + nrule + 1, nfact + nrule + 1), dtype=int)

            if "FAIL" in proof:
                nodes, edges = get_proof_graph_with_fail(proof)
            else:
                nodes, edges = get_proof_graph(proof)

            component_index_map = {}
            for (i, index) in enumerate(sentence_scramble):
                if index <= nfact:
                    component = "triple" + str(index)
                else:
                    component = "rule" + str(index - nfact)
                component_index_map[component] = i
            component_index_map["NAF"] = nfact+nrule

            for node in nodes:
                index = component_index_map[node]
                node_label[index] = 1

            edges = list(set(edges))
            for edge in edges:
                start_index = component_index_map[edge[0]]
                end_index = component_index_map[edge[1]]
                edge_label[start_index][end_index] = 1

            # Mask impossible edges
            for i in range(len(edge_label)):
                for j in range(len(edge_label)):
                    # Ignore diagonal
                    if i == j:
                        edge_label[i][j] = -100
                        continue

                    # Ignore edges between non-nodes
                    if node_label[i] == 0 or node_label[j] == 0:
                        edge_label[i][j] = -100
                        continue

                    is_fact_start = False
                    is_fact_end = False
                    if i == len(edge_label)-1 or sentence_scramble[i] <= nfact:
                        is_fact_start = True
                    if j == len(edge_label)-1 or sentence_scramble[j] <= nfact:
                        is_fact_end = True

                    # No edge between fact/NAF -> fact/NAF
                    if is_fact_start and is_fact_end:
                        edge_label[i][j] = -100
                        continue

                    # No edge between Rule -> fact/NAF
                    if not is_fact_start and is_fact_end:
                        edge_label[i][j] = -100
                        continue

            # Return the shortest proof
            node_labels.append(node_label)
            edge_labels.append(list(edge_label.flatten()))
            if sum(node_label) < max_len:
                max_len = sum(node_label)
                proof_idx = n

        return node_labels[proof_idx], edge_labels[proof_idx]

    def _create_examples(self, records, meta_records, get_proof, debug_num, one_proof):
        examples = []
        idx = 0
        for (i, (record, meta_record)) in tqdm(enumerate(zip(records, meta_records))):
            #print(i)
            assert record["id"] == meta_record["id"]
            context = record["context"]
            sentence_scramble = record["meta"]["sentenceScramble"]
            for (j, question) in enumerate(record["questions"]):
                # Uncomment to train/evaluate at a certain depth
                #if question["meta"]["QDep"] != 5:
                #    continue
                # Uncomment to test at a specific subset of Birds-Electricity dataset
                #if not record["id"].startswith("AttPosElectricityRB4"):
                #    continue
                id = question["id"]
                label = question["label"]
                qdep = question["meta"]["QDep"]
                qlen = question["meta"]["QLen"]
                question = question["text"]
                meta_data = meta_record["questions"]["Q"+str(j+1)]

                assert (question == meta_data["question"])

                proofs = meta_data["proofs"]
                nfact = meta_record["NFact"]
                nrule = meta_record["NRule"]

                if one_proof and 'OR' in proofs:
                    continue

                if get_proof:
                    node_label, edge_label = self._get_node_edge_label_constrained(proofs, sentence_scramble, nfact, nrule)
                else:
                    node_label, edge_label = None, None

                examples.append(RRInputExample(id, context, question, node_label, edge_label, label, qdep, qlen, meta_record, sentence_scramble))

                idx += 1 

            if debug_num is not None and debug_num > 0 and idx >= debug_num:
                print('DEBUG BREAK TRIGGERED @'+str(debug_num))
                break

        return examples

