try:
    from allennlp_models.dataset_readers.transformer_binary_qa_reader import TransformerBinaryReader
    from allennlp_models.dataset_readers.rule_reasoning_reader import RuleReasoningReader
    from allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader
    from allennlp_models.dataset_readers.proof_reader import ProofReader
    from allennlp_models.models.transformer_binary_qa_model import TransformerBinaryQA
    from allennlp_models.models.esim_binary_qa_model import ESIM
    from allennlp_models.predictors.transformer_binary_qa_predictor import *
    from allennlp_models.train.custom_train import *
    from allennlp_models.models.transformer_binary_qa_retriever import TransformerBinaryQARetriever
    from allennlp_models.models.retriever import RetrievalScorer
    from allennlp_models.models.policy_gradients import PolicyGradientsAgent
except:
    from ruletaker.allennlp_models.dataset_readers.transformer_binary_qa_reader import TransformerBinaryReader
    from ruletaker.allennlp_models.dataset_readers.rule_reasoning_reader import RuleReasoningReader
    from ruletaker.allennlp_models.dataset_readers.retrieval_reasoning_reader import RetrievalReasoningReader
    from ruletaker.allennlp_models.dataset_readers.proof_reader import ProofReader
    from ruletaker.allennlp_models.models.transformer_binary_qa_model import TransformerBinaryQA
    from ruletaker.allennlp_models.models.esim_binary_qa_model import ESIM
    from ruletaker.allennlp_models.predictors.transformer_binary_qa_predictor import *
    from ruletaker.allennlp_models.train.custom_train import *
    from ruletaker.allennlp_models.models.transformer_binary_qa_retriever import TransformerBinaryQARetriever
    from ruletaker.allennlp_models.models.retriever import RetrievalScorer
    from ruletaker.allennlp_models.models.policy_gradients import PolicyGradientsAgent

#from allennlp_models.models.transformer_mc_qa_model import *
#from allennlp_models.models.transformer_maksed_lm_model import *
#from allennlp_models.models.esim_baseline import *
#from allennlp_models.models.bert_binary_class import *

#from allennlp_models.dataset_readers.bert_mc_qa import *
#from allennlp_models.dataset_readers.baseline_reader import *
#from allennlp_models.dataset_readers.transformer_mc_qa_reader import *
#from allennlp_models.dataset_readers.transformer_masked_lm_reader import *
#from allennlp_models.dataset_readers.rule_reasoning_reader import RuleReasoningReader

#from allennlp_models.predictors.bert_binary_class import *
