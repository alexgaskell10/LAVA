# This file is a hack to make the OpenAttack codebase compatible with
# the allennlp main codebase as they use different versions of
# transformers package
try:
    import OpenAttack
except AttributeError:
    patch = '\nfrom .tokenization_utils import PreTrainedTokenizer as PreTrainedTokenizerBase'
    import transformers
    file = transformers.__file__
    print(f'Logging error. Adding the following patch {patch} to the file {file} to fix')
    with open(file, 'a') as f:
        f.write(patch)
    import importlib
    importlib.reload(transformers)
    import OpenAttack