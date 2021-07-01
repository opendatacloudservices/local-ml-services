from .universal_sentence_encoder import Universal_sentence_encoder

class Nnlm(Universal_sentence_encoder):
  module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
  prefix = "nnlm-"