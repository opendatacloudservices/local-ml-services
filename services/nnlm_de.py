from .universal_sentence_encoder import Universal_sentence_encoder

class Nnlm_de(Universal_sentence_encoder):
  module_url = "https://tfhub.dev/google/nnlm-de-dim128/2"
  prefix = "nnlm_de-"