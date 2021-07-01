from .universal_sentence_encoder import Universal_sentence_encoder

class Nnlm_de_norm(Universal_sentence_encoder):
  module_url = "https://tfhub.dev/google/nnlm-de-dim128-with-normalization/2"
  prefix = "nnlm_de_norm-"