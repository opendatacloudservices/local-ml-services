from .universal_sentence_encoder import Universal_sentence_encoder
import tensorflow_text

class Universal_sentence_encoder_multi(Universal_sentence_encoder):
  # requires tons of memory !!!!
  # "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
  module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
  prefix = "use_multi-"