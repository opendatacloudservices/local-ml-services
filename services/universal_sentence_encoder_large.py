from .universal_sentence_encoder import Universal_sentence_encoder
import tensorflow_text

class Universal_sentence_encoder_large(Universal_sentence_encoder):
  module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
  prefix = "use_large-"