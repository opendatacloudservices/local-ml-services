from flask import Flask, request, jsonify

import os
from dotenv import load_dotenv
load_dotenv()

from services.nnlm import Nnlm
from services.nnlm_de import Nnlm_de
from services.nnlm_de_norm import Nnlm_de_norm
from services.nnlm_norm import Nnlm_norm
from services.universal_sentence_encoder import Universal_sentence_encoder
from services.universal_sentence_encoder_large import Universal_sentence_encoder_large
from services.universal_sentence_encoder_multi import Universal_sentence_encoder_multi

app = Flask(__name__)

import json
with open("./tmp/10-translate.json") as jsonFile:
  jsonObject = json.load(jsonFile)
  jsonFile.close()
texts = []
texts_de = []
for object in jsonObject:
  texts.append(object['labelEn'])
  texts_de.append(object['label'])

# use_de = Universal_sentence_encoder()
# use_de.process(texts, max_pca=512)

# use_de = Universal_sentence_encoder_large()
# use_de.process(texts, max_pca=512)

# use_de = Universal_sentence_encoder_multi()
# use_de.process(texts_de, max_pca=512)

use_de = Nnlm()
use_de.process(texts, max_pca=128)

use_de = Nnlm_norm()
use_de.process(texts, max_pca=128)

use_de = Nnlm_de()
use_de.process(texts_de, max_pca=128)

use_de = Nnlm_de_norm()
use_de.process(texts_de, max_pca=128)

@app.route('/USE/cluster', methods=['POST'])
def teams():
  """Create universal sentence encoder and cluster based on a distance matrix
    ---
    parameters:
      - name: json
        in: body
        schema:
          type: object
          properties:
            text:
              type: array
              items:
                type: string
              required: true
              description: list of texts
    produces:
      - application/json
    accepts:
      - application/json
    responses:
      400:
        description: Input missing
      200:
        description: List of cluster ids corresponding to index of send strings (-1 > no cluster)
        examples:
          application/json: [1, 1, 2, -1, 2, 1]
  """
  if 'text' not in request.json or type(request.json['text']) is not list or len(request.json['text']) == 0:
    return 'Missing input', 400
  
  return jsonify(use_de.process(request.json['text'])), 200

if __name__ == '__main__':
  app.run()