import tensorflow_hub as hub
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import tensorflow as tf
import os
from sklearn.manifold import TSNE, MDS
import hdbscan
from datetime import datetime
from sklearn.decomposition import PCA

class Universal_sentence_encoder:
  module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
  prefix = "use-"

  def __init__(self, prefix = None):
    if prefix and len(prefix) > 0:
      self.prefix = prefix
    self.model = hub.load(self.module_url)

  def embed(self, txt):
    if txt is str:
      txt = [txt]
    
    embeddings = self.model(txt)
    embeddings_list = np.array(embeddings)

    return embeddings_list

  def distance_statistics(self, matrix):
    min = None
    max = None
    sum = 0
    count = 0
    for idx1, m1 in enumerate(matrix):
      for idx2, m2 in enumerate(m1):
        if idx1 != idx2 and idx1 < idx2:
          if min == None or min > m2:
            min = m2
          if max == None or max < m2:
            max = m2
          sum += m2
          count += 1
    return {
      "min": min,
      "max": max,
      "avg": sum / count
    }

  def mds_matrix(self, matrix, method='mds', perplexity=50):
    if method == 'mds':
      model = MDS(
        n_components=2,
        dissimilarity='precomputed',
        n_jobs=4,
        # quick test
        # n_init=1,
        # max_iter=100,
        # random_state=1,
      )
    else:
      model = TSNE(
        n_components=2,
        metric='precomputed',
        perplexity=perplexity,
        n_jobs=4,
        square_distances=True,
        # quick test
        # n_iter=250,
      )
    
    out = model.fit_transform(matrix)
    return np.array(out).tolist()
  
  def mds(self, embeddings, method='mds'):
    if method == 'mds':
      model = MDS(
        n_components=2,
        n_jobs=-1
      )
    else:
      model = TSNE(
        n_components=2,
        n_jobs=-1,
        square_distances=True
      )
    
    out = model.fit_transform(embeddings)
    return np.array(out).tolist()

  def cluster(self, matrix, limit):
    db = DBSCAN(eps=limit, metric='precomputed', min_samples=2).fit(matrix)
    return np.array(db.labels_).tolist()

  def cluster_2d(self, positions):
    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(positions)
    return np.array(clusterer.labels_).tolist()


  def process(self, txt, log = False):
    if log:
      log_dir=os.getenv('LOG_DIR')
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    start_time = datetime.now()

    embeddings = self.embed(txt)
    print('embeds {}'.format(datetime.now() - start_time))

    if log:
      np.savetxt(os.path.join(log_dir, self.prefix + 'metadata.tsv'), np.array(txt), fmt='%s', delimiter='\t')
      np.savetxt(os.path.join(log_dir, self.prefix + 'embeddings.tsv'), embeddings, fmt='%f', delimiter='\t')
      print('embeds saved {}'.format(datetime.now() - start_time))

    txt = None

    # reduce dimensionality through PCA
    pca = PCA(n_components=64)
    embeddings = pca.fit_transform(embeddings)
    print('pca {}'.format(datetime.now() - start_time))
    
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html
    # matrix_euclidean = pairwise_distances(embeddings, metric='euclidean')
    # matrix_cosine = pairwise_distances(embeddings, metric='cosine')
    # for some reason pairwise_distances sometimes returns asymetrical matrices, although being a bit faster than scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    distances_euclidean = pdist(embeddings, metric='euclidean')
    matrix_euclidean = squareform(distances_euclidean)
    distances_euclidean = None
    if log:
      np.savetxt(os.path.join(log_dir, self.prefix + 'matrix-euclidean-small.tsv'), matrix_euclidean, fmt='%.3f', delimiter='\t')
    print('matrix - euclidean {}'.format(datetime.now() - start_time))

    distances_cosine = pdist(embeddings, metric='cosine')
    matrix_cosine = squareform(distances_cosine)
    distances_cosine = None
    embeddings = None
    if log:
      np.savetxt(os.path.join(log_dir, self.prefix + 'matrix-cosine-small.tsv'), matrix_cosine, fmt='%.3f', delimiter='\t')
    print('matrix - cosine {}'.format(datetime.now() - start_time))

    # generate list of top 100 for each item
    top_distances = []
    for r_idx, row in enumerate(matrix_euclidean):
      top_distances.append(np.argsort(row)[:100])
    np.savetxt(os.path.join(log_dir, self.prefix + 'matrix-matrix-mini.tsv'), top_distances, fmt='%d', delimiter='\t')
    print('top 100 - euclidean {}'.format(datetime.now() - start_time))

    top_distances = []
    for r_idx, row in enumerate(matrix_cosine):
      top_distances.append(np.argsort(row)[:100])
    np.savetxt(os.path.join(log_dir, self.prefix + 'matrix-cosine-mini.tsv'), top_distances, fmt='%d', delimiter='\t')
    top_distances = None
    print('top 100 - cosine {}'.format(datetime.now() - start_time))

    # MDS results prove to be not really helpful
    # mds_outputs = self.mds_matrix(matrix_euclidean, method='mds')
    # if log:
    #   np.savetxt(os.path.join(log_dir, self.prefix + 'euclidean-mds.tsv'), mds_outputs, fmt='%.4f', delimiter='\t')
    # print('mds - euclidean {}'.format(datetime.now() - start_time))
    
    # clusters = self.cluster_2d(mds_outputs)
    # if log:
    #   np.savetxt(os.path.join(log_dir, self.prefix + 'euclidean-mds-cluster.tsv'), clusters, fmt='%d', delimiter='\t')
    # print('mds - euclidean - cluster {}'.format(datetime.now() - start_time))

    # mds_outputs = self.mds_matrix(matrix_cosine, method='mds')
    # if log:
    #   np.savetxt(os.path.join(log_dir, self.prefix + 'cosine-mds.tsv'), mds_outputs, fmt='%.4f', delimiter='\t')
    # print('mds - cosine {}'.format(datetime.now() - start_time))

    # clusters = self.cluster_2d(mds_outputs)
    # if log:
    #   np.savetxt(os.path.join(log_dir, self.prefix + 'cosine-mds-cluster.tsv'), clusters, fmt='%d', delimiter='\t')
    # print('mds - cosine - cluster {}'.format(datetime.now() - start_time))

    # TODO: Experiment with perplexity
    perpelexities = [5, 20, 50]
    for perplexity in perpelexities:

      mds_outputs = self.mds_matrix(matrix_euclidean, method='tsne', perplexity=perplexity)
      if log:
        np.savetxt(os.path.join(log_dir, self.prefix + 'euclidean-tsne-{}.tsv'.format(perplexity)), mds_outputs, fmt='%.4f', delimiter='\t')
      print('tsne - euclidean {}'.format(datetime.now() - start_time))

      clusters = self.cluster_2d(mds_outputs)
      if log:
        np.savetxt(os.path.join(log_dir, self.prefix + 'euclidean-tsne-cluster-{}.tsv'.format(perplexity)), clusters, fmt='%d', delimiter='\t')
      print('tsne - euclidean - cluster {}'.format(datetime.now() - start_time))

      mds_outputs = self.mds_matrix(matrix_cosine, method='tsne', perplexity=perplexity)
      if log:
        np.savetxt(os.path.join(log_dir, self.prefix + 'cosine-tsne-{}.tsv'.format(perplexity)), mds_outputs, fmt='%.4f', delimiter='\t')
      print('tsne - cosine {}'.format(datetime.now() - start_time))

      clusters = self.cluster_2d(mds_outputs)
      if log:
        np.savetxt(os.path.join(log_dir, self.prefix + 'cosine-tsne-cluster-{}.tsv'.format(perplexity)), clusters, fmt='%d', delimiter='\t')
      print('tsne - cosine - cluster {}'.format(datetime.now() - start_time))

      mds_outputs = None

    statistics = self.distance_statistics(matrix_euclidean)
    print('statistics {}'.format(datetime.now() - start_time))
    clusters = self.cluster(matrix_euclidean, statistics["min"] + (statistics["avg"]-statistics["min"]) * 0.75)
    if log:
      np.savetxt(os.path.join(log_dir, self.prefix + 'euclidean-cluster.tsv'), clusters, fmt='%d', delimiter='\t')
    print('cluster - euclidean {}'.format(datetime.now() - start_time))

    statistics = self.distance_statistics(matrix_cosine)
    print('statistics {}'.format(datetime.now() - start_time))
    clusters = self.cluster(matrix_cosine, statistics["min"] + (statistics["avg"]-statistics["min"]) * 0.75)
    if log:
      np.savetxt(os.path.join(log_dir, self.prefix + 'cosine-cluster.tsv'), clusters, fmt='%d', delimiter='\t')
    print('cluster - cosine {}'.format(datetime.now() - start_time))

    # return clusters
    return []