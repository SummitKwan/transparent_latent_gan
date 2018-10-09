""" test script """

import importlib
import numpy as np
import src.tl_gan.feature_axis as feature_axis

importlib.reload(feature_axis)

vectors = np.random.rand(10, 4)

print(np.sum(vectors**2, axis=0))

vectors_normalized = feature_axis.normalize_feature_axis(vectors)

print(np.sum(vectors_normalized**2, axis=0))

print(feature_axis.orthogonalize_one_vector(np.array([1, 0, 0 ]), np.array([1,1,1])))

print(vectors_normalized)

vectors_orthogonal = feature_axis.orthogonalize_vectors(vectors_normalized)
vectors_disentangled = feature_axis.disentangle_feature_axis_by_idx(vectors, idx_base=[0], idx_target=[2,3])

print(np.dot(vectors_normalized[:, -2], vectors_normalized[:, -1]))
print(np.dot(vectors_orthogonal[:, -2], vectors_orthogonal[:, -1]))

feature_axis.plot_feature_cos_sim(vectors)
feature_axis.plot_feature_cos_sim(vectors_orthogonal)
feature_axis.plot_feature_cos_sim(vectors_disentangled)

