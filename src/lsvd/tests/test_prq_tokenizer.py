import numpy as np


def test_projection_removes_component_then_unit_normalizes():
    from lsvd.prq_tokenizer import projection_residual

    points = np.array([[2., 3.], [4., 0.]], dtype=np.float32)
    axis = np.array([5., 0.], dtype=np.float32)
    np.testing.assert_allclose(projection_residual(points, axis), [[0., 1.], [0., 0.]])


def test_soft_refinement_matches_weighted_mean():
    from lsvd.prq_tokenizer import refine_centroids

    points = np.eye(2, dtype=np.float32)
    centers, empty = refine_centroids(points, points, neighbors=2, beta=np.log(3), chunk=1)
    np.testing.assert_allclose(centers, [[.75, .25], [.25, .75]], atol=1e-7)
    assert empty == 0


def test_encoding_uses_hard_assignment_and_removes_the_selected_direction():
    from lsvd.prq_tokenizer import PrqModel

    centers = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
    model = PrqModel(np.array([0., 0., 1.]), [centers, centers, centers])
    points = np.array([[2., 1., 5.], [1., 3., 5.], [0., 0., 1.]], dtype=np.float32)
    np.testing.assert_array_equal(model.encode(points), [[0, 1, 0], [1, 0, 0], [0, 0, 0]])
