import numpy as np
import pytest

from human_detect.identity import IdentityStore, normalize


def test_identity_persistence_ambiguity_and_model_binding(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = IdentityStore(path, "model-one", 3)
    store.enroll("a", "Alice", [1, 0, 0])
    store.enroll("a", "Alice", [0.98, 0.02, 0])
    assert store.match([1, 0, 0], 0.65, 0.08).person_id == "a"
    store.enroll("b", "Bob", [0.99, 0.01, 0])
    assert store.match([1, 0, 0], 0.65, 0.08).status == "unknown"
    assert store.match([0, 0, 1], 0.65, 0.08).status == "unknown"
    store.close()
    with pytest.raises(RuntimeError, match="model changed"):
        IdentityStore(path, "model-two", 3)
    store = IdentityStore(path, "model-one", 3)
    assert len(store.list_people()) == 2
    store.delete("b")
    assert store.match([1, 0, 0], 0.65, 0.08).person_id == "a"
    store.close()
    assert b"JFIF" not in path.read_bytes()


@pytest.mark.parametrize("vector", [[0, 0], [np.nan, 1], [np.inf, 1]])
def test_invalid_embeddings_rejected(vector):
    with pytest.raises(ValueError):
        normalize(vector)
