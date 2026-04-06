"""Minimal import and pipeline smoke tests."""

import numpy as np


def test_import_package():
    import quagua

    assert quagua.__version__
    assert hasattr(quagua, "encoders")
    assert hasattr(quagua, "evaluation")


def test_encode_then_mix_pipeline_shape():
    from quagua.encoders import (
        RandomOneToManyEncoder,
        ClassicalXORMixer,
        EncodeThenMixPipeline,
    )

    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, size=(30, 4)).astype(float)
    enc = RandomOneToManyEncoder(n_encodings_per_value=2, encoding_dim=3)
    mix = ClassicalXORMixer()
    pipe = EncodeThenMixPipeline(enc, mix)
    out = pipe.fit_transform(X)
    assert out.shape[0] == X.shape[0]
    assert out.shape[1] == X.shape[1] * 3


def test_privacy_framework_instantiable():
    from quagua.evaluation import PrivacyAITestFramework

    fw = PrivacyAITestFramework()
    assert fw.results == {}
