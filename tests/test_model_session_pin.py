"""The cached Mask R-CNN model is usable from any worker thread.

TensorFlow's default graph is a property of the thread and Keras keeps its
session per thread; every Detect batch runs in a fresh thread. Without the
pin, the second batch lost the graph, rebuilt the model and ran out of GPU
memory.
"""
from __future__ import annotations

import threading

import numpy as np
import tensorflow as tf


def test_a_cache_hit_pins_the_graph_and_session_in_the_calling_thread(monkeypatch):
    from functions import clasts_detection as CD
    seen = []

    class _Sess:
        graph = tf.Graph()

        def close(self):
            seen.append("closed")

    sess = _Sess()

    class _K:
        @staticmethod
        def set_session(s):
            seen.append((threading.current_thread().name, s))

        @staticmethod
        def get_session():
            return sess

    monkeypatch.setattr(CD, "_keras_backend", lambda: _K)
    saved = dict(CD._MODEL_CACHE)
    try:
        CD._MODEL_CACHE.update(key=("gpu", 0, 0.7), model=object(), loaded_at=1.0,
                               build_args=("gpu", 0, 0.7), session=sess)
        got = {}

        def worker():
            got["model"] = CD._build_model("gpu", 0, 0.7)
            got["default_graph"] = tf.compat.v1.get_default_graph()
        t = threading.Thread(target=worker, name="batch-2")
        t.start()
        t.join()
        assert got["model"] is CD._MODEL_CACHE["model"], "the cache was reused, not rebuilt"
        assert seen == [("batch-2", sess)], "the worker thread was bound to the model's session"
        assert got["default_graph"] is sess.graph, "and to the model's graph"

        # Clearing the cache closes the session before Keras clears it.
        monkeypatch.setattr(CD.tf.keras.backend, "clear_session", lambda: None)
        CD.clear_model_cache()
        assert seen[-1] == "closed" and CD._MODEL_CACHE["session"] is None
    finally:
        CD._MODEL_CACHE.clear()
        CD._MODEL_CACHE.update(saved)


def test_a_model_built_in_one_thread_predicts_in_another_once_pinned():
    """The real mechanism, on a two-layer stand-in for Mask R-CNN (the
    module import puts TensorFlow in graph mode, as the app runs it)."""
    from functions import clasts_detection as CD
    K = tf.compat.v1.keras.backend
    x = np.zeros((1, 4), np.float32)
    built = {}

    def build():
        tf.keras.backend.clear_session()
        inp = tf.keras.Input((4,), name="input_image")
        model = tf.keras.Model(inp, tf.keras.layers.Dense(2)(inp))
        model.predict(x)
        built["model"] = model
        built["session"] = K.get_session()

    t = threading.Thread(target=build, name="batch-1")
    t.start()
    t.join()

    def use(pin, res):
        try:
            if pin:
                res["pinned"] = CD._pin_cached_session()
            res["shape"] = built["model"].predict(x).shape
            res["same_session"] = K.get_session() is built["session"]
        except Exception as ex:
            res["error"] = f"{type(ex).__name__}: {ex}"

    saved = dict(CD._MODEL_CACHE)
    try:
        CD._MODEL_CACHE["session"] = built["session"]
        unpinned, pinned = {}, {}
        for pin, res in ((False, unpinned), (True, pinned)):
            t = threading.Thread(target=use, args=(pin, res), name="batch-2")
            t.start()
            t.join()
        assert "not found in the Graph" in unpinned.get("error", ""), (
            "the control run must reproduce the failure: " + repr(unpinned))
        assert pinned == {"pinned": True, "shape": (1, 2), "same_session": True}, pinned
    finally:
        CD._MODEL_CACHE.clear()
        CD._MODEL_CACHE.update(saved)
