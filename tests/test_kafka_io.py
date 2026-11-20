from types import SimpleNamespace

import pytest

from audiodf.config import Settings
from audiodf.streaming.kafka_io import consumer_config, ensure_topics


class _Future:
    def __init__(self, error=None):
        self.error = error

    def result(self):
        if self.error:
            raise self.error


class FakeAdmin:
    def __init__(self, have, race=False):
        self.have, self.created, self.race = set(have), [], race

    def list_topics(self, timeout=None):
        return SimpleNamespace(topics={t: object() for t in self.have})

    def create_topics(self, new):
        self.created += [(t.name, t.partitions) for t in new]
        err = RuntimeError("KafkaError{code=TOPIC_ALREADY_EXISTS}") if self.race else None
        return {t.name: _Future(err) for t in new}


def _new(name, num_partitions, replication_factor):
    return SimpleNamespace(name=name, partitions=num_partitions, replication=replication_factor)


def test_topics_are_created_up_front_with_the_configured_partitions():
    s = Settings()
    s.kafka.partitions = 3
    admin = FakeAdmin(have=["__consumer_offsets", "unrelated"])
    created = ensure_topics(s, admin, _new)
    assert set(created) == {s.kafka.audio_topic, s.kafka.result_topic}
    assert dict(admin.created) == {s.kafka.audio_topic: 3, s.kafka.result_topic: 1}


def test_existing_topics_are_left_alone_and_a_creation_race_is_not_an_error():
    s = Settings()
    assert ensure_topics(s, FakeAdmin(have=[s.kafka.audio_topic, s.kafka.result_topic]), _new) == []
    assert ensure_topics(s, FakeAdmin(have=[s.kafka.result_topic], race=True), _new) == [s.kafka.audio_topic]  # created by someone else first
    with pytest.raises(RuntimeError):  # any other failure is raised
        admin = FakeAdmin(have=[])
        admin.create_topics = lambda new: {t.name: _Future(RuntimeError("boom")) for t in new}
        ensure_topics(s, admin, _new)


def test_a_new_group_reads_from_the_start_so_calls_sent_while_it_was_joining_are_not_lost():
    cfg = consumer_config(Settings())
    assert cfg["auto.offset.reset"] == "earliest" and cfg["enable.auto.commit"] is True and cfg["group.id"] == "deepfake-detector"


def test_the_worker_publishes_a_final_verdict_for_every_call_however_it_ends(engine, monkeypatch):
    """Capacity: a new call pushes out an open one; eof: a normal end; timeout: no eof ever came. Each final verdict is keyed
    by its own call (the capacity one arrives while handling another call's chunk)."""
    import json
    from types import SimpleNamespace

    from audiodf.data.audio import float_to_pcm16
    from audiodf.streaming import kafka_io

    from conftest import tone

    chunk = float_to_pcm16(tone(2.5, noise=0.1))
    script = [("a", chunk, b"0"), ("b", chunk, b"0"), ("b", b"", b"1"), ("c", chunk, b"0")]
    published, clock = [], {"t": 0.0}

    class Msg:
        def __init__(self, key, value, eof):
            self._k, self._v, self._h = key, value, [("eof", eof)]
        def error(self): return None
        def key(self): return self._k.encode()
        def value(self): return self._v
        def headers(self): return self._h

    class Consumer:
        def __init__(self, conf): self.left = list(script)
        def subscribe(self, topics): pass
        def close(self): pass
        def poll(self, timeout):
            if self.left:
                return Msg(*self.left.pop(0))
            if clock["t"] == 0.0:
                clock["t"] = 100.0  # time passes with no audio: the next sweep closes call c
                return None
            raise KeyboardInterrupt

    class Producer:
        def __init__(self, conf): pass
        def produce(self, topic, key, value): published.append((key, json.loads(value)))
        def poll(self, timeout): pass
        def flush(self): pass

    settings = engine.settings
    monkeypatch.setattr(settings.stream, "max_sessions", 1)
    monkeypatch.setattr(settings.stream, "session_idle_seconds", -1)
    monkeypatch.setattr(kafka_io, "_kafka", lambda: SimpleNamespace(Consumer=Consumer, Producer=Producer))
    monkeypatch.setattr(kafka_io, "ensure_topics", lambda s: [])
    monkeypatch.setattr(kafka_io, "time", SimpleNamespace(monotonic=lambda: clock["t"]))
    kafka_io.run_consumer(settings, engine, metrics_port=None, log=lambda *a: None)

    finals = {(k, v["ended"]) for k, v in published if v["final"]}
    assert finals == {("a", "capacity"), ("b", "eof"), ("c", "timeout")}
    assert all(k == v["call_id"] for k, v in published)
