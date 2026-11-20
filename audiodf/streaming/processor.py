"""Broker-agnostic chunk handling: routes PCM16 chunks to per-call sessions. Kafka plugs in on top."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import replace

from audiodf.data.audio import pcm16_to_float
from audiodf.inference.engine import DetectionEngine, Verdict
from audiodf.inference.session import CallSession
from audiodf.monitoring import metrics


class ChunkProcessor:
    def __init__(self, engine: DetectionEngine):
        self.engine = engine
        self.cfg = engine.settings.stream
        self.sessions: OrderedDict[str, CallSession] = OrderedDict()

    def handle(self, call_id: str, pcm16: bytes, eof: bool = False) -> list[Verdict]:
        """One chunk for one call -> zero or more verdicts. Chunks of a call must arrive in order
        (Kafka guarantees this when messages are keyed by call_id)."""
        out = []
        session = self.sessions.get(call_id)
        if session is None:
            out += self._make_room()  # final verdicts of calls closed to make room (their own call_id)
            session = self.sessions[call_id] = CallSession(self.engine, call_id)
        self.sessions.move_to_end(call_id)
        metrics.ACTIVE_SESSIONS.set(len(self.sessions))

        if pcm16:
            verdict = session.push(pcm16_to_float(pcm16))
            if verdict:
                out.append(verdict)
        if eof:
            verdict = self.close(call_id, "eof")
            if verdict:
                out.append(verdict)
        return out

    def close(self, call_id: str, reason: str) -> Verdict | None:
        """End one call's session and return its final verdict tagged with why it ended (None if no audio arrived)."""
        session = self.sessions.pop(call_id, None)
        metrics.ACTIVE_SESSIONS.set(len(self.sessions))
        if session is None:
            return None
        verdict = session.finalize()
        if verdict is None:  # no audio ever arrived: nothing to publish, but count the call (record_verdict counts the rest)
            metrics.CALLS_ENDED.labels(reason).inc()
            return None
        return replace(verdict, ended=reason)

    def evict_idle(self) -> list[Verdict]:
        """Close calls with no audio for stream.session_idle_seconds. A call platform that never sends eof (dropped
        connection, crashed connector) still gets a final verdict, marked ended="timeout", instead of vanishing."""
        now = time.monotonic()
        stale = [cid for cid, s in self.sessions.items() if now - s.last_active > self.cfg.session_idle_seconds]
        return [v for v in (self.close(cid, "timeout") for cid in stale) if v]

    def _make_room(self) -> list[Verdict]:
        out = []
        while len(self.sessions) >= self.cfg.max_sessions:
            verdict = self.close(next(iter(self.sessions)), "capacity")
            if verdict:
                out.append(verdict)
        return out
