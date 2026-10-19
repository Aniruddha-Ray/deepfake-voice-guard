# Deepfake voice guard (`audiodf`)

Real-time deepfake voice detection for banking call security. *Imagine Cup 2026, individual entry by Aniruddha Ray.*

It listens to a call and, from 2 seconds of speech on, returns a score and one action: **allow**, **verify** (ask for an extra check) or **escalate** (send to a human fraud analyst; never an automatic reject). It can explain a verdict (SHAP) and add a transcript-based scam-intent check. Runs as a CLI, a FastAPI service, a WebSocket stream or a Kafka worker, with Prometheus and Grafana.

- What the model is, its results and its limits: [MODEL_CARD.md](MODEL_CARD.md)
- How to run a demo: [DEMO_GUIDE.md](DEMO_GUIDE.md)
- Development log (phases, decisions, mistakes): [docs/AUDIT.md](docs/AUDIT.md)

## The model
**WavLM-Base+ with its top 4 layers fine-tuned**, attentive-statistics pooling and a small head (run 7, seed 0). Scores 2 s windows with a 1 s hop; the call's score is the mean over the windows in the last 10 s. Trained on ASVspoof 5 and ASVspoof 2019 LA with simulated phone impairments (background noise, room echo, bursty packet loss, G.711). Weights are not in git (`artifacts.dvc` is the pointer; see "Model weights").

Risk levels (stored in `artifacts/bundle.json`): verify at score >= 0.0060 (about 10% of genuine ASVspoof 2021 phone callers flagged), escalate at >= 0.90. The score is a ranking value, not a probability.

```
audio chunks (0.5 s, PCM16)
   |  Kafka topic keyed by call_id      or      FastAPI  POST /predict | WS /stream/{call_id}
   v
CallSession (per call): skip silence before speech -> rolling 10 s buffer -> 2 s windows, 1 s hop
   v
WavLM-Base+ per window -> mean of the windows in the last 10 s -> score
   v
risk engine: allow < 0.0060 <= verify < 0.90 <= escalate       (+ Prometheus metrics)
   v
Verdict JSON  ->  optional: SHAP explanation (/explain), scam intent from the transcript (intent)
```

Results (EER and fakes missed at the verify level, per test set) are in [MODEL_CARD.md](MODEL_CARD.md), with the raw numbers in `results/`.

## Around the detector
- **Explanation agent** (`audiodf/explain/`): KernelSHAP over 2 s x 4-band regions, a faithfulness check, plain-language reasons from a template or an optional cloud LLM (Groq, Gemini or OpenRouter; the numbers in the text are checked against the facts). It shows where the score came from, not proof that a voice is fake.
- **Scam intent** (`audiodf/intent/`): rules over a local Whisper transcript, optionally a bounded LLM. It can only raise the check, never lower it, and never decides alone.
- **Serving** (`audiodf/serving/`, `streaming/`, `deploy/`): FastAPI, Kafka worker, Docker (CPU and GPU images), Prometheus, Grafana, Kubernetes manifests.

## Install and use
```bash
pip install -r requirements-dev.txt          # install the torch build you want first (CPU or CUDA)
# put the final model in artifacts/ (wavlm.pt + bundle.json), or point AUDIODF_ARTIFACTS at it

python -m audiodf predict some.flac                       # one file -> verdict JSON
python -m audiodf explain some.flac --plot out.png        # verdict + SHAP regions + reasons (--provider groq|gemini|openrouter, key in AUDIODF_LLM_API_KEY)
python -m audiodf intent --audio some.flac                # local speech-to-text + scam intent + voice verdict -> one action
python -m audiodf intent --text "caller: ... receiver: ..."
python -m audiodf serve --port 8000                       # API: /predict, /explain, /stream/{id}, /health, /metrics
python -m audiodf consume                                 # Kafka worker (needs a broker)
python -m audiodf produce call.wav --realtime             # simulate a call into Kafka
python -m audiodf benchmark                               # per-stage latency
python -m pytest                                          # 200+ tests, no dataset or GPU needed
```
Stream over a WebSocket: send binary frames of 16 kHz mono PCM16; a JSON verdict arrives each time a 2 s window completes (from 2 s of speech on, then every 1 s); send the text frame `end` for the final verdict. The thresholds in `artifacts/bundle.json` override the config at serving time.

Docker (from the repository root):
```bash
docker compose -f deploy/docker-compose.yml up -d                                   # Kafka, API, worker, Prometheus, Grafana
# GPU: add  -f deploy/docker-compose.gpu.yml  and  --build
python deploy/smoke_test.py some.flac other.flac --kafka                            # 12 checks end to end
```

## Retrain and re-evaluate
The datasets are not in the repository and must be downloaded: ASVspoof 5, ASVspoof 2019 LA, ASVspoof 2021 LA eval, In-the-Wild, and for the noise recipe MUSAN, RIRS_NOISES, DEMAND, ESC-50 and the MIT Impulse Response Survey. Paths: `configs/default.yaml` or `AUDIODF_ASV5`, `AUDIODF_DATA`, `AUDIODF_NOISE` and so on.
```bash
python run.py --config configs/run7.yaml --eval-utts 30000     # the final recipe: audit -> index -> train -> tune -> test
python -m audiodf audit --dataset asv5                         # integrity audit (training refuses to start on hard errors)
python -m audiodf evaluate --dataset asv5 --split eval --save-scores
python -m audiodf prepare21 && python -m audiodf evaluate --dataset asv21 --eval-utts 30000 --save-scores   # real phone channels
python -m audiodf render-calls                                 # simulated VoIP calls from clean ASV5 eval clips
python -m audiodf calibrate --scores S.csv --protocol P.tsv --tag T --repeat 20   # risk thresholds on speaker halves
python -m audiodf live-check --scores S.csv --bundle DIR --tag T                  # streaming path vs batched scores
```
Older SVM and RCNN branches and the Whisper branch are still in the code (`--branches`) but are not part of the served model: WavLM alone was as good or better.

## Training and evaluation discipline
- **Pool:** ASV5 train + dev and ASVspoof 2019 train + dev. Three ASV5 dev attacks (A10, A12, A15) and all clips of 25% of ASV5 dev's genuine speakers form the **tuning set**; everything tuned (epoch, thresholds) is chosen on attacks and voices the model never trained on. ASV5 eval and ASVspoof 2019 eval are scored once at the end and never used for tuning. Test splits are refused as training data.
- **Preprocessing** (identical for both classes, mirrored in serving): digital silence filled with +-1 LSB noise, VAD trim of both ends (clip-edge silence was a shortcut: genuine ASV5 clips carry 4-10x more of it), then label-blind impairments in training only.
- **Decision measure:** fakes missed at the verify level, each system's verify level set to flag 10% of genuine ASVspoof 2021 callers. EER is reported but never decides.
- **Thresholds** were set on all 67 ASVspoof 2021 speakers; the escalate level came from a call-audio threshold study.

## Layout
```
audiodf/
  config.py                settings (defaults + YAML + env overrides)
  data/                    protocol readers, audio, VAD, augmentation, impairments, call simulation, integrity audit
  models/                  wavlm.py (served); svm.py, rcnn.py, whisper.py (not served)
  training/ evaluation/    training loops; metrics, calibration, call thresholds, live-check, serving cost
  inference/               engine.py (scoring), session.py (per-call streaming state)
  explain/ intent/ gating/ SHAP explanation agent; scam intent; channel-quality gating (measured, not adopted)
  serving/ streaming/ monitoring/   FastAPI; Kafka glue; Prometheus metrics
  risk.py  artifacts.py    risk actions; versioned model bundle
configs/  deploy/ (Dockerfiles, compose, Prometheus, Grafana, k8s, smoke test)  tests/  results/  docs/
```

## Model weights
`artifacts/` is not in git. `artifacts.dvc` is the DVC pointer; a DVC remote has not been set up yet, so for now the weights live on the author's machine only. Treat a missing `artifacts/` as "model not available" and retrain with the recipe above.

## Known limitations
1. **No validation on live bank calls.** The thresholds come from ASVspoof 2021 phone channels and simulated calls; collect labelled production audio before relying on them. Packet loss, voice-activity gaps and echo in production will differ.
2. **Real phone channels are the weakest set** (ASVspoof 2021): 83 of 1000 fakes stay below the verify level. Unseen attacks and codecs can do worse.
3. **Escalate is not a verdict.** At 1 deepfake per 1000 calls, about 96% of calls at the highest level would be genuine customers.
4. **Intent rules** flag 37.8% of normal calls at the highest intent level on an independent dialogue set (0.6% on the set they were built on); the cloud-LLM path did not improve this and is not adopted.
5. **Serving:** GPU about 63 ms per 10 s call in the container, about 235 calls per GPU from the serving-cost study (not a load test of the container); CPU about 5 calls. The Kubernetes manifests are schema-valid but have never run on a cluster.
6. Mostly English read speech in training; other languages untested.
