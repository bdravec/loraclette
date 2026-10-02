# LoRAclette

Teaching Apertus to resolve merge conflicts with a LoRA adapter.
Hack Apertus hackathon, Track 2B (own project). **Deadline: Fri 2026-10-16, 12:00 CEST.**
Two people work on this, each with their own Claude Code session. This file is the shared context.

## The question
In the thesis, a SKILL.md (instructions in the prompt) mostly changed the *form* of Apertus's
answers, not its ability. The one clear gain was Apertus-8B **v1** on ConGra: +7.1 pp solved-rate
(21.5% baseline). Does a LoRA adapter, which changes the weights, do better?

## Model: Apertus v1.5 8B (`swiss-ai/Apertus-v1.5-8B`)
- The hackathon template names v1.5. It is a NEW architecture: multimodal
  (`Apertus1p5ForConditionalGeneration`, with vision + audio tokenizers), gated on Hugging Face.
- The swiss-ai LoRA recipe was written for v1. Its compatibility with v1.5 is **unverified**.
- Thesis numbers are v1. **Never present them as v1.5 results**; v1.5 needs its own baselines.
- Fallback if v1.5 can't be trained in time: v1 (`swiss-ai/Apertus-8B-Instruct-2509`), if the
  organisers accept it.

## The comparison (all v1.5, on the same 3,597 ConGra python-tiny cases)
| Condition | Solved-rate |
|---|---|
| v1.5, no skill | ? |
| v1.5 + skill v2.1 | ? |
| v1.5 + LoRA | ? |
| v1.5 + LoRA + skill | ? (optional) |

Solved = max(edit similarity, winnowing) > 0.8, as in the thesis.

## Hard rules
- **Secrets live only in `track_2b/.env`** (gitignored; template in `.env.example`). Never print,
  echo, log or commit the API key, and never paste it into chat, issues or the report.
- **Template structure is fixed.** `track_2b/` is the project root. Don't rename or move its files
  (`README.md`, `technical_report.md`, `Makefile`, `src/`, `data/`, `docs/`). Don't edit
  `track_2b/README.md`; that's the organisers' challenge text. Our code goes in `track_2b/src/`.
- **Never train on eval cases.** The 3,597 python-tiny cases are held out. `build_dataset.py` must
  assert zero overlap. Prefer excluding whole repos, not just cases.
- **One prompt.** Training data and eval both use `src/loraclette/prompt.py`. Never write a prompt
  anywhere else.
- **Configs, not code edits.** A new experiment is a new file in `src/configs/train/`.
- **Log every run** in `docs/experiments.md` (commit, config, data, result), including failures.
- `track_2b/data/` is committed and has a **100 MB limit**: the demo sample only. Full data goes in
  `data/raw/` and `data/processed/`, which are gitignored. Adapters go to the Hugging Face Hub.
- Use **+7.1 pp** for the v1 skill gain, never +7.4. Write "pp", never "ppt".
- `src/loraclette/train.py` is Apache-2.0 code from swiss-ai; record any change in its header.

## Deliverables (submit at hackapertus.ch/online-hack/submissions, NOT Devpost)
1. This public repo. Judges run `make run` from `track_2b/`, **in Docker**, end-to-end on a clean
   checkout. The model comes in through `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`.
2. `track_2b/technical_report.md` + a PDF of it, `LoRAclette_Report.pdf`, max 6 pages.
3. A demo video, max 2 min.
4. Optional: a Hugging Face dataset from their template (test cases, model responses, metadata
   with per-case licensing).

Judging (0–5 each): purposeful use of AI, technical rigour, value/cost/scalability, sovereign
deployability, implementation feasibility. Target architecture must be one of on-premise,
air-gapped, or sovereign Swiss cloud; say which in the report.

## Layout (inside `track_2b/`)
- `src/loraclette/train.py`: training (TRL `SFTTrainer` + `peft`)
- `src/loraclette/prompt.py`: the one prompt
- `src/loraclette/data/build_dataset.py`: ConGra → `data/processed/{train,test}.jsonl`
- `src/loraclette/eval/`: port of the thesis eval
- `src/configs/train/`: TRL configs (based on the swiss-ai recipe)
- `src/scripts/`: entry points (smoke test, serving)
- `docs/experiments.md`: the run log

## Commands (from `track_2b/`)
```bash
uv sync
uv run python src/loraclette/train.py --config src/configs/train/lora_r8_baseline.yaml
```
**CSCS inference API** (`https://api.inference.cscs.ch/v1`, OpenAI-compatible, key in `.env`)
serves both `swiss-ai/Apertus-8B-Instruct-2509` (v1) and `swiss-ai/Apertus-v1.5-8B`, so baselines
run there with no local GPU. It **cannot serve our adapter** (no private models), so LoRA evals
run locally.

vLLM runs in a **separate** environment (it pins its own torch). Start it yourself in your own
shell, not from a Claude background job.

## Workflow
- Board: github.com/users/bdravec/projects/5. **Assign yourself and move the card to In Progress
  before starting.**
- A GitHub issue first for every task; one branch + PR per issue with `closes #N`; the other person
  reviews.
- No `Co-Authored-By` trailers in commits.

## Sources
- Thesis eval + prompt: `merge-conflict-skill/scripts/pilot.py` (github.com/bdravec/merge-conflict-skill)
- Training recipe: github.com/swiss-ai/apertus-finetuning-recipes
- Template: github.com/HackApertus/project-template

Machine-specific paths and hosts go in `CLAUDE.local.md` (gitignored), not here.
