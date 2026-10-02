# LoRAclette 🧀

Teaching [Apertus](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) to resolve merge conflicts with a LoRA adapter.

A [Hack Apertus](https://hackapertus.ch/) 2026 project, Track 2B (own project).
The project lives in [`track_2b/`](track_2b/): start with its
[technical report](track_2b/technical_report.md).

## Why
In earlier work, SKILL.md instruction files mostly changed the *form* of Apertus's answers, not its
ability. The only clear gain was Apertus-8B (v1) on the ConGra merge-conflict benchmark (+7.1 pp).
LoRAclette asks whether a lightweight LoRA adapter can change what Apertus can actually do.

## Status
🚧 Work in progress (October 2026).

## Run it
From `track_2b/`:
```bash
make run
```
Set `LLM_NAME`, `LLM_BASE_URL` and `LLM_API_KEY` first. (Not implemented yet.)

## Credits
Created from the [Hack Apertus project template](https://github.com/HackApertus/project-template).
Training code is adapted from
[swiss-ai/apertus-finetuning-recipes](https://github.com/swiss-ai/apertus-finetuning-recipes)
(Apache 2.0). Benchmark: ConGra (NeurIPS 2024).

## License
All Hack Apertus projects are open source; see the
[Terms & Conditions](https://hackapertus.ch/terms-and-conditions) (6. What you build is open source).
Code: Apache 2.0 ([LICENSE](LICENSE)).
