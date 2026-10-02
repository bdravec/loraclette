# Technical report — LoRAclette

A deeper write-up than the README: what you built, how it works, and what the
numbers say.

- **Track:** Track 2B — LoRAclette
- **Event:** Online
- **Team:** LoRAclette — Barbara Dravec, Mariam
- **Demo:** `link to video`

## 1. Summary

The problem, your approach, and the headline result in one paragraph.

<!-- Draft from the Devpost submission (Inspiration); rewrite once we have results. -->
In earlier work, we tested whether SKILL.md files (instruction files that steer AI agents) make Apertus better at software engineering, using benchmarks for merge conflict resolution, secure code generation and program repair. Skills mostly changed the form of Apertus's answers, not its ability. Only Apertus-8B on ConGra improved; everything else stayed flat or got worse.
So we asked: if prompts only reshape output, can a lightweight LoRA adapter change what Apertus can actually do?

## 2. Architecture

Components, data flow, and where each one runs. Put diagrams in `docs/` and
reference them here.

### Target architecture (mandatory)

State which of the three architectures your project is deployable in, and how
it meets that constraint:

- **a) On-premise** — on the organisation's own infrastructure, under its own administration.
- **b) Air-gapped** — with no external network connection at runtime.
- **c) Sovereign Swiss cloud** — on a cloud platform operated in Switzerland, under Swiss jurisdiction, with Swiss data residency.

List any external dependencies, and separate build time from runtime.

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-8B`
- **How it is used:** inference | fine-tuning | evaluation | red-teaming | agents / tool use
- **Where it runs:** `local weights, hosted endpoint, ...`

Prompts, adapters, quantisation, serving stack — whatever a reader needs to
rebuild your setup.

## 4. Data

What you used, where it came from, and its licence. Flag anything personal or
non-redistributable, and keep it out of the repository (see `.gitignore`).
If data comes from human subjects or contains personal information, describe
how consent was obtained.

## 5. Evaluation

How you measured success: task, metric, baseline.

| Setup    | Metric | Result |
|----------|--------|--------|
| Baseline |        |        |
| Ours     |        |        |

## 6. Limitations

Where it breaks, what you did not test, and known failure modes.

## 7. Reproducibility

What a judge needs to get your numbers back: hardware, runtime, seeds, and the
exact commit. `make run` should do the rest.

## 8. Next steps

What you would build with another month.

## License

Creative Commons Attribution 4.0 (CC-BY-4.0). All HackApertus projects are open-sourced.

## References
