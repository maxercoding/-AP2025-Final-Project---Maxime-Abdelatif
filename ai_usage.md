# AI_USAGE.md — AI Tool Disclosure

## Tools used
- **ChatGPT (OpenAI)**
- **Claude (Anthropic)**
- **Gemini (Google)**

## How they were used
AI tools were used as support for:
- **Code organization / refactoring** (notebook → modular `main.py` + `src/` structure)
- **Debugging and code review suggestions** (pitfalls like index alignment, probability ordering, reproducibility settings)
- **Methodology sanity checks** (leakage risks, split design, baselines, uncertainty/threshold ideas)
- **Documentation and writing clarity** (README, comments/docstrings, wording improvements)

## What they were not used for
AI tools were not used to:
- Generate data, fabricate results, or invent interpretations
- Decide the research question, experimental protocol, or what to report
- Produce the first draft of the project: the author wrote the initial version and only consulted AI when needed, with specific, directed questions
- Run experiments autonomously (all runs were executed by the author)

## Verification
- All reported metrics/figures/tables come from running the final code (`python main.py`) and are saved under `results/`.
- AI suggestions were manually reviewed and only kept when validated by code execution and outputs.

## Author responsibility
The author remains fully responsible for the research design, implementation, experiments, and interpretation, and for the correctness of all reported results.

**Author:** Maxime Abdelatif  
**Student ID:** 20416384

