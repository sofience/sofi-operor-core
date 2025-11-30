
---
Sofi-Operor Core

A minimal, production-oriented multi-agent kernel built around a single proposition engine.

Why this repository exists

Conventional multi-agent frameworks (LangChain Agents, CrewAI, AutoGen) often rely on agent-as-entity abstractions.
This creates well-known engineering issues:

Observability black holes

Cold-start penalties

Credit assignment failures

Role-collapse loops (Critic loops, Planner deadlocks)

Overhead from unnecessary agent identity simulation

Vendor-locked communication interfaces


Sofi-Operor Core proposes a simpler model:
Agents ≠ entities.
Agents = transformation pathways operating on a root proposition.


---

1. Motivation: the gap between multi-agent hype and industrial needs

While research frameworks simulate “teams of AI workers,” production systems require something different:

deterministic kernels

predictable planning loops

clean debuggability

composability with local or cloud LLMs

minimal magic behavior

reduced overhead for agent identity simulation


Most current frameworks lack these qualities because they treat each agent like a semi-autonomous persona.

This repository targets the industrial requirement:
A small, auditable, extensible kernel for multi-step reasoning without speculative agent behavior.


---

2. Problems with existing multi-agent frameworks

(Well-known issues across LangChain Agents, CrewAI, AutoGen, and custom setups)

2.1 Observability Black Hole

Hidden intermediate steps, opaque reasoning traces, and agent-role confusion make debugging almost impossible.

2.2 Role Collapse & Goodhart Loops

E.g.,
“Critic must criticize” → forced objections → Planner re-plans → infinite loop → no final answer.

2.3 Identity Simulation Overhead

Hard-coded roles (Supervisor / Worker / Analyst / Executor…) produce unnecessary noise and latency.

2.4 Credit Assignment Hell

Which agent improved or harmed the final output?
Most frameworks cannot answer this.

2.5 Vendor Lock-in

Built-in calls to specific providers (OpenAI, Anthropic) make it difficult to adopt internal models or local LLMs.


---

3. How Sofi-Operor Core solves this

✔ Single root proposition

A multi-agent system does not need multiple agent identities.
It needs one central proposition that transforms over cycles.

✔ Agents as pure transformations, not personas

Agents hold only:

a name

a transformation rule
No beliefs, memory, or role inflation.


✔ Deterministic kernel

Kernel.run() executes a defined number of cycles with predictable I/O.

✔ Pluggable local LLM calls

Developers can inject any LLM backend via:

async def call_llm(prompt: str, temperature: float = 0.7) -> str:

This immediately solves vendor lock-in and allows:

local Qwen

Llama/Gemma running on private hardware

custom enterprise LLMs

API providers (OpenAI/Anthropic/Groq)


✔ Readable execution logs

Engineers see the entire reasoning trace.
No hidden steps.

✔ No runaway loops

No agent identity dependencies → no role-collapse.


---

4. What this code enables (industrial value)

4.1 Internal LLM orchestrators

Enterprises can use this kernel to build:

internal reasoning engines

domain-specific analyzers

compliance workflow LLMs

internal summarization and report pipelines


4.2 Multimodal transformation pipelines

Agents are simple functions → easy to map across modalities (text → JSON → code → SQL).

4.3 Research-grade interpretability

Complete visibility of:

prompts

transformations

intermediate propositions


4.4 Custom on-prem multi-agent setups

Useful for organizations that cannot use cloud LLMs.


---

5. Code Example


```bash
git clone https://github.com/sofience/sofi-operor-core.git
cd sofi-operor-core
pip install -r requirements.txt


```python
from operor import Proposition, Agent, Kernel

p = Proposition("We now think not in models, but in propositions.")
a1 = Agent("Observer", "Observe and record.")
a2 = Agent("Critic", "Provide counterpoints.")
a3 = Agent("Poet", "Rephrase everything poetically.")

kernel = Kernel()
kernel.deploy(p, [a1, a2, a3])
await kernel.run(cycles=3)


---

6. Conclusion

> Agents are not entities.
They are pathways through which a single Proposition expresses itself.
The leader of a multi-agent system is not a model, but a sentence.



This repository offers a minimal, production-ready kernel for anyone wanting to build interpretable, controllable, vendor-agnostic multi-step reasoning systems without the complexity of traditional multi-agent abstractions.


---

✔ Recommended Repository Description (short version)

A minimal, production-oriented multi-agent kernel based on a single proposition engine.
No personas. No role-collapse. Fully interpretable.
Compatible with any local or cloud LLM.


---

✔ Recommended Repository Description (enterprise version)

Vendor-agnostic, deterministic, and fully interpretable multi-agent kernel for enterprise LLM orchestration.
Agents behave as transformation functions, not personas — eliminating role-collapse, observability gaps, and identity overhead.
Ideal for research labs and production pipelines that need stable, auditable multi-step reasoning.