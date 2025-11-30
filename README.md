---

Sofi-Operor Core

A 663-line multi-agent kernel that doesn’t hallucinate roles — and stays controllable.

Sofi-Operor Core is a minimal, production-oriented multi-agent kernel designed for LLM-based systems that require high observability, predictable role behavior, and deterministic alignment logic.

Unlike many multi-agent frameworks—where agents drift, contradict each other, or recursively collapse due to role confusion—Sofi-Operor Core treats “roles” as execution pathways, not personalities.
This avoids the Critic-loop, Goodhart drift, role-hallucination, and recursive over-correction failure modes common in CrewAI, LangChain Agents, AutoGen, and custom agent stacks.


---

1. Why existing multi-agent stacks fail

Most multi-agent systems in the industry break down for predictable structural reasons:

🔹 1. Role Hallucination

Agents interpret their role instructions too literally (“critic must always criticize”), producing contradictory or low-quality outputs.

🔹 2. Goodhart Loops

Supervisor → Worker chains collapse into “optimize the instruction instead of the task.”

🔹 3. Infinite Recursion

Critic → Planner → Critic cycles repeat indefinitely because the Proposition (root directive) never updates.

🔹 4. Observability Black Holes

Developers can’t inspect why an agent made a decision. No trace logs, no Δφ(change-rate) metrics, no alignment signals.

🔹 5. Credit Assignment Hell

When a system succeeds or fails, no one knows which agent caused it.

Sofi-Operor Core is designed to eliminate these classes of failures at the architecture level.


---

2. What this kernel does differently

✔ 2.1 A single Proposition leads the system

Instead of building a cluster of “AI personalities,” the kernel routes every agent’s behavior through a single Root Proposition Node.

This removes role drift and ensures global coherence.

✔ 2.2 Agents are execution channels, not entities

“Analysis / Planner / Critic / Safety” channels do not simulate personalities.
They perform predictable transformations on the same state.

✔ 2.3 Deterministic alignment scoring

Every plan is evaluated via:

Alignment score

Risk score

Ethical check

Δφ(change-rate) vector

PhaseState snapshot

Trace-based recursion control


✔ 2.4 Full observability

Every decision is logged via a TraceLog entry with:

Context

Goal

Scored Plans

Δφ semantic/ethical/strategic vector

Phase transitions


This is the part that usually impresses engineers:
You can finally see why the system made a decision.


---

3. Industrial applications

Sofi-Operor Core is intentionally small but can scale to:

• Workflow automation agents

Predictable Planner-Critic-Safety cycles with no hallucinated behaviors.

• Enterprise copilots

Consistent multi-step reasoning with visible decision traces.

• Research assistants

Deterministic “analysis → proposal → critique → refine” loops.

• AI governance / compliance

Replace black-box agent chains with transparent scoring + trace logging.

• Product prototyping

A compact, extensible kernel for experimenting with multi-agent orchestration.


---

4. Installation

Method 1 — Standard clone

```bash
git clone https://github.com/sofience/sofi-operor-core.git
cd sofi-operor-core
pip install -r requirements.txt
```


---

5. Install & Run (auto-setup script)

Place the following in install_and_run.sh:


```bash
#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the kernel with a simple test LLM function
python3 sofi_operor_core.py
```

Give execution permission:

```bash
chmod +x install_and_run.sh
```

Then run:

```bash
./install_and_run.sh
```

---

6. Example: Running with a local LLM

If you have a local LLM endpoint:

```python 
async def call_llm(prompt: str, temperature: float = 0.7) -> str:
    import httpx
    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:32b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature}
        },
        timeout=120.0
    )
    return response.json()["response"]

Plug it into call_llm() inside sofi_operor_core.py and the system runs end-to-end.
```

---

7. Conclusion

Sofi-Operor Core is a compact, fully inspectable multi-agent kernel that solves the major structural weaknesses of existing frameworks.

It is:

Small enough to be readable

Modular enough to extend

Deterministic enough for production

Transparent enough for research

Flexible enough to integrate with any LLM API or local model


If you want agents that stay aligned, stay observable, and don’t hallucinate identities, this kernel is the right foundation.


---