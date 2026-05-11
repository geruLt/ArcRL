# ArcRL Framework

**Adaptive Reasoning Curriculum for Reinforcement Learning (ArcRL)**

*Accepted at IEEE Conference on Games (CoG) 2026.*

ArcRL is a modular, domain-agnostic framework designed to bridge Large Language Models (LLMs) and Reinforcement Learning architectures. It enables the creation of adaptive learning curricula where an LLM functions as a "Curriculum Architect", designing and adjusting training tasks dynamically based on agent performance.

This repository provides the core open-source framework, shared state management, and base toolsets.

## Environment-Specific Components

In order to maintain the domain-agnostic nature of the repository, environment-specific task encoders, orchestrator loops, and data models have been decoupled from the core framework. 

You will need to supply an environment-specific orchestrator (e.g., `orchestrator_llm.py`), environment-specific config (`config.json`), and environment-specific adapter (`adapter.py`) tailored to your specific application domain natively. 

### Get the Reference Implementation
The reference implementations (environment-specific data and orchestrator files used in our research) are available via the link below:

**[Placeholder: Google Drive Link to Environment-Specific Orchestrators and Data]**

1. Download the environment-specific files from the drive link.
2. Drop them into the root of this repository.
3. Configure your `.env` (see `.env.example`).
4. Run your orchestrator natively!

## Directory Structure

* `core/`: Base adapters and orchestrator configurations.
* `examples/`: Example configurations and adapter skeletons.
* `tools/`: Utility toolsets for the framework.

## License
See the `LICENSE` file for details.
