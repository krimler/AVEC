# AVEC: Adaptive Verifiable Edge Control for Private LLM Interactions

## Overview

AVEC is a novel framework designed to enable **verifiable private interactions with standard Large Language Models (LLMs)** while addressing challenges like operational efficiency, inference latency, and the "bootstrap problem" for local LLM development. It achieves this by shifting privacy control to the user's device and introducing explicit verifiability in data transformations.

## Key Features

* **Adaptive Edge-Based Privacy Budgeting (AEPB):** Dynamically computes and expends a Differential Privacy ($\epsilon$) budget on the local device, adapting to query sensitivity, user preferences, and historical consumption.
* **Local-First Fulfillment:** Prioritizes handling queries on-device to minimize external data exposure and remote costs.
* **Verifiable Privacy Transformations:** Uses a central Translation Agent to apply $\epsilon$-calibrated privacy transformations, with a `proof_of_transformation` for on-device cryptographic validation by the Local Agent.
* **Accelerated Local LLM Development:** Enables incremental, privacy-preserving local LLM development by safely utilizing privacy-enhanced prompts and responses as telemetry for on-device model improvement.
* **Statistical Robustness:** Designed for rigorous empirical evaluation, with a simulation framework supporting diverse users, privacy-significant queries, and statistical analysis.

## Architecture

AVEC operates with a three-tiered architecture:

1.  **Local Agent:** Resides on the user's device. Manages user privacy settings, attempts local query fulfillment, and executes the AEPB algorithm to determine delegation and privacy budgets.
2.  **Translation Agent:** A central service that receives delegated queries and determined $\epsilon$ budgets. It applies privacy-enhancing transformations and generates a verifiable proof.
3.  **Remote Agent:** A standard, untrusted LLM inference engine that receives privacy-enhanced queries from the Local Agent.

## Getting Started (Simulation)

This repository contains a simulation of the AVEC framework to demonstrate its principles and performance.

### Prerequisites

* Python 3.x
* `pandas` (for data handling)
* `scipy` (for statistical analysis)

### Installation
pip install pandas scipy
python aepb_simulation3.py

The simulation will execute multiple trials across various scenarios (AVEC Dynamic, Always Delegate, Fixed Epsilon, Always Local) and output statistical analysis results.

### Assumptions
The simulation is developed and tested on a Linux (Ubuntu) environment.
LLM functionalities are spoofed with deterministic or randomized heuristics for controlled experimentation.
Cryptographic verification is conceptual; actual cryptographic primitives are not implemented in this simulation.

### License: MIT
