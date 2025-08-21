# Awesome AI for Physics

## 💥 News
-  **Coming Soon**: An AI4Physics survey is currently in development. Stay tuned! 📢

## 📖 Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [1. Physics Reasoning AI](#1-physics-reasoning-ai)
  - [1.1 Benchmarks (***Please add links to dataset and/or leaderboard***)](#11-benchmarks-please-add-links-to-dataset-andor-leaderboard)
    - [SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](#seephys-does-seeing-help-thinking----benchmarking-vision-based-physics-reasoning)
    - [PHYSICSEVAL: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems](#physicseval-inference-time-techniques-to-improve-the-reasoning-proficiency-of-large-language-models-on-physics-problems)
    - [ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems](#abench-physics-benchmarking-physical-reasoning-in-llms-via-high-difficulty-and-dynamic-physics-problems)
    - [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](#scienceagentbench-toward-rigorous-assessment-of-language-agents-for-data-driven-scientific-discovery)
    - [SciCode: A Research Coding Benchmark Curated by Scientists:](#scicode-a-research-coding-benchmark-curated-by-scientists)
  - [1.2 Datasets (***Please add links to dataset and/or leaderboard***)](#12-datasets-please-add-links-to-dataset-andor-leaderboard)
  - [1.3 Physics-Inspired AI (RL, SFT, agent, PINN, LLMs/MLLMs)](#13-physics-inspired-ai-rl-sft-agent-pinn-llmsmllms)
    - [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](#lnn-pinn-a-unified-physics-only-training-framework-with-liquid-residual-blocks)
    - [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](#ai-newton-a-concept-driven-physical-law-discovery-system-without-prior-physical-knowledge)
  - [1.4 AI-driven Theoretical Physics Research](#14-ai-driven-theoretical-physics-research)
    - [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](#specclip-aligning-and-translating-spectroscopic-measurements-for-stars)
  - [1.5 AI-driven Experimental Physics Research](#15-ai-driven-experimental-physics-research)
- [2. Physical Reasoning AI](#2-physical-reasoning-ai)
  - [2.1 Benchmarks](#21-benchmarks)
  - [2.2 Video Generation and Reconstruction](#22-video-generation-and-reconstruction)
    - [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](#pbr-nerf-inverse-rendering-with-physics-based-neural-fields)
    - [IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing](#intrinsicavatar-physically-based-inverse-rendering-of-dynamic-humans-from-monocular-videos-via-explicit-ray-tracing)
    - [Generative AI for Validating Physics Laws](#generative-ai-for-validating-physics-laws)
  - [2.3 Robotics](#23-robotics)
  - [2.4 Navigation](#24-navigation)
  - [2.5 Autonomous Driving](#25-autonomous-driving)
  - [2.6 Game Playing (single/multiple-player )](#26-game-playing-singlemultiple-player-)
- [3. Cross Domain Applications and Future Directions](#3-cross-domain-applications-and-future-directions)
  - [3.1 Engineering](#31-engineering)
    - [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](#achieving-precise-and-reliable-locomotion-with-differentiable-simulation-based-system-identification)
  - [3.2 Healthcare and Biophysics](#32-healthcare-and-biophysics)
    - [Calibrating Biophysical Models for Grape Phenology Prediction via Multi-Task Learning](#calibrating-biophysical-models-for-grape-phenology-prediction-via-multi-task-learning)
    - [HeartUnloadNet: A Weakly-Supervised Cycle-Consistent Graph Network for Predicting Unloaded Cardiac Geometry from Diastolic States](#heartunloadnet-a-weakly-supervised-cycle-consistent-graph-network-for-predicting-unloaded-cardiac-geometry-from-diastolic-states)
  - [3.3 Architecture](#33-architecture)
  - [3.4 Aerospace Science](#34-aerospace-science)
  - [3.5 Education](#35-education)
    - [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](#the-alphaphysics-term-rewriting-system-for-marking-algebraic-expressions-in-physics-exams)
    - [Creating a customisable freely-accessible Socratic AI physics tutor](#creating-a-customisable-freely-accessible-socratic-ai-physics-tutor)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## 1. Physics Reasoning AI
### 1.1 Benchmarks (***Please add links to dataset and/or leaderboard***)
#### [SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](https://arxiv.org/abs/2505.19099)
- **Date:** 2025.05
- **Description:** It comprises 2,000 rigorously validated questions covering a comprehensive range of knowledge levels from middle school to PhD qualifying exam levels.Through meticulous selection of 21 diagram types by domain experts, each problem challenges frontier MLLMs to integrate domain knowledge with visual understanding of physics diagrams (e.g., Feynman diagrams for particle interactions and Circuit diagrams for Electromagnetism).
- **Link:** https://seephys.github.io/
- **Domain:** `MLLM` `Physics VQA`

#### [PHYSICSEVAL: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems](https://arxiv.org/pdf/2508.00079)
- **Date:** 2025.08
- **Description:** A new 19,609-problem physics benchmark, reveals that multi-agent verification frameworks enhance LLM performance on complex physics reasoning tasks. 
- **Link:** https://github.com/areebuzair/PhysicsEval
- **Domain:** `LLM` `Physics VQA`

#### [ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems](https://arxiv.org/pdf/2507.04766)
- **Date:** 2025.07
- **Description:**  ABench-Physics exposes LLMs' physics reasoning limitations through 500 challenging problems (400 static + 100 dynamic variants). 
- **Link:** https://github.com/inclusionAI/ABench/tree/main/Physics
- **Domain:** `LLM` `Physics VQA`

#### [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/pdf/2410.05080)
- **Date:** 2024.10
- **Description:** ScienceAgentBench evaluates LLM agents on 102 validated scientific tasks, showing even top-performing models (32.4-42.2% success rates) remain far from reliable automation, with performance gains requiring prohibitive cost increases. 
- **Link:** https://osu-nlp-group.github.io/ScienceAgentBench/
- **Domain:** `Agent` `Computational Physics`

#### [SciCode: A Research Coding Benchmark Curated by Scientists](https://arxiv.org/abs/2407.13168):
- **Date:** 2024.04
- **Description:** A scientist-curated benchmark (338 subproblems from 80 research challenges) with coding components. 
- **Link:** https://scicode-bench.github.io/
- **Domain:** `LLM` `Computational Physics`

### 1.2 Datasets (***Please add links to dataset and/or leaderboard***)

### 1.3 Physics-Inspired AI (RL, SFT, agent, PINN, LLMs/MLLMs)
  #### [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](https://arxiv.org/pdf/2508.08935)
- **Date:** 2025.08
- **Description:** LNN-PINN, a physics-informed neural network framework, combines a liquid residual gating architecture while retaining the original physical modeling and optimization process to improve prediction accuracy.
- **Domain:** `PINN` `Liquid Neural Network`

#### [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](https://arxiv.org/pdf/2504.01538)
- **Date:** 2025.04
- **Description:** AI-Newton, a concept-driven discovery system, can autonomously derive physical laws from raw data—without supervision or prior physical knowledge. The system integrates a knowledge base and knowledge representation centered around physical concepts, as well as an autonomous discovery workflow.
- **Domain:** `Survey` `Symbolic-AI`

### 1.4 AI-driven Theoretical Physics Research
#### [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](https://arxiv.org/pdf/2507.01939)
- **Date:** 2025.07
- **Description:** A CLIP-inspired foundation model for stellar spectral analysis that leverages cross-instrument contrastive pre-training and spectrum-aware decoders to enable precise spectral alignment, parameter estimation, and anomaly detection across diverse astronomical applications.
- **Domain:** `Contrastive Learning` `Astrophysics`


### 1.5 AI-driven Experimental Physics Research

## 2. Physical Reasoning AI

### 2.1 Benchmarks

### 2.2 Video Generation and Reconstruction

#### [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](https://arxiv.org/pdf/2412.09680)
- **Date:** 2024.12
- **Description:** A physics-aware NeRF that jointly recovers shape, materials and lighting via two novel PBR priors, achieving state-of-the-art material accuracy without hurting view synthesis.
- **Domain:** `Generation` `Physics-Based Rendering`

#### [IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing](https://arxiv.org/pdf/2312.05210)
- **Date:** 2023.12
- **Description:** IntrinsicAvatar, a novel method for recovering the intrinsic properties of clothed human avatars, including geometry, albedo, material, and ambient lighting, from monocular video alone. Recent advances in eye-based neural rendering have enabled high-quality reconstruction of clothed human geometry and appearance from monocular video alone.
- **Domain:** `Reconstruction` `Ray Tracing`

#### [Generative AI for Validating Physics Laws](https://arxiv.org/pdf/2503.17894)
- **Date:** 2025.03
- **Description:** A generative artificial intelligence (AI) approach was proposed to empirically verify fundamental laws of physics, focusing on the Stefan-Boltzmann law linking stellar temperature and luminosity. The approach simulates the counterfactual luminosity of each star under hypothetical temperature conditions and iteratively refines the temperature-luminosity relationship within a deep learning architecture.
- **Domain:** `Generation` `Astrophysics`

### 2.3 Robotics

### 2.4 Navigation

### 2.5 Autonomous Driving

### 2.6 Game Playing (single/multiple-player )


## 3. Cross Domain Applications and Future Directions

### 3.1 Engineering
  #### [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](https://arxiv.org/html/2508.04696v1)
- **Date:** 2025.08
- **Description:** It combines system identification with RL training to optimize physical parameters from trajectory data, achieving 75% reduction in rotational drift and 46% improvement in directional movement for bipedal locomotion compared to baseline methods.
- **Domain:** `Robotics` `Differentiable Simulator`

### 3.2 Healthcare and Biophysics
  #### [Calibrating Biophysical Models for Grape Phenology Prediction via Multi-Task Learning](https://arxiv.org/pdf/2508.03898)
- **Date:** 2025.08
- **Description:** It proposes a hybrid modeling approach that combines multi-task learning with a recurrent neural network to parameterize a differentiable biophysical model. This method significantly outperforms both conventional biophysical models and baseline deep learning approaches in predicting phenological stages, as well as other crop state variables such as cold-hardiness and wheat yield.
- **Domain:** `Graph Neural Networks ` `Biophysical Model`

#### [HeartUnloadNet: A Weakly-Supervised Cycle-Consistent Graph Network for Predicting Unloaded Cardiac Geometry from Diastolic States](https://arxiv.org/pdf/2507.18677)
- **Date:** 2025.07
- **Description:** HeartUnloadNet is a deep learning framework that predicts unloaded left ventricular (LV) shape directly from end-diastolic (ED) meshes while explicitly incorporating biophysical priors. The network accepts meshes of arbitrary size and physiological parameters such as ED pressure, myocardial stiffness, and fiber helicity orientation, and outputs the corresponding unloaded mesh. It employs a graph attention architecture and a cycle consistency strategy for bidirectional (loaded and unloaded) prediction, enabling partial self-supervision, which improves accuracy and reduces the need for large training datasets.
- **Domain:** `Graph Neural Networks ` `Biophysical Model`

### 3.3 Architecture

### 3.4 Aerospace Science

### 3.5 Education
#### [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](https://arxiv.org/pdf/2507.18337)
- **Date:** 2025.08
- **Description:** This automated physics grading system integrates LLM preprocessing with dual verification pathways (general SMT solvers and physics-specific term rewriting), successfully processing 1500+ Olympiad exam responses by combining natural language understanding with formal mathematical validation of student solutions.
- **Domain:** `Agent` `Physics Grading`

#### [Creating a customisable freely-accessible Socratic AI physics tutor](https://arxiv.org/pdf/2507.05795)
- **Date:** 2025.07
- **Description:** It customizes Gemini into a physics tutor that promotes Socratic dialogue rather than direct answers, successfully guiding students through diagram analysis and conceptual reasoning while acknowledging persistent limitations in accuracy.
- **Domain:** `Agent` `Physics Tutor`

