# Awesome-AI4Physics

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

## Table of Contents
<!-- - [1. AI for Physics Research](#1-ai-for-physics-research) -->
- [1. Theoretical Physics](#11-theoretical-physics)
  <!-- - [1.1 Theoretical Physics](#11-theoretical-physics) -->
  - [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](#1-specclip-aligning-and-translating-spectroscopic-measurements-for-stars)
- [2. Experimental Physics](#12-experimental-physics)
  <!-- - [1.2 Experimental Physics](#12-experimental-physics) -->
  - [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](#1-scienceagentbench-toward-rigorous-assessment-of-language-agents-for-data-driven-scientific-discovery)
  - [SciCode: A Research Coding Benchmark Curated by Scientists:](#2-scicode-a-research-coding-benchmark-curated-by-scientists)

- [3. Application](#2-ai-for-physical-application)
<!-- - [2. AI for Physical Application](#2-ai-for-physical-application) -->
  - [2.1 Engineering](#21-engineering)
    - [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](#1-achieving-precise-and-reliable-locomotion-with-differentiable-simulation-based-system-identification)
  - [2.2 Healthcare and Biophysics](#22-healthcare-and-biophysics)

- [4. Benchmarks](#3-ai-for-physics-qa-and-education)
<!-- - [3. AI for Physics QA and Education](#3-ai-for-physics-qa-and-education) -->
  - [3.1 Physics QA](#31-physics-qa)
    - [1. SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](#1-seephys-does-seeing-help-thinking----benchmarking-vision-based-physics-reasoning)
    - [2. PHYSICSEVAL: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems](#2-physicseval-inference-time-techniques-to-improve-the-reasoning-proficiency-of-large-language-models-on-physics-problems)
    - [3. ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems](#3-abench-physics-benchmarking-physical-reasoning-in-llms-via-high-difficulty-and-dynamic-physics-problems)
  - [3.2 Education](#32-education)
    - [1. The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](#1-the-alphaphysics-term-rewriting-system-for-marking-algebraic-expressions-in-physics-exams)
    - [2. Creating a customisable freely-accessible Socratic AI physics tutor](#2-creating-a-customisable-freely-accessible-socratic-ai-physics-tutor)

- [5. From the law of Physics Towards Physical World Model](#4-from-the-law-of-physics-towards-physical-world-model)
  - [4.1 Physics-Inspired Artificial Intelligence](#41-physics-inspired-artificial-intelligence)
  - [4.2 AI-Driven Reconstruction of Physical Scenario](#42-ai-driven-reconstruction-of-physical-scenario)
  - [4.3 Towards World Modelling by Understanding the Law of Physics](#43-towards-world-modelling-by-understanding-the-law-of-physics)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


 Template: [time] title [paper link]: description `tag/domain` 

## 1. AI for Physics Research

### 1.1 Theoretical Physics

Quantum Mechanics, Gravitation, etc.
#### 1. [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](https://arxiv.org/pdf/2507.01939)
- **Date:** 2025.07
- **Description:** A CLIP-inspired foundation model for stellar spectral analysis that leverages cross-instrument contrastive pre-training and spectrum-aware decoders to enable precise spectral alignment, parameter estimation, and anomaly detection across diverse astronomical applications.

### 1.2 Experimental Physics
#### 1. [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/pdf/2410.05080)
- **Date:** 2024.10
- **Description:** ScienceAgentBench evaluates LLM agents on 102 validated scientific tasks, showing even top-performing models (32.4-42.2% success rates) remain far from reliable automation, with performance gains requiring prohibitive cost increases. 

#### 2. [SciCode: A Research Coding Benchmark Curated by Scientists](https://arxiv.org/abs/2407.13168):
- **Date:** 2024.04
- **Description:** A scientist-curated benchmark (338 subproblems from 80 research challenges) with coding components. 

## 2. AI for Physical Application

### 2.1 Engineering
  #### 1. [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](https://arxiv.org/html/2508.04696v1)
- **Date:** 2024.10
- **Description:** It combines system identification with RL training to optimize physical parameters from trajectory data, achieving 75% reduction in rotational drift and 46% improvement in directional movement for bipedal locomotion compared to baseline methods.
  
### 2.2 Healthcare and Biophysics


## 3. AI for Physics QA and Education
### 3.1 Physics QA
#### 1. [SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](https://arxiv.org/abs/2505.19099)
- **Date:** 2025.05
- **Description:** It comprises 2,000 rigorously validated questions covering a comprehensive range of knowledge levels from middle school to PhD qualifying exam levels. These questions span 7 major fields of both classical and modern physics. Through meticulous selection of 21 diagram types by domain experts, each problem challenges frontier MLLMs to integrate domain knowledge with visual understanding of physics diagrams (e.g., Feynman diagrams for particle interactions and Circuit diagrams for Electromagnetism).
  
#### 2. [PHYSICSEVAL: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems](https://arxiv.org/pdf/2508.00079)
- **Date:** 2025.08
- **Description:** A new 19,609-problem physics benchmark, reveals that multi-agent verification frameworks enhance LLM performance on complex physics reasoning tasks. 

#### 3. [ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems](https://arxiv.org/pdf/2507.04766)
- **Date:** 2025.07
- **Description:**  ABench-Physics exposes LLMs' physics reasoning limitations through 500 challenging problems (400 static + 100 dynamic variants). 


### 3.2 Education
#### 1. [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](https://arxiv.org/pdf/2507.18337)
- **Date:** 2025.08
- **Description:** This automated physics grading system integrates LLM preprocessing with dual verification pathways (general SMT solvers and physics-specific term rewriting), successfully processing 1500+ Olympiad exam responses by combining natural language understanding with formal mathematical validation of student solutions.
  
#### 2. [Creating a customisable freely-accessible Socratic AI physics tutor](https://arxiv.org/pdf/2507.05795)
- **Date:** 2025.07
- **Description:** It customizes Gemini into a physics tutor that promotes Socratic dialogue rather than direct answers, successfully guiding students through diagram analysis and conceptual reasoning while acknowledging persistent limitations in accuracy.


## 4. From the law of Physics Towards Physical World Model

### 4.1 Physics-Inspired Artificial Intelligence

Note: PINN, Equivariance Network should be a subsubsection here


### 4.2 AI-Driven Reconstruction of Physical Scenario

Note: This includes Physical Engine


### 4.3 Towards World Modelling by Understanding the Law of Physics
