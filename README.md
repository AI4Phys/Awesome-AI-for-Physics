# Awesome AI for Physics

## 💥 News
-  **Coming Soon**: An AI4Physics survey is currently in development. Stay tuned! 📢

## 📖 Content
<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [1. Physics Reasoning AI](#1-physics-reasoning-ai)
  - [1.1 Database (Dataset and benchmarks)](#11-database-dataset-and-benchmarks)
    - [SeePhys: Does Seeing Help Thinking? -- Benchmarking Vision-Based Physics Reasoning](#seephys-does-seeing-help-thinking----benchmarking-vision-based-physics-reasoning)
    - [PHYSICSEVAL: Inference-Time Techniques to Improve the Reasoning Proficiency of Large Language Models on Physics Problems](#physicseval-inference-time-techniques-to-improve-the-reasoning-proficiency-of-large-language-models-on-physics-problems)
    - [ABench-Physics: Benchmarking Physical Reasoning in LLMs via High-Difficulty and Dynamic Physics Problems](#abench-physics-benchmarking-physical-reasoning-in-llms-via-high-difficulty-and-dynamic-physics-problems)
    - [ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](#scienceagentbench-toward-rigorous-assessment-of-language-agents-for-data-driven-scientific-discovery)
    - [SciCode: A Research Coding Benchmark Curated by Scientists:](#scicode-a-research-coding-benchmark-curated-by-scientists)
  - [1.2 Training Methods (RL, SFT, etc.)](#12-training-methods-rl-sft-etc)
  - [1.3 Inference Methods (CoT, etc.)](#13-inference-methods-cot-etc)
- [2. Physical Reasoning AI](#2-physical-reasoning-ai)
  - [2.1 General Understanding](#21-general-understanding)
    - [ContPhy: Continuum Physical Concept Learning and Reasoning from Videos](#contphy-continuum-physical-concept-learning-and-reasoning-from-videos)
    - [GRASP: A novel benchmark for evaluating language GRounding And Situated Physics understanding in multimodal language models](#grasp-a-novel-benchmark-for-evaluating-language-grounding-and-situated-physics-understanding-in-multimodal-language-models)
    - [IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments](#intphys-2-benchmarking-intuitive-physics-understanding-in-complex-synthetic-environments)
    - [LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models](#llmphy-complex-physical-reasoning-using-large-language-models-and-world-models)
    - [PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding](#physbench-benchmarking-and-enhancing-vision-language-models-for-physical-world-understanding)
    - [Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties](#physion-evaluating-physical-scene-understanding-that-requires-online-inference-of-different-physical-properties)
  - [2.2 World Model (Video Generation and 3D Reconstruction)](#22-world-model-video-generation-and-3d-reconstruction)
    - [PBR-NeRF: Inverse Rendering with Physics-Based Neural Fields](#pbr-nerf-inverse-rendering-with-physics-based-neural-fields)
    - [IntrinsicAvatar: Physically Based Inverse Rendering of Dynamic Humans from Monocular Videos via Explicit Ray Tracing](#intrinsicavatar-physically-based-inverse-rendering-of-dynamic-humans-from-monocular-videos-via-explicit-ray-tracing)
    - [Generative AI for Validating Physics Laws](#generative-ai-for-validating-physics-laws)
    - [Morpheus: Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments](#morpheus-benchmarking-physical-reasoning-of-video-generative-models-with-real-physical-experiments)
    - [T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation](#t2vphysbench-a-first-principles-benchmark-for-physical-consistency-in-text-to-video-generation)
    - [VideoPhy: Evaluating Physical Commonsense for Video Generation](#videophy-evaluating-physical-commonsense-for-video-generation)
    - [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](#videophy-2-a-challenging-action-centric-physical-commonsense-evaluation-in-video-generation)
  - [2.3 Robotics](#23-robotics)
    - [PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly](#phyblock-a-progressive-benchmark-for-physical-understanding-and-planning-via-3d-block-assembly)
  - [2.4 Navigation](#24-navigation)
  - [2.5 Autonomous Driving](#25-autonomous-driving)
  - [2.6 Game Playing (single/multiple-player )](#26-game-playing-singlemultiple-player-)
  - [2.7 Physics and Physical Engine](#27-physics-and-physical-engine)
    - [I-PHYRE: Interactive Physical Reasoning](#i-phyre-interactive-physical-reasoning)
- [3. Physics-Inspired AI (PINN series)](#3-physics-inspired-ai-pinn-series)
  - [3.1 (Category of PINN)](#31-category-of-pinn)
    - [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](#lnn-pinn-a-unified-physics-only-training-framework-with-liquid-residual-blocks)
    - [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](#ai-newton-a-concept-driven-physical-law-discovery-system-without-prior-physical-knowledge)
- [4. Cross Domain Applications and Future Directions](#4-cross-domain-applications-and-future-directions)
  - [4.1 AI for Physics (Theoretical and experimental)](#41-ai-for-physics-theoretical-and-experimental)
    - [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](#specclip-aligning-and-translating-spectroscopic-measurements-for-stars)
  - [4.2 Others (Healthcare, Biophysics, Architecture, Aerospace Science, Education)](#42-others-healthcare-biophysics-architecture-aerospace-science-education)
    - [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](#achieving-precise-and-reliable-locomotion-with-differentiable-simulation-based-system-identification)
    - [Calibrating Biophysical Models for Grape Phenology Prediction via Multi-Task Learning](#calibrating-biophysical-models-for-grape-phenology-prediction-via-multi-task-learning)
    - [HeartUnloadNet: A Weakly-Supervised Cycle-Consistent Graph Network for Predicting Unloaded Cardiac Geometry from Diastolic States](#heartunloadnet-a-weakly-supervised-cycle-consistent-graph-network-for-predicting-unloaded-cardiac-geometry-from-diastolic-states)
    - [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](#the-alphaphysics-term-rewriting-system-for-marking-algebraic-expressions-in-physics-exams)
    - [Creating a customisable freely-accessible Socratic AI physics tutor](#creating-a-customisable-freely-accessible-socratic-ai-physics-tutor)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## 1. Physics Reasoning AI
### 1.1 Database (Dataset and benchmarks)
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

### 1.2 Training Methods (RL, SFT, etc.)

### 1.3 Inference Methods (CoT, etc.)

## 2. Physical Reasoning AI

### 2.1 General Understanding

#### [ContPhy: Continuum Physical Concept Learning and Reasoning from Videos](https://arxiv.org/pdf/2402.06119)
- **Date:** 2024.02
- **Description:** A novel benchmark dataset of videos is designed to evaluate the physical reasoning capabilities of AI models on continuum substances, such as fluids and soft bodies, by requiring them to infer physical properties and predict dynamics in diverse and complex scenarios.
- **Link:** https://physical-reasoning-project.github.io/
- **Domain:** `Video Understanding` `Continuum Mechanics`

#### [GRASP: A novel benchmark for evaluating language GRounding And Situated Physics understanding in multimodal language models](https://arxiv.org/pdf/2311.09048)
- **Date:** 2023.11
- **Description:** A novel benchmark is built in a Unity-based simulation to evaluate how well video-based multimodal large language models can ground language in visual scenes and understand intuitive physics principles like object permanence and solidity.
- **Link:** https://github.com/i-machine-think/grasp
- **Domain:** `Video Understanding` `MLLM`

#### [IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments](https://arxiv.org/pdf/2506.09849)
- **Date:** 2025.06
- **Description:** IntPhys 2 is a large-scale benchmark that evaluates the intuitive physics understanding of AI models by challenging them to identify physically impossible events within complex, diverse, and realistic synthetic video scenes.
- **Link:** https://github.com/facebookresearch/IntPhys2
- **Domain:** `Video Understanding` `World Model`

#### [LLMPhy: Complex Physical Reasoning Using Large Language Models and World Models](https://arxiv.org/pdf/2411.08027)
- **Date:** 2024.11
- **Description:** LLMPhy is a zero-shot framework that synergizes the program synthesis abilities of Large Language Models with the simulation power of physics engines to solve complex physical reasoning tasks by iteratively estimating system parameters and predicting object dynamics.
- **Link:** 
- **Domain:** `Video Understanding` `World Model` `LLM` 

#### [PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding](https://arxiv.org/pdf/2501.16411)
- **Date:** 2025.01
- **Description:** PhysBench, a comprehensive benchmark with 10,002 entries, evaluates physical world understanding across 75 top Vision-Language Models, revealing their struggles with physics and introducing the PhysAgent framework to address this, which improves GPT-4o's performance by 18.4%.
- **Link:** https://physbench.github.io/
- **Domain:** `Embodied AI` `VLM`

#### [Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties](https://arxiv.org/pdf/2306.15668)
- **Date:** 2023.06
- **Description:** Physion++ is a novel benchmark that evaluates visual physical prediction models on their ability to perform online inference of latent properties like mass and friction from object dynamics, revealing a huge performance gap between current models and humans.
- **Link:** https://dingmyu.github.io/physion_v2/
- **Domain:** `Online Inference`

### 2.2 World Model (Video Generation and 3D Reconstruction)

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


#### [Morpheus: Benchmarking Physical Reasoning of Video Generative Models with Real Physical Experiments](https://arxiv.org/pdf/2504.02918)
- **Date:** 2025.04
- **Description:** Morpheus is a benchmark designed to evaluate the physical reasoning of video generative models, consisting of 80 meticulously filmed real-world videos that capture phenomena governed by physical conservation laws (like the conservation of energy and momentum).
- **Link:** https://physics-from-video.github.io/morpheus-bench/
- **Domain:** `Video Generation` `World Model`

#### [T2VPhysBench: A First-Principles Benchmark for Physical Consistency in Text-to-Video Generation](https://arxiv.org/pdf/2505.00337)
- **Date:** 2025.05
- **Description:** T2VPhysBench is a first-principles benchmark that uses rigorous human evaluation to assess text-to-video models on twelve core physical laws, revealing that all models perform poorly (scoring below 0.60) and consistently fail to generate physically plausible content even when given explicit hints.
- **Link:** 
- **Domain:** `T2V` `First-Principles`

#### [VideoPhy: Evaluating Physical Commonsense for Video Generation](https://arxiv.org/pdf/2406.03520)
- **Date:** 2024.06
- **Description:** VideoPhy is a benchmark designed to assess the physical commonsense of text-to-video models using diverse material interaction prompts, revealing through human evaluation that even the best-performing model, CogVideoX-5B, generates videos that are both text-adherent and physically plausible only 39.6% of the time.
- **Link:** https://videophy2.github.io/
- **Domain:** `T2V` `World Simulation`

#### [VideoPhy-2: A Challenging Action-Centric Physical Commonsense Evaluation in Video Generation](https://arxiv.org/pdf/2503.06800)
- **Date:** 2025.03
- **Description:** VideoPhy-2 is a challenging, action-centric benchmark that uses human evaluation to assess physical commonsense in video generation, revealing that even the best models achieve only 22% joint semantic and physical accuracy on its hard subset, particularly struggling with conservation laws.
- **Link:** https://videophy2.github.io/
- **Domain:** `Video Generation`


### 2.3 Robotics

#### [PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly](https://arxiv.org/pdf/2506.08708)
- **Date:** 2025.06
- **Description:** Tests on 21 top VLMs using the 2600-task PhyBlock benchmark revealed weak high-level physical planning skills, which Chain-of-Thought (CoT) prompting failed to effectively improve.
- **Link:** https://phyblock.github.io/
- **Domain:** `Embodied AI` `VLM` `VQA`

### 2.4 Navigation

### 2.5 Autonomous Driving

### 2.6 Game Playing (single/multiple-player )

#### [I-PHYRE: Interactive Physical Reasoning](https://arxiv.org/pdf/2312.03009)
- **Date:** 2023.12
- **Description:** I-PHYRE is a benchmark that evaluates an agent's physical reasoning by requiring it to actively interact with an environment to uncover latent physical properties and solve tasks that are impossible to complete from passive observation alone. It contains 40 interactive physics games mainly consisting of gray blocks, black blocks, blue blocks, and red balls.
- **Link:** https://lishiqianhugh.github.io/IPHYRE/
- **Domain:** `Interactive Reasoning` `Embodied AI `

### 2.7 Physics and Physical Engine

  
## 3. Physics-Inspired AI (PINN series)
### 3.1 (Category of PINN)
#### [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](https://arxiv.org/pdf/2508.08935)
- **Date:** 2025.08
- **Description:** LNN-PINN, a physics-informed neural network framework, combines a liquid residual gating architecture while retaining the original physical modeling and optimization process to improve prediction accuracy.
- **Domain:** `PINN` `Liquid Neural Network`

#### [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](https://arxiv.org/pdf/2504.01538)
- **Date:** 2025.04
- **Description:** AI-Newton, a concept-driven discovery system, can autonomously derive physical laws from raw data—without supervision or prior physical knowledge. The system integrates a knowledge base and knowledge representation centered around physical concepts, as well as an autonomous discovery workflow.
- **Domain:** `Survey` `Symbolic-AI`


## 4. Cross Domain Applications and Future Directions

### 4.1 AI for Physics (Theoretical and experimental)
#### [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](https://arxiv.org/pdf/2507.01939)
- **Date:** 2025.07
- **Description:** A CLIP-inspired foundation model for stellar spectral analysis that leverages cross-instrument contrastive pre-training and spectrum-aware decoders to enable precise spectral alignment, parameter estimation, and anomaly detection across diverse astronomical applications.
- **Domain:** `Contrastive Learning` `Astrophysics`

### 4.2 Others (Healthcare, Biophysics, Architecture, Aerospace Science, Education)
  #### [Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification](https://arxiv.org/html/2508.04696v1)
- **Date:** 2025.08
- **Description:** It combines system identification with RL training to optimize physical parameters from trajectory data, achieving 75% reduction in rotational drift and 46% improvement in directional movement for bipedal locomotion compared to baseline methods.
- **Domain:** `Robotics` `Differentiable Simulator`

#### [Calibrating Biophysical Models for Grape Phenology Prediction via Multi-Task Learning](https://arxiv.org/pdf/2508.03898)
- **Date:** 2025.08
- **Description:** It proposes a hybrid modeling approach that combines multi-task learning with a recurrent neural network to parameterize a differentiable biophysical model. This method significantly outperforms both conventional biophysical models and baseline deep learning approaches in predicting phenological stages, as well as other crop state variables such as cold-hardiness and wheat yield.
- **Domain:** `Graph Neural Networks ` `Biophysical Model`

#### [HeartUnloadNet: A Weakly-Supervised Cycle-Consistent Graph Network for Predicting Unloaded Cardiac Geometry from Diastolic States](https://arxiv.org/pdf/2507.18677)
- **Date:** 2025.07
- **Description:** HeartUnloadNet is a deep learning framework that predicts unloaded left ventricular (LV) shape directly from end-diastolic (ED) meshes while explicitly incorporating biophysical priors. The network accepts meshes of arbitrary size and physiological parameters such as ED pressure, myocardial stiffness, and fiber helicity orientation, and outputs the corresponding unloaded mesh. It employs a graph attention architecture and a cycle consistency strategy for bidirectional (loaded and unloaded) prediction, enabling partial self-supervision, which improves accuracy and reduces the need for large training datasets.
- **Domain:** `Graph Neural Networks ` `Biophysical Model`

#### [The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams](https://arxiv.org/pdf/2507.18337)
- **Date:** 2025.08
- **Description:** This automated physics grading system integrates LLM preprocessing with dual verification pathways (general SMT solvers and physics-specific term rewriting), successfully processing 1500+ Olympiad exam responses by combining natural language understanding with formal mathematical validation of student solutions.
- **Domain:** `Agent` `Physics Grading`

#### [Creating a customisable freely-accessible Socratic AI physics tutor](https://arxiv.org/pdf/2507.05795)
- **Date:** 2025.07
- **Description:** It customizes Gemini into a physics tutor that promotes Socratic dialogue rather than direct answers, successfully guiding students through diagram analysis and conceptual reasoning while acknowledging persistent limitations in accuracy.
- **Domain:** `Agent` `Physics Tutor`

