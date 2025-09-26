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
    - [PhySense: Principle-Based Physics Reasoning Benchmarking for Large Language Models](#physense-principle-based-physics-reasoning-benchmarking-for-large-language-models)
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
    - [I-PHYRE: Interactive Physical Reasoning](#i-phyre-interactive-physical-reasoning)
  - [2.7 Physics and Physical Engine](#27-physics-and-physical-engine)
- [C. Physics Reasoning AI](#c-physics-reasoning-ai)
  - [C.2 Symbolic Reasoning Frameworks](#c2-symbolic-reasoning-frameworks)
      - [Physics-AI symbiosis](#physics-ai-symbiosis)
      - [AI meets physics: a comprehensive survey](#ai-meets-physics-a-comprehensive-survey)
    - [C.2.1 Equation-based Reasoning Systems (equation discovery, symbolic regression)](#c21-equation-based-reasoning-systems-equation-discovery-symbolic-regression)
      - [Toward an AI Physicist for Unsupervised Learning](#toward-an-ai-physicist-for-unsupervised-learning)
      - [Intelligence, physics and information – the tradeoff between accuracy and simplicity in machine learning](#intelligence-physics-and-information--the-tradeoff-between-accuracy-and-simplicity-in-machine-learning)
      - [Discovering physical concepts with neural networks](#discovering-physical-concepts-with-neural-networks)
      - [AI Feynman: A physics-inspired method for symbolic regression](#ai-feynman-a-physics-inspired-method-for-symbolic-regression)
      - [AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity](#ai-feynman-20-pareto-optimal-symbolic-regression-exploiting-graph-modularity)
      - [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](#ai-newton-a-concept-driven-physical-law-discovery-system-without-prior-physical-knowledge)
      - [Bayesian symbolic regression: Automated equation discovery from a physicists' perspective](#bayesian-symbolic-regression-automated-equation-discovery-from-a-physicists-perspective)
    - [C.2.2 Neuro-Symbolic Integration, Differentiable Physics Engines](#c22-neuro-symbolic-integration-differentiable-physics-engines)
      - [KAN: Kolmogorov-Arnold Networks](#kan-kolmogorov-arnold-networks)
      - [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](#kan-20-kolmogorov-arnold-networks-meet-science)
  - [C.3 Physics-Informed Neural Networks](#c3-physics-informed-neural-networks)
    - [C.3.1 Macro Perspectives on Physics-Informed Machine Learning (PIML)](#c31-macro-perspectives-on-physics-informed-machine-learning-piml)
      - [Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data](#theory-guided-data-science-a-new-paradigm-for-scientific-discovery-from-data)
      - [Physics-informed machine learning](#physics-informed-machine-learning)
      - [Hybrid physics-based and data-driven models for smart manufacturing: Modelling, simulation, and explainability](#hybrid-physics-based-and-data-driven-models-for-smart-manufacturing-modelling-simulation-and-explainability)
      - [Physics-Informed Machine Learning: A Survey on Problems, Methods and Applications](#physics-informed-machine-learning-a-survey-on-problems-methods-and-applications)
      - [Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing](#physics-informed-neural-networks-a-review-of-methodological-evolution-theoretical-foundations-and-interdisciplinary-frontiers-toward-next-generation-scientific-computing)
      - [Training of physical neural networks](#training-of-physical-neural-networks)
    - [C.3.2 PINNs for Scalability: Addressing Large-Scale and Complex Problems](#c32-pinns-for-scalability-addressing-large-scale-and-complex-problems)
      - [General Frameworks & Foundations:](#general-frameworks--foundations)
        - [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](#physics-informed-deep-learning-part-i-data-driven-solutions-of-nonlinear-partial-differential-equations)
        - [Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](#physics-informed-deep-learning-part-ii-data-driven-discovery-of-nonlinear-partial-differential-equations)
        - [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](#physics-informed-neural-networks-a-deep-learning-framework-for-solving-forward-and-inverse-problems-involving-nonlinear-partial-differential-equations)
      - [Domain Decomposition & Parallelism:](#domain-decomposition--parallelism)
        - [PPINN: Parareal physics-informed neural network for time-dependent PDEs](#ppinn-parareal-physics-informed-neural-network-for-time-dependent-pdes)
        - [Finite Basis Physics-Informed Neural Networks (FBPINNs): A scalable domain decomposition approach for solving differential equations](#finite-basis-physics-informed-neural-networks-fbpinns-a-scalable-domain-decomposition-approach-for-solving-differential-equations)
        - [When Do Extended Physics-Informed Neural Networks (XPINNs) Improve Generalization?](#when-do-extended-physics-informed-neural-networks-xpinns-improve-generalization)
        - [Parallel Physics-Informed Neural Networks via Domain Decomposition](#parallel-physics-informed-neural-networks-via-domain-decomposition)
        - [Improved Deep Neural Networks with Domain Decomposition in Solving Partial Differential Equations](#improved-deep-neural-networks-with-domain-decomposition-in-solving-partial-differential-equations)
      - [Complex Geometries & Architectures:](#complex-geometries--architectures)
        - [Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries](#physics-informed-pointnet-a-deep-learning-solver-for-steady-state-incompressible-flows-and-thermal-fields-on-multiple-sets-of-irregular-geometries)
        - [INN: Interfaced neural networks as an accessible meshless approach for solving interface PDE problems](#inn-interfaced-neural-networks-as-an-accessible-meshless-approach-for-solving-interface-pde-problems)
        - [Deep neural network methods for solving forward and inverse problems of time fractional diffusion equations with conformable derivative](#deep-neural-network-methods-for-solving-forward-and-inverse-problems-of-time-fractional-diffusion-equations-with-conformable-derivative)
        - [Neural homogenization and the physics-informed neural network for the multiscale problems](#neural-homogenization-and-the-physics-informed-neural-network-for-the-multiscale-problems)
        - [A High-Efficient Hybrid Physics-Informed Neural Networks Based on Convolutional Neural Network](#a-high-efficient-hybrid-physics-informed-neural-networks-based-on-convolutional-neural-network)
        - [A hybrid physics-informed neural network for nonlinear partial differential equation](#a-hybrid-physics-informed-neural-network-for-nonlinear-partial-differential-equation)
        - [Separable Physics-Informed Neural Networks](#separable-physics-informed-neural-networks)
        - [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](#lnn-pinn-a-unified-physics-only-training-framework-with-liquid-residual-blocks)
      - [Specialized Applications:](#specialized-applications)
        - [ModalPINN: An extension of physics-informed Neural Networks with enforced truncated Fourier decomposition for periodic flow reconstruction using a limited number of imperfect sensors](#modalpinn-an-extension-of-physics-informed-neural-networks-with-enforced-truncated-fourier-decomposition-for-periodic-flow-reconstruction-using-a-limited-number-of-imperfect-sensors)
        - [Novel Physics-Informed Artificial Neural Network Architectures for System and Input Identification of Structural Dynamics PDEs](#novel-physics-informed-artificial-neural-network-architectures-for-system-and-input-identification-of-structural-dynamics-pdes)
        - [Physics-informed Neural Motion Planning on Constraint Manifolds](#physics-informed-neural-motion-planning-on-constraint-manifolds)
    - [C.3.3 PINNs for Robustness: Accelerating and Stabilizing Training](#c33-pinns-for-robustness-accelerating-and-stabilizing-training)
      - [Multi-Objective Loss Optimization：](#multi-objective-loss-optimization)
        - [A Dual-Dimer method for training physics-constrained neural networks with minimax architecture](#a-dual-dimer-method-for-training-physics-constrained-neural-networks-with-minimax-architecture)
        - [On Theory-training Neural Networks to Infer the Solution of Highly Coupled Differential Equations](#on-theory-training-neural-networks-to-infer-the-solution-of-highly-coupled-differential-equations)
        - [Self-adaptive loss balanced Physics-informed neural networks for the incompressible Navier-Stokes equations](#self-adaptive-loss-balanced-physics-informed-neural-networks-for-the-incompressible-navier-stokes-equations)
        - [Adversarial Multi-task Learning Enhanced Physics-informed Neural Networks for Solving Partial Differential Equations](#adversarial-multi-task-learning-enhanced-physics-informed-neural-networks-for-solving-partial-differential-equations)
        - [Multi-Objective Loss Balancing for Physics-Informed Deep Learning](#multi-objective-loss-balancing-for-physics-informed-deep-learning)
      - [Adaptive Input and Gradient Strategies：](#adaptive-input-and-gradient-strategies)
        - [Learning in Sinusoidal Spaces with Physics-Informed Neural Networks](#learning-in-sinusoidal-spaces-with-physics-informed-neural-networks)
        - [Robust Learning of Physics Informed Neural Networks](#robust-learning-of-physics-informed-neural-networks)
        - [Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](#gradient-enhanced-physics-informed-neural-networks-for-forward-and-inverse-pde-problems)
        - [Accelerated Training of Physics-Informed Neural Networks (PINNS) using Meshless Discretizations](#accelerated-training-of-physics-informed-neural-networks-pinns-using-meshless-discretizations)
        - [RPINNS: Rectified-physics informed neural networks for solving stationary partial differential equations](#rpinns-rectified-physics-informed-neural-networks-for-solving-stationary-partial-differential-equations)
        - [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](#a-comprehensive-study-of-non-adaptive-and-residual-based-adaptive-sampling-for-physics-informed-neural-networks)
        - [A Novel Adaptive Causal Sampling Method for Physics-Informed Neural Networks](#a-novel-adaptive-causal-sampling-method-for-physics-informed-neural-networks)
        - [Is L2 Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Network?](#is-l2-physics-informed-loss-always-suitable-for-training-physics-informed-neural-network)
      - [Robustness & Uncertainty Quantification:](#robustness--uncertainty-quantification)
        - [A physics-aware, probabilistic machine learning framework for coarse-graining high-dimensional systems in the Small Data regime](#a-physics-aware-probabilistic-machine-learning-framework-for-coarse-graining-high-dimensional-systems-in-the-small-data-regime)
        - [Adversarial uncertainty quantification in physics-informed neural networks](#adversarial-uncertainty-quantification-in-physics-informed-neural-networks)
        - [B-PINNS: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data](#b-pinns-bayesian-physics-informed-neural-networks-for-forward-and-inverse-pde-problems-with-noisy-data)
        - [PID-GAN: A GAN Framework based on a Physics-informed Discriminator for Uncertainty Quantification with Physics](#pid-gan-a-gan-framework-based-on-a-physics-informed-discriminator-for-uncertainty-quantification-with-physics)
        - [Wasserstein Generative Adversarial Uncertainty Quantification in Physics-Informed Neural Networks](#wasserstein-generative-adversarial-uncertainty-quantification-in-physics-informed-neural-networks)
        - [Flow Field Tomography with Uncertainty Quantification using a Bayesian Physics-Informed Neural Network](#flow-field-tomography-with-uncertainty-quantification-using-a-bayesian-physics-informed-neural-network)
        - [Stochastic Physics-Informed Neural Ordinary Differential Equations](#stochastic-physics-informed-neural-ordinary-differential-equations)
        - [A Physics-Data-Driven Bayesian Method for Heat Conduction Problems](#a-physics-data-driven-bayesian-method-for-heat-conduction-problems)
        - [Spectral PINNS: Fast Uncertainty Propagation with Physics-Informed Neural Networks](#spectral-pinns-fast-uncertainty-propagation-with-physics-informed-neural-networks)
        - [Multi-output physics-informed neural networks for forward and inverse PDE problems with uncertainties](#multi-output-physics-informed-neural-networks-for-forward-and-inverse-pde-problems-with-uncertainties)
        - [Bayesian Physics Informed Neural Networks for real-world nonlinear dynamical systems](#bayesian-physics-informed-neural-networks-for-real-world-nonlinear-dynamical-systems)
        - [Delta-PINNs: A new class of physics-informed neural networks for solving forward and inverse problems with noisy data](#delta-pinns-a-new-class-of-physics-informed-neural-networks-for-solving-forward-and-inverse-problems-with-noisy-data)
        - [Robust Regression with Highly Corrupted Data via Physics Informed Neural Networks](#robust-regression-with-highly-corrupted-data-via-physics-informed-neural-networks)
    - [C.3.4 PINNs for Generalization: Learning Families of PDEs](#c34-pinns-for-generalization-learning-families-of-pdes)
      - [Transfer Learning:](#transfer-learning)
        - [Transfer learning enhanced physics informed neural network for phase-field modeling of fracture](#transfer-learning-enhanced-physics-informed-neural-network-for-phase-field-modeling-of-fracture)
        - [A physics-aware learning architecture with input transfer networks for predictive modeling](#a-physics-aware-learning-architecture-with-input-transfer-networks-for-predictive-modeling)
        - [Transfer learning based multi-fidelity physics informed deep neural network](#transfer-learning-based-multi-fidelity-physics-informed-deep-neural-network)
      - [Meta-Learning & Hypernetworks:](#meta-learning--hypernetworks)
        - [Meta-learning PINN loss functions](#meta-learning-pinn-loss-functions)
        - [HyperPINN: Learning parameterized differential equations with physics-informed hypernetworks](#hyperpinn-learning-parameterized-differential-equations-with-physics-informed-hypernetworks)
        - [A Meta learning Approach for Physics-Informed Neural Networks (PINNs): Application to Parameterized PDEs](#a-meta-learning-approach-for-physics-informed-neural-networks-pinns-application-to-parameterized-pdes)
        - [META-PDE: LEARNING TO SOLVE PDES QUICKLY WITHOUT A MESH](#meta-pde-learning-to-solve-pdes-quickly-without-a-mesh)
        - [GPT-PINN: Generative Pre-Trained Physics-Informed Neural Networks toward non-intrusive Meta-learning of parametric PDEs](#gpt-pinn-generative-pre-trained-physics-informed-neural-networks-toward-non-intrusive-meta-learning-of-parametric-pdes)
    - [C.3.5 Theoretical Foundations, Convergence, and Failure Mode Analysis of PINNs](#c35-theoretical-foundations-convergence-and-failure-mode-analysis-of-pinns)
      - [Estimates on the generalization error of physics-informed neural networks for approximating a class of inverse problems for PDES](#estimates-on-the-generalization-error-of-physics-informed-neural-networks-for-approximating-a-class-of-inverse-problems-for-pdes)
      - [Error analysis for physics informed neural networks (PINNs) approximating Kolmogorov PDEs](#error-analysis-for-physics-informed-neural-networks-pinns-approximating-kolmogorov-pdes)
      - [Simultaneous Neural Network Approximations in Sobolev Spaces](#simultaneous-neural-network-approximations-in-sobolev-spaces)
      - [Characterizing possible failure modes in physics-informed neural networks](#characterizing-possible-failure-modes-in-physics-informed-neural-networks)
      - [Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks](#understanding-and-mitigating-gradient-flow-pathologies-in-physics-informed-neural-networks)
      - [Estimates on the generalization error of physics-informed neural networks for approximating PDEs](#estimates-on-the-generalization-error-of-physics-informed-neural-networks-for-approximating-pdes)
      - [Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)](#investigating-and-mitigating-failure-modes-in-physics-informed-neural-networks-pinns)
    - [C.3.6 Alternative Physics-Inspired Paradigms](#c36-alternative-physics-inspired-paradigms)
      - [Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling](#physics-guided-neural-networks-pgnn-an-application-in-lake-temperature-modeling)
      - [DGM: A deep learning algorithm for solving partial differential equations](#dgm-a-deep-learning-algorithm-for-solving-partial-differential-equations)
      - [Physics-Informed Generative Adversarial Networks for Stochastic Differential Equations](#physics-informed-generative-adversarial-networks-for-stochastic-differential-equations)
      - [Convergence Rate of DeepONets for Learning Operators Arising from Advection-Diffusion Equations](#convergence-rate-of-deeponets-for-learning-operators-arising-from-advection-diffusion-equations)
      - [Variational physics informed neural networks: The role of quadratures and test functions](#variational-physics-informed-neural-networks-the-role-of-quadratures-and-test-functions)
      - [SPINN: Sparse, Physics-based, and partially Interpretable Neural Networks for PDEs](#spinn-sparse-physics-based-and-partially-interpretable-neural-networks-for-pdes)
      - [Error Analysis of Deep Ritz Methods for Elliptic Equations](#error-analysis-of-deep-ritz-methods-for-elliptic-equations)
      - [Theory-guided hard constraint projection (HCP): A knowledge-based data-driven scientific machine learning method](#theory-guided-hard-constraint-projection-hcp-a-knowledge-based-data-driven-scientific-machine-learning-method)
      - [Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces](#learning-partial-differential-equations-in-reproducing-kernel-hilbert-spaces)
      - [A rate of convergence of physics informed neural networks for the linear second order elliptic PDEs](#a-rate-of-convergence-of-physics-informed-neural-networks-for-the-linear-second-order-elliptic-pdes)
      - [Physics-Augmented Learning: A New Paradigm Beyond Physics-Informed Learning](#physics-augmented-learning-a-new-paradigm-beyond-physics-informed-learning)
      - [Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems](#physics-informed-graph-neural-galerkin-networks-a-unified-framework-for-solving-pde-governed-forward-and-inverse-problems)
- [F. Future Direction](#f-future-direction)
  - [C.2 Symbolic System for Solving Physics Problems](#c2-symbolic-system-for-solving-physics-problems)
    - [C.2.0 General Surveys & Foundational Concepts](#c20-general-surveys--foundational-concepts)
      - [Machine learning and the physical sciences](#machine-learning-and-the-physical-sciences)
      - [AI meets physics: a comprehensive survey](#ai-meets-physics-a-comprehensive-survey-1)
    - [C.2.1 Physics-Inspired Generative Models](#c21-physics-inspired-generative-models)
      - [Poisson Flow Generative Models](#poisson-flow-generative-models)
      - [Flow Matching for Generative Modeling](#flow-matching-for-generative-modeling)
      - [PFGM++: Unlocking the Potential of Physics-Inspired Generative Models](#pfgm-unlocking-the-potential-of-physics-inspired-generative-models)
    - [C.2.2 Quantum and Particle Physics](#c22-quantum-and-particle-physics)
      - [Searching for Exotic Particles in High-Energy Physics with Deep Learning](#searching-for-exotic-particles-in-high-energy-physics-with-deep-learning)
      - [Machine learning phases of matter](#machine-learning-phases-of-matter)
      - [Discovering Phase Transitions with Unsupervised Learning](#discovering-phase-transitions-with-unsupervised-learning)
      - [Learning phase transitions by confusion](#learning-phase-transitions-by-confusion)
      - [Using a Recurrent Neural Network to Reconstruct Quantum Dynamics of a Superconducting Qubit from Physical Observations](#using-a-recurrent-neural-network-to-reconstruct-quantum-dynamics-of-a-superconducting-qubit-from-physical-observations)
      - [Graph Neural Networks in Particle Physics: Implementations, Innovations, and Challenges](#graph-neural-networks-in-particle-physics-implementations-innovations-and-challenges)
      - [Application of machine learning in solid state physics](#application-of-machine-learning-in-solid-state-physics)
      - [From Architectures to Applications: A Review of Neural Quantum States](#from-architectures-to-applications-a-review-of-neural-quantum-states)
      - [Ultra-high-granularity detector simulation with intra-event aware generative adversarial network and self-supervised relational reasoning](#ultra-high-granularity-detector-simulation-with-intra-event-aware-generative-adversarial-network-and-self-supervised-relational-reasoning)
      - [Neural-network quantum states for many-body physics](#neural-network-quantum-states-for-many-body-physics)
      - [Review of Machine Learning for Real-Time Analysis at the Large Hadron Collider experiments ALICE, ATLAS, CMS and LHCb](#review-of-machine-learning-for-real-time-analysis-at-the-large-hadron-collider-experiments-alice-atlas-cms-and-lhcb)
      - [Solving the Hubbard model with Neural Quantum States](#solving-the-hubbard-model-with-neural-quantum-states)
      - [Foundation Neural-Networks Quantum States as a Unified Ansatz for Multiple Hamiltonians](#foundation-neural-networks-quantum-states-as-a-unified-ansatz-for-multiple-hamiltonians)
      - [Advancing AI-Scientist Understanding: Multi-Agent LLMs with Interpretable Physics Reasoning](#advancing-ai-scientist-understanding-multi-agent-llms-with-interpretable-physics-reasoning)
    - [C.2.3 Fluid Mechanics & Geosciences](#c23-fluid-mechanics--geosciences)
      - [Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data](#surrogate-modeling-for-fluid-flows-based-on-physics-constrained-deep-learning-without-simulation-data)
      - [Physics-informed neural networks for high-speed flows](#physics-informed-neural-networks-for-high-speed-flows)
      - [Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations](#hidden-fluid-mechanics-learning-velocity-and-pressure-fields-from-flow-visualizations)
      - [NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](#nsfnets-navier-stokes-flow-nets-physics-informed-neural-networks-for-the-incompressible-navier-stokes-equations)
      - [Prediction of porous media fluid flow using physics informed neural networks](#prediction-of-porous-media-fluid-flow-using-physics-informed-neural-networks)
      - [Inverse modeling of nonisothermal multiphase poromechanics using physics-informed neural networks](#inverse-modeling-of-nonisothermal-multiphase-poromechanics-using-physics-informed-neural-networks)
      - [Coupled Lattice Boltzmann Modeling Framework for Pore-Scale Fluid Flow and Reactive Transport](#coupled-lattice-boltzmann-modeling-framework-for-pore-scale-fluid-flow-and-reactive-transport)
    - [C.2.4 Solid Mechanics & Materials Science](#c24-solid-mechanics--materials-science)
      - [Recent advances and applications of machine learning in solid-state materials science](#recent-advances-and-applications-of-machine-learning-in-solid-state-materials-science)
      - [Theory-training deep neural networks for an alloy solidification benchmark problem](#theory-training-deep-neural-networks-for-an-alloy-solidification-benchmark-problem)
      - [A Physics Informed Neural Network Approach to Solution and Identification of Biharmonic Equations of Elasticity](#a-physics-informed-neural-network-approach-to-solution-and-identification-of-biharmonic-equations-of-elasticity)
      - [Physics-informed neural networks for the shallow-water equations on the sphere](#physics-informed-neural-networks-for-the-shallow-water-equations-on-the-sphere)
      - [A mixed formulation for physics-informed neural networks as a potential solver for engineering problems in heterogeneous domains: Comparison with finite element method](#a-mixed-formulation-for-physics-informed-neural-networks-as-a-potential-solver-for-engineering-problems-in-heterogeneous-domains-comparison-with-finite-element-method)
      - [A physically consistent framework for fatigue life prediction using probabilistic physics-informed neural network](#a-physically-consistent-framework-for-fatigue-life-prediction-using-probabilistic-physics-informed-neural-network)
      - [State-of-the-art review on the use of AI-enhanced computational mechanics in geotechnical engineering](#state-of-the-art-review-on-the-use-of-ai-enhanced-computational-mechanics-in-geotechnical-engineering)
    - [C.2.5 Energy Systems & Thermodynamics](#c25-energy-systems--thermodynamics)
      - [Physics-Informed Neural Networks for AC Optimal Power Flow](#physics-informed-neural-networks-for-ac-optimal-power-flow)
      - [A Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature](#a-physics-informed-machine-learning-approach-for-estimating-lithium-ion-battery-temperature)
      - [Physics-Informed Neural Network for Discovering Systems with Unmeasurable States with Application to Lithium-Ion Batteries](#physics-informed-neural-network-for-discovering-systems-with-unmeasurable-states-with-application-to-lithium-ion-batteries)
      - [PE-GPT: A Physics-Informed Interactive Large Language Model for Power Converter Modulation Design](#pe-gpt-a-physics-informed-interactive-large-language-model-for-power-converter-modulation-design)
      - [Recent progress of artificial intelligence for liquid-vapor phase change heat transfer](#recent-progress-of-artificial-intelligence-for-liquid-vapor-phase-change-heat-transfer)
      - [Mapping the design of electrolyte additive for stabilizing zinc anode in aqueous zinc ion batteries](#mapping-the-design-of-electrolyte-additive-for-stabilizing-zinc-anode-in-aqueous-zinc-ion-batteries)
      - [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](#physics-informed-neural-network-for-lithium-ion-battery-degradation-stable-modeling-and-prognosis)
      - [Recent Advances in CO (2) Electroreduction Driven by Artificial Intelligence and Machine Learning](#recent-advances-in-co-2-electroreduction-driven-by-artificial-intelligence-and-machine-learning)
    - [C.2.6 Interdisciplinary & Complex Systems](#c26-interdisciplinary--complex-systems)
      - [Deep Learning for Plasma Tomography and Disruption Prediction from Bolometer Data](#deep-learning-for-plasma-tomography-and-disruption-prediction-from-bolometer-data)
      - [PHYSICS-INFORMED NEURAL NETWORK FOR NONLINEAR DYNAMICS IN FIBER OPTICS](#physics-informed-neural-network-for-nonlinear-dynamics-in-fiber-optics)
      - [Investigating a New Approach to Quasinormal Modes: Physics-Informed Neural Networks](#investigating-a-new-approach-to-quasinormal-modes-physics-informed-neural-networks)
      - [Towards neural Earth system modelling by integrating artificial intelligence in Earth system science](#towards-neural-earth-system-modelling-by-integrating-artificial-intelligence-in-earth-system-science)
      - [Explicit physics-informed neural networks for nonlinear closure: The case of transport in tissues](#explicit-physics-informed-neural-networks-for-nonlinear-closure-the-case-of-transport-in-tissues)
      - [Physically guided deep learning solver for time-dependent Fokker-Planck equation](#physically-guided-deep-learning-solver-for-time-dependent-fokker-planck-equation)
      - [Deep learning for intrinsically disordered proteins: From improved predictions to deciphering conformational ensembles](#deep-learning-for-intrinsically-disordered-proteins-from-improved-predictions-to-deciphering-conformational-ensembles)
      - [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](#specclip-aligning-and-translating-spectroscopic-measurements-for-stars)
- [4. Cross Domain Applications and Future Directions](#4-cross-domain-applications-and-future-directions)
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

#### [PhySense: Principle-Based Physics Reasoning Benchmarking for Large Language Models](https://arxiv.org/pdf/2505.24823)
- **Date:** 2025.05
- **Description:** This paper introduces PhySense, a novel benchmark designed to evaluate the physics reasoning capabilities of Large Language Models (LLMs) based on core principles. The authors find that current LLMs often fail to emulate the concise, principle-based reasoning characteristic of human experts, instead generating lengthy and opaque solutions. PhySense provides a systematic way to investigate this limitation, aiming to guide the development of AI systems with more efficient, robust, and interpretable scientific reasoning.
- **Domain:** `Large Language Models` `AI for Science` `Physics Reasoning` `Benchmarking` `Explainable AI`

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

## C. Physics Reasoning AI

### C.2 Symbolic Reasoning Frameworks

##### [Physics-AI symbiosis](https://doi.org/10.1088/2632-2153/ac9215)
- **Date:** 2022.09
- **Description:** Proposes the concept of "Physics-AI Symbiosis," a comprehensive review of the bidirectional and mutually beneficial relationship between physics and artificial intelligence.
- **Domain:** `AI for Physics` `Physics for AI` `Review` `Interdisciplinary` `Scientific Discovery`

##### [AI meets physics: a comprehensive survey](https://doi.org/10.1007/s10462-024-10874-4)
- **Date:** 2024.08
- **Description:** A comprehensive survey on the bidirectional relationship between AI and physics, detailing how physics principles inspire AI models (Physics for AI) and how AI empowers physical science research (AI for Physics).
- **Domain:** `AI for Physics` `Physics for AI` `Review` `Interdisciplinary` `Machine Learning Theory`

#### C.2.1 Equation-based Reasoning Systems (equation discovery, symbolic regression)

##### [Toward an AI Physicist for Unsupervised Learning](http://arxiv.org/abs/1810.10525)
- **Date:** 2019.09
- **Description:** Proposes a new paradigm for unsupervised learning centered on an 'AI Physicist' agent that discovers, simplifies, and organizes 'theories' (prediction function + domain of applicability) from observational data. Key innovations include a generalized-mean loss for unsupervised domain specialization (divide-and-conquer), a differentiable Minimum Description Length (MDL) objective for simplification (Occam's Razor), and a 'theory hub' for unification and lifelong learning. This work serves as the conceptual precursor to the AI Feynman algorithm.
- **Domain:** `AI for Science` `Unsupervised Learning` `Equation Discovery` `Occam's Razor` `Divide and Conquer`

##### [Intelligence, physics and information – the tradeoff between accuracy and simplicity in machine learning](https://doi.org/10.48550/arXiv.2001.03780)
- **Date:** 2020.01
- **Description:** A foundational thesis that frames machine learning through the lens of physics and information theory, proposing the information bottleneck principle as a universal framework for navigating the accuracy-simplicity tradeoff inherent in scientific discovery.
- **Domain:** `Information Theory` `Machine Learning Theory` `Scientific Discovery` `Symbolic Regression` `Interpretable AI`

##### [Discovering physical concepts with neural networks](https://doi.org/10.1103/PhysRevLett.124.010508)
- **Date:** 2020.01
- **Description:** Introduces a science-agnostic neural network that discovers fundamental physical concepts, like conserved quantities, from raw observational data in an unsupervised manner by using an information bottleneck.
- **Domain:** `Scientific Discovery` `Unsupervised Learning` `Interpretable AI` `Conceptual Physics` `Information Bottleneck`

##### [AI Feynman: A physics-inspired method for symbolic regression](https://www.science.org/doi/10.1126/sciadv.aay2631)
- **Date:** 2020.04
- **Description:** Introduces a novel, physics-inspired algorithm for symbolic regression that recursively discovers symbolic expressions from data. The method uses a neural network to approximate the unknown function and then applies a suite of physics-inspired techniques (e.g., dimensional analysis, symmetry detection, separability) to recursively break the problem down into simpler ones, which are finally solved by a brute-force search. It successfully rediscovered 100 equations from the Feynman Lectures on Physics.
- **Domain:** `Symbolic Regression` `AI for Science` `Equation Discovery` `Neural Networks` `Physics-Inspired AI`

##### [AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity](http://arxiv.org/abs/2006.10782)
- **Date:** 2020.06
- **Description:** An improved version of the AI Feynman algorithm introducing three key innovations: (1) seeking a Pareto-optimal frontier of formulas that balance accuracy and complexity, (2) using neural network gradients to discover generalized symmetries and arbitrary graph modularity, and (3) employing Normalizing Flows to extend symbolic regression to probability distributions. The method demonstrates significantly enhanced robustness to noise and solves more complex problems than its predecessor.
- **Domain:** `Symbolic Regression` `AI for Science` `Pareto Optimality` `Normalizing Flows` `Equation Discovery`

##### [AI-Newton: A Concept-Driven Physical Law Discovery System without Prior Physical Knowledge](https://arxiv.org/pdf/2504.01538)
- **Date:** 2025.04
- **Description:** AI-Newton, a concept-driven discovery system, can autonomously derive physical laws from raw data—without supervision or prior physical knowledge. The system integrates a knowledge base and knowledge representation centered around physical concepts, as well as an autonomous discovery workflow.
- **Domain:** `Survey` `Symbolic-AI`

##### [Bayesian symbolic regression: Automated equation discovery from a physicists' perspective](https://arxiv.org/abs/2507.19540)
- **Date:** 2025.07
- **Description:** Discusses a probabilistic approach to symbolic regression rooted in Bayesian inference, arguing for model plausibility and ensembles over heuristic criteria to provide a more rigorous framework for automated equation discovery.
- **Domain:** `Symbolic Regression` `Bayesian Inference` `Automated Scientific Discovery` `Model Selection` `Review`

#### C.2.2 Neuro-Symbolic Integration, Differentiable Physics Engines

##### [KAN: Kolmogorov-Arnold Networks](https://openreview.net/pdf?id=Ozo7qJ5vZi)
- **Date:** 2024.04
- **Description:** Inspired by the Kolmogorov-Arnold representation theorem, this paper introduces Kolmogorov-Arnold Networks (KANs) as a powerful and interpretable alternative to Multi-Layer Perceptrons (MLPs). KANs feature learnable activation functions on the edges (parameterized as splines) instead of fixed activations on the nodes. This fundamental architectural shift results in superior accuracy and better scaling laws on various tasks, including function fitting and PDE solving. Most importantly, their structure is inherently interpretable, making them a promising tool for scientific discovery.
- **Domain:** `Neural Network Architecture` `Kolmogorov-Arnold Theorem` `Interpretability` `Symbolic Regression` `AI for Science`

##### [KAN 2.0: Kolmogorov-Arnold Networks Meet Science](https://arxiv.org/abs/2408.10205)
- **Date:** 2024.08
- **Description:** This work elevates KANs from a neural network architecture to a comprehensive, bidirectional framework for scientific discovery. It establishes a synergy between science and KANs, enabling both the incorporation of scientific knowledge into KANs (via auxiliary variables, modular structures, and a novel "kanpiler" for compiling formulas) and the extraction of scientific insights from them (via feature attribution, a "tree converter" for modularity, and symbolic simplification). The paper also introduces MultKAN, an extension that includes native multiplication nodes, enhancing interpretability and efficiency.
- **Domain:** `KAN` `AI for Science` `Interpretability` `Scientific Discovery` `Symbolic Regression`

### C.3 Physics-Informed Neural Networks 

#### C.3.1 Macro Perspectives on Physics-Informed Machine Learning (PIML)

##### [Theory-Guided Data Science: A New Paradigm for Scientific Discovery from Data](https://doi.org/10.1109/TKDE.2017.2720168)
- **Date:** 2017.06
- **Description:** A foundational paper that conceptualizes Theory-Guided Data Science (TGDS), proposing a taxonomy of five themes for integrating scientific knowledge with data science models to improve generalization and interpretability.
- **Domain:** `Conceptual Framework` `Theory-Guided Data Science` `Scientific Discovery` `Review` `PIML`

##### [Physics-informed machine learning](https://www.nature.com/articles/s42254-021-00314-5)
- **Date:** 2021.06
- **Description:** A comprehensive review that frames Physics-Informed Machine Learning (PIML) as a new paradigm for integrating data and physical models. It categorizes the methods for embedding physics into machine learning into three types of biases: observational, inductive (architectural), and learning (loss function). The paper surveys the capabilities, limitations, and diverse applications of PIML, positioning PINNs as a key component within this broader field.
- **Domain:** `Physics-Informed Machine Learning` `Review` `PINN` `Inductive Bias` `Scientific Machine Learning`

##### [Hybrid physics-based and data-driven models for smart manufacturing: Modelling, simulation, and explainability](https://doi.org/10.1016/j.jmsy.2022.04.004)
- **Date:** 2022.04
- **Description:** This review paper provides a systematic overview of hybrid physics-based and data-driven models specifically for smart manufacturing applications. It categorizes these hybrid models into three main types: (1) physics-informed machine learning (e.g., PINNs), where physical laws constrain the learning process; (2) machine learning-assisted simulation (e.g., surrogate modeling), where ML accelerates or enhances traditional physics-based simulations; and (3) explainable AI (XAI), which aims to interpret the behavior of complex models. The paper highlights the complementary strengths of physics and data, offering a practical framework for developing more transparent, interpretable, and accurate models in industrial settings.
- **Domain:** `Review` `Hybrid Modeling` `Physics-Informed Machine Learning` `Explainable AI` `Smart Manufacturing`

##### [Physics-Informed Machine Learning: A Survey on Problems, Methods and Applications](https://arxiv.org/abs/2211.08064)
- **Date:** 2022.11
- **Description:** A comprehensive survey that systematically reviews the field of Physics-Informed Machine Learning (PIML) from a machine learning perspective, proposing a taxonomy based on tasks, types of physical priors, and methods of incorporation.
- **Domain:** `Survey` `PIML` `Physics-Informed Machine Learning` `Taxonomy` `Review`

##### [Physics-Informed Neural Networks: A Review of Methodological Evolution, Theoretical Foundations, and Interdisciplinary Frontiers Toward Next-Generation Scientific Computing](https://doi.org/10.3390/app15148092)
- **Date:** 2025.07
- **Description:** A comprehensive and up-to-date review of the Physics-Informed Neural Network (PINN) landscape. The paper systematically categorizes the field's progress into three main axes: methodological evolution (innovations in loss functions, architectures, and training strategies), theoretical foundations (error analysis and connections to classical methods), and interdisciplinary frontiers (applications beyond traditional physics). It provides a structured roadmap for understanding the state-of-the-art and future directions of PINNs as a next-generation scientific computing tool.
- **Domain:** `Review` `PINN` `Scientific Machine Learning` `Physics-Informed AI` `Computational Science`

##### [Training of physical neural networks](https://doi.org/10.1038/s41586-025-09384-2)
- **Date:** 2025.09
- **Description:** A comprehensive review of training methodologies for Physical Neural Networks (PNNs), which use analogue physical systems for computation, categorizing them into in-situ and ex-situ backpropagation strategies and outlining the field's future potential.
- **Domain:** `Physical Neural Networks` `Analogue Computing` `Physics-Aware Training` `Review`

#### C.3.2 PINNs for Scalability: Addressing Large-Scale and Complex Problems

##### General Frameworks & Foundations:

###### [Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/pdf/1711.10561)
- **Date:** 2017.11
- **Description:** This is the first part of the original two-part treatise that introduced Physics-Informed Neural Networks. It specifically focuses on the **forward problem**: using PINNs as data-efficient function approximators to infer the solutions of PDEs. The framework embeds physical laws as soft constraints in the neural network's loss function, demonstrating how to obtain accurate solutions from sparse boundary and initial condition data. It introduces both continuous-time and discrete-time (with Runge-Kutta schemes) models.
- **Domain:** `PINN` `Deep Learning` `Partial Differential Equations` `Forward Problems` `Data-driven Scientific Computing`

###### [Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/pdf/1711.10566)
- **Date:** 2017.11
- **Description:** This work establishes the use of Physics-Informed Neural Networks (PINNs) for solving inverse problems, specifically the data-driven discovery of parameters in nonlinear PDEs. By treating the unknown PDE parameters as trainable variables alongside the neural network's weights, the framework can accurately identify these parameters from sparse and noisy data. It presents a unified approach that handles both forward and inverse problems, demonstrating its power on various physical systems like fluid dynamics, and introduces continuous and discrete time models for different data scenarios.
- **Domain:** `PINN` `Inverse Problems` `Parameter Identification` `Equation Discovery` `Data-driven Scientific Computing`

###### [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://doi.org/10.1016/j.jcp.2018.10.045)
- **Date:** 2018.11
- **Description:** A seminal framework that introduces Physics-Informed Neural Networks (PINNs), which embed physical laws described by general nonlinear PDEs directly into the neural network's loss function. This acts as a regularization agent, enabling the solution of both forward (data-driven solution) and inverse (data-driven discovery) problems from sparse and noisy data.
- **Domain:** `PINN` `Deep Learning` `Partial Differential Equations` `Inverse Problems` `Data-driven Scientific Computing`

##### Domain Decomposition & Parallelism:

###### [PPINN: Parareal physics-informed neural network for time-dependent PDEs](https://doi.org/10.1016/j.cma.2020.113250)
- **Date:** 2020.07
- **Description:** This paper introduces the Parareal Physics-Informed Neural Network (PPINN), a framework designed to accelerate the solution of long-time integration problems for time-dependent PDEs. It combines the classical Parareal parallel-in-time algorithm with PINNs. The method uses a computationally cheap "coarse" solver to provide a global approximation, while multiple "fine" PINN solvers work in parallel to correct the solution on short time sub-intervals. This hybrid, iterative approach significantly reduces the wall-clock time for long-time simulations by decomposing a serial problem into parallelizable sub-problems.
- **Domain:** `PINN` `PINN Acceleration` `Parallel Computing` `Parareal Algorithm` `Time-dependent PDEs`

###### [Finite Basis Physics-Informed Neural Networks (FBPINNs): A scalable domain decomposition approach for solving differential equations](https://arxiv.org/pdf/2107.07871)
- **Date:** 2021.07
- **Description:** This paper proposes Finite Basis PINNs (FBPINNs), a scalable domain decomposition framework for solving large-scale and multiscale PDEs. The method decomposes the domain and assigns an independent PINN to each subdomain. Critically, it constructs a smooth global solution by treating the local PINN solutions as basis functions and stitching them together using a partition of unity weighting. This approach, inspired by finite element methods, significantly improves the scalability and efficiency of PINNs for challenging problems where standard PINNs fail.
- **Domain:** `PINN` `Domain Decomposition` `Scalability` `PINN Acceleration` `Finite Element Method`

###### [When Do Extended Physics-Informed Neural Networks (XPINNs) Improve Generalization?](https://arxiv.org/pdf/2109.09444)
- **Date:** 2021.09
- **Description:** This paper provides the first theoretical generalization analysis for Extended Physics-Informed Neural Networks (XPINNs). By deriving and comparing the generalization error bounds for both standard PINNs and XPINNs, the work reveals a fundamental trade-off in domain decomposition: while decomposing a complex function into simpler ones on subdomains reduces the necessary model complexity (improving generalization), the need to enforce interface conditions introduces an additional learning cost that can harm generalization. The authors conclude that XPINNs outperform PINNs if and only if the gain from the reduced function complexity outweighs the cost of learning the interface constraints.
- **Domain:** `PINN` `Domain Decomposition` `Generalization Theory` `Machine Learning Theory` `Scientific Machine Learning`

###### [Parallel Physics-Informed Neural Networks via Domain Decomposition](https://doi.org/10.1016/j.jcp.2021.110600)
- **Date:** 2021.10
- **Description:** This paper introduces a distributed and parallel framework for PINNs based on the classical domain decomposition method (DDM). The core idea is to break a large computational domain into smaller subdomains, assign an independent, smaller PINN to each subdomain, and enforce physical conservation laws at the interfaces. This approach, implemented for both spatial (cPINN) and spatio-temporal (XPINN) decompositions, enables massive parallelization, significantly accelerating the training process for large-scale problems and offering greater flexibility for multi-physics and multi-scale systems.
- **Domain:** `PINN` `Parallel Computing` `Domain Decomposition` `High-Performance Computing` `PINN Acceleration`

###### [Improved Deep Neural Networks with Domain Decomposition in Solving Partial Differential Equations](https://doi.org/10.1007/s10915-022-01980-y)
- **Date:** 2022.09
- **Description:** This paper proposes an improved domain decomposition method for PINNs to tackle large-scale and complex problems. The approach decomposes the computational domain into subdomains, assigning an individual neural network to each. A key insight of the work is framing domain decomposition as a strategy to mitigate the "gradient pathology" issue prevalent in large, single-network PINNs. The method demonstrates superior performance over classical PINNs in terms of training effectiveness, accuracy, and computational cost.
- **Domain:** `PINN` `Domain Decomposition` `PINN Acceleration` `Gradient Pathology` `Scalability`

##### Complex Geometries & Architectures:

###### [Physics-informed PointNet: A deep learning solver for steady-state incompressible flows and thermal fields on multiple sets of irregular geometries](https://doi.org/10.1016/j.jcp.2022.111510)
- **Date:** 2022.07
- **Description:** This paper introduces Physics-Informed PointNet (PI-PointNet), a novel deep learning framework that equips PINNs with the ability to handle irregular and varied geometries. It replaces the standard MLP backbone with a PointNet encoder, which learns a global feature representation of the domain's shape from a point cloud of its boundary. This geometry-aware feature is then concatenated with spatial coordinates and fed into an MLP decoder to predict the physical field. This "encode-decode" architecture allows the model to be trained on a collection of different geometries and then generalize to solve PDEs on new, unseen shapes without retraining.
- **Domain:** `PINN Architecture` `PointNet` `Geometric Deep Learning` `Irregular Domains` `AI for Engineering`

###### [INN: Interfaced neural networks as an accessible meshless approach for solving interface PDE problems](https://doi.org/10.1016/j.jcp.2022.111588)
- **Date:** 2022.12
- **Description:** This paper proposes Interfaced Neural Networks (INNs) to solve PDE problems with discontinuous coefficients and irregular interfaces, where standard PINNs typically fail. The core idea is a physics-driven domain decomposition: the domain is split along the known interfaces, and a separate neural network is assigned to each subdomain. Crucially, the physical interface conditions (e.g., continuity of the solution and jumps in the flux) are explicitly enforced as loss terms, enabling the framework to accurately capture discontinuities in the solution's derivatives.
- **Domain:** `PINN` `Interface Problems` `Domain Decomposition` `Discontinuous Solutions` `Scientific Machine Learning`

###### [Deep neural network methods for solving forward and inverse problems of time fractional diffusion equations with conformable derivative](https://arxiv.org/pdf/2108.07490)
- **Date:** 2021.08
- **Description:** This paper pioneers the application of Physics-Informed Neural Networks (PINNs) to solve time-fractional diffusion equations involving the conformable derivative, a newer definition in fractional calculus. The work demonstrates that the PINN framework can effectively handle both forward (solution) and inverse (parameter estimation) problems for this class of non-standard PDEs. To address accuracy degradation when the fractional order approaches integer values, the authors introduce a weighted PINN (wPINN) that adjusts the loss function to mitigate the effects of singularities, thereby enhancing the model's robustness.
- **Domain:** `PINN` `Fractional Calculus` `Conformable Derivative` `Inverse Problems` `Scientific Machine Learning`

###### [Neural homogenization and the physics-informed neural network for the multiscale problems](https://arxiv.org/pdf/2108.12942)
- **Date:** 2021.08
- **Description:** This paper introduces Neural Homogenization-PINN (NH-PINN), a method that combines classical homogenization theory with PINNs to solve complex multiscale PDEs. Instead of tackling the challenging multiscale problem directly, NH-PINN employs a three-step process: (1) using PINNs with a proposed oversampling strategy to accurately solve the periodic cell problems at the microscale, (2) computing the effective homogenized coefficients, and (3) using another PINN to solve the much simpler macroscopic homogenized equation. This theoretically-grounded approach significantly improves the accuracy of PINNs for multiscale problems where standard PINNs typically fail.
- **Domain:** `PINN` `Multiscale Modeling` `Homogenization Theory` `Scientific Machine Learning` `Hybrid Modeling`

###### [A High-Efficient Hybrid Physics-Informed Neural Networks Based on Convolutional Neural Network](https://doi.org/10.1109/TNNLS.2021.3070878)
- **Date:** 2021.04
- **Description:** Proposes a hybrid PINN that replaces automatic differentiation with a discrete differential operator learned via a "local fitting method," providing the first theoretical convergence rate for a machine learning-based PDE solver.
- **Domain:** `PINN` `Hybrid Method` `Numerical Stencil` `CNN` `Convergence Rate`

###### [A hybrid physics-informed neural network for nonlinear partial differential equation](https://arxiv.org/abs/2112.01696)
- **Date:** 2021.12
- **Description:** Proposes a hybrid PINN (hPINN) that uses a discontinuity indicator to switch between automatic differentiation for smooth regions and a classical WENO scheme to capture discontinuities, improving performance on PDEs with shock solutions.
- **Domain:** `PINN` `Hybrid Method` `WENO` `Discontinuous Solutions` `Burgers Equation`

###### [Separable Physics-Informed Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/4af827e7d0b7bdae6097d44977e87534-Paper-Conference.pdf)
- **Date:** 2023.12
- **Description:** Proposes Separable Physics-Informed Neural Networks (SPINN) to address the spectral bias issue in solving multiscale PDEs. The core idea is to decompose the solution into multiple components with different characteristic scales (e.g., low- and high-frequency) and use separate, specialized neural network streams for each component. These streams are trained jointly, allowing the model to efficiently and accurately learn complex solutions that standard PINNs fail to capture.
- **Domain:** `PINN Architecture` `Spectral Bias` `Multiscale Modeling` `Physics-Informed Machine Learning` `Fourier Features`

###### [LNN-PINN: A Unified Physics-Only Training Framework with Liquid Residual Blocks](https://arxiv.org/pdf/2508.08935)
- **Date:** 2025.08
- **Description:** LNN-PINN, a physics-informed neural network framework, combines a liquid residual gating architecture while retaining the original physical modeling and optimization process to improve prediction accuracy.
- **Domain:** `PINN` `Liquid Neural Network`

##### Specialized Applications:

###### [ModalPINN: An extension of physics-informed Neural Networks with enforced truncated Fourier decomposition for periodic flow reconstruction using a limited number of imperfect sensors](https://doi.org/10.1016/j.jcp.2022.111271)
- **Date:** 2022.05
- **Description:** This paper introduces ModalPINN, a specialized PINN architecture designed for reconstructing periodic flows from sparse and noisy sensor data. Instead of learning the high-dimensional spatio-temporal field directly, ModalPINN enforces a strong inductive bias by assuming the solution can be represented by a truncated Fourier series. The neural network's task is reduced to learning the low-dimensional time-dependent coefficients of these Fourier modes. This modal decomposition significantly improves the model's robustness and accuracy in data-limited regimes, outperforming standard PINNs for periodic problems.
- **Domain:** `PINN Architecture` `Flow Reconstruction` `Fourier Analysis` `Inductive Bias` `Scientific Machine Learning`

###### [Novel Physics-Informed Artificial Neural Network Architectures for System and Input Identification of Structural Dynamics PDEs](https://www.mdpi.com/2075-5309/13/3/650)
- **Date:** 2023.02
- **Description:** Proposes novel parallel and sequential PINN architectures to solve output-only system and input identification problems in structural dynamics. The method first discretizes the governing PDE into a set of modal ODEs using the Eigenfunction Expansion Method, then assigns individual, cooperating PINNs to each mode, significantly improving computational efficiency, flexibility, and accuracy for complex engineering inverse problems.
- **Domain:** `PINN Architecture` `Structural Dynamics` `System Identification` `Inverse Problems` `Hybrid Model`

###### [Physics-informed Neural Motion Planning on Constraint Manifolds](https://arxiv.org/pdf/2403.05765)
- **Date:** 2024.03
- **Description:** This work introduces a novel physics-informed framework for constrained motion planning (CMP) in robotics. It reformulates the CMP problem as solving the Eikonal PDE on the constraint manifold, which is then solved using a Physics-Informed Neural Network (PINN). This approach is entirely data-free, requiring no expert demonstrations, and learns a neural function that can generate optimal, collision-free paths in sub-seconds. The method significantly outperforms state-of-the-art CMP techniques in speed and success rate on complex, high-dimensional tasks.
- **Domain:** `Physics-Informed Neural Networks` `Robotics` `Motion Planning` `Eikonal Equation` `AI for Engineering`

#### C.3.3 PINNs for Robustness: Accelerating and Stabilizing Training

##### Multi-Objective Loss Optimization：

###### [A Dual-Dimer method for training physics-constrained neural networks with minimax architecture](https://doi.org/10.1016/j.neunet.2020.12.028)
- **Date:** 2021.01
- **Description:** Proposes a novel minimax architecture (PCNN-MM) that formulates PINN training as a saddle-point problem to systematically adjust loss weights, and introduces an efficient "Dual-Dimer" algorithm to solve it.
- **Domain:** `PINN` `Loss Balancing` `Minimax Optimization` `Saddle Point Search` `Training Optimization`

###### [On Theory-training Neural Networks to Infer the Solution of Highly Coupled Differential Equations](https://arxiv.org/abs/2102.04890)
- **Date:** 2021.02
- **Description:** Proposes a Partial Regularization Technique (PRT) to eliminate training oscillations and provides systematic guidelines for finding optimal network architectures, significantly improving the accuracy and robustness of PINNs for highly coupled systems.
- **Domain:** `PINN` `Training Strategy` `Robustness` `Network Architecture` `Regularization`

###### [Self-adaptive loss balanced Physics-informed neural networks for the incompressible Navier-Stokes equations](https://doi.org/10.1007/s10409-021-01053-7)
- **Date:** 2021.04
- **Description:** Proposes a self-adaptive method (lbPINNs) that automatically balances multiple loss components in PINNs by modeling each term's contribution through a learnable uncertainty parameter, significantly improving accuracy for complex fluid dynamics.
- **Domain:** `PINN` `Loss Balancing` `Uncertainty` `Navier-Stokes` `Training Optimization`

###### [Adversarial Multi-task Learning Enhanced Physics-informed Neural Networks for Solving Partial Differential Equations](https://arxiv.org/abs/2104.14320)
- **Date:** 2021.05
- **Description:** Proposes enhancing PINNs by combining multi-task learning (jointly training on a related auxiliary PDE) and adversarial training (generating high-loss samples) to improve generalization and accuracy in highly non-linear domains.
- **Domain:** `PINN` `Multi-task Learning` `Adversarial Training` `Generalization`

###### [Multi-Objective Loss Balancing for Physics-Informed Deep Learning](https://doi.org/10.13140/RG.2.2.20057.24169)
- **Date:** 2021.10
- **Description:** Proposes a novel self-adaptive algorithm, ReLoBRaLo, which dynamically balances multiple loss terms in PINNs based on their relative progress, an exponential moving average, and a unique random lookback mechanism to improve training speed and accuracy.
- **Domain:** `PINN` `Loss Balancing` `Multi-Objective Optimization` `Training Optimization`

##### Adaptive Input and Gradient Strategies：

###### [Learning in Sinusoidal Spaces with Physics-Informed Neural Networks](https://arxiv.org/pdf/2109.13901)
- **Date:** 2021.09
- **Description:** This paper diagnoses a key failure mode in training PINNs, showing that expressive networks are initialized with a bias towards flat, near-zero functions, trapping them in trivial local minima of the PDE residual loss. To overcome this, it proposes sf-PINN, an architecture that preprocesses inputs with a sinusoidal feature mapping. The authors theoretically prove that this mapping increases input gradient variability at initialization, providing effective gradients for the optimizer to escape these deceptive local minima. This simple, non-intrusive modification is shown to significantly improve the training stability and final accuracy of PINNs.
- **Domain:** `PINN` `Training Pathology` `Spectral Bias` `Initialization` `Input Representation`

###### [Robust Learning of Physics Informed Neural Networks](https://arxiv.org/abs/2110.13330)
- **Date:** 2021.10
- **Description:** Proposes a robust training framework where a Gaussian Process (GP) is first used to smooth/denoise noisy training data, and the resulting clean "proxy" data is then used to train the PINN, effectively decoupling data denoising from PDE solving.
- **Domain:** `PINN` `Robustness` `Noisy Data` `Gaussian Process` `Data Pre-processing`

###### [Gradient-enhanced physics-informed neural networks for forward and inverse PDE problems](https://arxiv.org/abs/2111.02801)
- **Date:** 2021.11
- **Description:** Proposes Gradient-enhanced PINNs (gPINNs), a method that improves the accuracy and efficiency of PINNs by adding the gradient of the PDE residual as an additional term in the loss function.
- **Domain:** `PINN` `Gradient-enhanced` `Loss Function` `Training Optimization` `RAR`

###### [Accelerated Training of Physics-Informed Neural Networks (PINNS) using Meshless Discretizations](https://arxiv.org/abs/2205.09332)
- **Date:** 2022.05
- **Description:** Proposes Discretely-Trained PINNs (DT-PINNs), which accelerate training by replacing expensive automatic differentiation for spatial derivatives with a pre-computed, high-order accurate meshless RBF-FD operator, achieving 2-4x speedups.
- **Domain:** `PINN` `Training Acceleration` `RBF-FD` `Meshless Method` `Numerical Differentiation`

###### [RPINNS: Rectified-physics informed neural networks for solving stationary partial differential equations](https://doi.org/10.1016/j.compfluid.2022.105583)
- **Date:** 2022.06
- **Description:** Proposes a Rectified-PINN (RPINN) which, inspired by multigrid methods, uses a second neural network to learn and correct the error of an initial PINN solution, leading to higher final accuracy.
- **Domain:** `PINN` `Iterative Refinement` `Multigrid` `High Precision`

###### [A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks](https://arxiv.org/abs/2207.10289)
- **Date:** 2022.07
- **Description:** Presents a comprehensive benchmark of 10 sampling methods for PINNs and proposes two new residual-based adaptive sampling algorithms (RAD and RAR-D) that significantly improve accuracy by dynamically redistributing points based on the PDE residual.
- **Domain:** `PINN` `Adaptive Sampling` `RAR` `Training Optimization` `Benchmark`

###### [A Novel Adaptive Causal Sampling Method for Physics-Informed Neural Networks](https://arxiv.org/abs/2210.12914)
- **Date:** 2022.10
- **Description:** Proposes an Adaptive Causal Sampling Method (ACSM) that incorporates temporal causality into the sampling process by weighting the residual-based sampling probability with a causal term, preventing training failure in time-dependent PDEs.
- **Domain:** `PINN` `Adaptive Sampling` `Causal Training` `Time-Dependent PDEs`

###### [Is L2 Physics-Informed Loss Always Suitable for Training Physics-Informed Neural Network?](https://arxiv.org/abs/2206.02016)
- **Date:** 2022.06
- **Description:** Theoretically proves that the standard L2 loss is unstable for training PINNs on high-dimensional HJB equations and proposes a new adversarial training algorithm to effectively minimize a more suitable L-infinity loss.
- **Domain:** `PINN` `Loss Function` `PDE Stability` `HJB Equation` `Adversarial Training`

##### Robustness & Uncertainty Quantification:

###### [A physics-aware, probabilistic machine learning framework for coarse-graining high-dimensional systems in the Small Data regime](https://doi.org/10.1016/j.jcp.2019.05.053)
- **Date:** 2019.07
- **Description:** Proposes a physics-aware Bayesian framework using variational inference to construct coarse-grained models from sparse, high-dimensional data, enabling robust uncertainty quantification in the small data regime.
- **Domain:** `Probabilistic Modeling` `Uncertainty Quantification` `Bayesian Inference` `Coarse-Graining` `Small Data`

###### [Adversarial uncertainty quantification in physics-informed neural networks](https://doi.org/10.1016/j.jcp.2019.05.027)
- **Date:** 2019.07
- **Description:** Proposes a framework for uncertainty quantification in PINNs using deep generative models (VAEs and GANs), trained via an adversarial inference procedure where the generator is constrained by physical laws.
- **Domain:** `Uncertainty Quantification` `PINN` `Adversarial Inference` `GAN` `Probabilistic Modeling`

###### [B-PINNS: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data](https://doi.org/10.1016/j.jcp.2020.109913)
- **Date:** 2020.10
- **Description:** Proposes a Bayesian Physics-Informed Neural Network (B-PINN) framework that uses Hamiltonian Monte Carlo (HMC) or Variational Inference (VI) to infer the posterior distribution of network weights, enabling robust uncertainty quantification for PDE problems with noisy data.
- **Domain:** `Bayesian PINN` `Uncertainty Quantification` `Hamiltonian Monte Carlo` `Variational Inference` `Noisy Data`

###### [PID-GAN: A GAN Framework based on a Physics-informed Discriminator for Uncertainty Quantification with Physics](https://arxiv.org/abs/2106.02993)
- **Date:** 2021.06
- **Description:** Proposes PID-GAN, a novel GAN framework where physics constraints are embedded into both the generator and the discriminator, enabling more robust and accurate uncertainty quantification for physical systems.
- **Domain:** `Uncertainty Quantification` `GAN` `PINN` `Probabilistic Modeling` `Adversarial Training`

###### [Wasserstein Generative Adversarial Uncertainty Quantification in Physics-Informed Neural Networks](https://arxiv.org/abs/2108.13054)
- **Date:** 2021.08
- **Description:** Proposes a physics-informed Wasserstein Generative Adversarial Network (WGAN) for uncertainty quantification, and provides a theoretical generalization error bound for the framework.
- **Domain:** `Uncertainty Quantification` `WGAN` `PINN` `Probabilistic Modeling` `Error Analysis`

###### [Flow Field Tomography with Uncertainty Quantification using a Bayesian Physics-Informed Neural Network](https://arxiv.org/abs/2108.09247)
- **Date:** 2021.08
- **Description:** Proposes a Bayesian PINN framework for flow field tomography that reconstructs a 2D flow field from sparse line-of-sight data without boundary conditions, by incorporating both the measurement model and the Navier-Stokes equations into the loss function, while providing uncertainty quantification.
- **Domain:** `Bayesian PINN` `Uncertainty Quantification` `Inverse Problems` `Tomography` `Fluid Dynamics`

###### [Stochastic Physics-Informed Neural Ordinary Differential Equations](https://arxiv.org/abs/2109.01621)
- **Date:** 2021.09
- **Description:** Proposes SPINODE, a framework that learns hidden physics in Stochastic Differential Equations (SDEs) by first deriving deterministic ODEs for the statistical moments, and then training a neural network within this moment-dynamics system using neural ODE solvers.
- **Domain:** `Stochastic Differential Equations` `Uncertainty Quantification` `Neural ODEs` `System Identification` `Probabilistic Modeling`

###### [A Physics-Data-Driven Bayesian Method for Heat Conduction Problems](https://arxiv.org/abs/2109.00996)
- **Date:** 2021.09
- **Description:** Proposes a Heat Conduction Equation assisted Bayesian Neural Network (HCE-BNN) that embeds the PDE into the loss function of a BNN, enabling uncertainty quantification for forward and inverse heat conduction problems.
- **Domain:** `Bayesian PINN` `Uncertainty Quantification` `Bayesian Neural Network` `Heat Transfer`

###### [Spectral PINNS: Fast Uncertainty Propagation with Physics-Informed Neural Networks](https://arxiv.org/abs/2109.00996)
- **Date:** 2021.09
- **Description:** Proposes Spectral PINNs, a method that learns the spectral coefficients of a Polynomial Chaos Expansion (PCE) of a stochastic PDE's solution, enabling fast uncertainty propagation by decoupling the spatiotemporal and stochastic domains.
- **Domain:** `Uncertainty Quantification` `Polynomial Chaos Expansion` `Stochastic PDEs` `Operator Learning`

###### [Multi-output physics-informed neural networks for forward and inverse PDE problems with uncertainties](https://doi.org/10.1016/j.cma.2022.115041)
- **Date:** 2022.05
- **Description:** Proposes a Multi-Output PINN (MO-PINN) that approximates the posterior distribution of the solution by training a single network with multiple output heads, each corresponding to a bootstrapped realization of the noisy data, enabling efficient uncertainty quantification.
- **Domain:** `Uncertainty Quantification` `PINN` `Bootstrap` `Multi-Output Network`

###### [Bayesian Physics Informed Neural Networks for real-world nonlinear dynamical systems](https://doi.org/10.1016/j.cma.2022.115346)
- **Date:** 2022.07
- **Description:** Extends the Bayesian PINN (B-PINN) framework to real-world nonlinear dynamical systems, such as biological growth and epidemic models, providing robust uncertainty quantification and inference of unobservable parameters from sparse, noisy data.
- **Domain:** `Bayesian PINN` `Uncertainty Quantification` `Dynamical Systems` `Epidemiology` `Biomechanics`

###### [Delta-PINNs: A new class of physics-informed neural networks for solving forward and inverse problems with noisy data](https://doi.org/10.1016/j.jcp.2022.111271)
- **Date:** 2022.10
- **Description:** This paper introduces Delta-PINNs, a new training paradigm for PINNs designed to be highly robust to noisy data. Instead of minimizing the standard Mean Squared Error (MSE) of the PDE residuals, Delta-PINNs optimize a novel loss function based on the change ("Delta") of the mean of the residuals over training epochs. This approach focuses on driving the expectation of the residuals to zero in a monotonically decreasing fashion, effectively averaging out the effects of noise rather than fitting to it. The method demonstrates remarkable robustness, successfully solving forward and inverse problems even when the training data is corrupted with up to 100% noise.
- **Domain:** `PINN` `Robustness` `Noisy Data` `Loss Functions` `Scientific Machine Learning`

###### [Robust Regression with Highly Corrupted Data via Physics Informed Neural Networks](https://arxiv.org/pdf/2210.10646)
- **Date:** 2022.10
- **Description:** This paper introduces a new class of PINNs designed to be robust against data with a high percentage of outliers. The authors first propose LAD-PINN, which replaces the standard Mean Squared Error (L2 norm) data loss with a Least Absolute Deviation (L1 norm) loss, making it inherently less sensitive to outliers. Building on this, they propose a two-stage MAD-PINN framework, which first uses LAD-PINN to identify and then screen out outliers based on the Median Absolute Deviation (MAD), and subsequently trains a standard PINN on the cleaned data for high accuracy. This approach is shown to be effective even when more than 50% of the data is corrupted.
- **Domain:** `PINN` `Robustness` `Outlier Detection` `Loss Functions` `Robust Regression`

#### C.3.4 PINNs for Generalization: Learning Families of PDEs

##### Transfer Learning:

###### [Transfer learning enhanced physics informed neural network for phase-field modeling of fracture](https://doi.org/10.1016/j.tafmec.2019.102447)
- **Date:** 2019.12
- **Description:** Proposes a new variational energy-based PINN paradigm (VE-PINN) for more stable fracture problem solving, and innovatively uses transfer learning to significantly accelerate the sequential solving process under multiple load steps.
- **Domain:** `Variational PINN` `Phase-Field Fracture` `Transfer Learning` `Computational Acceleration`

###### [A physics-aware learning architecture with input transfer networks for predictive modeling](https://doi.org/10.1016/j.asoc.2020.106665)
- **Date:** 2020.08
- **Description:** Proposes a novel hybrid architecture called OPTMA, whose core idea is to train a neural network to transform input features, enabling a simple "partial physics model" to make predictions matching a high-fidelity model.
- **Domain:** `Hybrid Modeling` `Transfer Learning` `Physics-Aware ML`

###### [Transfer learning based multi-fidelity physics informed deep neural network](https://doi.org/10.1016/j.jcp.2020.109942)
- **Date:** 2020.10
- **Description:** Proposes the MF-PIDNN framework, which first pre-trains a network on approximate physical equations without data using the PINN method, and then fine-tunes the model with a few high-fidelity data points via transfer learning.
- **Domain:** `Multi-Fidelity` `PINN` `Transfer Learning` `Data-Efficient`

##### Meta-Learning & Hypernetworks:

###### [Meta-learning PINN loss functions](https://arxiv.org/abs/2107.05544)
- **Date:** 2021.07
- **Description:** Proposes a gradient-based meta-learning framework to automatically discover an optimal, shared PINN loss function from a family of related PDE tasks, aiming to improve performance and efficiency on new tasks.
- **Domain:** `Meta-Learning` `PINN` `Loss Function` `Task Distribution`

###### [HyperPINN: Learning parameterized differential equations with physics-informed hypernetworks](https://openreview.net/pdf?id=LxUuRDUhRjM)
- **Date:** 2021.10
- **Description:** This paper introduces HyperPINN, a meta-learning framework that leverages hypernetworks to efficiently solve parameterized PDEs. Instead of training a new PINN for each parameter instance, HyperPINN trains a small "hypernetwork" that takes a physical parameter as input and outputs the weights for a smaller "main" PINN. This main network then solves the PDE for that specific parameter. This approach creates a single, compact model capable of instantly generating a specialized solver for any parameterization, offering a highly efficient and memory-saving alternative for multi-query and real-time applications.
- **Domain:** `PINN` `Meta-Learning` `Hypernetworks` `Parametric PDEs` `Scientific Machine Learning`

###### [A Meta learning Approach for Physics-Informed Neural Networks (PINNs): Application to Parameterized PDEs](https://arxiv.org/abs/2110.13361)
- **Date:** 2021.10
- **Description:** Proposes a model-aware metalearning approach that trains a surrogate model to learn the mapping from PDE parameters to optimal PINN initial weights, providing a high-quality starting point to accelerate training for new tasks.
- **Domain:** `Meta-Learning` `PINN` `Weight Initialization` `Parameterized PDEs`

###### [META-PDE: LEARNING TO SOLVE PDES QUICKLY WITHOUT A MESH](https://arxiv.org/abs/2211.01604)
- **Date:** 2022.11
- **Description:** Proposes a framework called Meta-PDE that uses meta-learning (MAML/LEAP) to find an optimal PINN weight initialization, enabling rapid convergence in just a few gradient steps when solving new, related PDE tasks, even with varying geometries.
- **Domain:** `Meta-Learning` `PINN` `Model Initialization` `Fast PDE Solver`

###### [GPT-PINN: Generative Pre-Trained Physics-Informed Neural Networks toward non-intrusive Meta-learning of parametric PDEs](https://arxiv.org/pdf/2303.14878)
- **Date:** 2023.03
- **Description:** This paper introduces GPT-PINN, a novel meta-learning framework to drastically accelerate the solution of parametric PDEs for multi-query and real-time applications. It treats fully pre-trained PINNs, solved at adaptively selected parameter points, as activation functions or "neurons" in a hyper-reduced meta-network. This "network of networks" learns to generate solutions for new parameters by linearly combining a very small set of these pre-trained basis solutions. The framework is non-intrusive and results in an extremely lightweight and fast surrogate model.
- **Domain:** `PINN` `Meta-Learning` `Model Reduction` `Parametric PDEs` `Scientific Machine Learning`

#### C.3.5 Theoretical Foundations, Convergence, and Failure Mode Analysis of PINNs

##### [Estimates on the generalization error of physics-informed neural networks for approximating a class of inverse problems for PDES](https://doi.org/10.1093/imanum/drab032)
- **Date:** 2021.06
- **Description:** Provides the first rigorous generalization error estimates for PINNs solving data assimilation (unique continuation) inverse problems by leveraging conditional stability estimates from classical PDE theory.
- **Domain:** `Generalization Error` `PINN Theory` `Inverse Problems` `Conditional Stability`

##### [Error analysis for physics informed neural networks (PINNs) approximating Kolmogorov PDEs](https://arxiv.org/abs/2106.14473)
- **Date:** 2021.06
- **Description:** Provides a comprehensive error analysis for PINNs approximating Kolmogorov PDEs, proving that the total error is bounded by the training error and that the required network size and sample complexity grow only polynomially with dimension.
- **Domain:** `Error Analysis` `PINN Theory` `Kolmogorov PDEs` `Curse of Dimensionality`

##### [Simultaneous Neural Network Approximations in Sobolev Spaces](https://arxiv.org/abs/2109.00161)
- **Date:** 2021.09
- **Description:** Provides the first rigorous, non-asymptotic error bounds for deep ReLU networks approximating a smooth function and its derivatives simultaneously in Sobolev norms, establishing a key theoretical foundation for why PINNs are feasible.
- **Domain:** `Approximation Theory` `Neural Network Theory` `Sobolev Norms` `PINN Theory`

##### [Characterizing possible failure modes in physics-informed neural networks](https://arxiv.org/abs/2109.01050)
- **Date:** 2021.09
- **Description:** Systematically identifies and characterizes key failure modes in PINN training, attributing them to "gradient pathologies" from imbalanced loss terms and "spectral bias" where networks fail to learn high-frequency components of the solution.
- **Domain:** `PINN Theory` `Failure Modes` `Gradient Pathologies` `Spectral Bias`

##### [Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks](https://doi.org/10.1137/20M1318043)
- **Date:** 2021.09
- **Description:** Identifies gradient pathologies, caused by imbalanced back-propagated gradients from different loss terms, as a key failure mode in PINNs and proposes a learning rate annealing algorithm that uses gradient statistics to adaptively balance the training.
- **Domain:** `PINN Theory` `Failure Modes` `Gradient Pathologies` `Loss Balancing` `Adaptive Training`

##### [Estimates on the generalization error of physics-informed neural networks for approximating PDEs](https://doi.org/10.1093/imanum/drab093)
- **Date:** 2022.01
- **Description:** Provides rigorous upper bounds on the generalization error for PINNs approximating forward problems for a broad class of (nonlinear) PDEs by leveraging stability estimates of the underlying PDE.
- **Domain:** `Generalization Error` `PINN Theory` `Forward Problems` `PDE Stability`

##### [Investigating and Mitigating Failure Modes in Physics-informed Neural Networks (PINNs)](https://arxiv.org/abs/2209.09988)
- **Date:** 2022.09
- **Description:** Identifies that high-order derivatives contaminate backpropagated gradients, causing training failure, and proposes a novel method to mitigate this by decomposing a high-order PDE into a first-order system using auxiliary variables.
- **Domain:** `PINN` `Failure Modes` `High-Order PDEs` `System Decomposition` `Gradient Contamination`

#### C.3.6 Alternative Physics-Inspired Paradigms

##### [Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling](https://arxiv.org/pdf/1710.11431)
- **Date:** 2017.10
- **Description:** This paper introduces Physics-Guided Neural Networks (PGNN), a framework that synergizes physics-based models and deep learning. PGNNs leverage the outputs of existing physics-based models as input features for a neural network. Crucially, they incorporate a physics-based loss function, which penalizes predictions that are inconsistent with known physical laws (e.g., density-temperature relationships in water) on a large set of unlabeled data. This approach ensures scientific consistency and significantly improves the model's generalization performance, especially in data-scarce scenarios.
- **Domain:** `Physics-Guided AI` `Hybrid Modeling` `PINN Alternatives` `Scientific Machine Learning` `Domain Knowledge Integration`

##### [DGM: A deep learning algorithm for solving partial differential equations](https://doi.org/10.1016/j.jcp.2018.08.029)
- **Date:** 2018.08
- **Description:** This paper introduces the Deep Galerkin Method (DGM), a pioneering deep learning framework for solving high-dimensional partial differential equations (PDEs). Similar to PINNs, DGM approximates the PDE solution with a neural network trained to satisfy the differential operator and boundary/initial conditions. Its key contribution lies in its meshfree nature, achieved by training on batches of randomly sampled points, which allows it to overcome the curse of dimensionality. The paper demonstrates DGM's effectiveness by accurately solving complex, high-dimensional free boundary PDEs in up to 200 dimensions.
- **Domain:** `PINN` `Deep Learning` `High-Dimensional PDEs` `Scientific Machine Learning` `Meshfree Methods`

##### [Physics-Informed Generative Adversarial Networks for Stochastic Differential Equations](http://arxiv.org/abs/1811.02033)
- **Date:** 2018.11
- **Description:** A novel framework that embeds physical laws, in the form of Stochastic Differential Equations (SDEs), into the architecture of a Generative Adversarial Network (GAN). This Physics-Informed GAN (PI-GAN) uses generators to model unknown stochastic processes (e.g., solution, coefficients), with some generators being induced by the SDE to enforce physical consistency. It provides a unified method for solving forward, inverse, and mixed stochastic problems from sparse data, and is capable of handling high-dimensional stochasticity.
- **Domain:** `Physics-Informed Machine Learning` `Generative Adversarial Networks` `Stochastic Differential Equations` `Inverse Problems` `Uncertainty Quantification`

##### [Convergence Rate of DeepONets for Learning Operators Arising from Advection-Diffusion Equations](https://arxiv.org/abs/2102.10621)
- **Date:** 2021.02
- **Description:** Provides the first convergence rate analysis for DeepONets when learning solution operators for advection-diffusion equations, showing that the error depends polynomially on the input dimension and revealing the importance of the solution operator's structure.
- **Domain:** `DeepONet` `Operator Learning` `Convergence Rate` `Error Analysis`

##### [Variational physics informed neural networks: The role of quadratures and test functions](https://doi.org/10.1093/imanum/drab032)
- **Date:** 2021.06
- **Description:** Provides a rigorous a priori error estimate for Variational PINNs (VPINNs) based on an inf-sup condition, revealing the counter-intuitive optimal strategy: using lowest-degree polynomial test functions with high-precision quadrature rules.
- **Domain:** `Variational PINN` `Error Analysis` `PINN Theory` `Inf-Sup Condition` `Petrov-Galerkin`

##### [SPINN: Sparse, Physics-based, and partially Interpretable Neural Networks for PDEs](https://doi.org/10.1016/j.jcp.2021.110600)
- **Date:** 2021.07
- **Description:** This paper introduces Sparse, Physics-based, and partially Interpretable Neural Networks (SPINN) as a bridge between traditional meshless numerical methods and dense PINNs. Instead of a standard MLP, SPINN employs a sparse, shallow network where each hidden neuron's activation function is a learnable basis function (e.g., a radial basis function) inspired by classical function approximation theory. The network is trained using a physics-informed loss. This architecture is inherently sparse and offers partial interpretability, as the learned basis functions and their weights directly correspond to a classical solution expansion.
- **Domain:** `PINN Architecture` `Interpretability` `Sparse Neural Networks` `Meshfree Methods` `Scientific Machine Learning`

##### [Error Analysis of Deep Ritz Methods for Elliptic Equations](https://arxiv.org/abs/2107.14478)
- **Date:** 2021.07
- **Description:** Provides the first rigorous, nonasymptotic convergence rate in the H1 norm for the Deep Ritz Method, establishing how network depth and width should be set relative to the number of training samples to achieve a desired accuracy.
- **Domain:** `Deep Ritz Method` `Error Analysis` `Convergence Rate` `Machine Learning Theory`

##### [Theory-guided hard constraint projection (HCP): A knowledge-based data-driven scientific machine learning method](https://doi.org/10.1016/j.jcp.2021.110624)
- **Date:** 2021.08
- **Description:** This paper introduces the Theory-guided Hard Constraint Projection (HCP) framework, an alternative to the "soft constraint" approach of PINNs. HCP decouples data-driven learning from physics enforcement. First, a standard machine learning model makes a preliminary prediction based on data. Then, this prediction is "projected" onto a manifold representing the feasible solution space defined by physical laws. This two-step process mathematically guarantees that the final output strictly satisfies the imposed physical constraints, addressing a key limitation of soft-constraint methods and enhancing the scientific fidelity of the predictions.
- **Domain:** `Physics-Inspired AI` `Hard Constraints` `Constrained Optimization` `Scientific Machine Learning` `Hybrid Modeling`

##### [Learning Partial Differential Equations in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2108.11580)
- **Date:** 2021.08
- **Description:** Proposes a data-driven method to learn the Green's functions of linear PDEs by framing the problem as functional linear regression in a Reproducing Kernel Hilbert Space (RKHS).
- **Domain:** `Operator Learning` `Reproducing Kernel Hilbert Space` `Green's Function` `Data-Driven Solver`

##### [A rate of convergence of physics informed neural networks for the linear second order elliptic PDEs](https://arxiv.org/abs/2109.01780)
- **Date:** 2021.09
- **Description:** Provides a rigorous convergence rate analysis for PINNs solving second-order elliptic PDEs, establishing upper bounds on the required training samples, network depth, and width to achieve a desired accuracy by analyzing approximation and statistical errors.
- **Domain:** `PINN Theory` `Convergence Rate` `Error Analysis` `Rademacher Complexity`

##### [Physics-Augmented Learning: A New Paradigm Beyond Physics-Informed Learning](https://arxiv.org/pdf/2109.13901)
- **Date:** 2021.09
- **Description:** This paper introduces Physics-Augmented Learning (PAL), a new paradigm to complement the popular Physics-Informed Learning (PIL) framework (e.g., PINNs). The authors draw a crucial distinction between "discriminative" physical properties (like PDEs, suitable for PIL's loss function) and "generative" properties (like conservation laws, which are hard to enforce as residuals). For generative properties, PAL proposes using them to augment a small initial dataset by generating a large number of new, physically-consistent pseudo-data points. This physics-based data augmentation allows neural networks to learn from generative physical laws that are inaccessible to the standard PINN approach.
- **Domain:** `Physics-Inspired AI` `Data Augmentation` `PINN Alternatives` `Scientific Machine Learning` `Generative Physics`

##### [Physics-informed graph neural Galerkin networks: A unified framework for solving PDE-governed forward and inverse problems](https://doi.org/10.1016/j.cma.2021.110602)
- **Date:** 2022.01
- **Description:** This paper introduces Physics-informed graph neural Galerkin networks (PGN) to address the scalability and geometry-handling limitations of standard PINNs. The framework discretizes the problem domain into a graph (mesh) and employs a Graph Neural Network (GNN) to learn the solution at the graph nodes. Critically, its loss function is not based on the strong-form PDE residual, but is inspired by the weak-form Galerkin method from classical numerical analysis. This discrete, GNN-based approach significantly improves training efficiency and naturally handles irregular geometries with unstructured meshes.
- **Domain:** `PINN Architecture` `Graph Neural Networks` `Galerkin Method` `Scientific Machine Learning` `Discrete PINN`

## F. Future Direction

### C.2 Symbolic System for Solving Physics Problems

#### C.2.0 General Surveys & Foundational Concepts

##### [Machine learning and the physical sciences](https://doi.org/10.48550/arXiv.1903.10563)
- **Date:** 2019.12
- **Description:** This comprehensive review article systematically surveys the wide-ranging applications of machine learning in the physical sciences, exploring the current state, challenges, and future directions of the field.
- **Domain:** `Machine Learning` `Physics` `Review` `Statistical Physics` `Quantum Physics` `Particle Physics`

##### [AI meets physics: a comprehensive survey](https://doi.org/10.1007/s10462-024-10874-4)
- **Date:** 2024.08
- **Description:** This is a comprehensive survey of the 'AI meets physics' field, presenting a new PS4AI paradigm and classifying the intersection based on classical mechanics, electromagnetism, statistical physics, and quantum mechanics, while also outlining major challenges and future directions.
- **Domain:** `AI for Physics` `Physics-Inspired AI` `Survey` `Deep Learning`

#### C.2.1 Physics-Inspired Generative Models

##### [Poisson Flow Generative Models](https://papers.neurips.cc/paper_files/paper/2022/file/6ad68a54eaa8f9bf6ac698b02ec05048-Paper-Conference.pdf)
- **Date:** 2022.12
- **Description:** Proposes a novel generative modeling paradigm, Poisson Flow Generative Models (PFGM), that does not require a predefined prior noise distribution. It embeds the data manifold into a higher-dimensional space and constructs a vector field, derived from the solution to a classic Poisson PDE, that deterministically transports the data distribution to a uniform distribution on a hemisphere. To generate samples, one simply samples from this uniform distribution and solves the corresponding ODE backward in time. PFGM achieves state-of-the-art likelihood scores with extremely high sampling efficiency.
- **Domain:** `Generative Modeling` `Poisson Equation` `Continuous Normalizing Flows` `Physics-Inspired AI` `Differential Equations`

##### [Flow Matching for Generative Modeling](http.arxiv.org/abs/2210.02747)
- **Date:** 2023.02
- **Description:** Introduces Flow Matching (FM), a new, simulation-free paradigm for training Continuous Normalizing Flows (CNFs) at scale. The method regresses a vector field that generates a predefined probability path from noise to data. Its core innovation, Conditional Flow Matching (CFM), makes the objective tractable and efficient by leveraging simple per-sample conditional paths. The paper also proposes using Optimal Transport (OT) paths, which are more efficient than standard diffusion paths, leading to faster training, faster sampling, and state-of-the-art performance on large-scale image generation tasks.
- **Domain:** `Generative Modeling` `Continuous Normalizing Flows` `Flow Matching` `Optimal Transport` `Diffusion Models`

##### [PFGM++: Unlocking the Potential of Physics-Inspired Generative Models](https://arxiv.org/abs/2302.04265)
- **Date:** 2023.02
- **Description:** This paper introduces PFGM++, a new family of physics-inspired generative models that unifies and generalizes Poisson Flow Generative Models (PFGM) and diffusion models. By allowing the dimension `D` of the augmented space to be a flexible hyperparameter, PFGM++ can interpolate between the original PFGM (when D=1) and diffusion models (as D approaches infinity). The work also introduces an unbiased, perturbation-based training objective, resolving a key limitation of the original PFGM, and provides a method to transfer hyperparameters from well-tuned diffusion models. PFGM++ with intermediate `D` values is shown to achieve state-of-the-art results on image generation benchmarks.
- **Domain:** `Generative Modeling` `Poisson Flow` `Diffusion Models` `Physics-Inspired AI` `Unified Models`

#### C.2.2 Quantum and Particle Physics

##### [Searching for Exotic Particles in High-Energy Physics with Deep Learning](https://arxiv.org/abs/1402.4735)
- **Date:** 2014.02
- **Description:** This seminal paper demonstrates that deep learning can significantly improve signal-versus-background classification in high-energy physics, outperforming traditional shallow methods without requiring manually-constructed features.
- **Domain:** `High Energy Physics` `Particle Physics` `Deep Learning` `Classification`

##### [Machine learning phases of matter](https://arxiv.org/abs/1605.01735)
- **Date:** 2016.05
- **Description:** This paper demonstrates that supervised learning with neural networks and CNNs can effectively classify conventional and topological phases of matter directly from raw configurations, even without prior knowledge of the underlying physics.
- **Domain:** `Condensed Matter Physics` `Phase Transition` `Supervised Learning` `Ising Model`

##### [Discovering Phase Transitions with Unsupervised Learning](https://arxiv.org/abs/1606.00318)
- **Date:** 2016.06
- **Description:** This paper pioneers the use of unsupervised learning, specifically PCA and clustering, to discover phases and phase transitions in the Ising model, demonstrating that machine learning can find fundamental physical concepts like order parameters without prior knowledge.
- **Domain:** `Unsupervised Learning` `Statistical Mechanics` `Phase Transition` `Ising Model`

##### [Learning phase transitions by confusion](https://arxiv.org/abs/1610.02048)
- **Date:** 2016.10
- **Description:** This paper proposes a novel "confusion scheme" using neural networks and deliberately mislabeled data to detect phase transitions in physical systems without prior knowledge of order parameters.
- **Domain:** `Condensed Matter Physics` `Quantum Physics` `Phase Transition` `Machine Learning`

##### [Using a Recurrent Neural Network to Reconstruct Quantum Dynamics of a Superconducting Qubit from Physical Observations](https://doi.org/10.1103/PhysRevX.10.011006)
- **Date:** 2020.01
- **Description:** A work demonstrating that an RNN (LSTM) can be trained to perform real-time, model-agnostic quantum filtering and reconstruct the full quantum state (including coherence) of an open superconducting qubit from experimental measurements.
- **Domain:** `Quantum Dynamics` `Quantum Control` `RNN/LSTM` `Superconducting Qubit`

##### [Graph Neural Networks in Particle Physics: Implementations, Innovations, and Challenges](https://arxiv.org/pdf/2203.12852)
- **Date:** 2022.03
- **Description:** A comprehensive review of the application of Graph Neural Networks (GNNs) in particle physics. This work highlights that physical systems like particle jets and detector signals can be naturally represented as graphs, making GNNs a particularly powerful and physically-motivated architecture. The paper surveys the successful use of GNNs across a wide range of tasks, including particle reconstruction, jet tagging, and event generation, demonstrating how this specialized architecture unlocks new capabilities in analyzing complex experimental data.
- **Domain:** `Graph Neural Networks` `Particle Physics` `AI for Science` `Experimental Data Analysis` `Scientific Machine Learning`

##### [Application of machine learning in solid state physics](https://doi.org/10.1016/bs.ssp.2023.08.001)
- **Date:** 2023.08
- **Description:** A systematic review covering the application of ML models (RBM, ARNN, GNN, RL) to symbolic solid state and statistical physics problems, specifically focusing on multi-body systems, phase transitions, and ground state optimization (e.g., spin glasses).
- **Domain:** `Statistical Mechanics` `Phase Transition` `Spin Models` `Multi-Body Physics`

##### [From Architectures to Applications: A Review of Neural Quantum States](https://arxiv.org/abs/2402.09402)
- **Date:** 2024.02
- **Description:** A comprehensive review of Neural Quantum States (NQS), detailing the architectures, training methods, and wide-ranging applications in simulating quantum many-body systems, from finding ground states to modeling quantum dynamics.
- **Domain:** `Neural Quantum States` `Quantum Many-Body Physics` `Variational Monte Carlo` `Computational Physics` `Review`

##### [Ultra-high-granularity detector simulation with intra-event aware generative adversarial network and self-supervised relational reasoning](https://doi.org/10.1038/s41467-024-49104-4)
- **Date:** 2024.06
- **Description:** Proposes IEA-GAN, a novel generative model using a Transformer-based relational reasoning module and self-supervised learning to achieve ultra-fast, high-fidelity simulation of high-granularity particle detectors.
- **Domain:** `Generative Models` `Particle Physics` `Detector Simulation` `Transformer` `Self-Supervised Learning`

##### [Neural-network quantum states for many-body physics](https://arxiv.org/abs/2402.11014)
- **Date:** 2024.08
- **Description:** A methodological review of Neural Quantum States, deriving the core equations for Variational Monte Carlo approaches and emphasizing the role of the quantum geometric tensor in optimizing networks for many-body systems.
- **Domain:** `Neural Quantum States` `Quantum Many-Body Physics` `Variational Monte Carlo` `Computational Physics` `Review`

##### [Review of Machine Learning for Real-Time Analysis at the Large Hadron Collider experiments ALICE, ATLAS, CMS and LHCb](https://arxiv.org/abs/2506.14578)
- **Date:** 2025.06
- **Description:** A comprehensive whitepaper reviewing the use of machine learning techniques for real-time data analysis and triggering systems in the major LHC experiments, highlighting methods for fast inference on hardware like FPGAs.
- **Domain:** `Machine Learning` `Particle Physics` `LHC` `Real-Time Analysis` `Trigger System`

##### [Solving the Hubbard model with Neural Quantum States](https://arxiv.org/pdf/2507.02644)
- **Date:** 2025.07
- **Description:** This work tackles the fundamental challenge of solving the Hubbard model, a cornerstone of condensed matter physics, by leveraging Neural Quantum States (NQS). The core innovation lies in parameterizing the quantum many-body wavefunction with an advanced neural network architecture inspired by Transformers, specifically incorporating a self-attention mechanism. This allows the NQS to efficiently capture the complex, long-range correlations and entanglement in strongly correlated electron systems. Optimized via the Variational Monte Carlo method, this approach achieves state-of-the-art accuracy in determining the ground state energy of the 2D Hubbard model, outperforming traditional numerical methods.
- **Domain:** `Neural Quantum States` `Hubbard Model` `Quantum Many-Body Physics` `Computational Physics` `AI for Science`

##### [Foundation Neural-Networks Quantum States as a Unified Ansatz for Multiple Hamiltonians](https://arxiv.org/abs/2502.09488)
- **Date:** 2025.08
- **Description:** Introduces Foundation Neural-Network Quantum States (FNQS), a single, versatile Transformer-based architecture that takes Hamiltonian parameters as input, enabling it to generalize and solve for ground states of quantum systems unseen during training.
- **Domain:** `Neural Quantum States` `Foundation Models` `Quantum Many-Body Physics` `Meta-Learning` `Generalization`

##### [Advancing AI-Scientist Understanding: Multi-Agent LLMs with Interpretable Physics Reasoning](https://arxiv.org/abs/2504.01911)
- **Date:** 2025.08
- **Description:** Introduces a multi-agent LLM framework that transforms free-form physics reasoning into an interpretable and executable model, enhancing reliability and human-AI collaboration.
- **Domain:** `Large Language Models` `Interpretability` `AI Scientist` `Multi-Agent Systems` `Human-AI Collaboration`

#### C.2.3 Fluid Mechanics & Geosciences

##### [Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data](https://doi.org/10.1016/j.cma.2019.112732)
- **Date:** 2019.11
- **Description:** Develops a physics-constrained DNN surrogate for parametric fluid flows that requires no simulation data for training, using the Navier-Stokes equations as the sole source of supervision and a novel architecture for hard boundary condition enforcement.
- **Domain:** `PINN` `Fluid Dynamics` `Surrogate Modeling` `Data-Free Learning` `Uncertainty Quantification`

##### [Physics-informed neural networks for high-speed flows](https://doi.org/10.1016/j.cma.2019.112789)
- **Date:** 2019.12
- **Description:** Applies PINNs to solve forward and inverse problems for the Euler equations in high-speed flows, demonstrating the ability to capture shocks and infer full flow fields from sparse, experimentally-inspired data like density gradients.
- **Domain:** `PINN` `Fluid Dynamics` `Euler Equations` `Shock Capturing` `Inverse Problem`

##### [Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations](https://doi.org/10.1126/science.aaw4741)
- **Date:** 2020.02
- **Description:** Introduces "Hidden Fluid Mechanics," a PINN framework that infers hidden velocity and pressure fields from visualized scalar concentration data by embedding the Navier-Stokes equations as a physical constraint.
- **Domain:** `PINN` `Fluid Dynamics` `Inverse Problem` `Data Assimilation` `Flow Visualization`

##### [NSFnets (Navier-Stokes flow nets): Physics-informed neural networks for the incompressible Navier-Stokes equations](https://doi.org/10.1016/j.jcp.2020.109951)
- **Date:** 2020.11
- **Description:** Systematically investigates PINNs for solving the incompressible Navier-Stokes equations by comparing velocity-pressure (VP) and vorticity-velocity (VV) formulations, and presents a pioneering attempt at direct turbulence simulation.
- **Domain:** `PINN` `Fluid Dynamics` `Navier-Stokes` `Turbulence` `Inverse Problem`

##### [Prediction of porous media fluid flow using physics informed neural networks](https://doi.org/10.1016/j.petrol.2021.109205)
- **Date:** 2021.07
- **Description:** Applies PINNs to the Buckley-Leverett problem for two-phase flow in porous media, demonstrating superior extrapolation capabilities over standard ANNs and the ability to solve inverse problems for multiphase flow parameters.
- **Domain:** `PINN` `Porous Media Flow` `Reservoir Engineering` `Buckley-Leverett` `Inverse Problem`

##### [Inverse modeling of nonisothermal multiphase poromechanics using physics-informed neural networks](https://arxiv.org/abs/2209.03276)
- **Date:** 2022.09
- **Description:** Proposes a sequential PINN solver for complex thermo-hydro-mechanical (THM) inverse problems in porous media by decoupling the multiphysics system and training separate networks in sequence.
- **Domain:** `PINN` `Poromechanics` `Geosciences` `Inverse Problem` `Multiphysics`

##### [Coupled Lattice Boltzmann Modeling Framework for Pore-Scale Fluid Flow and Reactive Transport](https://doi.org/10.1021/acsomega.2c07643)
- **Date:** 2023.04
- **Description:** A framework coupling LBM fluid flow and PHREEQCRM geochemical solvers, featuring an AI (ANN/MLP) optimization workflow for automatically calibrating reaction constants (log K) in complex pore-scale reactive transport models without domain knowledge.
- **Domain:** `Fluid Mechanics/LBM` `Geosciences` `Pore-Scale Modeling` `AI Optimization/ANN`

#### C.2.4 Solid Mechanics & Materials Science

##### [Recent advances and applications of machine learning in solid-state materials science](https://doi.org/10.1038/s41524-019-0221-0)
- **Date:** 2019.08
- **Description:** This review provides a comprehensive overview of the latest advancements in applying machine learning to solid-state materials science, covering material discovery, property prediction, force fields, and key challenges like interpretability and data scarcity.
- **Domain:** `Materials Science` `Review` `Machine Learning` `Solid State Physics`

##### [Theory-training deep neural networks for an alloy solidification benchmark problem](https://arxiv.org/abs/1912.09800)
- **Date:** 2019.12
- **Description:** Presents the first application of PINNs to alloy solidification, demonstrating the ability to solve coupled phase-change equations and learn implicit variables like solid fraction without any simulation data.
- **Domain:** `PINN` `Materials Science` `Solidification` `Phase Change` `Thermodynamics`

##### [A Physics Informed Neural Network Approach to Solution and Identification of Biharmonic Equations of Elasticity](https://arxiv.org/abs/2108.07243)
- **Date:** 2021.08
- **Description:** First applies PINNs to solve fourth-order biharmonic equations in elasticity by constructing a novel "Airy-Network" whose architecture is directly guided by classical analytical solutions (e.g., Airy stress functions), leading to superior accuracy and efficiency over standard PINNs.
- **Domain:** `PINN` `Solid Mechanics` `Elasticity Theory` `Biharmonic Equation` `Feature Engineering`

##### [Physics-informed neural networks for the shallow-water equations on the sphere](https://doi.org/10.1016/j.jcp.2022.111024)
- **Date:** 2022.02
- **Description:** Solves the shallow-water equations on a sphere using a novel multi-model approach, where the time domain is decomposed and a sequence of PINNs are trained to handle long integration intervals, a key challenge for standard PINNs.
- **Domain:** `PINN` `Geophysical Fluid Dynamics` `Shallow-Water Equations` `Domain Decomposition` `Meteorology`

##### [A mixed formulation for physics-informed neural networks as a potential solver for engineering problems in heterogeneous domains: Comparison with finite element method](https://doi.org/10.1016/j.cma.2022.115616)
- **Date:** 2022.09
- **Description:** Proposes a novel mixed formulation for PINNs, inspired by FEM, that uses separate networks for the primary variable and its spatial gradient and combines energy-based and strong-form losses to accurately solve problems in heterogeneous solids while avoiding high-order derivatives.
- **Domain:** `PINN` `Mixed Formulation` `Finite Element Method` `Solid Mechanics` `Heterogeneous Materials`

##### [A physically consistent framework for fatigue life prediction using probabilistic physics-informed neural network](https://doi.org/10.1016/j.ijfatigue.2022.107234)
- **Date:** 2022.09
- **Description:** Proposes a probabilistic PINN for fatigue life prediction that ensures physical consistency by encoding fatigue principles (e.g., S-N curve monotonicity and heteroscedasticity) as first and second-order derivative constraints in the loss function.
- **Domain:** `PINN` `Solid Mechanics` `Fatigue Life Prediction` `Probabilistic Modeling` `Uncertainty Quantification`

##### [State-of-the-art review on the use of AI-enhanced computational mechanics in geotechnical engineering](https://doi.org/10.1007/s10462-024-10836-w)
- **Date:** 2024.07
- **Description:** This state-of-the-art review comprehensively analyzes the use of AI-enhanced computational mechanics in geotechnical engineering, categorizing applications and identifying future research directions, particularly the integration of physically-guided and adaptive learning.
- **Domain:** `Geotechnical Engineering` `Computational Mechanics` `Solid Mechanics` `Review`

#### C.2.5 Energy Systems & Thermodynamics

##### [Physics-Informed Neural Networks for AC Optimal Power Flow](https://doi.org/10.1016/j.epsr.2022.108412)
- **Date:** 2022.07
- **Description:** First applies PINNs to the AC-OPF problem by incorporating KKT optimality conditions into the loss function, and introduces formal verification methods to provide rigorous worst-case guarantees on constraint violations.
- **Domain:** `PINN` `Power Systems` `Optimal Power Flow` `Worst-case Guarantees` `Energy Systems`

##### [A Physics-Informed Machine Learning Approach for Estimating Lithium-Ion Battery Temperature](https://doi.org/10.1109/ACCESS.2022.3199652)
- **Date:** 2022.08
- **Description:** Applies a PINN with a novel architecture and adaptive loss to predict lithium-ion battery temperature using a lumped thermal model and sparse data.
- **Domain:** `PINN` `Energy Systems` `Battery Management` `Thermodynamics` `Data-driven Modeling`

##### [Physics-Informed Neural Network for Discovering Systems with Unmeasurable States with Application to Lithium-Ion Batteries](https://arxiv.org/abs/2311.16374)
- **Date:** 2023.11
- **Description:** Proposes a novel PINN training method that embeds a numerical solver for the governing equations into the forward pass of the loss function, enabling the discovery of unmeasurable internal states in complex systems like Li-ion batteries from observable data alone.
- **Domain:** `PINN` `System Identification` `Unmeasurable States` `Differentiable Physics` `Energy Systems`

##### [PE-GPT: A Physics-Informed Interactive Large Language Model for Power Converter Modulation Design](https://arxiv.org/pdf/2403.14059)
- **Date:** 2024.03
- **Description:** This paper introduces PE-GPT, a novel system that synergizes a Large Language Model (LLM) with Physics-Informed Neural Networks (PINNs) to create an interactive design assistant for power electronics. The LLM (GPT-4) acts as a natural language interface, guiding users through the design process via in-context learning. The backend consists of a custom, hierarchical PINN architecture that accurately models the converter's physics with high data efficiency. This framework significantly enhances the accessibility, explainability, and efficiency of the power converter modulation design process.
- **Domain:** `Large Language Models` `Physics-Informed Neural Networks` `Human-AI Interaction` `Engineering Design` `Power Electronics`

##### [Recent progress of artificial intelligence for liquid-vapor phase change heat transfer](https://doi.org/10.1038/s41524-024-01223-8)
- **Date:** 2024.03
- **Description:** This review provides a comprehensive overview of how AI and machine learning are revolutionizing liquid-vapor phase change heat transfer research by enabling advanced meta-analysis, physical feature extraction from visual data, and real-time data stream analysis for smart thermal systems.
- **Domain:** `Phase Change` `Heat Transfer` `AI for Physics` `Review`

##### [Mapping the design of electrolyte additive for stabilizing zinc anode in aqueous zinc ion batteries](https://doi.org/10.1016/j.ensm.2024.103364)
- **Date:** 2024.03
- **Description:** This review article systematically classifies and analyzes the mechanisms of electrolyte additives for stabilizing zinc anodes in aqueous zinc-ion batteries, highlighting how AI and ML can accelerate the discovery and design of next-generation battery technologies.
- **Domain:** `Zinc-ion Batteries` `Electrochemistry` `Energy Storage` `Review`

##### [Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis](https://doi.org/10.1038/s41467-024-48779-z)
- **Date:** 2024.05
- **Description:** Proposes a PINN for battery state-of-health (SOH) prognosis by learning the single-cycle degradation dynamics constrained by an empirical state-space degradation model.
- **Domain:** `PINN` `Energy Systems` `Battery Management` `State of Health` `Prognosis`

##### [Recent Advances in CO (2) Electroreduction Driven by Artificial Intelligence and Machine Learning](https://doi.org/10.1002/aenm.202503560)
- **Date:** 2025.09
- **Description:** This review provides a comprehensive survey of how AI and ML are driving the field of CO₂ electroreduction, focusing on catalyst design, reaction mechanism investigation, and knowledge graph construction to accelerate the discovery of sustainable energy materials.
- **Domain:** `Electrocatalysis` `CO2 Reduction` `Generative AI` `Energy Systems`

#### C.2.6 Interdisciplinary & Complex Systems

##### [Deep Learning for Plasma Tomography and Disruption Prediction from Bolometer Data](https://doi.org/10.1109/TPS.2020.3010833)
- **Date:** 2020.06
- **Description:** Utilizes Convolutional Neural Networks (CNNs) for real-time plasma radiation tomography and Recurrent Neural Networks (RNNs) for disruption prediction from bolometer data in the JET tokamak.
- **Domain:** `Deep Learning` `Plasma Physics` `Nuclear Fusion` `Tomography` `Disruption Prediction`

##### [PHYSICS-INFORMED NEURAL NETWORK FOR NONLINEAR DYNAMICS IN FIBER OPTICS](https://arxiv.org/abs/2109.00526)
- **Date:** 2021.09
- **Description:** Develops a generalizable PINN for the Nonlinear Schrödinger Equation by embedding physical parameters (e.g., pulse power) as inputs, enabling a single model to solve multiple fiber optic dynamics scenarios.
- **Domain:** `PINN` `Nonlinear Optics` `Fiber Optics` `NLSE` `Generalizability`

##### [Investigating a New Approach to Quasinormal Modes: Physics-Informed Neural Networks](https://arxiv.org/abs/2108.05867)
- **Date:** 2021.08
- **Description:** Pioneers the use of PINNs for calculating black hole quasinormal modes by framing the problem of finding the unknown quasinormal frequency as an inverse parameter identification problem within the perturbation equation.
- **Domain:** `PINN` `Black Hole Physics` `General Relativity` `Quasinormal Modes` `Inverse Problem`

##### [Towards neural Earth system modelling by integrating artificial intelligence in Earth system science](https://doi.org/10.1038/s42256-021-00374-3)
- **Date:** 2021.08
- **Description:** Proposes the concept of 'neural Earth system modelling' (NESYM), a methodological vision for deeply integrating AI and Earth System Models (ESMs) into learning, self-validating, and interpretable hybrids to tackle grand challenges in climate science.
- **Domain:** `Earth System Science` `Climate Modeling` `AI for Science` `Hybrid Modeling` `Review`

##### [Explicit physics-informed neural networks for nonlinear closure: The case of transport in tissues](https://doi.org/10.1016/j.jcp.2021.110781)
- **Date:** 2021.10
- **Description:** Proposes a novel 'explicit' PINN approach for upscaling, where a neural network learns the nonlinear closure term (effectiveness factor) by explicitly using macroscale variables (concentration and its gradient) as input features, demonstrating excellent generalizability for transport in complex biological tissues.
- **Domain:** `PINN` `Multiscale Modeling` `Upscaling` `Nonlinear Closure` `Biophysics`

##### [Physically guided deep learning solver for time-dependent Fokker-Planck equation](https://doi.org/10.1016/j.ijnonlinmec.2022.104202)
- **Date:** 2022.08
- **Description:** Proposes a PINN-based solver for the Fokker-Planck equation, uniquely using dropout during inference to assess model robustness and demonstrating that sparse observation data can effectively substitute for missing boundary conditions.
- **Domain:** `PINN` `Fokker-Planck Equation` `Stochastic Dynamics` `Robustness`

##### [Deep learning for intrinsically disordered proteins: From improved predictions to deciphering conformational ensembles](https://doi.org/10.1016/j.sbi.2024.102950)
- **Date:** 2024.07
- **Description:** Reviews the impact of deep learning on the study of intrinsically disordered proteins (IDPs), highlighting the shift from predicting disordered regions to using generative models for deciphering their complex conformational ensembles.
- **Domain:** `Deep Learning` `Biophysics` `Structural Biology` `Intrinsically Disordered Proteins` `Generative Models`

##### [SpecCLIP: Aligning and Translating Spectroscopic Measurements for Stars](https://arxiv.org/pdf/2507.01939)
- **Date:** 2025.07
- **Description:** A CLIP-inspired foundation model for stellar spectral analysis that leverages cross-instrument contrastive pre-training and spectrum-aware decoders to enable precise spectral alignment, parameter estimation, and anomaly detection across diverse astronomical applications.
- **Domain:** `Contrastive Learning` `Astrophysics`


## 4. Cross Domain Applications and Future Directions

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

