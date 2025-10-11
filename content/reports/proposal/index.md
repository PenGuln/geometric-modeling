---
title: "Project Proposal"
date: 2025-10-11
tags: ["geometric modeling", "computer graphics", "implicit modeling", "neural networks", "optimization"]
author: "Siqi Guo"
description: "Project proposal for geometric modeling" 
summary: "This proposal outlines a project to accelerate MLP training for 3D curve surfacing through smoothness energy computation optimization." 
cover:
    image: "proposal.png"
    alt: "Project Proposal"
    relative: true
showToc: true
disableAnchoredHeadings: false

---

# Project Proposal

**Title:** Accelerating Training for 3D Curve Surfacing through Optimized Implicit Modeling

**Student Name:** Siqi Guo

## Summary

### Description of Problem
Surfacing a collection of 3D curves is a fundamental challenge in geometric modeling. Neural implicit modeling offers a promising approach to interpolate input curves by representing the surface $S$ as the zero-level set of a neural Signed Distance Function: $f(x;\Theta) : \mathbb{R}^3 \rightarrow \mathbb{R}$, where $\Theta$ denotes the network parameters. Formally, $S = \{x \in \mathbb{R}^3 \mid f(x; \Theta) = 0\}$.

Recent work by Wang et al. [2025] introduced NeuVAS, which optimizes the network $f(x;\Theta)$ using a combined loss function consisting of:
- **Eikonal Condition**: Ensures regularization of the implicit field
- **Dirichlet Condition**: Guarantees that the implicit surface $S$ passes through the input curves
- **Weighted Thin-Plate Energy**: Produces smooth surfaces while preserving sharpness on selected feature curves

While NeuVAS demonstrates strong modeling performance compared to existing methods, there are notable limitations not addressed by the authors:

1. **Computational Cost**: Thin-plate energy requires Hessian matrix computation of $f$, which incurs significant computational overhead (though it remains more efficient than curvature variation energy, which involves third-order derivatives).

2. **Weight Function Design**: The choice of squared Euclidean distance $d^2(s,\mathcal{P}_f)$ as the weight for thin-plate energy lacks clear justification, where $s$ is a surface point and $\mathcal{P}_f$ represents the set of feature curve points. Alternative distance metrics such as $d(s,\mathcal{P}_f)$ or Manhattan distance warrant investigation.


### Importance of Problem
3D curve surfacing is fundamental to computer graphics, CAD systems, and geometric modeling applications. While implicit modeling offers distinct advantages in representing complex surfaces and handling topological changes, the computational cost of training neural networks remains a significant barrier to real-time applications and large-scale geometric modeling tasks. Accelerating the training process would enhance the practicality of implicit modeling for industrial applications and enable more sophisticated geometric modeling scenarios.

### Your Proposal
I propose to develop an optimized training system for neural implicit modeling with specific focus on addressing the computational bottlenecks in 3D curve surfacing. The approach encompasses:

1. **Computational Analysis**: Systematically analyze the computational structure and bottlenecks of the NeuVAS loss function
2. **Algorithm Development**: Design efficient algorithms for Hessian matrix approximation or develop alternative smoothness measures
3. **Validation**: Evaluate the approach on diverse test cases with comprehensive runtime performance analysis

### Originality

This work will contribute novel optimization techniques for Hessian computation in neural implicit modeling, with potential applications to broader geometric modeling problems involving neural networks. Additionally, the investigation of alternative weight functions may provide insights into better surface quality control mechanisms.


## List of Goals

### First Update
- Complete comprehensive literature review on neural implicit modeling and surface optimization techniques
- Implement baseline MLP training pipeline using the NeuVAS loss formulation (original code not publicly available)
- Profile the implementation to identify and quantify specific computational bottlenecks


### Second Update

- Develop the evaluation framework including surface visualization and run-time performance checking.
- Explore and implement alternative approaches: 
    - Efficient smoothness energy compuation
    - Alternative Weight function
    
### Final Report

Identify optimal optimization strategy through systematic evaluation. Complete comprehensive validation on diverse 3D curve surfacing problem. Perform ablation studies.