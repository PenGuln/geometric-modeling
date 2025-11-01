---
title: "First Project Update"
date: 2025-10-11
tags: ["geometric modeling", "computer graphics", "implicit modeling", "neural networks", "optimization", "project update"]
author: "Siqi Guo"
description: "First project update for accelerating training in 3D curve surfacing" 
summary: "First progress update covering literature review completion and initial implementation work." 
showToc: true
disableAnchoredHeadings: false
---

# First Project Update

**Project Title:** Accelerating Training for 3D Curve Surfacing through Optimized Implicit Modeling

**Student Name:** Siqi Guo

## Summary of Work to Date

### Literature Review

I have completed a comprehensive literature review examining neural implicit modeling for 3D curve surfacing, with particular focus on computational optimization techniques. The review organized into key areas: traditional geometric modeling approaches, neural implicit representations (occupancy networks, DeepSDF, SIREN), optimization strategies (Eikonal condition, Dirichlet conditions, thin-plate energy), NeuVAS and related methods, and computational acceleration techniques.

### Implementation Progress

I have implemented the baseline MLP training pipeline using the NeuVAS loss formulation. The implementation includes:

- **Framework Setup**: PyTorch-based training framework with modular components for network definition, loss computation, and optimization.

- **Loss Function Components**: Two of the three components of the NeuVAS loss have been implemented:
  1. **Eikonal condition**: ✓ Completed - Regularization term enforcing $|\nabla f| = 1$ at sample points
  2. **Dirichlet condition**: ✓ Completed - Boundary constraint ensuring $f(x) = 0$ for points on input curves
  3. **Weighted thin-plate energy**: In progress - Smoothness term requiring Hessian computation, which presents significant implementation complexity due to the complex forward/backward processes needed for second-order derivatives through the automatic differentiation system

### Visualization Implementation

I am currently implementing surface visualization tools. Meshes will be extracted from the zero-level set of the learned neural implicit function using the Marching Cubes algorithm [Lorensen and Cline 1987]. I have recently studied the Marching Cubes algorithm to understand how to properly extract surface meshes from the implicit representation for visualization and evaluation purposes.

## Analysis of Work

### First Update Goals Assessment

The original goals for the first update were:

1. **Complete comprehensive literature review**: The literature review has been completed and compiled into a comprehensive PDF document.

2. **Implement baseline MLP training pipeline using the NeuVAS loss formulation**: The baseline MLP training pipeline framework has been implemented, and two of the three NeuVAS loss components (Eikonal condition and Dirichlet condition) are complete. The weighted thin-plate energy component is in progress, as it requires complex forward/backward processes for Hessian computation. Evaluation of the partial implementation is currently in process.

3. **Profile the implementation to identify and quantify specific computational bottlenecks**: Profiling setup has been initiated, but comprehensive quantitative analysis remains to be completed. Once evaluation of the baseline implementation is complete, detailed profiling will be performed to quantify specific computational bottlenecks.

### Status Assessment

Overall, the project is **on schedule**. The literature review is complete, and the baseline implementation framework is established with two of the three loss components implemented. The remaining work for the first update goals includes completing the weighted thin-plate energy component, evaluation of the full baseline implementation, and comprehensive profiling to quantify computational bottlenecks.

## Plan for Completion

### Completion of First Update Goals (Remaining Work)

- **Complete weighted thin-plate energy implementation** (0.5 week): Finish implementing the weighted thin-plate energy component, including the complex forward/backward processes for Hessian computation through the automatic differentiation system.

### Second Update Goals

- **Complete visualization framework** (0.5 week): Finish implementing Marching Cubes algorithm for mesh extraction and create rendering pipeline for surface visualization.

- **Develop evaluation framework** (0.5 week): Implement quantitative evaluation metrics and create automated testing pipeline.

- **Complete evaluation of baseline implementation**: Finish testing the full training pipeline with all three loss components on various curve networks to ensure functionality and stability.

- **Explore and implement optimization strategies** (1 week):
  - Alternative smoothness measures
  - Weight function alternatives
