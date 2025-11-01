---
title: "First Project Update"
date: 2025-10-30
tags: ["geometric modeling", "computer graphics", "implicit modeling", "neural networks", "optimization", "project update"]
author: "Siqi Guo"
description: "First project update for accelerating training in 3D curve surfacing" 
summary: "First progress update covering literature review completion and initial implementation work." 
showToc: true
disableAnchoredHeadings: false
---

# First Project Update

**Student Name:** Siqi Guo

**Project Title:** Accelerating Training for 3D Curve Surfacing through Optimized Implicit Modeling

## Summary of Work to Date

### Literature Review (Completed)

I have completed a comprehensive literature review examining neural implicit modeling for 3D curve surfacing, with focus on computational optimization techniques. The review covers traditional geometric modeling approaches, neural implicit representations (occupancy networks, DeepSDF, SIREN), optimization strategies for surface reconstruction, NeuVAS and related methods, and computational acceleration techniques. This work has been compiled into a comprehensive PDF document.

### Baseline Implementation (Partially Completed)

I have implemented the core training framework in PyTorch with modular components for network definition, loss computation, and optimization. Two of the three NeuVAS loss components are fully functional:

1. **Eikonal condition** (completed): Regularization term enforcing unit gradient magnitude at sample points
2. **Dirichlet condition** (completed): Boundary constraint ensuring the implicit function equals zero on input curves
3. **Weighted thin-plate energy** (in progress): Smoothness term requiring Hessian computation through automatic differentiation

The thin-plate energy component has proven more complex than anticipated due to the need for second-order derivatives through the autodiff system. I am currently working through the implementation of the forward/backward passes for this component.

### Visualization Tools (In Progress)

I have studied the Marching Cubes algorithm [Lorensen and Cline 1987] for extracting surface meshes from the learned implicit function's zero-level set. Implementation is underway but not yet complete.

## Analysis of Work

My original goals for this first update were:

1. **Complete comprehensive literature review** - ✓ Achieved
2. **Implement baseline MLP training pipeline using NeuVAS loss** - Partially achieved (2 of 3 loss components complete)
3. **Profile implementation to identify computational bottlenecks** - Not yet achieved

The literature review proceeded as planned and provided a solid foundation for understanding the problem space. The baseline implementation has taken longer than expected, primarily due to the complexity of implementing the weighted thin-plate energy term with its second-order derivative requirements. I initially underestimated the technical challenges involved in computing Hessians efficiently through PyTorch's automatic differentiation system.

Profiling has not yet begun, as I decided it would be more productive to first complete a fully functional baseline before conducting detailed performance analysis. This represents a minor adjustment to my timeline but does not fundamentally impact the project goals.

**Status:** I consider the project to be on schedule overall, though with some internal reordering of tasks. The core framework is solid, and completing the remaining loss component should not require more than another few days of focused work.

## Plan for Completion

### Immediate Goals (Next 1-2 Weeks)

1. **Complete weighted thin-plate energy implementation** (3-4 days): Finish the Hessian computation and integrate it into the loss function. Test on simple curve networks to verify correctness.

2. **Complete visualization framework** (2-3 days): Finish Marching Cubes implementation and create a rendering pipeline for visualizing reconstructed surfaces.

3. **Initial baseline evaluation** (2-3 days): Run the complete training pipeline on several test cases to verify functionality and stability. Collect initial qualitative results.

### Second Update Goals (Next 2-3 Weeks)

1. **Comprehensive profiling** (3-4 days): Use PyTorch profiler and other tools to identify and quantify computational bottlenecks. Generate detailed performance metrics for each component of the training pipeline.

2. **Develop quantitative evaluation metrics** (2 days): Implement metrics for measuring reconstruction quality and create an automated testing framework.

3. **Explore optimization strategies** (1-1.5 weeks): Based on profiling results, investigate and implement optimization approaches such as:
   - Alternative network architectures or activation functions
   - Sampling strategies for loss computation
   - Potential acceleration through vectorization or GPU optimization
   - Alternative formulations for the smoothness term

### Timeline for Remainder of Project

I anticipate remaining on schedule for the rest of the project. The main risk is that profiling may reveal optimization challenges that are more difficult to address than expected. However, the literature review has already identified several promising acceleration techniques, so I have multiple approaches to explore if initial optimization attempts prove insufficient.

If necessary, I can adjust the scope by limiting the number of optimization strategies explored in depth, focusing on the most promising 2-3 approaches rather than attempting a comprehensive comparison of all possible techniques. The core deliverable—a working implementation with demonstrated speedup over the baseline—remains feasible within the project timeline.