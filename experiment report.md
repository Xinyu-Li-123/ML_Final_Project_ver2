# Experiment and Report

## Introduction

### Goal

Do style transfer on ligne art style.

### Motivation





### Model



### Key Idea

* Feature Visualization
* 



## Problem 01: How to preserve color?

source:

 

## Problem 02: Other Backbones?

VGG seems to be a common choice.

Maybe because it can't catch non-robust features (compared to ResNet)



## Problem 03: Artifacts?

#### Types of Artifacts

* Overlapping between content and style
* Checkerboard
* Distorted edges

### Possible reason

* Focus on non-robust features

### Solution

To sum up, I want to weaken / eliminate non-robust features

* Use average pooling instead of max pooling
* Add regularization to content / style loss

## Interesting Findings

### Checkerboard Artifacts

### Same content and style with random initialized network

