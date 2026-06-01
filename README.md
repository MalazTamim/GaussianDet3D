# GaussianDet3D: Bridging Gaussian Splatting and Sparse LiDAR Detection for Multi-View 3D Object Detection

**[Malaz Tamim](https://malaztamim.com/)<sup>1,3*</sup>, [Wenzhao Zheng](https://wzzheng.net/)<sup>2</sup>, [Johannes Meier](https://cvg.cit.tum.de/members/mejo)<sup>1,3,4</sup>, [Daniel Cremers](https://cvg.cit.tum.de/members/cremers)<sup>1,3</sup>, [Kurt Keutzer](https://www2.eecs.berkeley.edu/Faculty/Homepages/keutzer.html)<sup>2</sup>**

<sup>1</sup>Technical University of Munich &nbsp; <sup>2</sup>UC Berkeley &nbsp; <sup>3</sup>Munich Center for Machine Learning &nbsp; <sup>4</sup>DeepScenario



**DriveX Workshop @ CVPR 2026 — Oral Presentation**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://drivex-workshop.github.io/cvpr2026/static/pdf/30_GaussianDet3D_Bridging_Gaus.pdf)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://malaztamim.com/GaussianDet3D/)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey)](#)

---

> 🚧 **Code coming soon!** We are preparing the codebase for release. Stay tuned by watching/starring this repo.

---

## Overview

We present **GaussianDet3D**, the first method to apply 3D Gaussian Splatting from multi-view images to 3D object detection in autonomous driving. Gaussian primitives are treated as a pseudo-LiDAR point cloud fed directly into a sparse LiDAR detector, encoding geometry, orientation, opacity, and per-class semantics. Temporal aggregation across frames enables precise velocity estimation without explicit tracking. On the **nuScenes benchmark**, GaussianDet3D achieves **state-of-the-art translation error and velocity error** among all camera-based methods, outperforming BEVFormer by **8.1%** and **13.1%** respectively.

![GaussianDet3D Pipeline](fig/method.png)

Multi-view images are encoded (ResNet-101-DCN + FPN), lifted into 3D Gaussian primitives via depth estimation, refined by the Gaussian Encoder (sparse 3D conv + deformable cross-attention), and passed as a pseudo-LiDAR point cloud to FSD V2 for 3D bounding box prediction.

## Citation

Citation information will be available upon arXiv publication.

## Acknowledgements

The project page template is borrowed from [Nerfies](https://github.com/nerfies/nerfies.github.io).
