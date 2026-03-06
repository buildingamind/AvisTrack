AvisTrack 🐣
Background & Motivation
In our lab, our primary animal model is the chick. While our subject remains constant, our experimental environments and analytical needs are highly variable. From simple open-field tests to complex social interaction paradigms in various custom chambers, the demands on our computer vision systems change.

We frequently transition between:

Real-time vs. Offline analysis: Balancing the need for low-latency, on-the-fly interventions with high-fidelity, frame-by-frame post-hoc analysis.

Single vs. Multi-bird tracking: Scaling from tracking a single individual to managing complex occlusions and ID assignments in groups.

Macro vs. Micro movements: Switching between simple spatial positioning (bounding boxes) and intricate ethological observations like pecking or preening (keypoint extraction).

The Problem
Because our setups vary so drastically, we rely on a diverse arsenal of machine learning models—YOLO, DeepLabCut (DLC), and Vision Transformers (ViT).

Historically, every new experimental paradigm required us to write a new tracking script from scratch. Every time we fine-tuned a new YOLO model for a specific lighting condition, or trained a DLC network to recognize a specific body part, we found ourselves tangled in boilerplate code—rewriting video reading logic, re-implementing coordinate smoothing, and hardcoding model paths. Comparing the performance of multiple fine-tuned models became an administrative nightmare.

Our Philosophy
Researchers should spend time analyzing behavior, not rewriting video processing loops.

AvisTrack was built to be a permanent, stable foundation for our ever-changing experiments. It treats the core mechanics of tracking (reading frames, handling outputs, smoothing data) as a constant, while treating the experimental variables (the chamber, the model, the weights) as easily swappable configurations.

Whether we are deploying a newly fine-tuned YOLO model for a quick multi-bird pilot study, or loading a heavy ViT for rigorous offline pose estimation, AvisTrack provides the standardized testing ground to make it happen effortlessly.
