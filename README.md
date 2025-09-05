

<h1 align="center"> Machine Learning Notes</h1>
<p align="center">
  A clear, concise guide for curious minds transitioning into Machine Learning from a theoretical CS background.
</p>

---

### About These Notes

I'm currently pursuing a PhD in theoretical computer science. As I started exploring machine learning, I often struggled to find resources that were both clear and comprehensive. Many notes were either too informal to be useful for in depth understanding, or too dense to absorb quickly.

This repository is my attempt to create what I wish I had:
-  Concise, intuitive explanations that capture the core ideas  
-  Formal definitions and equations included for completeness, but never the whole picture  
-  Visual aids, examples, and code snippets where they clarify the intuition  

The goal is not to oversimplify, but to explain things in a way that makes the underlying structure transparent especially for others making the same transition.


### Current State

The notes are currently in the beginning stages and growing (albeit slower than I would like due to many concurrent projects). For an idea of what kind of explanations I am aiming for look at my notes on [HDBSCAN](notes/unsupervised_learning_models.ipynb#iii--hierarchical-density-based-spatial-clustering-of-applications-with-noise-hdbscan). I will add figures as my progress expands. For complementary code see my implementation in [clustering.py](unsupervised_learning/HDBSCAN.py).

---

###  Table of Contents
Here we provide a skeleton of what we intend to include in the near future. For current files we include links.

| Section | Description |
|--------|-------------|
| Machine Learning Paradigms | Overview of supervised, unsupervised, self-supervised learning, reinforcement learning, and generative paradigms | #[Machine Learning Paradigms](notes/ml_paradigms.ipynb)
| Supervised Learning | Covers classification and regression tasks, common model families and evaluation metrics | #(notes/supervised_learning_models.ipynb)
| [ Unsupervised Learning](notes/unsupervised_learning_models.ipynb) | Clustering, dimensionality reduction, representation learning, self-supervised learning, evaluation of methods|
| Self-Supervised Learning | Overview of contrastive, generative, and pretext-task-based approaches to learning representations without labels (e.g., SimCLR, BERT, BYOL) | #[Self-Supervised Learning](notes/self_supervised_learning.ipynb)
| Loss Functions | Common loss functions grouped by task: regression, classification, probabilistic models. | #[Loss Functions](notes/loss_functions.ipynb)
| Data Preprocessing | Practical techniques for preparing raw data: scaling, encoding, handling missing values, and preventing data leakage. Includes their effects on optimization and model behavior | #[Data Preprocessing](notes/data_preprocessing.ipynb)
| Overfitting and Generalization | Explanation of overfitting, underfitting, bias-variance tradeoff, and generalization bounds |
| Optimization in ML | Gradient-based and gradient-free methods, convex vs non-convex problems, and optimization landscapes |
| Generative AI | Overview of models that learn to generate data: VAEs, GANs, Diffusion models, autoregressive models |
| Reinforcement Learning | Agents learning through trial and error: MDPs, policies, rewards, and exploration strategies |
| Interpretability and Explainability | Tools and concepts for understanding model predictions: feature importance, SHAP, LIME, saliency maps, counterfactuals | #[Interpretability and Explainability](notes/interpretability.ipynb)

