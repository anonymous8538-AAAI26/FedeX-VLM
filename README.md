# AAAI26-FedeX-VLM

Fedex-VLM is a novel FL framework tailored for VLMs that addresses heterogeneity by leveraging client-specific knowledge levels.
Specifically, we enhance the weighting mechanism for VLMs in FL by applying a weighted smoothing to client weights and balancing client's contributions, while amplifying the influence of clients with model qualities, which possess a broader answer space in VLMs and exhibit better training progress and convergence.
Furthermore, we incorporate explainable AI (XAI) techniques to improve interpretability on complex multi-modal data and decentralized FL settings, fostering better understanding and trust in the decision-making process.

This is an implementation for 'FedeX-VLM: Towards Knowledge Harmonization in Heterogeneous Federated Learning with Explainable Vision-Language Models' (under review).

---
## Approach
<img width="850" height="400" alt="image" src="https://github.com/user-attachments/assets/9c7e6f1c-a8b2-43bd-9d34-dc9a95c39182" />


## Ready for data
Please download the VQA v1 and VQA v2 datasets manually from the official VQA website and place them in the appropriate directory.


## Pre-trained model preperation

| Pre-trained Backbone | Link | 
| --- | --- | 
|  ViT-B | 행1 열2 | 
| Swin-B  | 행2 열2 | 
|  BERT | 행3 열2 | 
|  T5 | 행3 열2 | 
