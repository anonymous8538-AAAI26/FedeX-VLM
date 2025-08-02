# AAAI26-FedeX-VLM

Fedex-VLM is a novel FL framework tailored for VLMs that addresses heterogeneity by leveraging client-specific knowledge levels.
Specifically, we enhance the weighting mechanism for VLMs in FL by applying a weighted smoothing to client weights and balancing client's contributions, while amplifying the influence of clients with model qualities, which possess a broader answer space in VLMs and exhibit better training progress and convergence.
Furthermore, we incorporate explainable AI (XAI) techniques to improve interpretability on complex multi-modal data and decentralized FL settings, fostering better understanding and trust in the decision-making process.

This is an implementation for 'FedeX-VLM: Towards Knowledge Harmonization in Heterogeneous Federated Learning with Explainable Vision-Language Models' (under review).

---
## Approach
<img width="850" height="400" alt="image" src="https://github.com/user-attachments/assets/9c7e6f1c-a8b2-43bd-9d34-dc9a95c39182" />


## Datasets
The Datasets/ directory contains our knowledge-level heterogeneous splits of the VQA v1 and VQA v2 datasets, which are used in our experiments.

Details of each dataset split are summarized in the table below:

<img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/403c2a5b-29ad-4e11-b67e-76fdf864956d" />

<img width="600" height="250" alt="image" src="https://github.com/user-attachments/assets/ae7603cb-23ad-4a7c-bf99-a99c18a6578d" />

The structure of the dataset is as follows:

```
Datasets/
├── VQA_v1/
│   ├── 2clients/
│   │   ├── X_client1.csv
│   │   ├── y_client1.csv
│   │   ├── X_client2.csv
│   │   └── y_client2.csv
│   ├── 3clients/
│   │   ├── X_client1.csv
│   │   ├── y_client1.csv
│   │   ├── X_client2.csv
│   │   ├── y_client2.csv
│   │   ├── X_client3.csv
│   │   └── y_client3.csv
│   └── 4clients/
│       ├── ...
├── VQA_v2/
│   ├── 2clients/
│   │   ├── X_client1.csv
│   │   ├── y_client1.csv
│   │   ├── X_client2.csv
│   │   └── y_client2.csv
│   ├── 3clients/
│   │   ├── ...
│   └── 4clients/
│       ├── ...
```

## About the Heterogeneous Split
We define our dataset as:

D = {(q_img, a) | q_img ∈ Q_img} where Q_img represents the full set of all image-based questions,
and a denotes the corresponding set of possible answers for each question q_img.

From the full question set Q_img, we extract a subset of unique questions, denoted as {q_img,i} where i ∈ [1, N], where each q_img,i represents a unique question i associated with an image.

To create answer space heterogeneity, we first sort all unique question {q_img,i }where i ∈ [1, N] in the ascending order based on the number of possible answers (i.e., small to large).


## Pre-trained model preperation

| Pre-trained Backbone | Link | 
| --- | --- | 
|  ViT-B | [Link](https://huggingface.co/docs/transformers/model_doc/vit) | 
| Swin-B  | [Link](https://huggingface.co/docs/transformers/model_doc/swin) | 
|  BERT | [Link](https://huggingface.co/docs/transformers/model_doc/bert)  | 
|  T5 | [Link](https://huggingface.co/docs/transformers/model_doc/t5) | 
