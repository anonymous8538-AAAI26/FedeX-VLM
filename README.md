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

<img width="600" height="270" alt="image" src="https://github.com/user-attachments/assets/403c2a5b-29ad-4e11-b67e-76fdf864956d" />

<img width="600" height="270" alt="image" src="https://github.com/user-attachments/assets/ae7603cb-23ad-4a7c-bf99-a99c18a6578d" />

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


## STEP 1) Prepare Pre-trained Vision and Language Models

We use standard vision and language models as backbones. Download the pre-trained weights from the links below:

| Pre-trained Backbone | Link | 
| --- | --- | 
|  ViT-B | [Link](https://huggingface.co/docs/transformers/model_doc/vit) | 
| Swin-B  | [Link](https://huggingface.co/docs/transformers/model_doc/swin) | 
|  BERT | [Link](https://huggingface.co/docs/transformers/model_doc/bert)  | 
|  T5 | [Link](https://huggingface.co/docs/transformers/model_doc/t5) | 

These models serve as the backbone for our multimodal baseline architecture.

## STEP 2) Knowledge-level Data Splits for Federated VQA
Our dataset consists of knowledge-level heterogeneous splits based on the VQA v1 and VQA v2 datasets.

Each split is organized by number of clients:

```
from dataset_loader import load_split

X1_train, y2_train = load_split("Datasets/VQA_v1/2clients/X_client1.csv", 
                              "Datasets/VQA_v1/2clients/y_client1.csv")
...
```

## STEP 3) Set Hyperparameters
Before training, set appropriate training configurations. Example:

```yaml
learning_rate: 1e-5
epsilon: 1e-8
batch_size: 32
round_num: 50
alpha: 0.7
dataset: "VQA_v1"
num_clients: 5
normalize: "z_score"
eval_folder: "saved_model/z_score_alpha0.7_WeightedVQA_v2FEDUlen_clientall5_vit_bert_all_concat_bert_transformerepcoh_50soft_max"
...
```

## STEP 4) Train the Model
Run the training!
```python
python main.py --config config.yaml
```

## STEP 5) Evaluate
To evaluate the trained model:

```python
python evaluate.py --config config.yaml
```
Make sure to set the 'eval_folder' parameter in your config.yaml file to the directory containing the the model you want to evaluate on.

