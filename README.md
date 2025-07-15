# Computer Vision Center (CVC)
## Graph Neural Networks for Multimodal Brain Connectivity Analysis in Multiple Sclerosis

## Abstract
Accurately predicting subject status from brain network data is a complex task that requires advanced machine learning tech-niques. In this work, we propose a comprehensive methodology and pipeline for applying supervised graph learning models, specifically Graph Neural Networks, to this task using brain network information derived from diffusion tensor imaging, gray matter and resting-state func-tional MRI adjacency matrices. Our approach includes a graph pruning step to retain the most relevant edges while preserving crucial informa-tion, the generation of node features to enhance graph representations, the creation of synthetic data to balance the dataset and improve train-ing, and the design and training of GNN models for both multi-class and binary classification tasks. Experimental results in a cohort of peo-ple with multiple sclerosis and healthy volunteers demonstrate that our methodology effectively captures meaningful patterns in brain graphs, leading to improved classification performance.

## Reference
Subirà-Cribillers, M., Solé-Casaramona, J., Lladós, J., Casas-Roma, J. (2025). **Graph Neural Networks for Multimodal Brain Connectivity Analysis in Multiple Sclerosis**. In: Brun, L., Carletti, V., Bougleux, S., Gaüzère, B. (eds) *Graph-Based Representations in Pattern Recognition*. GbRPR 2025. Lecture Notes in Computer Science, vol 15727. Springer, Cham. [https://doi.org/10.1007/978-3-031-94139-9_9](https://doi.org/10.1007/978-3-031-94139-9_9)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/jcasasr/CVC_BrainNetsPipeline
    ```
2. Navigate to the project directory:
    ```bash
    cd CVC_BrainNetsPipeline
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Files Overview

- `data_duplication.py`: This script is used to augment the dataset by generating additional data points. The goal is to increase the dataset size for better generalization of the model.
- `graph_saving.py`: This script processes raw data to generate graph representations. It includes methods for thresholding, sparsification, and salience computation to extract meaningful graph structures from the data.
- `embeddings.py`: This script generates node embeddings from the created graphs. It extracts graph features and saves them in a structured format for later use in model training.
- `data_combination.py`: This script combines all the generated embeddings into a single dataset. The output file serves as input to the GNN model.
- `final_model.py`: This script implements the final Graph Neural Network (GNN) model. It loads the preprocessed data, trains a GNN model using PyTorch Geometric, and evaluates the classification performance.
