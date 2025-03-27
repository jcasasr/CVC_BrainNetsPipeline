# Computer Vision Center (CVC)
## Graph Neural Networks for Multimodal Brain Connectivity Analysis in Multiple Sclerosis
### Internship Merlès Subirà i Jan Solé

De moment el readme s'ha de fer. Afegeixo un exemple d'estructura bàsica de fitxer readme.

## Description
Accurately predicting subject status from brain network data
is a complex task that requires advanced machine learning techniques.
In this work, we propose a comprehensive methodology and pipeline for
applying supervised graph learning models, specifically Graph Neural
Networks, to this task using brain network information derived from
diffusion tensor imaging, gray matter and resting-state functional MRI
adjacency matrices. 

Our approach includes a graph pruning step to retain the most relevant edges while preserving crucial information, the
generation of node embeddings to enhance graph representations, the
creation of synthetic data to balance the dataset and improve training,
and the design and training of GNN models for both multi-class and binary classification tasks. Experimental results in a cohort of people with
multiple sclerosis and healthy volunteers demonstrate that our methodology effectively captures meaningful patterns in brain graphs, leading
to improved classification performance.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/username/repository.git
    ```
2. Navigate to the project directory:
    ```bash
    cd repository
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Files Overview

### `data_duplication.py`
This script is used to augment the dataset by generating additional data points. The goal is to increase the dataset size for better generalization of the model.

### `graph_saving.py`
This script processes raw data to generate graph representations. It includes methods for thresholding, sparsification, and salience computation to extract meaningful graph structures from the data.

### `embeddings.py`
This script generates node embeddings from the created graphs. It extracts graph features and saves them in a structured format for later use in model training.

### `data_combination.py`
This script combines all the generated embeddings into a single dataset. The output file serves as input to the GNN model.

### `final_model.py`
This script implements the final Graph Neural Network (GNN) model. It loads the preprocessed data, trains a GNN model using PyTorch Geometric, and evaluates the classification performance.


## Usage
Example of how to use the project:
```python
import project

project.do_something()
```
