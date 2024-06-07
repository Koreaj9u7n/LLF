# LLF (Leveraging Local Features) 

## Abstract <a name="Abstract"></a>
Binding affinity prediction has been considered as a fundamental task in drug discovery.
Although there has been much effort to predict the binding affinity of a pair of a protein se-
quence and a drug, which has provided valuable insights on developing machine-learning-
based predictors, the components or layers of the prediction models proposed by the prior
work are designed not to preserve comprehensive features of local structures of a drug and
a sequence of target protein. In this paper, we propose a deep learning model that concen-
trates more on local structures of both a drug and a sequence of target protein. To this end,
the proposed model employs two modules, R-CNN and R-GCN, each of which is responsible
for extracting the comprehensive features from subsequences of a target protein sequence and
subgraph of a drug, respectively. With multiple streams with different numbers of layers, both
modules not only computes the comprehensive features with multiple CNN and GCN layers,
but also preserve the local features computed by a single layer. Based on the evaluation with
two popular datasets, Davis and KIBA, we demonstrates that the proposed model shows the
competitive performance on both the datasets and keeping local features can play significant
roles of binding affinity prediction.

## Model Figure <a name="Model Figure"></a>

![alt text](https://github.com/Koreaj9u7n/LLF/blob/main/image/Figure.png "LLF")

## Dataset <a name="Dataset"></a>

![alt text](https://github.com/Koreaj9u7n/LLF/blob/main/image/dataset.png "Dataset")

## Result Table <a name="Result"></a>

![alt text](https://github.com/Koreaj9u7n/LLF/blob/main/image/Result%20Table.png "Result Table")

## The components of our model <a name="Environment"></a>

<B>data_creation.py :</B> converting datasets into PyTorch Geometric format. It's used for data preprocessing in Binding affinity prediction tasks.

<B>emetrics.py :</B> calculating various evaluation metrics such as Concordance Index, Mean Squared Error, R-squared, Pearson Correlation, and Area Under the Precision-Recall Curve (AUPR).

<B>gcn.py :</B> consisting of graph convolutional layers for processing molecular graphs (SMILES), convolutional layers for processing protein sequences, and fully connected layers for combining the features from both branches and making predictions.

<B>training.py :</B> training and testing a model on a given dataset using PyTorch.

<B>utils.py :</B> It preprocesses the input data (SMILES, target sequences, and affinities) into a format suitable for model training.

## How to use our codes <a name="Environment"></a>
All the requirements are listed in the requirements.txt file

<B>Step 1:</B> Download the file that matches the dataset you want to use from the Dataset download links.

<B>Step 2:</B> Use the (data_creation.py) file for data preprocessing.

<B>Step 3:</B> Use (the training.py) file to train the model using the provided data.

<B>Step 4:</B> You can check the model performance for a specific epoch using the score values specified in (emetrics.py).


### Dataset download links <a name="P-down"></a>
| Dataset   | Dataset download links |
| --------- | :------------------:|
| davis_train    |[Link](https://drive.google.com/file/d/1GD5RoLsOFaIvhzVRK2ikXSJl3EWW1bEu/view?usp=drive_link)|
| davis_test     |[Link](https://drive.google.com/file/d/1GWlyfLG9zSaP-OiWMQrMgbotorp0y3Ct/view?usp=drive_link)|
| kiba_train     |[Link](https://drive.google.com/file/d/1-dWKWqCqa_YKGmr6IV6uvi_CMrlNbvN7/view?usp=drive_link)|
| kiba_test      |[Link](https://drive.google.com/file/d/1lwVBGzqVvba4sdp71vOgpQnbNuGKiNyK/view?usp=drive_link)|

You can Download dataset from Google Drive for particular data(.csv).
