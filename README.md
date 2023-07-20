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

## Result Table <a name="Dataset"></a>

![alt text](https://github.com/Koreaj9u7n/LLF/blob/main/image/Result%20Table.png "Result Table")

### Dataset download links <a name="P-down"></a>
| Dataset   | Dataset download links |
| --------- | :------------------:|
| davis_train    |[Link](https://drive.google.com/file/d/1GD5RoLsOFaIvhzVRK2ikXSJl3EWW1bEu/view?usp=drive_link)|
| davis_test     |[Link](https://drive.google.com/file/d/1GWlyfLG9zSaP-OiWMQrMgbotorp0y3Ct/view?usp=drive_link)|
| kiba_train     |[Link](https://drive.google.com/file/d/1-dWKWqCqa_YKGmr6IV6uvi_CMrlNbvN7/view?usp=drive_link)|
| kiba_test      |[Link](https://drive.google.com/file/d/1lwVBGzqVvba4sdp71vOgpQnbNuGKiNyK/view?usp=drive_link)|

You can Download dataset from Google Drive for particular data(.csv).
