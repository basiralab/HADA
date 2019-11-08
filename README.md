# HADA

HADA (Hiearachical Adversarial Domain Alignment) for brain graph prediction code, created by Alaa Bessadok. 
Please contact alaa.bessadok@gmail.com for inquiries. Thanks.

![HADA pipeline](http://basira-lab.com/hada_fig/)

# Introduction

This work is published in MICCAI 2019 and it is selected as oral presentation at the “PRedictive Intelligence in MEdicine” (PRIME) workshop, Shenzhen, China. 

Hierarchical Adversarial Domain Alignment (HADA) is a generative adversarial network (GAN) based framework for predicting a brain graph from a source graph using a hierarchical domain alignment. Our HADA framework comprises three key steps (1) hierarchical domain alignment, (2) target graph prediction and, (3) disease classification. We have evaluated our method on ABIDE dataset (http://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html). Detailed information can be found in the original paper (https://link.springer.com/chapter/10.1007/978-3-030-32281-6_11) and our research paper video on the BASIRA Lab YouTube channel (https://www.youtube.com/watch?v=OJOtLy9Xd34&t=2s). In this repository, we release the code for training and testing HADA on a simulated dataset with paired source and target graphs drawn from two different distributions.

# Installation
The code has been tested with Python 2.7, Anaconda2-5.3.0 and TensorFlow 1.5 on Ubuntu 16.04. GPU is not needed to run the code. You also need some of the following Python packages, which can be installed via pip:

Tensorflow
Numpy
scikit-learn 
Scipy
SIMLR

# Run from Jupyter Notebook
We provide a demo code for the usage of HADA for target graph prediction from a source graph. In test_graph_prediction.py we run HADA on a simulated dataset with 150 subjects and each has 595 features (very similar to the connectomic data we used in our paper).

run HADA.py

# YouTube videos to install and run the code and understand how the method works

To install and run HADA, check the following YouTube video:

To learn about how HADA works, check the following YouTube video:
https://www.youtube.com/watch?v=OJOtLy9Xd34&t=10s

# Data
In order to use our framework, you need to provide:
•	a SourceGraph and a TargetGraph matrices each has a size of (n * m). We denote n the total number of subjects in the dataset and m the number of features.
•	a Label list denoting the label of each subject in the dataset such as healthy or disordered.

# Related references

Adversarially Regularized Graph Autoencoder (ARGA): 
Pan, S., Hu, R., Long, G., Jiang, J., Yao, L., Zhang, C.: Adversarially regularized graph autoencoder. [https://arxiv.org/abs/1802.04407] (2018) [https://github.com/Ruiqi-Hu/ARGA].

Single‐cell Interpretation via Multi‐kernel LeaRning (SIMLR):
Wang, B., Ramazzotti, D., De Sano, L., Zhu, J., Pierson, E., Batzoglou, S.: SIMLR: a tool for large-scale single-cell analysis by multi-kernel learning. [https://www.biorxiv.org/content/10.1101/052225v3] (2017) [https://github.com/bowang87/SIMLR_PY].

# Please cite the following paper when using HADA

@inproceedings{bessadok2019hierarchical,<br/>
  title={Hierarchical Adversarial Connectomic Domain Alignment for Target Brain Graph Prediction and Classification from a Source Graph},<br/>
  author={Bessadok, Alaa and Mahjoub, Mohamed Ali and Rekik, Islem},<br/>
  booktitle={International Workshop on PRedictive Intelligence In MEdicine},<br/>
  pages={105--114},<br/>
  year={2019},<br/>
  organization={Springer}<br/>
}

