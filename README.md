# ViT-Pruning

This is the official implementation of the paper: Efficient Token Pruning in Vision Transformers Using an Attention-Based Multilayer Network.

## Abstract


Vision Transformers (ViTs) have shown promising performance in a variety of computer vision tasks, leading to their deployment in many heterogeneous scenarios. However, their training and inferences require a significant amount of computational resources. To address this challenge, researchers have proposed solutions to limit the computational efforts needed to work with ViTs. These solutions are generally based on manipulating the input of the attention layers present in a ViT to reduce the data to be processed. However, to the best of our knowledge, no approach uses only the attention matrix of the patches and can work with most ViTs without requiring fine-tuning. In this paper, we fill this gap by proposing an approach called Token Reduction via an Attention-based Multilayer network (TRAM). TRAM represents the attention layers of a ViT via a multilayer network and derives the importance of a token by computing an appropriate centrality measure. Then, it removes the least relevant tokens based on this centrality. We also report the results of testing TRAM from scratch and on pre-trained ViTs using CIFAR10, Imagenette and FashionMNIST. The results demonstrate promising efficiency gains in terms of frame per second (FPS) and gigaflops (GFlops) while maintaining near-Vanilla model accuracy. Finally, we present a qualitative analysis that provides a visual representation of the selection process performed by TRAM to reduce the computational load of ViTs.
TRAM's workflow is shown in <a href="#Multilayer creation">Figure 1</a>. Specifically, for each layer, it first creates a multilayer network based on the current and previous attention matrices obtained from the input image. Then it uses the multilayer network previously created to generate tokens score. Only the best $\kappa$ tokens will be used in the next layers. 

![Multilayer creation](Readme_imgs/Workflow.png)


The following image shows our approach applied to a subset of the Imagenet dataset.

<div style="display: flex; justify-content: center; align-items: center; margin: 0;">
  <img src="https://github.com/DavideTraini/TRAM/blob/main/Readme_imgs/ImmagineGitHub.png" style="width: 570px; height: 430px;">
</div>


## Repository structure

This repository can be directly downloaded and executed locally. The required libraries are displayed in Section [Requirements](#requirements)
The **Code** folder contains 3 subfolders: 
- **Benchmark**: contains the code for testing the efficiency of our approach against the other ones. This benchmark it's just about FPS, GFlops, and parameters and is performed both for ViT and SimpleViT architecture.
- **Pretrained**: contains the code for testing TRAM and the other approaches with fine-tuning. It also contains the notebook for visualizing the patches selected by TRAM. 
- **Scratch**: contains the code for testing TRAM and the other approaches when trained from scratch. This benchmark is done both for ViT and SimpleViT architecture


> [!NOTE]
> Some classes are replicated in different folders because:
> - their implementation varies between ViT and SimpleViT architectures;
> - their implementation varies between scratch and fine-tuning.


## Results on pretrained Models

### ViTB

Below are the performance results for a pretrained **ViTB** fine-tuned for 5 epochs and evaluated with a retention rate of 75% and 50%.


#### Performances at 75% Evaluation

| Model         | FPS ↑ | GFlops ↓  | CIFAR10 ↑ | Imagenette ↑ | FMNIST ↑ |
|---------------|-------|-----------|-----------|--------------|----------|
| Vanilla case  | 358   | 16.8808   | 95.68     | 97.43        | 93.16    |
| TRAM          | 483   | 11.8498   | **95.62** | **96.36**    | **93.50**|
| ATS           | 504   | **10.2368**| 94.48     | 95.54        | 93.08    |
| PatchMerger   | **537**| 11.4968   | 87.49     | 84.97        | 91.16    |
| TopK          | 503   | 11.8510   | 94.52     | 95.39        | 93.28    |

*Table 1: Performance of the models evaluated at 75%.*

#### Performances at 50% Evaluation

| Model         | FPS ↑ | GFlops ↓  | CIFAR10 ↑ | Imagenette ↑ | FMNIST ↑ |
|---------------|-------|-----------|-----------|--------------|----------|
| Vanilla case  | 358   | 16.8808   | 95.68     | 97.43        | 93.16    |
| TRAM          | 660   | 8.4135    | **93.61** | **94.65**    | **93.00**|
| ATS           | 588   | **7.8111**| 93.00     | 94.45        | 92.83    |
| PatchMerger   | **781**| 7.9230    | 86.28     | 87.13        | 90.95    |
| TopK          | 698   | 8.4147    | 93.15     | 94.55        | 92.72    |

*Table 2: Performance of the models evaluated at 50%.*



### ViTS


Below are the performance results for a pretrained **ViTS** fine-tuned for 5 epochs and evaluated with a retention rate of 75% and 50%.


#### Performances at 75% Evaluation

| Model         | FPS ↑  | GFlops ↓ | CIFAR10 ↑ | Imagenette ↑ | FMNIST ↑ |
|---------------|--------|----------|-----------|--------------|----------|
| Vanilla case  | 888    | 4.2577   | 96.68     | 98.27        | 93.89    |
| TRAM          | 1142   | 2.9964   | **96.38** | **97.25**    | 93.86    |
| ATS           | 1043   | **2.6106** | 95.59     | 96.74        | 93.27    |
| PatchMerger   | **1318**| 2.9087   | 87.60     | 88.89        | 92.16    |
| TopK          | 1225   | 2.9973   | 95.54     | 97.20        | **94.12**|

*Table 1: Performance of the models evaluated at 75%.*

#### Performances at 50% Evaluation

| Model         | FPS ↑  | GFlops ↓ | CIFAR10 ↑ | Imagenette ↑ | FMNIST ↑ |
|---------------|--------|----------|-----------|--------------|----------|
| Vanilla case  | 888    | 4.2577   | 96.68     | 98.27        | 93.89    |
| TRAM          | 1521   | 2.1359   | **95.64** | **95.57**    | **93.61**|
| ATS           | 1086   | **1.9859** | 94.02     | 95.31        | 93.46    |
| PatchMerger   | **1894**| 2.0136   | 87.92     | 87.80        | 92.85    |
| TopK          | 1669   | 2.1368   | 94.31     | 95.52        | 93.25    |

*Table 2: Performance of the models evaluated at 50%.*




## Requirements <a name="requirements"></a>

In our notebook we used the following libraries:
```
timm=0.9.2
einops=0.4.1
torch=2.2.2+cu121  
torchvision=0.17.2+cu121  
numpy=1.24.4  
matplotlib=3.8.2
thop=0.1.1
```


## Citation

If you use this model for your research please cite our paper.
