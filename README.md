# Convolutional Autoencoder with Gamma Correction Module (GAE) for Automatic Image Colorization

## Computer Vision 2023/24
**Student:** Angelo Gianfelice 1851260

---

## 📌 Project Overview
This project presents a **Convolutional Autoencoder with a gamma correction module (GAE)** for automatic image colorization. The approach leverages deep learning techniques to enhance colorization results, improve feature extraction, and achieve visually appealing outputs.

---

## 📋 Outline
- Introduction
- Related Works
- Proposed Methods
- Datasets and Metrics
- Implementation Details
- Experimental Results
- Conclusion and Future Works

---

## 🎨 Deep Learning for Image Colorization
Deep learning-based colorization methods improve various computer vision tasks by enhancing:
- **Historical and cultural restoration**
- **Medical imaging enhancement**
- **Object detection & scene understanding**

![alb](https://github.com/user-attachments/assets/493614be-6953-48b1-843d-49094a3a4d0d)
---

## 📖 Related Works
### **Early Approaches**
- Levin et al. (2004): Scribble-based optimization
- Irony et al. (2005): Image retrieval for color transfer

### **CNN-Based Methods**
- Zhang et al. (2016): End-to-end CNN with classification loss
- Iizuka et al. (2016): Global-local context model

### **GAN-Based Colorization**
- Nazeri et al. (2018): Adversarial learning for vibrant colors
- Zhang et al. (2017): User-guided GAN for interactive colorization

### **Transformer & Diffusion Models**
- Kumar et al. (2021): Colorization Transformer
- Liu et al. (2023): Diffusion models for semantic colorization

---


## 🏗️ Proposed Architecture (GAE)
The proposed model follows an **autoencoder-based approach** with gamma correction for enhanced colorization:
- **Autoencoder:** Extracts and reconstructs features.
- **Gamma Correction Module:** Adjusts contrast and saturation.
- **Color Space Conversion:** RGB → LAB → RGB transformation.
![model](https://github.com/user-attachments/assets/e83a7f4b-42e6-4f30-9710-bb7cff718a25)

### **LAB Color Space**
- L*: Lightness (0 = black, 100 = white)
- A*: Green–Red axis (-128 = green, 127 = red)
- B*: Blue–Yellow axis (-128 = blue, 127 = yellow)
- Stores color information in only two channels (ab), making it suitable for colorization tasks.
  
<img src="https://github.com/user-attachments/assets/2cbda31e-e9d2-40bd-9739-dc18e3215f14" width="300" height="300">

---


## ⚙️ Model Implementation
- **Symmetric 10-layer convolutional autoencoder**
- **Batch normalization after every encoder layer**
- **Upsampling through deconvolution in the decoder**
- **Tanh final activation function** to predict a*b* channels (normalized in [-1,1])
![adad](https://github.com/user-attachments/assets/faf82944-9b60-4691-ac81-db4d9702beb1)

### **Gamma Correction Module**
- Addresses the issue of dull and unsaturated colors by applying a **learnable gamma correction** operator:
  
  $$ X_{out} = X_{in}^{\gamma} $$
  
- Enhances color vibrancy and contrast dynamically.

---

## 📊 Baseline Model
A baseline approach using **ResNet-18** as a fixed feature extractor:
- Features are fed into a **decoder-only architecture**.
- **Faster training but worse results** since only the decoder is trained.
![resnet](https://github.com/user-attachments/assets/539fc8f0-ac33-4486-9979-c60a2ab30596)

---

## 📂 Dataset
- **Subset of the Places dataset** (~10M images)

  ![places](https://github.com/user-attachments/assets/043ae7ab-e151-4111-852d-8a4a87b93714)

- **30K images** used for training/testing (70-15-15 split)
- **Data augmentation**: random rotation, flipping, cropping
- **Image size**: Resized to **224x224**

---

## 📈 Evaluation Metrics
### **Pixel-wise Metrics**
- **MSE (Mean Squared Error):** Measures pixel-wise difference.
- **PSNR (Peak Signal-to-Noise Ratio):** Evaluates image quality.

### **Perceptual Metrics**
- **SSIM (Structural Similarity Index):** Assesses structural similarity.
- **ΔE (Color Difference Measure):** Quantifies perceptual color differences.

---

## 🔬 Experimental Results
| Model | MSE (↓) | PSNR (↑) | SSIM (↑) | ΔE (↓) |
|--------|----------|----------|----------|----------|
| **Baseline** | 0.0063 | 23.7471 | 0.9189 | 13.9822 |
| **GAE (Proposed)** | 0.0057 | 24.5263 | 0.9408 | 12.9247 |

- **Test set:** 4500 images
- **GAE outperforms the baseline in all metrics**
  
  ![res](https://github.com/user-attachments/assets/8f6aebc3-92e3-43e6-815e-11dd8db2c4dc)

---

## 🖼️ More Experiments
- **Model trained on different datasets**:

| **4,000+ nature/landscape images** | **5K subset of CelebA dataset** |
|------------------------|----------------------|
| ![land](https://github.com/user-attachments/assets/b54570a5-be03-4c66-b282-565426f05b98)| ![faces](https://github.com/user-attachments/assets/6bd03df5-c3bb-48d1-84cd-76a793e72794) |

- **Historical Photograph Colorization:**
  
  ![old](https://github.com/user-attachments/assets/f46a3208-5eff-496d-b195-d0c64d3559a4)
  
  - No ground truth available, but results appear **promising**.

---

## 🔮 Conclusion & Future Work
The proposed model successfully colorizes grayscale images naturally and vibrantly while maintaining **lightweight performance**. Future improvements include:
- **Expanding training data** for better generalization
- **Using perceptual loss** for more realistic colors
- **Fine-tuning the model** for specific applications (e.g., medical imaging, historical photos)
- **Extending the model** for **video colorization**

---

## 📚 References
- Places: A 10 million Image Database for Scene Recognition B. Zhou, A. Lapedriza, A. Khosla, A. Oliva, and A. Torralba. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017
- Levin A., Lischinski D., Weiss Y. (2004). Colorization using optimization. In ACM SIGGRAPH 2004 Papers (pp. 689–694).
- Irony, R., Cohen-Or, D., & Lischinski, D. (2005). Colorization by example. Proceedings of the 16th Eurographics Conference on Rendering Techniques, 201–210. https://www.cs.tau.ac.il/~dcor/online_papers/papers/colorization05.pdf
- Zhang, R., Isola, P., & Efros, A. A. (2016). Colorful image colorization. European Conference on Computer Vision (ECCV), 649–666. https://doi.org/10.1007/978-3-319-46487-9_40
- Iizuka, S., Simo-Serra, E., & Ishikawa, H. (2016). Let there be color! Joint end-to-end learning of global and local image priors for automatic image colorization with simultaneous classification. ACM Transactions on Graphics (TOG), 35(4), 1–11. https://doi.org/10.1145/2897824.2925974
- Nazeri, K., Ng, E., & Ebrahimi, M. (2018). Image colorization using generative adversarial networks. arXiv preprint arXiv:1803.05400. https://arxiv.org/abs/1803.05400
- Zhang, R., Zhu, J. Y., Isola, P., Geng, X., Lin, A. S., Yu, T., & Efros, A. A. (2017). Real-time user-guided image colorization with learned deep priors. https://doi.org/10.48550/arXiv.1705.02999
- Kumar, M., Weissenborn, D., & Kalchbrenner, N. (2021). Colorization transformer. International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2102.04432
- Liu, H., Xing, J., Xie, M., Li, C., & Wong, T.-T. (2023). Improved diffusion-based image colorization via piggybacked models. arXiv preprint arXiv:2304.11105. https://arxiv.org/abs/2304.11105

---

## Running the Code

Follow these steps to install dependencies and run the project.

### 1. Install Dependencies
Ensure you have Python installed (preferably Python 3.x). Then, install the required dependencies using:

```sh
pip install -r requirements.txt
```

### 2. Run the Project
After installing the dependencies, you can run the main script with 2 required and one optional parameter:
- **mode**, which describe in which mode you want to run the model [train,test or predict]
- **model**, which descibe which model to use [model1 for GAE, model2 for baseline]
- [optional] **image_path**, path to image you want to colorize (only needed for predict mode)
  
```sh
python main.py --mode=[chosen model] --model=[chosen model] --image_path=[/path/to/your/image]
```
Note that model configs are stored in the **config.py** file and you can change it prior to executing the main script. Default canfiguration is the following:
- **batch size** = 256
- **learning rate** = 0.001
- **patience** = 10
- **seed** = 42
- **image size** = 224
- **split ratio** = (0.70, 0.15, 0.15)
- **epochs** = 200
