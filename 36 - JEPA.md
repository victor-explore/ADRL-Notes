# JEPA - Joint Embedding Prediction and Autoencoding
## Idea
Joint Embedding Predictive Architecture (JEPA) combines predictive learning with representation learning in a unified framework.

## Formulation
Let us formalize the JEPA framework:

1. Given:
   - Input space $\mathcal{X}$ 
   - Context patch $x_c \in \mathcal{X}$
   - Target patch $x_t \in \mathcal{X}$
   - Context encoder $f_\theta: \mathcal{X} \rightarrow \mathcal{Z}$
   - Target encoder $g_\phi: \mathcal{X} \rightarrow \mathcal{Z}$ 
   - Predictor network $h_\psi: \mathcal{Z} \rightarrow \mathcal{Z}$

2. The process works as follows:
   - Encode context: $z_c = f_\theta(x_c)$
   - Encode target: $z_t = g_\phi(x_t)$ 
   - Predict target embedding: $\hat{z}_t = h_\psi(z_c)$

3. The objective is to minimize prediction error in embedding space:
   $$\mathcal{L}(\theta,\phi,\psi) = \mathbb{E}_{(x_c,x_t) \sim p_\text{data}}\left[\|z_t - \hat{z}_t\|^2\right]$$
   
   where:
   - The expectation is over pairs of context and target patches
   - $\|\cdot\|$ denotes Euclidean distance
   - Parameters $\theta,\phi,\psi$ are learned jointly

4. Key properties:
   - Learns representations by predicting embeddings rather than raw inputs
   - Uses vision transformer architecture as underlying backbone
   - Can leverage any encoder architecture for inference
   - Enables both predictive and autoencoding objectives

This approach combines benefits of predictive learning and representation learning while avoiding pixel-level reconstruction.

  <div style="text-align: center;"><img src="6.png" alt="Image Description" width="1000" height="auto"/></div>  

## Training and Downstream Usage

1. During training, three components are trained jointly:
   - Context encoder $f_\theta$ that maps context patches to embeddings
   - Target encoder $g_\phi$ that maps target patches to embeddings  
   - Predictor network $h_\psi$ that predicts target embeddings from context embeddings

2. The training process optimizes:
   - Context encoder parameters $\theta$
   - Target encoder parameters $\phi$
   - Predictor network parameters $\psi$
   
   Through minimizing the prediction error in embedding space.

3. For downstream tasks:
   - The target encoder $g_\phi$ is typically used as the main feature extractor
   - This is because the target encoder learns to create embeddings that:
     - Capture meaningful semantic information about inputs
     - Are predictable from context, implying they encode important features
     - Don't rely on predicting low-level details like pixels
   - The context encoder can also be used but generally performs slightly worse since it's optimized for prediction rather than representation

4. Common downstream applications include:
   - Image classification: Use target encoder embeddings as input features
   - Object detection: Use target encoder as backbone network
   - Semantic segmentation: Use target encoder for dense feature extraction
   - Other vision tasks that benefit from pre-trained representations

