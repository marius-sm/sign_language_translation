# Sign language translation

This repo contains an implementation of [Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://arxiv.org/abs/2003.13830), [Learning and Using the Arrow of Time](https://www.robots.ox.ac.uk/~vgg/publications/2018/Wei18/wei18.pdf), [Self-supervised Spatiotemporal Learning via Video Clip Order Prediction](https://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf)

#### ⚠ To compute BLEU on embeddings please use this fork
`pip install git+git://github.com/marius-sm/transformers@v4.1.1-release`

## Training on embeddings2text
### Supervised pre-training
 - Embeddings from https://github.com/neccam/slt/ - Best accuracy in 20 epochs: 51.36
 - Embeddings from https://github.com/neccam/slt/, downsampled with stride 8 - Best accuracy in 20 epochs: 50.64
 - Embeddings from R(2+1)D pre-trained on Kinetics-400 - Best accuracy in 20 epochs: 45.64
 ### Self-supervised pre-training
 - Embeddings from R(2+1)D pre-trained on Kinetics-400 and fine tuned on arrow of time prediction
 - Embeddings from R(2+1)D pre-trained on Kinetics-400 and fine tuned on clip order prediction - Best accuracy in 20 epochs: 45.13
