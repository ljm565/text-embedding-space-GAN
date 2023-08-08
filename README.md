# TESGAN
## Requirements
* Python 3.6.9
* PyTorch 1.10.1+cu113
<br>

## Paper
* [TESGAN paper](https://arxiv.org/abs/2306.17181)

## Introduction
In this experiment, we apply the Generative Adversarial Networks to synthesizing text embedding space called a seed.
Text Embedding Space GAN (TESGAN) does not explicitly refer to training data because it trains in an unsupervised way which is similar to the original [GAN framework](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf).
Thus, data memorization does not appear when synthesizing sentences.
Additionally, unlike previous studies, TESGAN does not generate discrete tokens, but rather creates text embedding space, thereby solving the gradient backpropagation problem argued in [SeqGAN](https://github.com/LantaoYu/SeqGAN).
<br>

## Metric
* ### [Fréchet BERT Distance (FBD)](https://github.com/IAmS4n/TextGenerationEvaluationMetrics)
    We use a Fréchet BERT Distance (FBD) for comparing the results.
    The FBD calculates the quality and diversity of generated sentences, and a lower value means better.

* ### [Multi-sets Jaccard (MSJ)](https://github.com/IAmS4n/TextGenerationEvaluationMetrics)
    The Multi-sets Jaccard (MSJ) calculates the similarity between the generative model and the real distribution by comparing the generated text samples.
    MS-Jaccard focuses on the similarity of the n-grams frequencies between the two sets with considering the average frequency of the generated n-gram per sentence and a higher value means better.

* ### [Language Model score (LM)](https://github.com/pclucas14/GansFallingShort)
    LM measures generated sample quality which means that scores of the bad samples are poor under a well-trained language model.

* ### [Self-BLEU (SBL)](https://github.com/geek-ai/Texygen)
    SBL measures diversity of the generated sample based on the token combination.

* ### Data Synthesis Ratio (DSR)
    We calculate Data Synthesis Ratio (DSR) to evaluate the data memorization ratio and synthetic diversity.
    DSR is calculated as the harmonic mean of the ratio where data memorization does not occur and the ratio of diversity.
    A higher value means better.


## Testing
* ### Seed Interpretation Model
    ```
    python3 src/main.py -d gpu -m train
    ```
    * You have to set "model" and "max_len" in *src/config.json* as "interpretation" and 64, respectively.

    * For more detailed setup, you can refer *model/interp_sigmoid/interp_sigmoid.json* file.

* ### TESGAN
    ```
    python3 src/main.py -d gpu -m train
    ```
    * You have to set "model" and "max_len" in *src/config.json* as "tesgan" and 16, respectively.
    
    * If you want to train P-TESGAN, you have to set "perturbed" as 1 in *src/config.json*.

    * For more detailed setup, you can refer *model/tesgan_sigmoid/tesgan_sigmoid.json* or *model/tesganP_sigmoid/tesganP_sigmoid.json* files.

* ### Synthesized Result
    ```
    python3 src/main.py -d gpu -m syn -n {model file name}
    ```
    * After training both seed interpretation model and TESGAN, you can synthesize sentences based on your TESGAN model.
    
    * You must give the file name with a *.pt* extension as the -n argument.

    * For example
        ```
        python3 src/main.py -d gpu -m syn -n tesgan_sigmoid_17.pt
        ```
## Acknowledgement
* [multiset_distances.py](https://github.com/ljm565/TESGAN/blob/master/etc/multiset_distances.py) and [bert_distances.py](https://github.com/ljm565/TESGAN/blob/master/etc/bert_distances.py) is based on [IAmS4n](https://github.com/IAmS4n/TextGenerationEvaluationMetrics). Many thanks for the authors.
* [DailyDialog Dataset](https://github.com/facebookresearch/EmpatheticDialogues) were used in this experiment. Many thanks for the authors.