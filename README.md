# TESGAN
## Requirements
* Python 3.6.9
* PyTorch 1.10.1+cu113
<br>

## Paper
* [TESGAN paper](https://arxiv.org/abs/2306.17181) (Accpeted in NEJLT)

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

* ### Text Synthesizing
    ```
    python3 src/main.py -d gpu -m syn -n {model file name} --interp-name {seed interpretation model dir}
    ```
    * After training both seed interpretation model and TESGAN, you can synthesize sentences based on your TESGAN model.
    
    * You must give the file name with a *.pt* extension as the -n argument.

    * For example
        ```
        python3 src/main.py -d gpu -m syn -n tesgan_sigmoid_17.pt --interp-name interp_sigmoid
        ```

* ### Post-processing Synthesizing Results
    ```
    python3 etc/pp.py --input-path {synthesized txt path} --output-path {pp output txt path}
    ```
    * For more realistic synthesized text, we provide the simple post-processing code.
    
    * For example
        ```
        python3 etc/pp.py --input-path syn/syn.txt --output-path syn/pp/syn.txt
        ```

## Citation
Please cite the below bibtex style.
```
@inproceedings{lee-ha-2023-unsupervised,
    title = "Unsupervised Text Embedding Space Generation Using Generative Adversarial Networks for Text Synthesis",
    author = "Lee, Jun-Min  and
      Ha, Tae-Bin",
    editor = "Derczynski, Leon",
    booktitle = "Northern European Journal of Language Technology, Volume 9",
    year = "2023",
    address = {Link{\"o}ping, Sweden},
    publisher = {Link{\"o}ping University Electronic Press},
    url = "https://aclanthology.org/2023.nejlt-1.9/",
    doi = "https://doi.org/10.3384/nejlt.2000-1533.2023.4855",
    abstract = "Generative Adversarial Networks (GAN) is a model for data synthesis, which creates plausible data through the competition of generator and discriminator. Although GAN application to image synthesis is extensively studied, it has inherent limitations to natural language generation. Because natural language is composed of discrete tokens, a generator has difficulty updating its gradient through backpropagation; therefore, most text-GAN studies generate sentences starting with a random token based on a reward system. Thus, the generators of previous studies are pre-trained in an autoregressive way before adversarial training, causing data memorization that synthesized sentences reproduce the training data. In this paper, we synthesize sentences using a framework similar to the original GAN. More specifically, we propose Text Embedding Space Generative Adversarial Networks (TESGAN) which generate continuous text embedding spaces instead of discrete tokens to solve the gradient backpropagation problem. Furthermore, TESGAN conducts unsupervised learning which does not directly refer to the text of the training data to overcome the data memorization issue. By adopting this novel method, TESGAN can synthesize new sentences, showing the potential of unsupervised learning for text synthesis. We expect to see extended research combining Large Language Models with a new perspective of viewing text as an continuous space."
}
```


## Acknowledgement
* [multiset_distances.py](https://github.com/ljm565/TESGAN/blob/master/etc/multiset_distances.py) and [bert_distances.py](https://github.com/ljm565/TESGAN/blob/master/etc/bert_distances.py) is based on [IAmS4n](https://github.com/IAmS4n/TextGenerationEvaluationMetrics). Many thanks for the authors.
* [DailyDialog Dataset](https://github.com/facebookresearch/EmpatheticDialogues) were used in this experiment. Many thanks for the authors.
