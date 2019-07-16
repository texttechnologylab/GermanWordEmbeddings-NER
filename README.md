# German Morphological Processing for Word Embeddings &amp; Named Entity Recognition

This short script performs a grammar-dependent morphological processing of the raw text data. Such data can be either be a large text corpus used for computing the word embeddings or a smaller labeled dataset used for training the neural network according to a given downstream-task (e.g. named entity recognition). Using this script prior to any training process improves the quality of the original resources, utimately leading to an increase of the final performance.

The pre-trained word embeddings produced with this morphological processing are provided under the following [link](https://www.texttechnologylab.org/resources2018/). 

NOTE: The results of this script (i.e. (1) word embeddings & (2) labled datasets) can be used to train the [NER Tagger](https://github.com/glample/tagger) for reproducing and evaluating the performance boost. Further details can be found in the reference below. Please cite the reference if you happen to use it in your work.

## Requirements
- [spaCy](https://spacy.io)
- Python 3

## Data
### Unlabeled text corpora
- [Leipzig](http://wortschatz.uni-leipzig.de/en/download)
- [WMT 2010](http://www.statmt.org/wmt10/translation-task.html)
- [COW](http://corporafromtheweb.org)
### Labeled datasets for German named entity recognition
- [CoNLL 2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- [GermEval 2014](https://sites.google.com/site/germeval2014ner/data)
- [Europarl](https://nlpado.de/~sebastian/software/ner_german.shtml)
- [Europeana Newspaper](https://github.com/EuropeanaNewspapers/ner-corpora)
- [Tübingen Treebank 2018](http://www.sfs.uni-tuebingen.de/en/ascl/resources/corpora/tueba-dz.html)


## Cite
Sajawel Ahmed and Alexander Mehler, "Resource-Size matters: Improving Neural Named Entity Recognition with Optimized Large Corpora" in Proceedings of the 17th IEEE International Conference on Machine Learning and Applications (ICMLA), 2018. accepted [[PDF]](https://arxiv.org/pdf/1807.10675.pdf)

## BibTeX

```
@InProceedings{Ahmed:Mehler:2018,
author		= {Sajawel Ahmed and Alexander Mehler},
title		= {{Resource-Size matters: Improving Neural Named Entity Recognition with Optimized Large Corpora}},
booktitle	= {Proceedings of the 17th IEEE International Conference on Machine Learning and Applications (ICMLA)},
location	= {Orlando, Florida, USA},
note		= {accepted},
pdf		= {https://arxiv.org/pdf/1807.10675.pdf},
year		= 2018
}
```
