# Sequence classification with BERT

- BERT with logistic regression notebook is the simplest example of sequence classification with BERT. The final hidden states of the BERT base model are used as a feature embedding. Each example is passed through the model to obtain the feature embedding. These features are then used to train the logistic regression. This seems to work well due to BERT's large amount of pre-training which gives it a decent understanding of language. 

 - BERT classification custom notebook was an attempt at doing what was done in the logistic regression notebook but with a larger dataset and a different model. This is difficult to do with a larger dataset that has larger sequence lengths because the amount of memory required is huge, due to the size of BERT and the data. Therefore batching the data is a logical step and getting the feature embeddings for each batch, but at this point doing this in torchtext with the appropriate tools makes more sense than writing a custom batching function. 

- BERT classification with torchtext notebook achieves the goal of using a larger dataset and a different model by batching the data and training on feature embeddings for each batch. 

 - Fast Bert notebook is making use of the popular library by Kaushal Trivedi. This library allows us to easily fine-tune the BERT model on the downstream task of sequence classification. Training takes longer due to the base models weights being fine-tuned as well as the classification head, but the resulting model is more accurate. 

**References:** 

- https://github.com/kaushaltrivedi/fast-bert
- https://github.com/keitakurita/
- https://github.com/huggingface/transformers
- https://jalammar.github.io/
