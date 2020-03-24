# BERT for Sequence Classification

# Installation

If you are testing this on your own machine I would recommend you do the setup in a virtual environment, as not to affect the rest of your files. 

In Python3 you can set up a virtual environment with 

```bash
python3 -m venv /path/to/new/virtual/environment
```

Or by installing virtualenv with pip by doing 
```bash
pip3 install virtualenv
```
Then creating the environment with 
```bash
virtualenv venv
```
and finally activating it with
```bash
source venv/bin/activate
```

You must have Python3

Install the requirements with:
```bash
pip3 install -r requirements.txt
```

### Toxic Comment Fine-tuned model 

The toxic comment fine-tuned model is available in my [S3 Bucket](https://toxic-model.s3.eu-west-2.amazonaws.com/toxic_model.pt) or alternatively you can fine-tune the model yourself using one of the provided notebooks in the train folder.

You can load your fine-tuned model with:

```python
model = linear_model(bert_model, len(labels_list))
model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
```

# Making predictions

You can test the model using the `predict` function in `model.py` or by using the provided Flask user interface

![demo](/Users/oliverproud/bert-sequence-classification/static/images/demo.png)

# How to train (Distil)BERT

Check out my notebooks on how to fine-tune BERT [here]()

# References 

- <https://github.com/huggingface/transformers>
- <https://medium.com/huggingface/distilbert-8cf3380435b5>
