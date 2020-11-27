# Instrinsic Knowledge Evaluation on Chinese Language Models

Code, data, and models for intrinsic knowledge evalutation on Chinese Language Models introduced in the paper [Intrinsic Knowledge Evaluation on Chinese Language Models]()




## Evaluate
To evaluate a model on a specfic knowledge aspect, take facts as an example:
```bash
cd ./ChnEval/
python eval_fact.py \
  --data_path="./data/knowledge/fact.txt" \
  --pretrained_model_path="./data/models/bert.bin" \
  --target="bert"
```
Use eval_commonsese/syntax/semantic_pos.py when testing on the commonsense, syntactic, and semantic knowledge.
Meanwhile, alter your model by specify the `--pretrained_model_path` argument, along with the correct `--target` (bert, mlm, sbo, sop). 


## Data Set
We organized four data sets concerning knowledge of different aspects. [Fact]() and [Common Sense]() examines the world knowledge within models, while [Syntactic]() and [Semantic]() data focus on the nlignuistic knowledge.

By default, these data sets come along with this project, and posit under the `./ChnEval/data/knowledge/` directory.


## Model
For a fair comparison between BERT-variants, we performed incremental pre-training from BERT-base-chinese using four different objective settings, i.e. MLM, MLM+SBO, MLM+SOP, MLM+NSP. The 50w-step checkpoints can be downloaded from:
* [MLM]()
* [MLM + SBO]()
* [MLM + SOP]()
* [MLM + NSP]()
  
Also, we include (comparable) sota pre-trained Chinese Language Models:
* [BERT](): BERT-base-chinese in PyTorch.
* [BERT-wwm](): BERT using the whole-word-masking strategy.
* [BERT-wwm-ext](): BERT-wwm pre-trained using additional external corpus.
* [RoBERTa-wwm-ext](): a Chinese version RoBERTa using additional external corpus.
