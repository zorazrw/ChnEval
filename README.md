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
We organized four data sets concerning knowledge of different aspects. [Fact](https://github.com/ZhiruoWang/ChnEval/blob/master/data/knowledge/fact.txt) and [Common Sense](https://github.com/ZhiruoWang/ChnEval/blob/master/data/knowledge/commonsense.txt) examines the world knowledge within models, while [Syntactic](https://github.com/ZhiruoWang/ChnEval/tree/master/data/knowledge/syntax) and [Semantic](https://github.com/ZhiruoWang/ChnEval/blob/master/data/knowledge/semantic.txt) data focus on the lignuistic knowledge.

By default, these data sets come along with this project, and posit under the `./ChnEval/data/knowledge/` directory.


## Model
For a fair comparison between BERT-variants, we performed incremental pre-training from BERT-base-chinese using four different objective settings, i.e. MLM, MLM+SBO, MLM+SOP, MLM+NSP. The 50w-step checkpoints can be downloaded from (google drive/bnu cloud drive):
* MLM ([google](https://drive.google.com/file/d/1m5OhD6v8PceVBIqHocaMHlZMZ_6NRYdC/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/lu8ARy))
* MLM + SBO([google](https://drive.google.com/file/d/136c5QtERePqcUEUZLDR1rwWjQp-eqNeH/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/2nfi3O))
* MLM + SOP ([google](https://drive.google.com/file/d/19_O0UEQx42P9awcUDVdITjhAuBwWcxcj/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/Y0TMiz))
* MLM + NSP ([google](https://drive.google.com/file/d/1zS0jrw1-7K7oElBBRHP1LgjDZpJn3Hhg/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/aoMMFC))
  
Also, we include (comparable) sota pre-trained Chinese Language Models from [CLUE](https://github.com/CLUEbenchmark/CLUE):
* BERT ([google](https://drive.google.com/file/d/1xrBCC2gzYtlp2veCN2LSwhI12ZIZBXPV/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/DuUpHu)): BERT-base-chinese in PyTorch.
* BERT-wwm ([google](https://drive.google.com/file/d/1snprTrHIa3EcJdm4IZGbtuWAPzg-sD1c/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/R09Du8)): BERT using the whole-word-masking strategy.
* BERT-wwm-ext ([google](https://drive.google.com/file/d/15c4fNsIiY_t8gNHJ4ag8tL0MtzGVJ3xZ/view?usp=sharing)/[bnu](https://pan.bnu.edu.cn/l/Ou6oov)): BERT-wwm pre-trained using additional external corpus.
* RoBERTa-wwm-ext ([google](https://drive.google.com/file/d/1mMnMY8ZPzRTBhYYDzfxERRhOSPgifLEG/view?usp=sharing)[bnu](https://pan.bnu.edu.cn/l/lu8ARS)): a Chinese version RoBERTa using additional external corpus.
