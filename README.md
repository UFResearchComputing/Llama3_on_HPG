# Llama3_on_HPG

This tutorial is adopted from github repositories [meta-llama/llama](https://github.com/meta-llama/llama3) and [meta-llama/llama-recipes](https://github.com/meta-llama/llama-recipes).

## Llama3 Introduction

Llama 3, Meta’s latest family of open-source large language models (LLMs). Llama 3 comes in two sizes: 8B and 70B parameters. Each size has both base (pre-trained) and instruct-tuned versions. The context length for all variants is 8K tokens.
* Base Models:
- Meta-Llama-3-8b: The base 8B model.
- Meta-Llama-3-70b: The base 70B model.
* Fine-Tuned Versions:
- Meta-Llama-3-8b-instruct: Instruct fine-tuned version of the base 8B model.
- Meta-Llama-3-70b-instruct: Instruct fine-tuned version of the base 70B model.
* Llama Guard 2:
- Llama Guard 2, designed for production use cases, classifies LLM inputs and responses to detect unsafe content. It was fine-tuned on Llama 3 8B.

Llama 3 uses a new tokenizer with an expanded vocabulary size of 128,256 (compared to 32K tokens in Llama 2). This larger vocabulary improves text encoding efficiency and potentially enhances multilingualism. Grouped-Query Attention (GQA): The 8B version of Llama 3 now uses GQA, an efficient representation that helps with longer contexts.
Availability: Llama 3 models are freely available for research and commercial purposes.

For more details, you can check out the official [Llama 3 blog post](https://huggingface.co/blog/llama3) or this [article](https://ai.plainenglish.io/llama3-a-new-era-in-large-language-models-2270ca1d80c7).

## Download Llama 3

In order to download the model weights and tokenizer, please visit the [Meta Llama website](https://llama.meta.com/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

### Access to Hugging Face

We are also providing downloads on [Hugging Face](https://huggingface.co/meta-llama), in both transformers and native `llama3` formats. To download the weights from Hugging Face, please follow these steps:

- Visit one of the repos, for example [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).
- Read and accept the license. Once your request is approved, you'll be granted access to all the Llama 3 models. Note that requests use to take up to one hour to get processed.
- To download the original native weights to use with this repo, click on the "Files and versions" tab and download the contents of the `original` folder. You can also download them from the command line if you `pip install huggingface-hub`:

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --include "original/*" --local-dir meta-llama/Meta-Llama-3-8B-Instruct
```

- To use with transformers, the following [pipeline](https://huggingface.co/docs/transformers/en/main_classes/pipelines) snippet will download and cache the weights:

  ```python
  import transformers
  import torch

  model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

  pipeline = transformers.pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
  )
  ```

## Quick Start Llama3 on HiPerGator (HPG)

You can follow the steps below to quickly get up and running with Llama 3 models. These steps will let you run quick inference on HiPerGator. All the LLama3 models have already been downloaded to `/data/ai/models/nlp/llama/models_llama3`. You can run the model on HPG via the command line using `ml nlp/1.3`, or in the Jupyter Notebook with the `nlp-1.3` kernels.

* [01. Getting to know Llama_3](01_Getting_to_know_Llama_3.ipynb): This session is to provide a guided tour of Llama3, including understanding different Llama 3 models, how access them on HPG, Generative AI and Chatbot architectures, prompt engineering, RAG (Retrieval Augmented Generation), Fine-tuning and more.

* [02. Prompt Engineering with Llama_3](02_prompt_engineering_with_Llama_3.ipynb): This session interactive guide covers prompt engineering & best practices with Llama 3 use [Replicate API](https://replicate.com/meta/llama-2-70b-chat).

* [03. Running_Llama_3_on_HF_transformers](03_Running_Llama_3_on_HF_transformers.ipynb): This session shows how to run Llama 3 models with Hugging Face transformers

* [04. Deploy_Llama_3_with_TensorRT](04_Deploy_Llama_3_with_TensorRT.ipynb): This session shows how to deploy Llama 3 models with NVIDIA TensorRT-LLM.

* [05. Inference_Llama_3_locally](05_Inference_Llama_3_locally.ipynb): This session shows how to run inference with Llama 3 models using the command line.

## Finetuning and Inference

* If you insteaed in finetune Llama 3 on single-GPU and multi-GPU setups, you can find recipes at [finetuning](./finetuning).
* If you want to deploy Llama3 for inference locally and using model servers, you can find recipes at [inference](./inference).

For more examples, see the [Llama recipes repository](https://github.com/facebookresearch/llama-recipes).

## License  
All rights are reserved by the [Meta Llama team](https://llama.meta.com/) Please refer to the [Meta Llama License file](https://github.com/meta-llama/llama3/blob/main/LICENSE) and the Acceptable [Use Policy](https://github.com/meta-llama/llama3/blob/main/USE_POLICY.md).
