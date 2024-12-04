# Fine-Tuning Llama-3.2 1Bil Model Using Unsloth

## Introduction
In this lab, we used the **unsloth** library to fine-tune the **4-bit quantized** version of the [**Llama-3.2 1 Billion Instruction**](unsloth/llama-3.2-1b-instruct-bnb-4bit) on the [**FineTome-100k**](https://huggingface.co/datasets/mlabonne/FineTome-100k) data set. The fine-tuning was performed on the T4 GPU using a slightly modified version of the provided Colab Notebook. The fine-tuned model was then used for inference on 2 different User-Interfaces hosted on HuggingFace (HF) Spaces: 
1. [The ShallowLearning Chatbot](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference): A simple, traditional chatbot that primarily answers questions.
2. [The Shallow Storymaker](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker): A more creative use of the model, where users get to craft and participate in adventurous story-telling.

Both these models run on the **T4 GPU**. We tried to run them initially on the CPU hardware on HF Spaces, but ran in to several issues. We made a discussion post about this on the course canvas page. Ultimately, we decided to proceed with running our inference on GPU in the interest of time. 
