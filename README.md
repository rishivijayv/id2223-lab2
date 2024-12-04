# Fine-Tuning Llama-3.2 1Bil Model Using Unsloth

## Introduction
In this lab, we used the **unsloth** library to fine-tune the **4-bit quantized** version of the [**Llama-3.2 1 Billion Instruction**](unsloth/llama-3.2-1b-instruct-bnb-4bit) on the [**FineTome-100k**](https://huggingface.co/datasets/mlabonne/FineTome-100k) data set. The fine-tuning was performed on the T4 GPU using a slightly modified version of the provided Colab Notebook. You can find the fine-tuned model [here](https://huggingface.co/rishivijayvargiya/id2223_lab2_base_model). 

The fine-tuned model was then used for inference on 2 different User-Interfaces hosted on HuggingFace (HF) Spaces: 
1. [The ShallowLearning Chatbot](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference): A simple, traditional chatbot that primarily answers questions.
2. [The Shallow Storymaker](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker): A more creative use of the model, where users get to craft and participate in adventurous story-telling.

Both these inference programs run on the **T4 GPU**. We tried to run them initially on the CPU hardware on HF Spaces, but ran in to several issues. We made a discussion post about this on the course canvas page. Ultimately, we decided to proceed with running our inference on GPU in the interest of time. 

## Contents of Repository
This repository contains the following items:
1. **This `README.md`**: Describes our Lab
2. **The `Lab2_ID2223_GPU_Training_Unsloth.ipynb` Notebook**: Is the program we used to fine-tune our model. Is slightly modified for adding support for checkpointing weights and hyperparameter tuning tests (more on these later).
3. **The `shallow-learning-chatbot` Directory**: Just contains a `README.md` with a link to the repository on HF Spaces which contains the code for this inference program.
4. **The `shallow-storymaker` Directory**: Similar to (3), but the `README.md` has a link to the git repo on HF Spaces with code for the story-maker program instead.
5. **The `assets` Directory**: Contains some static assets used in this `README.md` file. 

The code for the inference programs is structured in this way so that there is only **one source of truth** for the code for the **inference programs**: The **HF Spaces Git Repository**. 

