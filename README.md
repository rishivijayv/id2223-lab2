# Fine-Tuning Llama-3.2 1Bil Model Using Unsloth

## Introduction
In this lab, we used the **unsloth** library to fine-tune the **4-bit quantized** version of the [**Llama-3.2 1 Billion Instruction**](unsloth/llama-3.2-1b-instruct-bnb-4bit) on the [**FineTome-100k**](https://huggingface.co/datasets/mlabonne/FineTome-100k) data set. The fine-tuning was performed on the T4 GPU using a slightly modified version of the provided Colab Notebook. You can find the fine-tuned model [here](https://huggingface.co/rishivijayvargiya/id2223_lab2_base_model). 

The fine-tuned model was then used for inference on 2 different User-Interfaces hosted on HuggingFace (HF) Spaces: 
1. [The ShallowLearning Chatbot](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference): A simple, traditional chatbot that primarily answers questions.
2. [The Shallow Storymaker](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker): A more creative use of the model, where users get to craft and participate in adventurous story-telling.

Both these inference programs run on the **T4 GPU**. We tried to run them initially on the CPU hardware on HF Spaces, but ran in to several issues. We made a discussion post about this on the course canvas page and discussed this with the course instructor: Jim Dowling. Ultimately, we decided to proceed with running our inference on GPU in the interest of time. 

## Contents of Repository
This repository contains the following items:
1. **This `README.md`**: Describes our Lab
2. **The `Lab2_ID2223_GPU_Training_Unsloth.ipynb` Notebook**: Is the program we used to fine-tune our model. Is slightly modified for adding support for checkpointing weights and hyperparameter tuning tests (more on these later).
3. **The `shallow-learning-chatbot` Directory**: Just contains a `README.md` with a link to the repository on HF Spaces which contains the code for this inference program.
4. **The `shallow-storymaker` Directory**: Similar to (3), but the `README.md` has a link to the git repo on HF Spaces with code for the story-maker program instead.
5. **The `assets` Directory**: Contains some static assets used in this `README.md` file. 

The code for the inference programs is structured in this way so that there is only **one source of truth** for the code for the **inference programs**: The **HF Spaces Git Repository**. 

## Part 1: Fine-Tuning on the T4 GPU

### Hyperparameters and Weight Checkpointing
We will give a brief description of how we fine-tuned the quantized Llama-3.2-1b model to work with the **FineTome-100k** dataset. 

We used the following hyperparameters for fine-tuning (the `TrainingArguments` to `SFTTrainer`)
```python3
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 3, 
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = output_dir,
        report_to = "none", 

        # GROUP: our params for weight-checkpointing
        save_steps=100,
        save_total_limit=3,
        save_strategy="steps"
    )
```
As evident from the the arguments, we fine-tuned the quantized Llama-3.2-1b model for **3 epochs**. This lead to a running time of around **36 hours**. So, we were not able to fine-tune this in 1 consecutive run. As a result of this, we had to rely on **checkpointing the intermediate weights** obtained by the fine-tuning. For this, we used a strategy of checkpointing every **100 steps** (ie, after every 100 batches have been processed). We stored up-to 3 checkpoints in **Google Drive**, where the checkpoints were saved in the `output_dir` variable in the Google Drive (this was defined earlier in the fine-tuning notebook). 

Then, to begin the training from the **latest** checkpoint, or to begin training from scratch (when no checkpoint existed), we used the following piece of code:
```python3
import glob
import os

checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))

resume_from_checkpoint = False
if checkpoint_dirs:
  resume_from_checkpoint = True

trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
```
Both these code snippets are available in full-context in the [fine-tuning notebook used](https://github.com/rishivijayv/id2223-lab2/blob/main/Lab2_ID2223_GPU_Training_Unsloth.ipynb).  

After the fine-tuning was performed, the fine-tuned model was saved as **LoRA Adapters** using the `unsloth` library, which can be found at this location: [`rishivijayvargiya/id2223_lab2_base_model`](https://huggingface.co/rishivijayvargiya/id2223_lab2_base_model)

### Choice of Base Model
We used the **4-bit quantized Llama-3.2-1b model** to fine-tune for 2 particular reasons. 
1. First, the model being **4-bit quantized** meant that fine-tuning would progress faster. This, we felt, was very important especially when working in a time-constrained environment.
2. Secondly, we felt that the model having 1-billion parameter would, again, ensure that we were able to fine-tune the model on our dataset over multiple epochs in a timely manner. It still took us 30+ hours to get 3 epochs out of the fine-tuning, and the time would only have been greater had we chosen a model with more parameters.
3. Finally, even though we ended up using T4 GPUs for inference, we started out assuming we would be using CPUs. On CPUs, larger base LLM models would perform even slower than the Llama-3.2-1b model. So, we also felt this would provide the best user experience given the limited resrouces. 

Thus, for these reasons, we decided to go with the **4-bit quantized Llama-3.2-1b** instruct model as our base model.

## Part 2: UIs for Inference
After the fine-tuning the base LLM model, we created **2 User Interfaces** on HF Spaces which run on the **T4 GPU**. As mentioned earlier, the original plan was to use CPUs for inference, but because of some unforseen hiccups in getting that set-up and in the interest of time, we eneded up using GPUs. We will now briefly describe the 2 inference programs. 

Both these UIs take heavy inspiration from a demo Gradio app found in the `unsloth` repository: https://github.com/unslothai/unsloth-studio/blob/main/unsloth_studio/chat.py, which was discovered from this GitHub discussion: https://github.com/unslothai/unsloth/issues/990. Reference to this file are present in the `app.py` file for both the inference programs.

### Program 1: [The ShallowLearning Chatbot](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference)
This is a chatbot with a simple, interactive UI which allows users to send messages to the fine-tuned LLM and receive responses in return. After playing around with this chatbot a little, we believe that the chatbot seems to be primarily focused with trying to answer user questions. This, we think, makes sense: since the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) data-set was obtained from a larger data set, [The Tome](https://huggingface.co/datasets/arcee-ai/The-Tome), which had a "focus on instruction following" (according to its README). 

Below is a **short video demo** of an interaction with the chatbot:

https://github.com/user-attachments/assets/e4692858-96c3-4bc0-b24c-c8e04e9f6b7a

**The Code**: https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference/tree/main

**Link to Chatbot**: https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-inference

### Program 2: [The Shallow Storymaker](https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker)
This is a slightly more sophisticated UI, with a theme([YTheme/Minecraft](https://huggingface.co/spaces/YTheme/Minecraft)), and a specialized function: creating a _choose your own adventure_ type experience for the user based on a _Title_ that the user picks for their adventure. This required us to use an initial **system prompt** to tell the fine-tuned LLM what its purpose was. 

Below is a **short video demo** of an adventure created by the storymaker: 

https://github.com/user-attachments/assets/2cbb0026-951c-4ec8-ae3e-0eb79eea85a0

**The Code**: https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker/tree/main

**Link to Storymaker**: https://huggingface.co/spaces/rishivijayvargiya/id2223-lab2-storymaker

## Model-Centric Approaches for Fine-Tuning Improvements
**TODO: Talk about some hyperparameters and show results for a couple (graphs for 5 epochs on 1/10th the dataset). Can also talk about "early stopping" to save resources, using a validation/evaluation set in addition to a training set, etc.**

## Data-Centric Approaches for Fine-Tuning Improvements
**TODO: Just talk about data-centric approaches for improvement**

## Future Work
**TODO: Talk a little about what more could have been done if time was not a constraint (eg: trying out different base models, changing even more hyperparameters, getting infrerence to work on CPUs using GGUF by trying out a workaround suggested in the discussion post, etc)**
