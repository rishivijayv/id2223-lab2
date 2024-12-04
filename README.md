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
When thinking about some **mode-centric** approaches of improving the performance of the performance of our model, we thought primarily of **changing the hyper-parameters of our model** to obtain **lower training losses**. As a recap, the hyperparameters for the `SFTTrainer` with which we fine-tuned our model were as follows:

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
Running for 3 epochs took us about 30+ hours on the original data-set with 100k examples. We wanted to experimentally approximate the impact of chaning some of the hyperparameters, but we felt that doing so on the original data-set with 100k examples would not be practical given the time constraints. Thus, to obtain _some_ approximation on what the impact of chaning a hyperparameter would be on the model performance (**judged here by average training loss per epoch**), we decided to fine-tune the model on a much **smaller** data-set of 1k examples instead of the 100k. These 1k examples were chosen from the original FineTome-100k dataset, and the code for doing so can be found in the `Lab2_ID2223_GPU_Training_Unsloth.ipynb` notebook. Given the smaller dataset, we felt that we could increase the number of epochs so that we were able to observe a trend in the average training loss per epoch more easily (having more epochs makes the trend more obvious, we felt). 

Thus, the **base** model performance against which we judged the performance of the fine-tuned model with different hyperparameters had the following configuration:
```python3
args = TrainingArguments(
  # ... same as above
  num_train_epochs = 5
  # ... same as above
)
```

With 5 epochs and 1k examples, each fine-tuning run with different hyperparameter settings took about **~30min** instead of the original 30+ hours (with 3 epochs and 100k examples). While we recognize that the performance of the model will likely not be similar with the same hyperparameter settings when trained over 100k examples instead of 1k, we felt that getting _some_ approximation might at least give us some idea on what could be changed, which would then allow us to more specifically target the hyperparameters that appeared the most promising. 

With this in mind, in the interest of time, our attention was drawn towards the following hyperparameters: `num_train_epochs`, `learning_rate`, `lr_scheduler_type`, `wegith_decay`, `per_device_train_batch_size`. We will now explain our observations of the average training losses per epoch. Note that when we say a model **seems to perform better/worse**, we are talking here only about **lower/higher training losses (respectively)** of the model per epoch.

### Impact of `num_train_epochs`
The more time that our model sees a specific training example, the morel likely it is that the model performs better (ie, has lower training loss) on that example. This was also evident in our observations, as can be seen from the graph below: the training loss seems to generally **decrease** as the number of **epochs** increase. Thus, one easy way to improve model performance would be to **increase the number of epochs**.

![epoch_impact](https://github.com/rishivijayv/id2223-lab2/blob/main/assets/epoch_impact.png)

However, there is a fine-line when it comes to simply increasing the number of epochs. If the model continues seeing the same examples over and over again (eg: 20 or 30 epochs, for instance), then it is very likely that the model has weights that **overfit** the training data. Such behvaious can be prevented by introducing a set of examples that form the **validation set**. Then, at the end of each epoch, we can compute the **validation loss** of our model against the examples in the validation set. If the **validation loss** does not seem to improve over multiple epochs, or if it starts to _increase_ in consecutive epochs, then we can use **early stopping** to stop the training and prevent the model from overfitting on the training data. Such an addition can also **save resources**, as model training/fine-tuning is a time-intensive process, so it would be beneficial for everyone if it can be cut short and have better performance on data it has not seen. 


### Impact of `learning_rate`
Learning Rate controls the **magnitude** of the adjustments made to the model's weight. A **higher learning rate** has the advantage of _pulling_ the model's weights out of local minima, but have the disadvantage of _overshooting_ (ie, adjusting weights too drastically) and missing the optimal minimum. A **lower learning rate** has the oppoiste advantage and disadvantage. We wanted to determine the impact of halving/doubling the learning rate on the average loss per epoch (again, on 1k examples). So, **keeping the other hyperparameters the same**, we halved the learning rate (`1e-4` from `2e-4`) and fine-tuned our model, doubled the learning rate (`4e-4` from `2e-4`) and fine-tuned our model, and then plotted the average training loss with the 3 learning rates (the 2 described above, and the default: `2e-4`), and observed the following

![learning_rate_impact](https://github.com/rishivijayv/id2223-lab2/blob/main/assets/learning_rate_impact.png)

From this, it is evident that the model seems to start with a similar training loss initially, but a higher learning rate seems to force the model _out_ of local minimums, as the training loss per epoch (after the first epoch) **decreases** as the learning rate **increases**. Thus, from this, we can see that another way to improve the model performance (ie, reduce training loss), could be to **increase the learning rate**.

### Impact of `lr_scheduler_type`
As the training progresses, we might want to reduce the magnitude of the weight adjustments made, as it is likely that the model's weights now need _minute refinements_ instead of _major adjustments_. This is the motivation behind the `lr_scheduler_type` hyperparameter: which is responsible for adjusting (in most cases, **decreasing**) the learning rate as the training progresses. The default `lr_scheduler` was `linear`: which, as the name suggests, means that the learning rate decreases in a linear fashion. We wanted to try out one other value for the `lr_scheduler`, which was `cosine`: meaning that the adjustments makes the learning rate follow a `cosine` curve, which is more smoother and could allow for better convergence. We observed the following results (note: we kept all other hyperparameters the same)

![lr_scheduler_impact](https://github.com/rishivijayv/id2223-lab2/blob/main/assets/lr_scheduler_impact.png)

Although there seems to be a _slight_ improvement in the training loss as the epochs increaqse, the loss at the final epoch seems to be identical for both the `lr_scheduler_types`. So, given this information, when making adjustments to the fine-tuning of the entire 100k dataset, we believe that even though this switch could be made since it does not seem to make the model perform _worse_,  we might not prioritize going for the `cosine` scheduler. This is because we feel it would also be a little more computationally expensive to compute `cosine` terms as opposed to `linear` terms, and so the model might **scale poorly**, and the training loss benefits might not be worth this cost. If given the time, we could try some other schedulers, howver, such as `constant` or `constant_with_warmup`, etc. 


### Impact of `weight_decay`
This determines how _much_ we decide to penalize larger weights during training: since large weights would make the model too complex and thus likely to overfit on the training data. The "default" value for this was 0.01. So, we deicded to test how the average training loss per epoch would change if, keeping all other hyperparameters the same, we **douobled** the `weight_decay` (to 0.02), or if we **halved** the `weight_decay` (to 0.005). We observed the following results

![weight_decay_impact](https://github.com/rishivijayv/id2223-lab2/blob/main/assets/weight_decay_impact.png)

There are 3 lines in the graph: but the lines for `weight_decay=0.01` and `weight_decay=0.02` seem to overlap too well. So, we feel there is **no significant impact** on the training loss for the 1k examples data set if we increase the weight decay. However, the training losses seem to generally be **lower** with a lower weight_decay, but not by much. So, it could be worth a try to **decrease** `weight_decay` in order to **decrease** training loss when using the 100k example data set. This might be a better balance between penalizing complex weights (to prevent overfitting) yet still allowing the model to learn some complex patterns from the training data. 

We would like to mention that in addition to decreasing the weight decay, we should also consider adding a **validation set** of examples that we talked about previously. This would be another way to help us monitor that our model is not overfitting on the training data.


### Impact of `per_device_train_batch_size`
This controls the effective **batch size** used during training, meaning how many examples will be processed by the model at once. The default batch size of 2 per device gave us an effective batch size of 8, and thus gave us 1000/8 = 125 steps to process all examples once (ie, 125 steps per epoch). We decided to monitor what would happen if we double the batch size from 2 to 4 per device, which would give us an effective batch size of 16, and thus 1000/16 ~ approx 62 steps per epoch. Below are our results

![batch_size_impact](https://github.com/rishivijayv/id2223-lab2/blob/main/assets/batch_size_impact.png)

Doubling the batch size seems to be **worse** when it comes to training loss per epoch. So, if we were to choose which hyperparameters to tune for the 100k model, we would probably **not** want to change the per-device-batch-size first, and experiment tuning some of the other hyperparameters we have mentioned above. 


### Some Other Model-Centric Changes
Above, we have mentioned one way in which one could go about improving the model. However, here are some other steps that could be taken if we had more time: 

1. **Experimenting With Other Hyperparameters or Values**: Above we only discussed a subset of hyperparameters that we used. We could also try experimenting with different values of Hyperparameters such as `gradient_accumulation_steps` or `optim`. We could also use different values for some of the hyperparameters above, such as using different `lr_scheduler_type` instead of just `cosine` and see which one has a lesser training loss
2. **Changing the _Model Architecture_ and Defintion of _Better Model_**: As we explaied earlier, our experiments above worked under the assumption that a "better model" is one which has a lower training loss per epoch. However, it could very well be the case that we are more concerned about the model generalizing well as opposed to learning patterns in the trianing data, for example. In this case, for instance, we should consider changing the **fine-tuning model architecture** we have and add a **validation set** which validates the performance of our model against a set of examples (not used in the training set) after each epoch. Using concepts such as **early stopping** could also help us save time and prevent overfitting. For example, we could use the `evaluation_strategy`, `eval_dataset` argument to specify this for `SFTTrainer`. 

## Data-Centric Approaches for Fine-Tuning Improvements
**TODO: Just talk about data-centric approaches for improvement**

## Future Work
**TODO: Talk a little about what more could have been done if time was not a constraint (eg: trying out different base models, changing even more hyperparameters, getting infrerence to work on CPUs using GGUF by trying out a workaround suggested in the discussion post, etc)**
