# MLP_proj2_Fall_2025
MLP_proj2_Fall_2025 Kaggle

## Step 3: How to run 'Step 3. llm-finetuning-deberta.ipynb'
1. Download the llm_finetuning_deberta.ipynb file prepared on GitHub and upload it to the Kaggle notebook environment.
2. Obtain the deberta-v3-base version from Hugging Face.
3. Perform fine-tuning using the train.csv dataset in the local environment.
4. Compress the lightweight weights of the trained model into the deberta_lora_weights.zip file. Compress the previously downloaded deberta-v3-base-local model into a zip file as well, and upload both to Kaggle as datasets.
4-1. Download and use the deberta-lora-weights.zip file from the arcroot branch.
5. Use these as input datasets in the Kaggle notebook environment.
6. Enable the GPU T4 x2 accelerator in the Kaggle notebook to proceed with training.