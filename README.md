# MLP_proj2_Fall_2025
MLP_proj2_Fall_2025 Kaggle

Step 3: How to run
Download the llm_finetuning_deberta.ipynb file prepared on GitHub and upload it to the Kaggle notebook environment.
First, preliminary work on the LLM model is required.
Obtain the DeBerta-v3-Base version from Hugging Face and perform fine-tuning using the train.csv dataset in a local environment. Compress the resulting lightweight weights of the trained model into a zip file named deberta_lora_weights.zip. Compress the resulting lightweight model weights into a deberta_lora_weights.zip file. Compress the previously downloaded deberta-v3-base-local model into a zip file as well. Upload both zip files to Kaggle as the dataset. Use these as the input dataset in the Kaggle Notebook environment.
Enable the GPU T4 x2 accelerator in the Kaggle Notebook and proceed with training.
