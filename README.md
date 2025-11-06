# MLP_proj2_Fall_2025
MLP_proj2_Fall_2025 Kaggle

## Step 2, 4: How to run 'Step 2, 4. llm-classification-finetuning.ipynb'
1. Upload 'Step 2, 4. llm-classification-finetuning.ipynb' file in the Kaggle Notebook environment.(change file name to llm-classification-finetuning is recommanded.)
2. 2× T4 GPUs are recommended, but only the CPU and the P100 GPU are allowed.
3. Before run, you should do "Input" setting.
    <br>3-1. Click '+ Add Input' button and add tag:'Competition Datasets' and search "LLM Classification Finetuning" and click '+' button.
    <br>3-2. Click '+ Add Input' button and add tag:'Datasets' and Search' and search "sentence-transformers/all-MiniLM-L6-v2" and click '+' button.
4. Run the code and submit the results.

## How to run 'Step 5. llm-finetuning.ipynb'
1. Upload or write the code 'llm-finetuning.ipynb' in the Kaggle Notebook environment.
2. This code requires GPU T4 x2, so verifiying phone number to access GPU is required.
3. Internet connection in Kaggle Notebook->Settings is not required. However, accelerator by GPU is still necessary.
4. This code uses Gemma-2 model.
5. Set the Settings -> Accelerator to GPU T4 x2.
6. Run the code and submit the results.

 * Note: This code requires 2 NVIDIA T4 Tensor Core GPUs, so running on local device is not recommended.
 * Python version inconsistency can occur errors. We suggest changing the version to 3.11.
 * Note: For more stable reproducibility, copy the code from here and change the certain parameters.
   * Original code: https://www.kaggle.com/code/blue0924/finetuning-test2
   * np.average weights=[0.5, 0.5, 0]
   * processor = ProcessorPAB max_length=4608 (gemma2-9b)
   * processor = ProcessorPAB max_length=4608 (llama3-8b)
