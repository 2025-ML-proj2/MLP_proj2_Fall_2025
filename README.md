# MLP_proj2_Fall_2025
MLP_proj2_Fall_2025 Kaggle

## How to run 'Step 5. llm-finetuning.ipynb'
1. Upload or write the code 'llm-finetuning.ipynb' in the Kaggle Notebook environment.
2. This code requires GPU T4 x2, so verifiying phone number to access GPU is required.
3. This code uses Gemma-2 model from Hugging Face, so getting permission from Hugging Face is required.
4. Set the Settings -> Accelerator to GPU T4 x2.
5. Run the code and submit the results.

 * Note: This code requires 2 NVIDIA T4 Tensor Core GPUs, so running on local device is not recommended.
 * Python version inconsistency can occur errors. We suggest changing the version to 3.11.
 * Note: For more stable reproducibility, copy the code from here and change the certain parameters.
   * Original code: https://www.kaggle.com/code/blue0924/finetuning-test2
   * np.average weights=[0.5, 0.5, 0]
   * processor = ProcessorPAB max_length=4608 (gemma2-9b)
   * processor = ProcessorPAB max_length=4608 (llama3-8b)
