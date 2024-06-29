from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch

checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, torch_dtype=torch.float32)

# Set the device to the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def llm_pipeline():
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=400, 
        min_length=50,
        device=device  # Add this line to specify the device
    )
    input_text = input("\n \n Enter the TExt: \n\n ")
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

print(f"\n \n The sammary is given below \n \n {llm_pipeline()} ")