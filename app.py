
import streamlit as st
import time
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pretrained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_story(input_text):
    # Tokenize input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Ensure attention mask and pad token ID are set
    #input_ids = input_ids.to(model.device)
    #attention_mask = input_ids.ne(tokenizer.pad_token_id).float()

    # Generate output
    output = model.generate(input_ids, 
                            num_beams=50, no_repeat_ngram_size=2,
                            early_stopping=True, max_length=100)

    # Decode output tokens to text
    generated_story = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_story

def main():
    st.title("Story Generation App")

    # User input
    input_text = st.text_input("Enter the starting sentence")

    # Generate story when button is clicked
    if st.button("Generate Story"):
        progress_text = "Generating story..."
        my_bar = st.progress(0)
        my_bar_text = st.empty()

        for percent_complete in range(100):
            time.sleep(.2)
            my_bar.progress(percent_complete + 1)
            my_bar_text.text(f"{progress_text} {percent_complete + 1}%")

        generated_story = generate_story(input_text)

        # Display the generated story
        #st.write("Generated Story:")
        st.text_area("Generated Story", generated_story, height=200)

if __name__ == "__main__":
    main()
