import gradio as gr
import torch
import os
from v1.usta_model import UstaModel
from v1.usta_tokenizer import UstaTokenizer

# Load the model and tokenizer
def load_model():
    try:
        u_tokenizer = UstaTokenizer("v1/tokenizer.json")
        
        # Model parameters - adjust these to match your trained model
        context_length = 32
        vocab_size = len(u_tokenizer.vocab)
        embedding_dim = 12
        num_heads = 4
        num_layers = 8
        
        # Load the model
        u_model = UstaModel(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim, 
            num_heads=num_heads, 
            context_length=context_length, 
            num_layers=num_layers
        )
        
        # Load the trained weights if available
        model_path = "./u_model.pth"
        if os.path.exists(model_path):
            try:
                u_model.load_state_dict(torch.load(model_path, map_location="cpu"))
                u_model.eval()
                print("✅ Model weights loaded successfully!")
            except Exception as e:
                print(f"⚠️ Warning: Could not load trained weights: {e}")
                print("Using random initialization.")
        else:
            print(f"⚠️ Model file not found at {model_path}. Using random initialization.")
        
        return u_model, u_tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# Initialize model and tokenizer globally
try:
    model, tokenizer = load_model()
    print("🚀 Model and tokenizer initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize model: {e}")
    model, tokenizer = None, None

def generate_response(message, history, max_new_tokens=20, temperature=1.0):
    """
    Generate a response using the UstaModel
    """
    if model is None or tokenizer is None:
        return "Sorry, the model is not available. Please try again later."
        
    try:
        # Encode the input message
        tokens = tokenizer.encode(message)
        
        # Make sure we don't exceed context length
        if len(tokens) > 25:  # Leave some room for generation
            tokens = tokens[-25:]
        
        # Generate response
        with torch.no_grad():
            if temperature > 0:
                # Use the model's built-in generate method with some modifications for temperature
                generated_tokens = model.generate(tokens, max_new_tokens)
            else:
                generated_tokens = model.generate(tokens, max_new_tokens)
        
        # Decode the generated tokens
        response = tokenizer.decode(generated_tokens)
        
        # Clean up the response (remove the original input)
        original_text = tokenizer.decode(tokens.tolist())
        if response.startswith(original_text):
            response = response[len(original_text):]
        
        # Clean up any unwanted tokens
        response = response.replace("<unk>", "").replace("<pad>", "").strip()
        
        if not response:
            response = "I'm not sure how to respond to that."
            
        return response
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def chat_fn(message, history):
    """
    Main chat function for Gradio
    """
    if not message.strip():
        return history, ""
    
    # Generate response
    response = generate_response(message, history)
    
    # Add to history
    history.append([message, response])
    
    return history, ""

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Usta Model Chat", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🤖 Usta Model Chat
            
            Chat with a custom language model trained from scratch! This model has a vocabulary focused on 
            countries, capitals, and geographic knowledge.
            
            **Note**: This is a small experimental model, so responses may be limited to the training vocabulary.
            """
        )
        
        chatbot = gr.Chatbot(
            [],
            elem_id="chatbot",
            bubble_full_width=False,
            height=400
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="Type your message here...",
                show_label=False,
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear", variant="secondary", scale=1)
        
        with gr.Accordion("Advanced Settings", open=False):
            max_tokens = gr.Slider(
                minimum=1,
                maximum=50,
                value=20,
                step=1,
                label="Max New Tokens",
                info="Maximum number of new tokens to generate"
            )
        
        # Event handlers
        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            chat_fn,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        )
        
        # Example inputs
        gr.Examples(
            examples=[
                "the capital of france",
                "tell me about spain",
                "what is the capital of united states",
                "paris is in",
                "germany and its capital"
            ],
            inputs=msg
        )
        
        gr.Markdown(
            """
            ### About this model
            This is a custom transformer model trained from scratch with a focused vocabulary 
            on geographical knowledge including countries, capitals, and cities.
            
            **Model Details:**
            - Vocabulary Size: 64 tokens
            - Embedding Dimension: 12
            - Number of Attention Heads: 4
            - Number of Layers: 8
            - Context Length: 32 tokens
            """
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    # Use share=True for Hugging Face Spaces
    demo.launch(share=True) 