import modal
app = modal.App("ragen-training")

image = (
    modal.Image.debian_slim()
    .pip_install("torch")  
    .add_local_dir(".", "/root/code") 
)

@app.function(
    image=image,
    gpu="A10G",  
    timeout=1800, 
)
def train():
    import sys
    sys.path.insert(0, '/root/code')
    
    from ragen.train_ragen_apo import train_ragen_apo
    
    print("Starting training on Modal GPU...")
    agent, tokenizer, action_space, max_len = train_ragen_apo()
    print("Training complete!")
    
    # ← CHANGE 3: Return success message
    return "Training finished - check Modal logs for results"

@app.local_entrypoint()
def main():
    result = train.remote()
    print(result)