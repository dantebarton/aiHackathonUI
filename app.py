from flask import Flask, render_template, request
import torch, random
import numpy as np
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Flask setup
app = Flask(__name__)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set random seeds for reproducibility
seed = 30
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Model and transformers configuration
model_name = 'facebook/opt-1.3b'
transformers_config = TransformersConfig(
    model=AutoModelForCausalLM.from_pretrained(model_name).to(device),
    tokenizer=AutoTokenizer.from_pretrained(model_name),
    vocab_size=50272,
    device=device,
    max_new_tokens=200,
    min_length=230,
    do_sample=True,
    no_repeat_ngram_size=4
)

# Load watermark algorithm
myWatermark = AutoWatermark.load('KGW', transformers_config=transformers_config)

@app.route("/", methods=["GET", "POST"])
def index():
    watermarked_text = None
    unwatermarked_text = None
    watermark_score = None
    if request.method == "POST":
        user_input = request.form.get("user_input")
        if user_input:
            # Generate watermarked and unwatermarked text
            watermarked_text = myWatermark.generate_watermarked_text(user_input)
            unwatermarked_text = myWatermark.generate_unwatermarked_text(user_input)

            # Detect watermark score for the watermarked text
            detect_result = myWatermark.detect_watermark(watermarked_text)
            watermark_score = detect_result.get("score", 0)

    #watermark_percentage = str(int(watermark_score)) + "%"
   # print(watermark_percentage)
    
    return render_template("index.html", 
                           watermarked_text=watermarked_text,
                           unwatermarked_text=unwatermarked_text,
                           watermark_score=watermark_score)

if __name__ == "__main__":
    app.run(debug=True)
