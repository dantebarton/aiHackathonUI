from flask import Flask, render_template, request
import torch
import random
import numpy as np
from markllm.watermark.auto_watermark import AutoWatermark
from markllm.utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from markllm.visualize.visualizer import DiscreteVisualizer
from markllm.visualize.font_settings import FontSettings
from markllm.visualize.legend_settings import DiscreteLegendSettings
from markllm.visualize.page_layout_settings import PageLayoutSettings
from markllm.visualize.color_scheme import ColorSchemeForDiscreteVisualization
from PIL import Image

# Flask setup
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for session management

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
myWatermark = AutoWatermark.load('KGW',
                                 algorithm_config='MarkLLM/config/KGW.json',
                                 transformers_config=transformers_config)

# Init visualizer
visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                font_settings=FontSettings(),
                                page_layout_settings=PageLayoutSettings(),
                                legend_settings=DiscreteLegendSettings())

@app.route("/", methods=["GET", "POST"])
def index():
    watermarked_text = None
    unwatermarked_text = None
    watermark_score = 0
    human_score = 100  # Initialize human score as 100 by default
    watermarked_img_path = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("user_input")
        action = request.form.get("action")
        prompt = user_input if user_input else "Describe world war1"  # Use user input or default prompt
        
        if action == "clear":
            return render_template("index.html")
        
        elif action == "generate":
            # Generate watermarked text
            watermarked_text = myWatermark.generate_watermarked_text(prompt)
            unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
            
            if not watermarked_text:  # Ensure text is generated
                return "No watermarked text generated."

        elif action == "analyze":
            watermarked_text = request.form.get("watermarked_text")  # Retrieve the watermarked text
            if user_input:
                human_result = myWatermark.detect_watermark(prompt)
                human_score = int((human_result.get("score", 0)) * 10)
            if watermarked_text:
                # Detect watermark score for watermarked text
                detect_result = myWatermark.detect_watermark(watermarked_text)
                watermark_score = int((detect_result.get("score", 0)) * 10)  # Scale to percentage
                if watermark_score > 100:
                    watermark_score = 99
                human_score = 100 - watermark_score  # Ensure total adds up to 100

                if user_input and watermarked_text:
                    combined_length = len(watermarked_text + user_input)
                    human_score = int((len(user_input) / combined_length) * 100)
                    watermark_score = 100 - human_score
                    watermarked_text = user_input
                
                # Get data for visualization
                watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
                watermarked_img = visualizer.visualize(data=watermarked_data,
                                                       show_text=True,
                                                       visualize_weight=True,
                                                       display_legend=True)

                # Save image to file
                watermarked_img_path = "static/KGW_watermarked.png"
                watermarked_img.save(watermarked_img_path)
            else:
                # If no watermarked text is provided, treat everything as human input
                watermark_score = 0  # No watermark detected
                human_score = 100  # Entirely human-generated text

    return render_template("index.html", 
                           watermarked_text=watermarked_text,
                           unwatermarked_text=unwatermarked_text,
                           watermark_score=watermark_score,
                           human_score=human_score,
                           user_input=user_input,
                           watermarked_img_path=watermarked_img_path)

if __name__ == "__main__":
    app.run(debug=True)
