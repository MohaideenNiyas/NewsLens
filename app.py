# combined_app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import json
import base64
import tempfile
import os
import re
import requests
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
from werkzeug.utils import secure_filename

# For Mistral OCR
from mistralai import Mistral, ImageURLChunk, TextChunk
import time
start_time = time.time()

app = Flask(__name__, static_folder='static')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create static/js directory if it doesn't exist
os.makedirs('static/js', exist_ok=True)

# Initialize API keys
# In production, use environment variables for API keys
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY', 'your_api_key_here')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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')

# Pydantic model for newspaper structure
class NewspaperStructure(BaseModel):
    """Structure to represent the extracted newspaper content"""
    headline: str
    source: Optional[str] = None
    location: Optional[str] = None
    body_text: List[str]
    date: Optional[str] = None
    summary: Optional[str] = None
    news_script: Optional[str] = None

# ---- FUNCTIONS FROM BIG_STEPPER.PY ----

def simple_sentence_tokenize(text):
    """Split text into sentences using regex"""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def rank_sentences(sentences):
    """Rank sentences based on position and features"""
    if not sentences:
        return []
    
    # Hardcoded keywords for content scoring
    keywords = ['arrested', 'charges', 'navy', 'officials', 'incident', 
                'killed', 'attack', 'explosion', 'conflict', 'military']
        
    scores = []
    for i, sentence in enumerate(sentences):
        position_score = 1.5 if i < len(sentences) * 0.2 else 1.2 if i > len(sentences) * 0.8 else 1.0
        length_score = 1.2 if 10 <= len(sentence.split()) <= 25 else 0.8
        content_score = 1.3 if any(keyword in sentence.lower() for keyword in keywords) else 1.0
        total_score = position_score * length_score * content_score
        scores.append((i, total_score))
    
    return scores

def extract_top_sentences(sentences, scores, num_sentences=5):
    """Extract top-ranked sentences"""
    if not sentences or not scores:
        return []
        
    # Hardcoded number of sentences to extract
    num_sentences = min(5, len(sentences))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in scores[:num_sentences]])
    return [sentences[i] for i in top_indices]

def add_war_reporter_style(sentences):
    """Add war correspondent style to sentences"""
    if not sentences:
        return ""

    # Hardcoded transition phrases
    transition_phrases = [
        "Reports from the front line indicate",
        "Our correspondents on the ground confirm",
        "Breaking news from the conflict zone:",
        "Eyewitness accounts suggest",
        "According to military analysts,",
        "The situation remains tense as",
    ]

    # Use a fixed seed for reproducible results
    np.random.seed(42)
    
    styled_text = f"{np.random.choice(transition_phrases)} {sentences[0]}\n\n"
    for i, sentence in enumerate(sentences[1:-1], 1):
        if i % 2 == 0 and len(sentences) > 3:
            styled_text += f"{np.random.choice(transition_phrases)} {sentence} "
        else:
            styled_text += f"{sentence} "

    if len(sentences) > 1:
        styled_text += f"\n\nThe situation continues to develop as {sentences[-1].lower()}"

    return styled_text

def rule_based_script_generation(summary, headline, location):
    """Generate a war reporter script using a rule-based approach (no LLM required)"""
    # Hardcoded reporter names
    reporter_names = [
        "Alex Harker", "Morgan Wells", "Jamie Frost", "Casey Rivers", 
        "Taylor Stone", "Jordan Reed", "Sam Fletcher", "Riley Hayes"
    ]
    
    # Hardcoded sign-off phrases
    sign_offs = [
        "Reporting live from the frontlines, {reporter}.",
        "Back to you in the studio, this is {reporter}.",
        "This is {reporter}, reporting from {location}.",
        "For World News Network, this is {reporter} signing off.",
        "The situation remains fluid. From {location}, I'm {reporter} reporting."
    ]
    
    # Hardcoded sound effects
    sounds = [
        "[distant explosions]", "[helicopter overhead]", "[sirens wailing]",
        "[crowd noise]", "[wind howling]", "[radio static]", "[gunfire in distance]"
    ]

    # Use a fixed seed for reproducible results
    np.random.seed(42)
    
    reporter = np.random.choice(reporter_names)
    sign_off = np.random.choice(sign_offs).format(location=location or "the conflict zone", reporter=reporter)
    opening_sound = np.random.choice(sounds)
    middle_sound = np.random.choice(sounds)

    # Split summary into parts
    summary_parts = summary.split('\n\n')
    first_part = summary_parts[0] if summary_parts else summary
    rest_parts = ' '.join(summary_parts[1:]) if len(summary_parts) > 1 else ""

    script = (
        f"{opening_sound} This is {reporter}, reporting live from {location or 'the conflict zone'}. (pause)\n\n"
        f"{headline}. (pause)\n\n"
        f"{first_part}\n\n"
        f"{middle_sound} (pause)\n\n"
    )
    
    if rest_parts:
        script += f"{rest_parts}\n\n"
    
    script += f"{sign_off}"
    
    return script

def generate_war_reporter_script_groq(summary, headline, location):
    """Use Groq API to generate a personalized war reporter script"""
    prompt = f"""You are an experienced war correspondent reporting from dangerous conflict zones.

LOCATION: {location if location else "the conflict zone"}
HEADLINE: {headline}
SUMMARY: {summary}

Create a dramatic news report script with the following:
1. Introduce yourself with a unique war reporter persona name and briefly describe where you're reporting from.
2. Present the news in a dramatic, tense style typical of frontline reporting.
3. End with a signature sign-off phrase and your reporter name.

FORMAT YOUR RESPONSE AS A COMPLETE SCRIPT READY FOR TEXT-TO-SPEECH:
"""

    if not GROQ_API_KEY or GROQ_API_KEY == "your_hardcoded_api_key_here":
        print("Using hardcoded script generation (no API call).")
        return rule_based_script_generation(summary, headline, location)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an AI specialized in creating dramatic war correspondent scripts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            print(f"Groq API Error: {response.status_code}, {response.text}")
    
    except requests.RequestException as e:
        print(f"Groq API request failed: {str(e)}")

    print("Falling back to rule-based generation due to API failure.")
    return rule_based_script_generation(summary, headline, location)

def process_json_news(json_data):
    """Process news from JSON data"""
    try:
        # Extract values from the JSON data
        headline = json_data.get('headline', 'Breaking News from Conflict Zone')
        
        # Handle different possible formats of body_text
        body_text = ''
        raw_body_text = json_data.get('body_text', '')
        if isinstance(raw_body_text, list):
            body_text = ' '.join(raw_body_text)
        elif isinstance(raw_body_text, str):
            body_text = raw_body_text
        else:
            body_text = str(raw_body_text)
            
        location = json_data.get('location', 'the conflict zone')
        
        full_text = f"{headline}. {body_text}".strip()
        cleaned_text = re.sub(r'\s+', ' ', full_text)

        sentences = simple_sentence_tokenize(cleaned_text)
        scores = rank_sentences(sentences)
        top_sentences = extract_top_sentences(sentences, scores)

        war_reporter_summary = add_war_reporter_style(top_sentences)
        news_script = generate_war_reporter_script_groq(war_reporter_summary, headline, location)

        return {
            'headline': headline,
            'location': location,
            'summary': war_reporter_summary,
            'news_script': news_script,
            'original_text': body_text
        }

    except Exception as e:
        print(f"Error processing JSON: {str(e)}")
        # Return hardcoded fallback values
        fallback_headline = "Crisis in Conflict Zone Intensifies"
        fallback_location = "an undisclosed location"
        fallback_summary = "Reports from the front line indicate fighting has intensified in the region. \n\nCivilian casualties are reported as forces clash near populated areas. \n\nThe situation continues to develop as international observers call for immediate ceasefire."
        fallback_script = rule_based_script_generation(fallback_summary, fallback_headline, fallback_location)
        
        return {
            'headline': fallback_headline,
            'location': fallback_location,
            'summary': fallback_summary,
            'news_script': fallback_script,
            'error': str(e)
        }

# ---- ORIGINAL APP.PY FUNCTIONS WITH IMPROVEMENTS FROM COLAB ----

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def preprocess_image(image_path: str, enhance_contrast=True, denoise=True, auto_rotate=True) -> str:
    """
    Preprocess the image for better OCR accuracy - improved version from Colab code
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    processed_image = gray
    
    # Apply enhancements based on options
    if enhance_contrast:
        # Apply contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Save enhanced version for reference
        enhanced_path = os.path.join(app.config['UPLOAD_FOLDER'], f"enhanced_{os.path.basename(image_path)}")
        cv2.imwrite(enhanced_path, enhanced)
    
    # Apply adaptive thresholding first (improved order from Colab)
    binary = cv2.adaptiveThreshold(
        processed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    if denoise:
        # Apply denoising (after thresholding, as in Colab)
        processed_image = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
    else:
        processed_image = binary
    
    if auto_rotate:
        # TODO: Implement auto-rotation using text orientation detection
        # This is a complex feature and might require additional libraries
        pass
    
    # Save the preprocessed image to a temporary file
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"preprocessed_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, processed_image)
    
    return output_path

def extract_text_with_mistral_ocr(image_path: str, api_key: str, max_retries=3) -> Dict:
    """
    Extract text from the image using Mistral OCR with retry logic from Colab code
    """
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Read and encode the image
    image_file = Path(image_path)
    encoded = base64.b64encode(image_file.read_bytes()).decode()
    base64_data_url = f"data:image/jpeg;base64,{encoded}"
    
    # Implement retry with exponential backoff
    for attempt in range(max_retries):
        try:
            # Process the image using OCR
            image_response = client.ocr.process(
                document=ImageURLChunk(image_url=base64_data_url), 
                model="mistral-ocr-latest"
            )
            print(f"OCR API Time: {time.time() - start_time} seconds")
            
            # Get the markdown text from OCR
            ocr_markdown = ""
            if hasattr(image_response, 'pages') and len(image_response.pages) > 0:
                ocr_markdown = image_response.pages[0].markdown
            
            return {
                "raw_ocr": ocr_markdown,
                "response": image_response
            }
        except Exception as e:
            if attempt < max_retries - 1:  # Don't sleep on the last attempt
                # Exponential backoff with jitter
                sleep_time = (2 ** attempt) + random.random()
                print(f"OCR attempt {attempt+1} failed. Retrying in {sleep_time:.2f} seconds...")
                print(f"Error: {str(e)}")
                time.sleep(sleep_time)
            else:
                # Last attempt failed, log the error and return empty result
                print(f"All {max_retries} OCR attempts failed: {str(e)}")
                return {
                    "raw_ocr": "",
                    "error": str(e)
                }

def extract_newspaper_data(ocr_result: Dict, api_key: str) -> NewspaperStructure:
    """
    Extract structured data from OCR results
    """
    client = Mistral(api_key=api_key)
    
    # Get the raw OCR text
    ocr_text = ocr_result.get("raw_ocr", "")
    
    if not ocr_text:
        return NewspaperStructure(
            headline="No text detected",
            body_text=[]
        )
    
    # Process the OCR text to extract structured information
    try:
        chat_response = client.chat.complete(
            model="ministral-8b-latest",  # Using the smaller model since we don't need image input now
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    This is the OCR text from a newspaper:
                    
                    {ocr_text}
                    
                    Extract the following information in JSON format:
                    - headline: The main headline
                    - source: The news source (e.g. Associated Press)
                    - location: Location mentioned in the article (dateline)
                    - body_text: Array containing paragraphs of the article body
                    - date: Publication date if available
                    
                    The output should be strictly JSON with no extra commentary.
                    """
                },
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        # Parse the structured response
        structured_data = json.loads(chat_response.choices[0].message.content)
        
        # Create NewspaperStructure instance
        return NewspaperStructure(
            headline=structured_data.get("headline", "No headline detected"),
            source=structured_data.get("source", None),
            location=structured_data.get("location", None),
            body_text=structured_data.get("body_text", []),
            date=structured_data.get("date", None)
        )
    except Exception as e:
        # Log the error and return a basic structure
        print(f"Extraction Error: {str(e)}")
        return NewspaperStructure(
            headline="Error extracting structured data",
            body_text=[ocr_text.strip() if ocr_text else "No text extracted"]
        )

def process_sample_taliban_article():
    """
    Process a sample article when OCR fails - from Colab code
    """
    # Return structured data for the Taliban article as a fallback
    result = NewspaperStructure(
        headline="Taliban reject court move to arrest its top officials",
        source="Associated Press",
        location="KABUL",
        body_text=[
            "The Taliban on Friday rejected a court move to arrest two of their top officials for persecuting women, accusing the court of baseless accusations and misbehaviour.",
            "The International Criminal Court's chief prosecutor Karim Khan announced on Thursday he had requested arrest warrants for two top Taliban officials, including the leader Hibatullah Akhundzada.",
            "Since they took back control of the country in 2021, the Taliban have barred women from jobs, most public spaces and education beyond sixth grade.",
            "A Foreign Ministry statement condemned the ICC request."
        ],
        date=None  # Date not visible in the image
    )
    
    # Generate war reporter summary and script
    structure_dict = result.dict()
    full_text = f"{result.headline}. {' '.join(result.body_text)}".strip()
    sentences = simple_sentence_tokenize(full_text)
    scores = rank_sentences(sentences)
    top_sentences = extract_top_sentences(sentences, scores)
    
    war_reporter_summary = add_war_reporter_style(top_sentences)
    news_script = generate_war_reporter_script_groq(war_reporter_summary, result.headline, result.location)
    
    result.summary = war_reporter_summary
    result.news_script = news_script
    
    return result

def generate_war_reporter_summary(structure: NewspaperStructure) -> NewspaperStructure:
    """
    Generate war reporter style summary and script from the newspaper structure
    """
    # Combine headline and body text
    full_text = f"{structure.headline}. {' '.join(structure.body_text)}".strip()
    cleaned_text = re.sub(r'\s+', ' ', full_text)
    
    # Tokenize and rank sentences
    sentences = simple_sentence_tokenize(cleaned_text)
    scores = rank_sentences(sentences)
    top_sentences = extract_top_sentences(sentences, scores)
    
    # Generate summary and script
    war_reporter_summary = add_war_reporter_style(top_sentences)
    location = structure.location or "the conflict zone"
    
    # Generate script using Groq API or fallback to rule-based
    news_script = generate_war_reporter_script_groq(
        war_reporter_summary, 
        structure.headline, 
        location
    )
    
    # Add summary and script to the structure
    structure.summary = war_reporter_summary
    structure.news_script = news_script
    
    return structure

@app.route('/api/process', methods=['POST'])
def process_image():
    # Check if the post request has the file part
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get processing options
        options = json.loads(request.form.get('options', '{}'))
        
        try:
            # Preprocess the image
            preprocessed_path = preprocess_image(
                filepath, 
                enhance_contrast=options.get('enhanceContrast', True),
                denoise=options.get('denoise', True),
                auto_rotate=options.get('autoRotate', True)
            )
            
            # Extract text with OCR
            ocr_result = extract_text_with_mistral_ocr(preprocessed_path, MISTRAL_API_KEY)
            
            # Handle OCR failure with sample data
            if not ocr_result.get("raw_ocr") and "error" in ocr_result:
                print(f"OCR failed: {ocr_result.get('error')}")
                print("Using sample article data as fallback")
                sample_structure = process_sample_taliban_article()
                result = sample_structure.dict()
                result['is_fallback'] = True  # Flag to indicate this is fallback data
                
                # Clean up temporary files
                try:
                    os.remove(filepath)
                    os.remove(preprocessed_path)
                except:
                    pass
                
                return jsonify(result)
            
            # Extract structured data if requested
            result = None
            if options.get('extractStructure', True):
                structure = extract_newspaper_data(ocr_result, MISTRAL_API_KEY)
                
                # Generate war reporter summary and script if enabled
                if options.get('generateWarReport', True):
                    result_with_summary = generate_war_reporter_summary(structure)
                    result = result_with_summary.dict()
                else:
                    result = structure.dict()
            else:
                # Just return the OCR text
                result = {
                    'raw_ocr': ocr_result.get('raw_ocr', ''),
                }
            
            # Clean up temporary files
            try:
                os.remove(filepath)
                os.remove(preprocessed_path)
            except:
                pass
                
            return jsonify(result)
            
        except Exception as e:
            print(f"Processing error: {str(e)}")
            # Try to use sample data as fallback on any error
            try:
                sample_structure = process_sample_taliban_article()
                result = sample_structure.dict()
                result['is_fallback'] = True
                result['error'] = str(e)
                return jsonify(result)
            except:
                # If even the fallback fails, return error
                return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/summarize', methods=['POST'])
def summarize_article():
    """
    Endpoint to summarize an already extracted article
    """
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Process the JSON data directly for war reporter summary
        result = process_json_news(data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_json', methods=['POST'])
def process_json():
    """
    New endpoint that directly processes a JSON file with OCR results
    """
    try:
        # Get JSON data from request
        json_data = request.json
        
        if not json_data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Process the JSON data
        result = process_json_news(json_data)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)