from flask import Flask, render_template, request, jsonify, send_file
import google.generativeai as genai
import os
import json
import re
from functools import wraps
import time
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from io import BytesIO
import base64
import hashlib

# Load environment variables
load_dotenv()

# ========================
# CONFIGURATION
# ========================
app = Flask(__name__)

# Secure API key handling
GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable must be set")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"

# Cache for generated images
image_cache = {}

# Rate limiting
request_timestamps = {}
RATE_LIMIT = 10
RATE_WINDOW = 60

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.remote_addr
        current_time = time.time()
        
        if client_ip not in request_timestamps:
            request_timestamps[client_ip] = []
        
        request_timestamps[client_ip] = [
            ts for ts in request_timestamps[client_ip]
            if current_time - ts < RATE_WINDOW
        ]
        
        if len(request_timestamps[client_ip]) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
        
        request_timestamps[client_ip].append(current_time)
        return f(*args, **kwargs)
    
    return decorated_function

# ========================
# IMAGE GENERATION
# ========================

def generate_geometric_figure(shape_type, params):
    """Generate geometric shapes"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    if shape_type == "triangle":
        # Triangle with given sides or angles
        points = np.array([[0, 0], [params.get('base', 5), 0], 
                          [params.get('base', 5)/2, params.get('height', 4)]])
        triangle = patches.Polygon(points, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(triangle)
        ax.plot(*points.T, 'o', color='red')
        
        # Add labels
        if params.get('show_labels'):
            ax.text(points[0][0], points[0][1]-0.5, 'A', ha='center', fontsize=12, fontweight='bold')
            ax.text(points[1][0], points[1][1]-0.5, 'B', ha='center', fontsize=12, fontweight='bold')
            ax.text(points[2][0], points[2][1]+0.5, 'C', ha='center', fontsize=12, fontweight='bold')
    
    elif shape_type == "rectangle":
        width = params.get('width', 6)
        height = params.get('height', 4)
        rect = patches.Rectangle((0, 0), width, height, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        
        if params.get('show_dimensions'):
            ax.text(width/2, -0.5, f'{width}', ha='center', fontsize=10)
            ax.text(-0.5, height/2, f'{height}', va='center', fontsize=10)
    
    elif shape_type == "circle":
        radius = params.get('radius', 3)
        circle = patches.Circle((radius, radius), radius, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(circle)
        
        if params.get('show_radius'):
            ax.plot([radius, radius+radius], [radius, radius], 'r--', linewidth=1.5)
            ax.text(radius+radius/2, radius+0.3, f'r={radius}', ha='center', fontsize=10)
        
        ax.set_xlim(-1, radius*2+1)
        ax.set_ylim(-1, radius*2+1)
    
    elif shape_type == "coordinate_points":
        points = params.get('points', [[1,2], [3,4], [5,1]])
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], s=100, c='red', zorder=5)
        
        for i, (x, y) in enumerate(points):
            ax.text(x+0.2, y+0.2, f'({x},{y})', fontsize=9)
    
    elif shape_type == "angle":
        angle_deg = params.get('angle', 45)
        # Draw angle
        ax.plot([0, 3], [0, 0], 'b-', linewidth=2)
        x_end = 3 * np.cos(np.radians(angle_deg))
        y_end = 3 * np.sin(np.radians(angle_deg))
        ax.plot([0, x_end], [0, y_end], 'b-', linewidth=2)
        
        # Draw arc
        theta = np.linspace(0, np.radians(angle_deg), 50)
        arc_x = 0.5 * np.cos(theta)
        arc_y = 0.5 * np.sin(theta)
        ax.plot(arc_x, arc_y, 'r-', linewidth=1.5)
        ax.text(0.7, 0.3, f'{angle_deg}°', fontsize=11, fontweight='bold')
        
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 3.5)
    
    ax.set_xlabel('X', fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title(params.get('title', 'Figure'), fontsize=12, fontweight='bold')
    
    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

def generate_chart(chart_type, data):
    """Generate charts and graphs"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if chart_type == "bar":
        categories = data.get('categories', ['A', 'B', 'C', 'D'])
        values = data.get('values', [10, 25, 15, 30])
        ax.bar(categories, values, color='steelblue', edgecolor='black')
        ax.set_ylabel('Values', fontsize=10)
        ax.set_xlabel('Categories', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    elif chart_type == "line":
        x = data.get('x', list(range(10)))
        y = data.get('y', [i**2 for i in range(10)])
        ax.plot(x, y, marker='o', linewidth=2, markersize=6, color='darkblue')
        ax.set_ylabel('Y', fontsize=10)
        ax.set_xlabel('X', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    elif chart_type == "pie":
        labels = data.get('labels', ['A', 'B', 'C', 'D'])
        sizes = data.get('sizes', [25, 30, 20, 25])
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    
    elif chart_type == "scatter":
        x = data.get('x', [1, 2, 3, 4, 5])
        y = data.get('y', [2, 4, 5, 4, 6])
        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black')
        ax.set_ylabel('Y', fontsize=10)
        ax.set_xlabel('X', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    ax.set_title(data.get('title', 'Chart'), fontsize=12, fontweight='bold')
    
    # Convert to base64
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return f"data:image/png;base64,{img_base64}"

# ========================
# VALIDATION
# ========================
def validate_quiz_params(data):
    """Validate quiz generation parameters"""
    errors = []
    
    topic = data.get("topic", "").strip()
    if not topic or len(topic) < 2:
        errors.append("Topic must be at least 2 characters")
    if len(topic) > 100:
        errors.append("Topic must be less than 100 characters")
    
    grade = data.get("grade", "").strip()
    if not grade:
        errors.append("Grade is required")
    try:
        grade_num = int(grade)
        if not (2 <= grade_num <= 12):
            errors.append("Grade must be between 2 and 12")
    except ValueError:
        errors.append("Grade must be a number")
    
    difficulty = data.get("difficulty", "Medium").strip()
    if difficulty not in ["Easy", "Medium", "Hard"]:
        errors.append("Difficulty must be Easy, Medium, or Hard")
    
    count = data.get("count", 5)
    try:
        count = int(count)
        if not (1 <= count <= 20):
            errors.append("Question count must be between 1 and 20")
    except (ValueError, TypeError):
        errors.append("Count must be a number")
    
    quiz_type = data.get("quizType", "written").strip()
    if quiz_type not in ["written", "mcq", "mixed"]:
        errors.append("Quiz type must be written, mcq, or mixed")
    
    return errors, {
        "topic": topic,
        "grade": grade,
        "difficulty": difficulty,
        "count": count,
        "quiz_type": quiz_type
    }

def extract_json_from_response(text):
    """Robustly extract JSON from Gemini response"""
    # Remove common markdown formatting
    text = text.strip()
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```, '', text)

# ========================
# ROUTES
# ========================

@app.route('/')
def index():
    return render_template('quiz.html')

@app.route('/generate', methods=['POST'])
@rate_limit
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        
        errors, validated_data = validate_quiz_params(data)
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400
        
        # Check if topic might need images
        needs_images = any(word in validated_data['topic'].lower() for word in 
                          ['geometry', 'graph', 'coordinate', 'shape', 'triangle', 
                           'circle', 'rectangle', 'angle', 'chart', 'data'])
        
        # Construct prompt - simplified to avoid image generation issues initially
        prompt = f"""
You are an expert math teacher creating quizzes for Grade {validated_data['grade']} students.

Generate {validated_data['count']} {validated_data['difficulty'].lower()} level math questions on the topic "{validated_data['topic']}".

The quiz type is "{validated_data['quiz_type']}".
- If type is "mcq": give 4 answer options and specify the correct one.
- If type is "written": give direct-answer questions (no options).
- If type is "mixed": mix both styles randomly.

CRITICAL: Use LaTeX math notation wrapped in \\( \\) for inline math.

Examples:
- Fractions: \\(\\frac{{2}}{{5}}\\) NOT 2/5
- Powers: \\(x^2\\) NOT x^2
- Square roots: \\(\\sqrt{{16}}\\) NOT sqrt(16)

Return ONLY a valid JSON array, no other text:
[
  {{"question": "What is \\\\(2^2 + 3^2\\\\)?", "options": ["8", "10", "13", "14"], "answer": "13"}},
  {{"question": "Simplify \\\\(\\\\frac{{6}}{{3}} + 2\\\\)", "options": [], "answer": "4"}}
]

Make sure each question has: question, options (array), and answer fields.
"""

        model = genai.GenerativeModel(MODEL_NAME)
        
        # Add generation config for more reliable JSON output
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if not response or not response.text:
            return jsonify({"error": "Empty response from AI model"}), 500
        
        # More robust JSON extraction
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        questions = extract_json_from_response(text)
        
        if not questions or not isinstance(questions, list):
            # Log the actual response for debugging
            print(f"Failed to parse response: {text[:500]}")
            return jsonify({"error": "Could not parse questions. Please try again."}), 500
        
        if len(questions) == 0:
            return jsonify({"error": "No questions were generated. Please try again."}), 500
        
        # Validate and process questions
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return jsonify({"error": f"Question {i+1} is not a valid object"}), 500
            
            if 'question' not in q:
                return jsonify({"error": f"Question {i+1} is missing 'question' field"}), 500
            
            if 'answer' not in q:
                return jsonify({"error": f"Question {i+1} is missing 'answer' field"}), 500
            
            # Ensure options field exists
            if 'options' not in q:
                q['options'] = []
            
            # Set image_data to None by default (images disabled for now to fix the error)
            q['image_data'] = None
            
            # TODO: Re-enable image generation after basic functionality is working
            # if q.get('image'):
            #     img_spec = q['image']
            #     try:
            #         if img_spec['type'] in ['triangle', 'rectangle', 'circle', 'angle', 'coordinate_points']:
            #             q['image_data'] = generate_geometric_figure(img_spec['type'], img_spec.get('params', {}))
            #     except Exception as e:
            #         print(f"Error generating image: {e}")
            #         q['image_data'] = None
        
        return jsonify({"questions": questions})
    
    except Exception as e:
        app.logger.error(f"Error in generate: {str(e)}")
        return jsonify({"error": "An error occurred while generating the quiz"}), 500

@app.route('/evaluate', methods=['POST'])
@rate_limit
def evaluate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        
        qa_pairs = data.get("qa_pairs", [])
        
        if not qa_pairs or not isinstance(qa_pairs, list):
            return jsonify({"error": "No answers provided"}), 400
        
        if len(qa_pairs) > 20:
            return jsonify({"error": "Too many questions to evaluate"}), 400
        
        eval_prompt = """
You are an intelligent math evaluator. For each question-answer pair, check if the student's answer is correct.

Be lenient with formatting differences (e.g., "1/2" vs "0.5" vs "½").
Consider mathematically equivalent answers as correct.

Return ONLY valid JSON with no additional text:
[
  {{"result": "Correct", "explanation": "2² + 3² = 4 + 9 = 13"}},
  {{"result": "Incorrect", "explanation": "Student wrote 4, but correct answer is 6"}}
]

Data:
""" + json.dumps(qa_pairs, indent=2)

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(eval_prompt)
        
        if not response or not response.text:
            return jsonify({"error": "Empty response from AI model"}), 500
        
        evaluation = extract_json_from_response(response.text.strip())
        
        if not evaluation or not isinstance(evaluation, list):
            return jsonify({"error": "Invalid response from evaluator"}), 500
        
        for i, e in enumerate(evaluation):
            if not isinstance(e, dict) or 'result' not in e or 'explanation' not in e:
                return jsonify({"error": f"Evaluation {i+1} has invalid structure"}), 500
        
        return jsonify({"evaluation": evaluation})
    
    except Exception as e:
        app.logger.error(f"Error in evaluate: {str(e)}")
        return jsonify({"error": "An error occurred while evaluating answers"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
, '', text, flags=re.MULTILINE)
    
    # Try to find JSON array with better regex
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Problematic text: {json_match.group(0)[:200]}")
    
    # Try to parse entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find content between first [ and last ]
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        try:
            json_str = text[first_bracket:last_bracket+1]
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Last resort: try to fix common JSON issues
    try:
        # Replace single quotes with double quotes
        text = text.replace("'", '"')
        # Remove trailing commas
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None

# ========================
# ROUTES
# ========================

@app.route('/')
def index():
    return render_template('quiz.html')

@app.route('/generate', methods=['POST'])
@rate_limit
def generate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        
        errors, validated_data = validate_quiz_params(data)
        if errors:
            return jsonify({"error": "; ".join(errors)}), 400
        
        # Check if topic might need images
        needs_images = any(word in validated_data['topic'].lower() for word in 
                          ['geometry', 'graph', 'coordinate', 'shape', 'triangle', 
                           'circle', 'rectangle', 'angle', 'chart', 'data'])
        
        # Construct prompt - simplified to avoid image generation issues initially
        prompt = f"""
You are an expert math teacher creating quizzes for Grade {validated_data['grade']} students.

Generate {validated_data['count']} {validated_data['difficulty'].lower()} level math questions on the topic "{validated_data['topic']}".

The quiz type is "{validated_data['quiz_type']}".
- If type is "mcq": give 4 answer options and specify the correct one.
- If type is "written": give direct-answer questions (no options).
- If type is "mixed": mix both styles randomly.

CRITICAL: Use LaTeX math notation wrapped in \\( \\) for inline math.

Examples:
- Fractions: \\(\\frac{{2}}{{5}}\\) NOT 2/5
- Powers: \\(x^2\\) NOT x^2
- Square roots: \\(\\sqrt{{16}}\\) NOT sqrt(16)

Return ONLY a valid JSON array, no other text:
[
  {{"question": "What is \\\\(2^2 + 3^2\\\\)?", "options": ["8", "10", "13", "14"], "answer": "13"}},
  {{"question": "Simplify \\\\(\\\\frac{{6}}{{3}} + 2\\\\)", "options": [], "answer": "4"}}
]

Make sure each question has: question, options (array), and answer fields.
"""

        model = genai.GenerativeModel(MODEL_NAME)
        
        # Add generation config for more reliable JSON output
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
        }
        
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if not response or not response.text:
            return jsonify({"error": "Empty response from AI model"}), 500
        
        # More robust JSON extraction
        text = response.text.strip()
        
        # Remove markdown code blocks if present
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        questions = extract_json_from_response(text)
        
        if not questions or not isinstance(questions, list):
            # Log the actual response for debugging
            print(f"Failed to parse response: {text[:500]}")
            return jsonify({"error": "Could not parse questions. Please try again."}), 500
        
        if len(questions) == 0:
            return jsonify({"error": "No questions were generated. Please try again."}), 500
        
        # Validate and process questions
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return jsonify({"error": f"Question {i+1} is not a valid object"}), 500
            
            if 'question' not in q:
                return jsonify({"error": f"Question {i+1} is missing 'question' field"}), 500
            
            if 'answer' not in q:
                return jsonify({"error": f"Question {i+1} is missing 'answer' field"}), 500
            
            # Ensure options field exists
            if 'options' not in q:
                q['options'] = []
            
            # Set image_data to None by default (images disabled for now to fix the error)
            q['image_data'] = None
            
            # TODO: Re-enable image generation after basic functionality is working
            # if q.get('image'):
            #     img_spec = q['image']
            #     try:
            #         if img_spec['type'] in ['triangle', 'rectangle', 'circle', 'angle', 'coordinate_points']:
            #             q['image_data'] = generate_geometric_figure(img_spec['type'], img_spec.get('params', {}))
            #     except Exception as e:
            #         print(f"Error generating image: {e}")
            #         q['image_data'] = None
        
        return jsonify({"questions": questions})
    
    except Exception as e:
        app.logger.error(f"Error in generate: {str(e)}")
        return jsonify({"error": "An error occurred while generating the quiz"}), 500

@app.route('/evaluate', methods=['POST'])
@rate_limit
def evaluate():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request data"}), 400
        
        qa_pairs = data.get("qa_pairs", [])
        
        if not qa_pairs or not isinstance(qa_pairs, list):
            return jsonify({"error": "No answers provided"}), 400
        
        if len(qa_pairs) > 20:
            return jsonify({"error": "Too many questions to evaluate"}), 400
        
        eval_prompt = """
You are an intelligent math evaluator. For each question-answer pair, check if the student's answer is correct.

Be lenient with formatting differences (e.g., "1/2" vs "0.5" vs "½").
Consider mathematically equivalent answers as correct.

Return ONLY valid JSON with no additional text:
[
  {{"result": "Correct", "explanation": "2² + 3² = 4 + 9 = 13"}},
  {{"result": "Incorrect", "explanation": "Student wrote 4, but correct answer is 6"}}
]

Data:
""" + json.dumps(qa_pairs, indent=2)

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(eval_prompt)
        
        if not response or not response.text:
            return jsonify({"error": "Empty response from AI model"}), 500
        
        evaluation = extract_json_from_response(response.text.strip())
        
        if not evaluation or not isinstance(evaluation, list):
            return jsonify({"error": "Invalid response from evaluator"}), 500
        
        for i, e in enumerate(evaluation):
            if not isinstance(e, dict) or 'result' not in e or 'explanation' not in e:
                return jsonify({"error": f"Evaluation {i+1} has invalid structure"}), 500
        
        return jsonify({"evaluation": evaluation})
    
    except Exception as e:
        app.logger.error(f"Error in evaluate: {str(e)}")
        return jsonify({"error": "An error occurred while evaluating answers"}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':

    app.run(debug=False, host='0.0.0.0', port=5000)
