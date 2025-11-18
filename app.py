from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import os
import json
import re
from functools import wraps
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========================
# CONFIGURATION
# ========================
app = Flask(__name__)

GENAI_API_KEY = os.environ.get("GENAI_API_KEY")
if not GENAI_API_KEY:
    raise ValueError("GENAI_API_KEY environment variable must be set")

genai.configure(api_key=GENAI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"

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
# HELPER FUNCTIONS
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
    """Extract JSON array from AI response"""
    text = text.strip()
    
    # Remove markdown code blocks
    text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text, flags=re.MULTILINE)
    
    # Find first [ and last ]
    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        text = text[first_bracket:last_bracket+1]
    else:
        return None
    
    # Try to parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try fixing common issues
    try:
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        result = json.loads(text)
        if isinstance(result, list):
            return result
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
        
        # Build prompt
        prompt = f"""Create {validated_data['count']} math questions for Grade {validated_data['grade']} about "{validated_data['topic']}".

Difficulty: {validated_data['difficulty']}
Quiz type: {validated_data['quiz_type']}

Rules:
- If quiz type is "mcq": provide 4 options
- If quiz type is "written": leave options empty []
- If quiz type is "mixed": alternate between mcq and written
- Use LaTeX notation: \\(\\frac{{a}}{{b}}\\) for fractions, \\(x^2\\) for powers, \\(\\sqrt{{x}}\\) for roots

RETURN ONLY THIS JSON (no extra text):
[
  {{"question": "What is \\\\(2 + 3\\\\)?", "options": ["3", "4", "5", "6"], "answer": "5"}},
  {{"question": "Solve \\\\(\\\\frac{{6}}{{2}}\\\\)", "options": [], "answer": "3"}}
]"""

        model = genai.GenerativeModel(MODEL_NAME)
        
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 40,
        }
        
        # Try up to 3 times
        questions = None
        last_error = None
        
        for attempt in range(3):
            try:
                response = model.generate_content(prompt, generation_config=generation_config)
                
                if not response or not response.text:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    return jsonify({"error": "Empty response from AI"}), 500
                
                text = response.text.strip()
                questions = extract_json_from_response(text)
                
                if questions and len(questions) > 0:
                    break
                
                last_error = f"Could not parse response (attempt {attempt + 1})"
                if attempt < 2:
                    time.sleep(1)
                    
            except Exception as e:
                last_error = str(e)
                if attempt < 2:
                    time.sleep(1)
                else:
                    return jsonify({"error": f"Failed to generate quiz: {last_error}"}), 500
        
        if not questions or not isinstance(questions, list) or len(questions) == 0:
            return jsonify({"error": "Could not generate valid questions. Please try again."}), 500
        
        # Validate each question
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                return jsonify({"error": f"Question {i+1} is invalid"}), 500
            
            if 'question' not in q:
                return jsonify({"error": f"Question {i+1} missing 'question' field"}), 500
            
            if 'answer' not in q:
                return jsonify({"error": f"Question {i+1} missing 'answer' field"}), 500
            
            if 'options' not in q:
                q['options'] = []
        
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
        
        eval_prompt = """Evaluate these student answers. Be lenient with formatting (e.g., "0.5" = "1/2").

Return ONLY this JSON (no extra text):
[
  {{"result": "Correct", "explanation": "Brief explanation"}},
  {{"result": "Incorrect", "explanation": "Brief explanation"}}
]

Data:
""" + json.dumps(qa_pairs, indent=2)

        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(eval_prompt)
        
        if not response or not response.text:
            return jsonify({"error": "Empty response from AI"}), 500
        
        evaluation = extract_json_from_response(response.text.strip())
        
        if not evaluation or not isinstance(evaluation, list):
            return jsonify({"error": "Could not evaluate answers. Please try again."}), 500
        
        for i, e in enumerate(evaluation):
            if not isinstance(e, dict) or 'result' not in e or 'explanation' not in e:
                return jsonify({"error": f"Evaluation {i+1} is invalid"}), 500
        
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