from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import subprocess
import webbrowser
import cv2
import detect as capture_age_and_gender  # Assuming this contains the capture logic

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for Flask sessions

video = cv2.VideoCapture(0)
global captured_age, captured_gender

@app.route('/')
def index():
    return render_template('index.html')  # First HTML page for starting the detection

@app.route('/run_python', methods=['POST'])
def run_python():
    try:
        # Run the Gender and Age Detection Python script
        result = subprocess.run(['python', 'detect.py'], capture_output=True, text=True)
        
        # Capture the output from the Python script (gender and age data)
        output = result.stdout.strip()
        
        # Debugging line: print the output to verify it's in the correct format
        print(f"Output from detection: {output}")
        
        # Check if the output contains the correct format
        if ", " not in output:
            return f"Error: Unexpected output format from detection script: {output}"

        # Assuming the output of your script contains gender and age information
        gender, age = output.split(', ')  # Example output: "Male, (25-32)"
        age = age[1:-1]  # Remove parentheses around age
        
        # Store the captured age and gender in session
        session['gender'] = gender
        session['age'] = age
        
        # Redirect to the form page with gender and age parameters
        return redirect(url_for('home'))
    except Exception as e:
        return f"Error: {str(e)}"
    
@app.route('/capture', methods=['POST'])
def capture():
      
    try:
        # Capture age and gender code here
        age, gender = capture_age_and_gender()
        
        if age is None or gender is None:
            return jsonify({"error": "No face detected or unable to capture data"}), 400  # Handle errors
        
        # Store the captured age and gender in session
        session['gender'] = gender
        session['age'] = age

        return jsonify({"success": True, "age": captured_age, "gender": captured_gender})
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Handle any errors
    
@app.route('/home')
def home():
    # Access the captured age and gender from session
    captured_age = session.get('age', 'Unknown')
    captured_gender = session.get('gender', 'Unknown')
    
    # Render your main page with the captured age and gender
    return render_template('capture_page.html', age=captured_age, gender=captured_gender)

# Main function to run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
