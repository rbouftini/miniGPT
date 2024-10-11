from flask import Flask, render_template, request, Response, stream_with_context
import sys
import os

# Add parent directory to sys.path to import generate.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import generate  # Make sure generate.py has the generate_completion function

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/generate', methods=['POST'])
def generate_text():
    prompt = request.form['prompt']
    
    def generate_tokens():
        for token in generate.generate_completion(prompt):
            yield token
            sys.stdout.flush()  # Ensure token is flushed out immediately

    return Response(stream_with_context(generate_tokens()), mimetype='text/plain')

if __name__ == '__main__':
    app.run(debug=True)
