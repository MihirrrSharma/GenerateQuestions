from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM

app = Flask(__name__)

# Load tokenizer and model for question generation
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = TFAutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return "Welcome to the Question Generation API!"

@app.route('/generate_questions', methods=['GET'])
def generate_questions():
    try:
        # Get input text from request parameters
        text = request.args.get('text', '')

        # Validate input text
        if not text:
            return jsonify({'error': 'Input text is required'}), 400

        # Generate questions using the pipeline
        generated_questions = generator(text)

        # Extract the list of generated questions from the pipeline response
        questions = [q['generated_text'].strip() for q in generated_questions]

        # Return the list of generated questions as a JSON response
        return jsonify({'questions': questions})

    except Exception as e:
        # Handle any exceptions gracefully and return an error response
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
