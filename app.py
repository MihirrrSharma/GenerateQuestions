from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = TFAutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

@app.route('/')
def home():
    return "Welcome to the Question Generation API!"

@app.route('/generate_questions', methods=['GET'])
def generate_questions():
    try:
        text = request.args.get('text', '')

        if not text:
            return jsonify({'error': 'Input text is required'}), 400

        generated_questions = generator(text)

        questions = [q['generated_text'].strip() for q in generated_questions]

        return jsonify({'questions': questions})

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return jsonify({'error': error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)
