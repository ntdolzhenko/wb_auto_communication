
from flask import Blueprint, render_template, request
from app.model import generate_answer  

main = Blueprint('main', __name__)

@main.route('/', methods=['GET'])
def index():
    return render_template('index_new.html')

@main.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = generate_answer(question) 
    return render_template('index_new.html', question=question, answer=answer)

from flask import jsonify

@main.route('/answer', methods=['POST'])
def api_answer():
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'Вопрос не передан'}), 400

    try:
        answer = generate_answer(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
