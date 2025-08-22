# Основная логика: загрузка данных, поиск, генерация ответа

# app/model.py
import faiss
import pandas as pd
import numpy as np
import ollama
import pickle
import scipy.sparse
import nltk
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


client = ollama.Client(host='http://k1.rxs77.net:14360')
embedding_dim = 768

# Загрузка эмбеддингов
embed_qa = pd.read_csv('data/embed_df_qa.csv')
embed_chunks = pd.read_csv('data/embed_df_chunk.csv')

# Загрузка индексов
index_qa = faiss.read_index('data/index_questions.faiss')
index_chunks = faiss.read_index('data/index_chunks.faiss')

# Загрузка текстов
df_qa = pd.read_csv('data/qa_df.csv')
df_chunks = pd.read_csv('data/chunk_df.csv')

# Загрузка TF-IDF индексов
# Загружаем векторизаторы
with open('data/vectorizer_q.pkl', 'rb') as f:
    vectorizer_q = pickle.load(f)

with open('data/vectorizer_kb.pkl', 'rb') as f:
    vectorizer_kb = pickle.load(f)

# Загружаем матрицы
tfidf_matrix_q = scipy.sparse.load_npz('data/tfidf_matrix_q.npz')
tfidf_matrix_kb = scipy.sparse.load_npz('data/tfidf_matrix_kb.npz')

# Загрузка стоп-слов для русского языка
nltk.download('stopwords')

# Инициализация лемматизатора
morph = MorphAnalyzer()

# Получаем множество стоп-слов русского языка
russian_stopwords = set(stopwords.words('russian'))

# получение эмбеддинга через Ollama
def getEmbeddingsOllama(text):

    response = client.embed(
        model='deeppavlov:latest',
        input=[text]
    )

    embedding = response['embeddings'][0]
    return np.array(embedding)

def lemmatize_and_remove_stopwords(text):
    tokens = text.split()
    
    # Лемматизация и удаление стоп-слов
    lemmas = []
    for token in tokens:
        if token not in russian_stopwords:
            lemma = morph.parse(token)[0].normal_form
            if lemma not in russian_stopwords:
                lemmas.append(lemma)
    return ' '.join(lemmas)


# Функция поиска релевантных QA и чанков

def retrieve_hybrid_context(question, top_q=5, top_kb=5):
    # Настройки
    SIMILARITY_THRESHOLD = 0.5
    ALPHA = 0.5  # вес семантической части
    BETA = 0.5   # вес лексической части

    # Эмбеддинг нового вопроса 
    q_embedding = getEmbeddingsOllama(question)
    q_embedding_norm = q_embedding / np.linalg.norm(q_embedding)

    # Семантический поиск (FAISS)
    scores_q, idxs_q = index_qa.search(q_embedding_norm.reshape(1, -1), top_q * 2)
    scores_kb, idxs_kb = index_chunks.search(q_embedding_norm.reshape(1, -1), top_kb * 2)

    question_for_tfidf = lemmatize_and_remove_stopwords(question)
    # Лексический поиск (TF-IDF) 
    tfidf_q = vectorizer_q.transform([question_for_tfidf])
    tfidf_kb = vectorizer_kb.transform([question_for_tfidf])

    lexical_scores_q = cosine_similarity(tfidf_q, tfidf_matrix_q).flatten()
    lexical_scores_kb = cosine_similarity(tfidf_kb, tfidf_matrix_kb).flatten()

    # Гибридное ранжирование вопросов
    hybrid_scores_q = []
    for i, sem_score in zip(idxs_q[0], scores_q[0]):
        lex_score = lexical_scores_q[i]
        hybrid_score = ALPHA * sem_score + BETA * lex_score
        hybrid_scores_q.append((i, hybrid_score))

    hybrid_scores_q = sorted(hybrid_scores_q, key=lambda x: -x[1])[:top_q]
    # similar_answers = [df_qa.iloc[i]['answer'] for i, s in hybrid_scores_q if s >= SIMILARITY_THRESHOLD]
    similar_answers = [df_qa.iloc[i]['answer'] for i, s in hybrid_scores_q]

    # similar_questions = [df_qa.iloc[i]['question'] for i, s in hybrid_scores_q if s >= SIMILARITY_THRESHOLD]
    similar_questions = [df_qa.iloc[i]['question'] for i, s in hybrid_scores_q]

    # Гибридное ранжирование чанков 
    hybrid_scores_kb = []
    for i, sem_score in zip(idxs_kb[0], scores_kb[0]):
        lex_score = lexical_scores_kb[i]
        hybrid_score = ALPHA * sem_score + BETA * lex_score
        hybrid_scores_kb.append((i, hybrid_score))

    hybrid_scores_kb = sorted(hybrid_scores_kb, key=lambda x: -x[1])[:top_kb]
    # similar_chunks = [chunked_df_kb_new.iloc[i]['chunk'] for i, s in hybrid_scores_kb if s >= SIMILARITY_THRESHOLD]
    similar_chunks = [df_chunks.iloc[i]['chunk'] for i, s in hybrid_scores_kb]

    return [similar_questions, similar_answers, similar_chunks]
    # return [similar_chunks]


# Генерация контекста
def build_context(qa_context):
    similar_questions, similar_answers, relevant_chunks = qa_context

    context_blocks = []

    for i, (q, a) in enumerate(zip(similar_questions, similar_answers), start=1):
        block = f"""--- Похожий вопрос №{i} ---
Вопрос: {q}
Ответ: {a}"""
        context_blocks.append(block)

    for i, chunk in enumerate(relevant_chunks, start=1):
        block = f"""--- Фрагмент регламента №{i} ---
{chunk}"""
        context_blocks.append(block)

    return "\n\n".join(context_blocks)

# Генерация промта пользователя
def generate_user_prompt(question, context):
    prompt = f"""Вопрос: <QUESTION>{question}</QUESTION>
Контекст: <CONTEXT>{context}</CONTEXT>"""
    return prompt


# Генерация ответа
def generate_answer(question):

    with open('data/system_rag_prompt.txt', 'rb') as f:
        system_with_rag_prompt = f.read()
    f.close()

    context = retrieve_hybrid_context(question)

    user_prompt = generate_user_prompt(question, context)

    client = ollama.Client(host='http://k1.rxs77.net:14360')

    response = client.chat(
        model='gemma3:12b', 
        messages=[
        {"role": "system", "content": system_with_rag_prompt},
        {"role": "user", "content": user_prompt}
    ])
    # print("\n\SUCCESSFUL ANSWER: ", response['message']['content'])
    # print("\nCONTEXT: ", user_prompt)
    return response['message']['content']