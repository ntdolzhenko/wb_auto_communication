import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import ollama


client = ollama.Client(host='http://k1.rxs77.net:14360')

df = pd.read_csv('results_with_model_answers.csv')
# получение эмбеддинга через Ollama
def getEmbeddingsOllama(text):

    response = client.embed(
        model='deeppavlov:latest',
        input=[text]
    )

    embedding = response['embeddings'][0]
    return np.array(embedding)


# Считаем cosine similarity между true_answer и model_answer
similarities = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        emb_true = getEmbeddingsOllama(row['true_answer'])
        emb_model = getEmbeddingsOllama(row['model_answer'])

        sim = cosine_similarity([emb_true], [emb_model])[0][0]
        similarities.append(sim)
    except Exception as e:
        print(f"[!] Ошибка на id={row['id']}: {e}")
        similarities.append(np.nan)

# Сохраняем результат
df['semantic_similarity'] = similarities

# Усреднённая метрика по всем успешным примерам
average_score = df['semantic_similarity'].mean()
print(f"\n\n Средняя семантическая близость (cosine): {average_score:.4f}")


df.to_csv('results_with_similarity.csv', index=False)

df = pd.read_csv('results_with_model_answers.csv')
df_clean = df.drop(df[df['model_answer'] == 'К сожалению, не удалось найти точный ответ на ваш вопрос. Передаю его оператору.'].index)

similarities = []

for idx, row in tqdm(df_clean.iterrows(), total=len(df_clean)):
    try:
        emb_true = getEmbeddingsOllama(row['true_answer'])
        emb_model = getEmbeddingsOllama(row['model_answer'])

        sim = cosine_similarity([emb_true], [emb_model])[0][0]
        similarities.append(sim)
    except Exception as e:
        print(f"[!] Ошибка на id={row['id']}: {e}")
        similarities.append(np.nan)

# Сохраняем результат
df_clean['semantic_similarity'] = similarities

# Усреднённая метрика по всем успешным примерам
average_score = df_clean['semantic_similarity'].mean()
print(f"\n\n Средняя семантическая близость чистая: {average_score:.4f}")

df_clean.to_csv('results_with_similarity_clean.csv', index=False)