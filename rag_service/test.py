import pandas as pd
import requests
from tqdm import tqdm
import time

MICROSERVICE_URL = "http://localhost:5000/answer"
INPUT_FILE = "paraphrased_questions.csv"
OUTPUT_FILE = "results_with_model_answers.csv"

def query_model(question: str) -> str:
    try:
        response = requests.post(MICROSERVICE_URL, json={"question": question}, timeout=10)
        if response.status_code == 200:
            return response.json().get("answer", "")
        else:
            print(f"[!] HTTP {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"[!] Ошибка запроса: {e}")
        return ""

def run_local_test():
    df = pd.read_csv(INPUT_FILE)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        paraphrased_q = row['paraphrased_question']
        model_answer = query_model(paraphrased_q)

        results.append({
            "id": row['id'],
            "original_question": row['original_question'],
            "paraphrased_question": paraphrased_q,
            "true_answer": row['true_answer'],
            "model_answer": model_answer
        })

        time.sleep(0.1)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n Результаты сохранены в {OUTPUT_FILE}")

run_local_test()
