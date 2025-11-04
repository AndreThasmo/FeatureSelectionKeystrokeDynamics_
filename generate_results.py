"""
Script principal para rodar a pipeline de avaliação de algoritmos
de autenticação baseada em digitação. Executa carregamento dos dados,
seleção de atributos, treinamento de modelos e salva os resultados.
"""

from app.keystroke_evaluator import KeystrokeEvaluator

import pandas as pd
from datetime import datetime

if __name__ == "__main__":
    all_results = []

    for seed in range(1, 31):
        evaluator = KeystrokeEvaluator(
            csv_path="data/DSL-StrongPasswordData.csv",
            random_state=seed
        )
        evaluator.run()
        for row in evaluator.results:
            row["random_state"] = seed
        all_results.extend(evaluator.results)
        print(f"Finished run with random_state={seed}")

    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"data/output/keystroke_results_all_{timestamp}.csv", index=False)
    df.to_excel(f"data/output/keystroke_results_all_{timestamp}.xlsx", index=False)
