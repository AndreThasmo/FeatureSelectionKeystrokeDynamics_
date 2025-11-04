
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    @staticmethod
    def plot_confusion_matrix(conf_matrix, user_id):
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['not_user', 'is_user'], yticklabels=['not_user', 'is_user'])
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão para o Usuário {user_id}')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_metrics(user_id, results):
        print(f"\nResultados para o Usuário: {user_id}\n")
        print(f"Acurácia: {results['accuracy']:.2f}")
        print(f"Acurácia Balanceada: {results['balanced_accuracy']:.2f}")
        print(f"Precisão: {results['precision']:.2f}")
        print(f"Recall: {results['recall']:.2f}")
        print(f"F1-Score: {results['f1_score']:.2f}")
        print(f"Especificidade: {results['specificity']:.2f}\n")
