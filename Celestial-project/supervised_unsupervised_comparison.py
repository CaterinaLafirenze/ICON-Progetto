import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from gaia_unsupervised_classifier import load_and_preprocess_data_gaia, evaluate_clustering


def evaluate_random_forest(X, y):
    #Valuta il modello Random Forest con K-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=33)
    accuracies = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestClassifier(n_estimators=100, random_state=33)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    return np.mean(accuracies), np.std(accuracies)



if __name__ == '__main__':
    file_name = 'gaia_data.csv'

    X, y, class_mapping = load_and_preprocess_data_gaia(file_name)

    print("----------------------------------------------------")
    print("Confronto tra Apprendimento Supervisionato e Non Supervisionato")
    print("----------------------------------------------------\n")

    # Valutazione del metodo Supervisionato (Random Forest)
    rf_accuracy_mean, rf_accuracy_std = evaluate_random_forest(X, y)
    print(f"Valutazione del Modello Supervisionato (Random Forest):")
    print(f"  - Accuratezza Media (K-Fold ): {rf_accuracy_mean:.4f}")
    print(f"  - Deviazione Standard: {rf_accuracy_std:.4f}\n")

    # Valutazione del metodo Non Supervisionato (K-Means Clustering)
    # riesegue calcolo di purezza per il confronto
    kmeans_purity = evaluate_clustering(X, y, n_clusters=2)
    print(f"Valutazione del Metodo Non Supervisionato (K-Means):")
    print(f"  - Purezza del Clustering: {kmeans_purity:.4f}\n")

    print("----------------------------------------------------")
    print("Conclusione:")
    print(f"L'accuratezza del Random Forest (supervisionato) è: {rf_accuracy_mean:.4f}")
    print(f"La purezza del K-Means (non supervisionato) è: {kmeans_purity:.4f}")
    print("----------------------------------------------------")

    if rf_accuracy_mean > kmeans_purity:
        print(
            "Il modello supervisionato ha ottenuto un risultato migliore. Questo è atteso, in quanto il modello ha avuto accesso alle etichette per l'addestramento.")
    else:
        print(
            "Il modello non supervisionato ha ottenuto un risultato sorprendentemente simile o migliore. Questo indica che la struttura dei dati è molto ben definita.")