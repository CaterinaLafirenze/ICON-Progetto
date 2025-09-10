import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(file_path):

    #Carica e prepara i dati per l'apprendimento automatico

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
        exit()

    df = df.head(1000)
    df.drop(['obj_ID', 'alpha', 'delta', 'spec_obj_ID', 'MJD', 'fiber_ID'], axis=1, inplace=True)
    df.dropna(inplace=True)

    # Mappa le classi di testo in numeri interi
    class_mapping = {'QSO': 0, 'STAR': 1, 'GALAXY': 2, 'RED_DWARF':3, 'WHITE_DWARF':4}
    df['class'] = df['class'].map(class_mapping)

    # Visualizza la distribuzione delle classi nel dataset
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='class', palette='viridis', hue='class', legend=False)
    plt.title('Distribuzione delle classi nel Dataset')
    plt.xlabel('Classi (0=QSO, 1=STAR, 2=GALAXY, 3=RED_DWARF, 4=WHITE_DWARF)')
    plt.ylabel('Numero di Oggetti')
    plt.xticks(ticks=[0, 1, 2, 3, 4], labels=['QSO', 'STAR', 'GALAXY', 'RED_DWARF', 'WHITE_DWARF'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Seleziona le feature (X) e la variabile target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # Normalizza le feature
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return pd.DataFrame(X, columns=df.drop('class', axis=1).columns), pd.Series(y, index=y.index), class_mapping


def train_and_evaluate_model(X, y):

    # Addestra e valuta un modello di Random Forest e identifica i casi ambigui

    print("Addestramento del modello di base (Random Forest)...")
    model = RandomForestClassifier(n_estimators=100, random_state=33)

    # Esegue la cross-validation e calcola accuratezza media e deviazione standard
    cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

    print("\n----------------------------------------------------")
    print("Valutazione del Modello di Base (Cross-Validation):")
    print(f"Accuratezza Media: {np.mean(cv_scores):.4f}")
    print(f"Deviazione Standard: {np.std(cv_scores):.4f}")
    print("----------------------------------------------------")

    # Ri-addestra il modello sul dataset completo per l'analisi successiva
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Ottiene le probabilità di confidenza per le previsioni
    probabilities = model.predict_proba(X_test)

    # Identifica i casi ambigui (es. confidenza inferiore al 90%)
    confidence_threshold = 0.90
    predicted_classes = model.predict(X_test)
    max_probabilities = np.max(probabilities, axis=1)

    # Crea un DataFrame con i casi ambigui
    ambiguous_indices = np.where(max_probabilities < confidence_threshold)[0]
    ambiguous_cases = X_test.iloc[ambiguous_indices]

    print(f"\nIdentificati {len(ambiguous_cases)} casi ambigui con confidenza < {confidence_threshold * 100}%")


    return model, ambiguous_cases, predicted_classes, y_test


if __name__ == '__main__':
    # Carica e prepara i dati
    X, y, class_mapping = load_and_preprocess_data('star_galaxy_quasar_sdss.csv')

    # Addestra e valuta il modello, ottenendo i casi ambigui
    trained_model, ambiguous_cases, predicted_classes, y_test = train_and_evaluate_model(X, y)

    print("\nModello di base addestrato con successo.")