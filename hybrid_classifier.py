import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from pyswip import Prolog


# --- Sezione 1: Funzioni di Preprocessing e Machine Learning ---
def load_and_preprocess_data(file_path):
    """Carica, pulisce e prepara i dati per la classificazione."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
        exit()

    df = df.head(1000)
    df.drop(['obj_ID', 'alpha', 'delta', 'spec_obj_ID', 'MJD', 'fiber_ID'], axis=1, inplace=True)
    df.dropna(inplace=True)

    class_mapping = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}
    df['class'] = df['class'].map(class_mapping)

    X = df.drop('class', axis=1)
    y = df['class']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), pd.Series(y, index=y.index), class_mapping


def train_and_get_ambiguous_cases(X, y):
    """
    Addestra il modello, esegue la cross-validation e identifica i casi ambigui.
    Ritorna anche il set di test completo per la valutazione.
    """
    # 1. Suddivisione dei dati in training e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Addestramento del modello solo sul training set
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 3. Valutazione rigorosa con cross-validation sul training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print("----------------------------------------------------")
    print("Valutazione del Modello di Base (Cross-Validation sul training set):")
    print(f"Accuratezza Media: {np.mean(cv_scores):.4f}")
    print(f"Deviazione Standard: {np.std(cv_scores):.4f}")
    print("----------------------------------------------------")

    # 4. Ottenere le probabilità di confidenza sul test set
    probabilities = model.predict_proba(X_test)
    max_probabilities = np.max(probabilities, axis=1)

    # 5. Identificare i casi ambigui (confidenza < 90%)
    confidence_threshold = 0.90
    ambiguous_indices = np.where(max_probabilities < confidence_threshold)[0]

    ambiguous_cases = X_test.iloc[ambiguous_indices]
    ambiguous_true_labels = y_test.iloc[ambiguous_indices]

    return model, ambiguous_cases, ambiguous_true_labels, y_test


# --- Sezione 2: Funzione per il Motore di Ragionamento Prolog ---
def resolve_ambiguities_with_prolog(ambiguous_cases, class_mapping, ambiguous_true_labels):
    """Risolve i casi ambigui usando un motore di ragionamento Prolog."""
    prolog = Prolog()
    prolog.consult("prolog_reasoner.pl")

    reverse_class_mapping = {v: k for k, v in class_mapping.items()}

    resolved_count = 0
    correctly_resolved_count = 0
    total_ambiguous_cases = len(ambiguous_cases)

    # Inizializza la lista per memorizzare i risultati dettagliati
    detailed_results = []

    print(f"\nTentativo di risolvere {total_ambiguous_cases} casi ambigui con Prolog...")

    for index, (case_id, row) in enumerate(ambiguous_cases.iterrows()):
        star_id = f"ambiguous_star_{index}"

        prolog.assertz(f"magnitude('{star_id}', u, {row['u']})")
        prolog.assertz(f"magnitude('{star_id}', g, {row['g']})")
        prolog.assertz(f"magnitude('{star_id}', r, {row['r']})")
        prolog.assertz(f"magnitude('{star_id}', i, {row['i']})")
        prolog.assertz(f"magnitude('{star_id}', z, {row['z']})")

        # Query aggiornata per ottenere anche la spiegazione
        query = f"risolvi_ambiguita('{star_id}', FinalClass, Explanation)"
        solution = list(prolog.query(query))

        prolog_class = solution[0]['FinalClass'] if solution and 'FinalClass' in solution[0] else 'Indefinito'
        prolog_explanation = solution[0]['Explanation'] if solution and 'Explanation' in solution[
            0] else 'Nessuna spiegazione.'

        if prolog_class != 'Indefinito':
            resolved_count += 1
            true_label = reverse_class_mapping[ambiguous_true_labels.iloc[index]]
            if prolog_class == true_label:
                correctly_resolved_count += 1

        # Aggiungi il risultato dettagliato alla lista
        detailed_results.append({
            'prolog_class': prolog_class,
            'prolog_explanation': prolog_explanation
        })

        prolog.retractall(f"magnitude('{star_id}', _, _)")

    print("\n----------------------------------------------------")
    print("Analisi Risultati del Sistema Ibrido:")
    print(f"Casi Ambigui Passati a Prolog: {total_ambiguous_cases}")
    print(f"Casi Ambigui Risolti da Prolog: {resolved_count}")
    print(
        f"Accuratezza di Prolog sui casi ambigui: {correctly_resolved_count / resolved_count:.4f}" if resolved_count > 0 else "Nessun caso risolto da Prolog.")
    print("----------------------------------------------------")
    print(
        f"\nIl progetto ibrido ha classificato correttamente {correctly_resolved_count} dei {total_ambiguous_cases} casi difficili.")

    # Restituisce anche i risultati dettagliati
    return resolved_count, correctly_resolved_count, detailed_results


# --- Sezione Principale ---
if __name__ == '__main__':
    # 1. Carica e prepara i dati
    X, y, class_mapping = load_and_preprocess_data('star_galaxy_quasar_sdss.csv')

    # 2. Esegui la pipeline ibrida
    model, ambiguous_cases, ambiguous_true_labels, y_test = train_and_get_ambiguous_cases(X, y)

    # 3. Risoluzione delle ambiguità con Prolog
    resolved_count, correctly_resolved_count, detailed_results = resolve_ambiguities_with_prolog(
        ambiguous_cases, class_mapping, ambiguous_true_labels
    )

    # --- Sezione di Esempi Chiari ---
    print("\n\n--- Esempi di Classificazione Ibrida in Dettaglio ---")

    # Inverti la mappatura delle classi per una facile lettura
    reverse_class_mapping = {v: k for k, v in class_mapping.items()}

    # Ottieni le previsioni originali del modello ML per i casi ambigui
    ml_predictions = model.predict(ambiguous_cases)
    ml_probabilities = model.predict_proba(ambiguous_cases)

    num_examples = min(5, len(ambiguous_cases))  # Mostra al massimo 5 esempi

    for i in range(num_examples):
        case_index = ambiguous_cases.index[i]

        # Dati del caso
        case_data = ambiguous_cases.iloc[i]
        true_label = reverse_class_mapping[ambiguous_true_labels.iloc[i]]

        # Previsione del modello ML
        ml_pred_label = reverse_class_mapping[ml_predictions[i]]
        confidence = np.max(ml_probabilities[i])

        # Usa i risultati dettagliati ottenuti da Prolog
        prolog_class = detailed_results[i]['prolog_class']
        prolog_explanation = detailed_results[i]['prolog_explanation']

        print(f"\nEsempio #{i + 1}:")
        print(
            f"  - Caratteristiche: u={case_data['u']:.4f}, g={case_data['g']:.4f}, r={case_data['r']:.4f}, i={case_data['i']:.4f}, z={case_data['z']:.4f}")
        print(f"  - Etichetta Reale: {true_label}")
        print(f"  - Previsione ML: {ml_pred_label} (Confidenza: {confidence:.2f})")
        print(f"  - Classificazione Prolog: {prolog_class}")
        print(f"  - Ragionamento di Prolog: {prolog_explanation}")