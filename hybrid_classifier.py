import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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

    df = df.head(5000)
    df.drop(['obj_ID', 'alpha', 'delta', 'spec_obj_ID', 'MJD', 'fiber_ID'], axis=1, inplace=True)
    df.dropna(inplace=True)

    class_mapping = {'STAR': 0, 'GALAXY': 1, 'QSO': 2}
    df['class'] = df['class'].map(class_mapping)

    X = df.drop('class', axis=1)
    y = df['class']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), pd.Series(y, index=y.index), class_mapping


# --- Sezione 2: Funzione per il Motore di Ragionamento Prolog ---
def resolve_ambiguities_with_prolog(ambiguous_cases, class_mapping, ambiguous_true_labels):
    """Risolve i casi ambigui usando un motore di ragionamento Prolog."""
    prolog = Prolog()
    prolog.consult("prolog_reasoner.pl")

    reverse_class_mapping = {v: k for k, v in class_mapping.items()}

    resolved_count = 0
    correctly_resolved_count = 0
    total_ambiguous_cases = len(ambiguous_cases)

    if total_ambiguous_cases == 0:
        return 0, 0, []

    detailed_results = []

    for index, (case_id, row) in enumerate(ambiguous_cases.iterrows()):
        star_id = f"ambiguous_star_{index}"

        prolog.assertz(f"magnitude('{star_id}', u, {row['u']})")
        prolog.assertz(f"magnitude('{star_id}', g, {row['g']})")
        prolog.assertz(f"magnitude('{star_id}', r, {row['r']})")
        prolog.assertz(f"magnitude('{star_id}', i, {row['i']})")
        prolog.assertz(f"magnitude('{star_id}', z, {row['z']})")

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

        detailed_results.append({
            'prolog_class': prolog_class,
            'prolog_explanation': prolog_explanation,
            'true_label': true_label
        })

        prolog.retractall(f"magnitude('{star_id}', _, _)")

    return resolved_count, correctly_resolved_count, detailed_results


# --- Sezione 3: Funzione per la Valutazione del Sistema Ibrido ---
def evaluate_hybrid_system(X, y, class_mapping, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_hybrid_accuracies = []
    total_prolog_accuracies = []

    print("----------------------------------------------------")
    print("Valutazione del Sistema Ibrido con K-Fold Cross-Validation:")
    print("----------------------------------------------------")

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)
        max_probabilities = np.max(probabilities, axis=1)
        confidence_threshold = 0.90
        ambiguous_indices = np.where(max_probabilities < confidence_threshold)[0]
        ambiguous_cases = X_test.iloc[ambiguous_indices]
        ambiguous_true_labels = y_test.iloc[ambiguous_indices]

        resolved_count, correctly_resolved_count, detailed_results = resolve_ambiguities_with_prolog(
            ambiguous_cases, class_mapping, ambiguous_true_labels
        )

        prolog_accuracy = correctly_resolved_count / resolved_count if resolved_count > 0 else 0
        total_prolog_accuracies.append(prolog_accuracy)

        ml_predictions = model.predict(X_test)
        ml_predictions[ambiguous_indices] = [class_mapping[res['prolog_class']] for res in detailed_results]
        hybrid_accuracy = np.mean(ml_predictions == y_test)
        total_hybrid_accuracies.append(hybrid_accuracy)

        print(f"Fold {fold + 1}:")
        print(f"  - Casi ambigui passati a Prolog: {len(ambiguous_cases)}")
        print(f"  - Accuratezza di Prolog sui casi ambigui: {prolog_accuracy:.4f}")
        print(f"  - Accuratezza ibrida totale: {hybrid_accuracy:.4f}\n")

    print("----------------------------------------------------")
    print("Valutazione Media Finale del Sistema Ibrido:")
    print(f"Accuratezza media di Prolog sui casi ambigui: {np.mean(total_prolog_accuracies):.4f}")
    print(f"Deviazione Standard di Prolog sui casi ambigui: {np.std(total_prolog_accuracies):.4f}")
    print(f"Accuratezza media complessiva del sistema ibrido: {np.mean(total_hybrid_accuracies):.4f}")
    print(f"Deviazione Standard complessiva del sistema ibrido: {np.std(total_hybrid_accuracies):.4f}")
    print("----------------------------------------------------")

    return detailed_results, ambiguous_cases, ambiguous_true_labels, model


# --- Sezione Principale ---
if __name__ == '__main__':
    X, y, class_mapping = load_and_preprocess_data('star_galaxy_quasar_sdss.csv')

    # Valutazione del sistema ibrido con K-fold cross-validation
    detailed_results, ambiguous_cases, ambiguous_true_labels, model = evaluate_hybrid_system(X, y, class_mapping)

    # --- Sezione di Esempi Dettagliati (dopo l'esecuzione del K-fold) ---
    print("\n\n--- Esempi di Classificazione Ibrida in Dettaglio (dall'ultimo fold) ---")

    reverse_class_mapping = {v: k for k, v in class_mapping.items()}
    ml_predictions = model.predict(ambiguous_cases)
    ml_probabilities = model.predict_proba(ambiguous_cases)

    num_examples = min(5, len(ambiguous_cases))

    for i in range(num_examples):
        case_data = ambiguous_cases.iloc[i]
        true_label = reverse_class_mapping[ambiguous_true_labels.iloc[i]]
        ml_pred_label = reverse_class_mapping[ml_predictions[i]]
        confidence = np.max(ml_probabilities[i])

        prolog_class = detailed_results[i]['prolog_class']
        prolog_explanation = detailed_results[i]['prolog_explanation']

        print(f"\nEsempio #{i + 1}:")
        print(
            f"  - Caratteristiche: u={case_data['u']:.4f}, g={case_data['g']:.4f}, r={case_data['r']:.4f}, i={case_data['i']:.4f}, z={case_data['z']:.4f}")
        print(f"  - Etichetta Reale: {true_label}")
        print(f"  - Previsione ML: {ml_pred_label} (Confidenza: {confidence:.2f})")
        print(f"  - Classificazione Prolog: {prolog_class}")
        print(f"  - Ragionamento di Prolog: {prolog_explanation}")