import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


    # Funz ioni di preprocessing e analisi di Clustering

def load_and_preprocess_data_gaia(file_path):

    # Carica e prepara i dati di gaia_data, etichettando gli oggetti in 5 classi.

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato.")
        exit()

    df = df.dropna()

    # Creazione degli indici di colore
    df['bp_rp'] = df['phot_bp_mean_mag'] - df['phot_rp_mean_mag']
    df['g_bp'] = df['phot_g_mean_mag'] - df['phot_bp_mean_mag']

    # Logica di etichettatura per creare le 5 classi
    df['class'] = 'UNKNOWN'

    # Classificazione iniziale basata sulla parallasse
    # Oggetti vicini (STAR, WHITE_DWARF, RED_DWARF)
    df.loc[df['parallax'] > 0.1, 'class'] = 'STAR'
    # Oggetti lontani (GALAXY, QSO)
    df.loc[df['parallax'] <= 0.1, 'class'] = 'EXTRAGALACTIC'

    # Suddivisione delle classi basata sugli indici di colore
    # Stelle Nane Rosse (più rosse)
    df.loc[(df['class'] == 'STAR') & (df['bp_rp'] > 1.8), 'class'] = 'RED_DWARF'
    # Stelle Nane Bianche (molto calde, blu)
    df.loc[(df['class'] == 'STAR') & (df['g_bp'] < 0.2), 'class'] = 'WHITE_DWARF'
    # Quasar (indici g-bp bassi)
    df.loc[(df['class'] == 'EXTRAGALACTIC') & (df['g_bp'] < 0.4), 'class'] = 'QSO'
    # Galassie
    df.loc[(df['class'] == 'EXTRAGALACTIC') & (df['g_bp'] >= 0.4), 'class'] = 'GALAXY'

    df.loc[(df['class'] == 'STAR') & (df['bp_rp'] <= 1.8) & (df['g_bp'] >= 0.2), 'class'] = 'STAR'

    # Rimuovi eventuali classi 'UNKNOWN' rimanenti
    df = df[df['class'] != 'UNKNOWN']

    # Seleziona le feature e la colonna di classe
    features = ['bp_rp', 'g_bp']
    X = df[features]
    y = df['class']

    # Mappatura delle 5 classi per i modelli
    class_mapping = {'STAR': 0, 'RED_DWARF': 1, 'WHITE_DWARF': 2, 'GALAXY': 3, 'QSO': 4}
    y = y.map(class_mapping)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return pd.DataFrame(X_scaled, columns=X.columns), y, class_mapping


def evaluate_clustering(X, y, n_clusters=10):
    # Esegue il clustering K-Means e valuta la purezza dei cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=33, n_init=10)
    # Addestramento del modello e organizza i cluster con le etichette
    kmeans.fit(X)
    cluster_labels = kmeans.labels_
    # Confronta le etichette trovare per i cluster con le etichette di classi reali
    y_true = np.array(y)
    purity = 0
    for i in range(n_clusters):
        true_labels_in_cluster = y_true[cluster_labels == i]
        if len(true_labels_in_cluster) > 0:
            most_frequent_class = np.argmax(np.bincount(true_labels_in_cluster))
            purity += np.sum(true_labels_in_cluster == most_frequent_class)

    total_samples = len(y)
    purity_score = purity / total_samples

    return purity_score


if __name__ == '__main__':
    file_name = 'gaia_data.csv'

    X, y, class_mapping = load_and_preprocess_data_gaia(file_name)

    print("----------------------------------------------------")
    print("Analisi di Clustering Non Supervisionato (K-Means)")
    print("----------------------------------------------------\n")

    # IL clustering cerca i 10 cluster
    kmeans_purity = evaluate_clustering(X, y, n_clusters=10)

    print(f"Purezza del Clustering: {kmeans_purity:.4f}\n")
    print("----------------------------------------------------")
    print("L'analisi di clustering è completata.")