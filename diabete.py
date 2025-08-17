import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

# === Constante ===
CSV_FILE = "diabetes.csv"

# === Entraînement du modèle ===
def train_model():
    if not os.path.exists(CSV_FILE):
        messagebox.showerror("Erreur", f"Fichier {CSV_FILE} non trouvé.\nVeuillez placer diabetes.csv dans le dossier du script.")
        exit(1)
    
    df = pd.read_csv(CSV_FILE)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=500, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# === Prédiction du diabète ===
def predire_diabete():
    try:
        features = []
        for e in entries:
            val = float(e.get())
            if val < 0:
                messagebox.showerror("Erreur", "Toutes les valeurs doivent être positives ou nulles.")
                return
            features.append(val)

        prediction = model.predict([features])[0]
        proba = model.predict_proba([features])[0][1]  # Proba classe 1 (diabète)

        msg = f"Probabilité estimée de diabète : {proba * 100:.2f}%\n\n"
        msg += "⚠️ Risque élevé de diabète détecté." if prediction == 1 else "✅ Pas de risque détecté."
        
        messagebox.showinfo("Résultat", msg)

    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des nombres valides dans tous les champs.")

# === Interface graphique Tkinter ===
app = tk.Tk()
app.title("Détection du diabète - IA")
app.geometry("500x820")
app.resizable(False, False)

tk.Label(app, text="Détection du diabète avec IA", font=("Helvetica", 20, "bold")).pack(pady=15)

frame_form = tk.Frame(app)
frame_form.pack(pady=10)

labels_text = [
    "Grossesses (Pregnancies)",
    "Glycémie (Glucose)",
    "Tension artérielle (BloodPressure)",
    "Épaisseur de peau (SkinThickness)",
    "Insuline (Insulin)",
    "IMC (BMI)",
    "Fonction héréditaire (DiabetesPedigreeFunction)",
    "Âge (Age)"
]

entries = []
for text in labels_text:
    tk.Label(frame_form, text=text, font=("Helvetica", 12)).pack(anchor="w", pady=5)
    e = tk.Entry(frame_form, font=("Helvetica", 14), width=30, bd=2, relief="groove", justify="center")
    e.pack(pady=3)
    entries.append(e)

btn_predire = tk.Button(app, text="🩺 Prédire le risque de diabète",
                        font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white",
                        command=predire_diabete)
btn_predire.pack(pady=30)

tk.Label(app, text="© 2025 Camara Mohamed", font=("Helvetica", 9), fg="#666").pack(side="bottom", pady=10)

# === Chargement du modèle IA ===
model = train_model()

app.mainloop()
