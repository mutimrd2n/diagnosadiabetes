import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    # 1. LOAD DATA
    df = pd.read_csv("diabetes.csv")
    print("Ukuran Dataset:", df.shape)
    print("5 data pertama:\\n", df.head())

    # 2. PREPROCESSING DATA
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_to_fix:
        median_value = df[col].median()
        df[col] = df[col].replace(0, median_value)
    
    for col in cols_to_fix:
        print(f"Jumlah nilai 0 pada kolom '{col}' setelah perbaikan: {(df[col] == 0).sum()}")

    # 3. MEMISAHKAN FITUR DAN TARGET
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 4. MEMBUAT DAN MELATIH MODEL DECISION TREE
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 5. EVALUASI MODEL
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("\\n===== Evaluasi Model =====")
    print("Akurasi: {:.2f}%".format(accuracy * 100))
    print("\\nConfusion Matrix:")
    print(conf_matrix)
    print("\\nClassification Report:")
    print(class_report)

    # 6. VISUALISASI MODEL (SIMPAN KE FILE, BUKAN SHOW)
    plt.figure(figsize=(20,10))
    plot_tree(model,
              feature_names=X.columns,
              class_names=["Non-Diabetes", "Diabetes"],
              filled=True)
    plt.title("Visualisasi Pohon Keputusan - Diagnosis Diabetes")
    plt.savefig("decision_tree.png")
    print("Plot pohon keputusan disimpan sebagai 'decision_tree.png'")

if __name__ == "__main__":
    main()
