import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.model_selection import train_test_split

# Baca dataset
balanced_dataset = pd.read_csv('final_preprocessed_dataset.csv')

# Vektorisasi menggunakan TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(balanced_dataset['text_content'])

# Asumsi label sudah ditentukan
y = balanced_dataset['is_gambling']

# Pembagian Data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model SVM
svm_model = SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train, y_train)

# Simpan model dan vektorizer
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')