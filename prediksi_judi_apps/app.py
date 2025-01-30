from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tldextract
import csv
import time
import urllib3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import nltk
from typing import List

app = Flask(__name__)

# Muat model dan vektorizer
svm_model = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Konfigurasi Google CSE
API_KEY = "APIKEY[REDACTED]"
CX = "CX-Key[REDACTED]"

# Disable warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TIMEOUT = 10
KEYWORDS = ["judi", "slot", "casino", "taruhan", "poker", "deposit", "bonus", "bet", "sportsbook"]
DELAY = 2  # Delay antar-request (dalam detik)
SEARCH_DELAY = 1  # Delay antar request ke Google CSE
MAX_RESULTS = 50  # Batas maksimal hasil pencarian

def extract_features(url):
    try:
        # Delay antar-request
        time.sleep(DELAY)

        # Ambil konten HTML
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT, verify=False)
        response.raise_for_status()  # Raise exception jika status code bukan 200
        soup = BeautifulSoup(response.text, 'html.parser')

        # Ekstrak domain dan subdomain
        ext = tldextract.extract(url)
        domain = ext.registered_domain
        subdomain = ext.subdomain

        # Ekstrak teks dan keyword
        text = soup.get_text(separator=' ', strip=True)
        keyword_count = sum(text.lower().count(keyword) for keyword in KEYWORDS)

        # Metadata HTML
        title = soup.title.string if soup.title else ""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        meta_desc = meta_desc['content'] if meta_desc else ""
        
        # Data akhir
        return {
            'url': url,
            'domain': domain,
            'subdomain': subdomain,
            'text_content': text[:2000],  # Batasi teks
            'keyword_count': keyword_count,
            'meta_title': title,
            'meta_description': meta_desc,
        }

    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {str(e)}")
        return None
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

stop_words_indonesian = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # Ubah ke huruf kecil
    text = text.lower()
        
    # Hapus URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
        
    # Hapus hashtags
    text = re.sub(r'#\w+', '', text)
        
    # Hapus mentions (misalnya pada Twitter)
    text = re.sub(r'@\w+', '', text)
        
    # Hapus simbol dan karakter spesial (termasuk emoji)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Hapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', '', text)
        
    # Hapus karakter kontrol
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
    # Hapus pengulangan kata
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
        
    # Hapus pengulangan tanda baca
    text = re.sub(r'([?.!,])\1+', r'\1', text)
        
    # Hapus spasi ekstra
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenisasi
    tokens = word_tokenize(text)

    # Hapus stop words
    tokens = [word for word in tokens if word.lower() not in stop_words_indonesian]

    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]

    # Gabungkan kembali menjadi string
    return ' '.join(tokens)

# Routes
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/single', methods=['GET', 'POST'])
def single():
    result = None
    details = None
    if request.method == 'POST':
        url = request.form['url']
        features = extract_features(url)
        
        if features:
            cleaned_text = preprocess_text(features['text_content'])
            tfidf_features = tfidf_vectorizer.transform([cleaned_text])
            prediction = svm_model.predict(tfidf_features)
            
            details = {
                'url': url,
                'domain': features['domain'],
                'status': 200,
                'keyword_count': features['keyword_count'],
                'prediction': "Gambling" if prediction[0] == 1 else "Non-Gambling"
            }
            result = details['prediction']
    
    return render_template('single.html', result=result, details=details)

@app.route('/multi')
def multi():
    return render_template('multi.html', DELAY=DELAY)

@app.route('/check', methods=['POST'])
def check_single():
    url = request.form['url']
    try:
        features = extract_features(url)
        if features:
            cleaned_text = preprocess_text(features['text_content'])
            tfidf_features = tfidf_vectorizer.transform([cleaned_text])
            prediction = svm_model.predict(tfidf_features)
            
            return {
                'url': url,
                'status': 200,
                'domain': features['domain'],
                'prediction': "Gambling" if prediction[0] == 1 else "Non-Gambling"
            }
        return {'url': url, 'status': 'Error', 'error': 'Gagal mengekstrak fitur'} 
    except Exception as e:
        return {'url': url, 'status': 'Error', 'error': str(e)}

@app.route('/scrape', methods=['GET', 'POST'])
def scrape():
    DEFAULT_QUERY = "slot OR gacor OR maxwin OR SITUS RESMI SLOT GACOR OR slot gacor OR judi online"
    MAX_RESULTS = 50 
    
    # Inisialisasi semua variabel template
    search_results = []
    links_to_check = []
    no_results = False
    
    if request.method == 'POST':
        action = request.form.get('action', 'search_only')
        domain = request.form['domain']
        query = request.form['query'].strip() or DEFAULT_QUERY

        links = []
        start_index = 1
        while True:
            url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX}&siteSearch={domain}&start={start_index}"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                new_links = [item['link'] for item in data.get('items', [])]
                links.extend(new_links)
                
                if len(new_links) < 10 or len(links) >= MAX_RESULTS:
                    break
                
                start_index += 10
                time.sleep(SEARCH_DELAY)
            else:
                break

        if action == 'search_only':
            return render_template('scrape.html',
                                 DELAY=DELAY,
                                 search_results=links,
                                 default_query=DEFAULT_QUERY,
                                 links_to_check=[],
                                 no_results=False)
        
        return render_template('scrape.html',
                             DELAY=DELAY,
                             links_to_check=links,
                             default_query=DEFAULT_QUERY,
                             search_results=[],
                             no_results=False)
    
    # Handle GET request
    return render_template('scrape.html',
                         DELAY=DELAY,
                         default_query=DEFAULT_QUERY,
                         search_results=[],
                         links_to_check=[],
                         no_results=False)

@app.route('/check_link', methods=['POST'])
def check_link():
    url = request.form['url']
    try:
        features = extract_features(url)
        if features:
            cleaned_text = preprocess_text(features['text_content'])
            tfidf_features = tfidf_vectorizer.transform([cleaned_text])
            prediction = svm_model.predict(tfidf_features)
            
            return {
                'url': url,
                'status': 200,
                'prediction': "Gambling" if prediction[0] == 1 else "Non-Gambling"
            }
        return {'url': url, 'status': 'Error', 'prediction': 'Gagal ekstrak fitur'}
    except Exception as e:
        return {'url': url, 'status': 'Error', 'prediction': str(e)}



if __name__ == '__main__':
    app.run(debug=True, port=5001)

