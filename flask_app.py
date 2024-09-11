# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
# import joblib
import pickle
# from utils import preprocess_text  # Assurez-vous que utils.py est dans le même répertoire
# import pandas



app = Flask(__name__)

app.config['DEBUG'] = True

# print("version de pandas installée:\n",pandas.__version__)

# pretraitement version spacy :
from bs4 import BeautifulSoup
#from flask import Flask, request, jsonify


#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
import nltk

import re
import spacy

# Ajouter le répertoire de données NLTK au chemin des données
nltk.data.path.append('./nltk_data')

# Telecharger les stopwords et tokenizer de NLTK
# nltk.download('stopwords')
# nltk.download('punkt')

try:
    from bs4 import BeautifulSoup
    print("BeautifulSoup importé avec succès.")
except ImportError as e:
    print(f"Erreur lors de l'importation de BeautifulSoup: {e}")

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    print("NLTK et ressources importés avec succès.")
except ImportError as e:
    print(f"Erreur lors de l'importation de NLTK: {e}")



# Fonction pour nettoyer le texte HTML et enlever les portions de code
def clean_html_code(text):
    # Supprimer les portions de code
    # Supprimer les balises <code> et leur contenu
    text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)
    # Supprimer les balises <p> en conservant le contenu
    text = re.sub(r'</?p>', '', text)
    text = re.sub(r'\WA+', ' ', text)
    # Utiliser BeautifulSoup pour nettoyer les balises HTML
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

# Fonction pour normaliser le texte
def normalize_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer la ponctuation et les caract�res sp�ciaux
    text = re.sub(r'\W+', ' ', text)
    # Supprimer plusieurs espaces par un espace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# # Fonction de pr�traitement complet du texte
# def preprocess_text(text,nlp):
# # Charger le mod�le anglais de SpaCy
#     #nlp = spacy.load('en_core_web_sm')

# Fonction de pr�traitement complet du texte
def preprocess_text(text):
# Charger le mod�le anglais de SpaCy
    nlp = spacy.load('en_core_web_sm')

# Initialiser le stemmer de NLTK
    stemmer = PorterStemmer()

# Ajouter des stopwords personnalis�s
    custom_stopwords = set([
    'like', 'question', 'use', 'want', 'one', 'know', 'work', 'example', 'code', 'seem',
    'using', 'instead', 'way', 'get', 'would', 'need', 'following', '1', '2', 'run',
    'something', 'trying', 'tried', 'also', 'new', 'could', 'see', 'line', 'however',
    'solution', '3', '4', '5', 'without', 'still', 'answer', 'say', 'another', 'help',
    'anyone', 'best', 'looking', 'show', 'give', 'better', 'many', 'good', 'even',
    'think', 'thing', 'look', 'problem', 'try', 'possible'
    ])
    nltk_stopwords = set(stopwords.words('english'))
    all_stopwords = nltk_stopwords.union(custom_stopwords)
    # Nettoyage du texte HTML et suppression des portions de code
    # st.write("texte brut:", text)
    text = clean_html_code(text)

    # Normalisation
    text = normalize_text(text)
    # st.write("texte nettoyé:", text)
    # Utilisation de SpaCy pour lemmatisation et POS tagging
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.pos_ == 'NOUN' and token.text not in all_stopwords]
    # Stemming des tokens (si n�cessaire)
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Fonction de pr�traitement complet du texte (pour use,bert,w2vec)
def preprocess_text_NN(text):
    # Nettoyage du texte HTML et suppression des portions de code
    text = clean_html_code(text)
    # Normalisation
    text = normalize_text(text)
    tokens=text.split()
    return tokens





# Chargement des modèles pré-entraînés
# try:
#     mlb_job = joblib.load('mlb_bow_model.pkl')
#     print("modele mlb_job bien chargé")
# except Exception as e:
#     raise Exception(f"Erreur lors du chargement du modèle mlb_job: {e}")

# try:
#     model = joblib.load('tag_predictor_bow_model_sept.pkl')
#     print("Modèle bien chargé")
# except Exception as e:
#     raise Exception(f"Erreur lors du chargement des modèles: {e}")


try:
    with open('mlb_bow_model_vpickle.pkl', 'rb') as mlb_job_file:
        mlb_job = pickle.load(mlb_job_file)
        print("modele mlb_job pickle bien chargé")
except Exception as e:
    raise Exception(f"Erreur lors du chargement des modèles: {e}")

print("Étiquettes binarisées:", mlb_job.classes_)



try:
    with open('tag_predictor_bow_model_vpickle.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        print("modele bien chargé")
        print("Étapes du pipeline: ", model.named_steps)
except Exception as e:
    raise Exception(f"Erreur lors du chargement des modèles: {e}")

#@app.route("/", methods=["GET,POST"])
@app.route("/")
def home():
    return jsonify({"message": "Bienvenue dans l'API de prédiction de tags. Consultez /docs pour plus d'informations. Test du 11Sept 15H"})


#@app.route("/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])

def predict_tags():

    # data = {'col1': [1, 2], 'col2': [3, 4]}
    # df = pandas.DataFrame(data)
    print("hello")

    try:
        data = request.get_json()
        print("msg de la methode post")
        print(data)
        #return jsonify({"message": "Bienvenue dans la méthode poste de 'API de prédiction de tags."})
        if "text" not in data:
            return jsonify({"error": "Le champ 'text' est requis."}), 400

        print("debut de la fonction predict sur main.py!§§sd")
        # Prétraiter le texte
        text_cleaned_list = preprocess_text(data["text"])
        text_cleaned_joined = ' '.join(text_cleaned_list)
        print("le texte cleané est:",text_cleaned_joined)


        # # debut debogue
        # # Le modèle s'attend à recevoir une liste ou un tableau de textes
        # try:
        #     vectorizer = model.named_steps['vectorizer']
        #     app.logger.info(f"vectorizer vocabulary: {vectorizer.vocabulary_}")
        #     print("vectorizer vocabulary: ", vectorizer.vocabulary_)
        # except KeyError as e:
        #     print(f"Erreur: l'étape 'vectorizer' est introuvable dans le pipeline. Détails: {e}")
        # # Vérifier quels termes de text_cleaned sont dans le vocabulaire
        # terms_in_vocab = [term for term in text_cleaned_list if term in vectorizer.vocabulary_]
        # print("terms_in_vocab: ", terms_in_vocab)

        # text_vectorized = vectorizer.transform(text_cleaned_list)
        # text_vectorized_array = text_vectorized.toarray()
        # print("texte vectorisé : ", text_vectorized_array)  # Afficher le tableau pour déboguer

        # # # Créer un DataFrame pour inspecter les termes activés
        # # df_vectorized = pd.DataFrame(text_vectorized_array, columns=vectorizer.get_feature_names_out())
        # # non_zero_columns = df_vectorized.loc[:, (df_vectorized != 0).any(axis=0)]
        # # print("Non-zero columns: \n", non_zero_columns)
        # # alternative a pandas:
        # # Transformer le texte vectorisé en une liste de listes
        # text_vectorized_list = text_vectorized_array.tolist()

        # # Récupérer les noms des features (colonnes)
        # feature_names = vectorizer.get_feature_names_out()

        # # Filtrer les colonnes non nulles (où il y a au moins une valeur différente de zéro)
        # non_zero_columns = [feature_names[i] for i in range(len(feature_names)) if any(row[i] != 0 for row in text_vectorized_list)]

        # # Afficher les colonnes avec des valeurs non nulles
        # print("Non-zero columns: \n", non_zero_columns)



        # # # Prédiction avec le modèle de classification
        # classifier = model.named_steps['classifier']
        # predicted_tags = classifier.predict(text_vectorized)
        # print("predicted_tags: \n", predicted_tags)  # Afficher les prédictions brutes pour déboguer

        # # # Inverse transform des prédictions
        # predicted_tags_inverse = mlb_job.inverse_transform(predicted_tags)
        # print("predicted_tags_inverse: \n", predicted_tags_inverse)  # Afficher les prédictions inverses pour déboguer



        # # Conversion des tags prédits en liste de chaînes de caractères
        # predicted_tags_list = [tag for tags in predicted_tags_inverse for tag in tags]
        # print("predicted_tags_list: ",predicted_tags_list)
        # #fin debogue





        try:
        # # Faire la prédiction
            print("Début du prédict")
            bow_predict_result = model.predict([text_cleaned_joined])
            if bow_predict_result is None:
                raise ValueError("Le modèle n'a pas produit de prédiction.")
            print("Résultat de la prédiction:", bow_predict_result)
        except Exception as e:
            raise Exception(f"Erreur lors de la fonction model.predict: {e}")
        print("Début du inverse")
        tags_predits = mlb_job.inverse_transform(bow_predict_result)
        print("resulat du inverse:",tags_predits)
        predicted_tags_list = [tag for tags in tags_predits for tag in tags]
        print("tags prédits :",predicted_tags_list)
        return jsonify({"tags": predicted_tags_list})
        # Pour le moment, retournons un message de succès pour tester
        return jsonify({"message": "Prédiction réussie"}), 200
    except Exception as e:
        return jsonify({"error": f"Erreur lors de la prédiction: {e}"}), 500

    #     return jsonify({"message": "Test réussi"})
    # except Exception as e:
    #     # Gestion d'erreur générique
    #     print(f"Erreur lors de la prédiction: {e}")
    #     return jsonify({"error": f"Erreur lors de la prédiction: {e}"}), 500


@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == 'POST':
        # Optionally, verify the event came from GitHub
        payload = request.json
        if payload.get('ref') == 'refs/heads/main':  # Only deploy on push to the main branch
            # Run the deploy script
            subprocess.Popen(["/bin/bash", "/home/Azamku/deploy.sh"])
            return 'Deployed successfully', 200
        else:
            return 'No deployment', 200
    else:
        abort(400)



if __name__ == "__main__":
    app.run()
