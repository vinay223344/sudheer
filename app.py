from flask import Flask, request, jsonify, render_template
import requests
from newspaper import Article
from transformers import pipeline
import nltk
from config import Config  # Ensure you have a config.py file with API keys

# Download required NLTK data
nltk.download('punkt')

app = Flask(__name__)
app.config.from_object(Config)

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_model = pipeline("question-answering")

article_data = {"text": "", "summary": ""}

def make_request(endpoint, payload):
    """Helper function to make API requests."""
    headers = {
        "ulcaApiKey": app.config['ULCA_API_KEY'],
        "userID": app.config['USER_ID'],
        "Content-Type": "application/json",
        "Authorization": app.config['AUTHORIZATION_TOKEN']
    }
    response = requests.post(endpoint, json=payload, headers=headers)
    return response.json()

def fetch_article(url):
    """Fetch and extract article text."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        return None

def summarize_text(text, max_len=300, min_len=200):
    """Summarize long articles by splitting into chunks."""
    if len(text) > 1000:
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = [summarizer(chunk, max_length=max_len//len(chunks), min_length=min_len//len(chunks), do_sample=False)[0]['summary_text'] for chunk in chunks]
        return " ".join(summaries)
    else:
        return summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)[0]['summary_text']

@app.route('/')
def index():
    """Serve the frontend."""
    return render_template("index.html")

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarizes a given article URL and translates if needed."""
    global article_data
    url = request.json.get("url", "")
    target_language = request.json.get("targetLanguage", "en")  # Default is English

    article_text = fetch_article(url)
    if not article_text:
        return jsonify({"error": "Failed to extract article"}), 400

    summary = summarize_text(article_text)
    article_data = {"text": article_text, "summary": summary}

    # If translation is requested
    if target_language in ["te", "hi"]:
        translation_payload = {
            "pipelineTasks": [
                {
                    "taskType": "translation",
                    "config": {
                        "language": {
                            "sourceLanguage": "en",
                            "targetLanguage": target_language
                        },
                        "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                    }
                }
            ],
            "inputData": {
                "input": [
                    {
                        "source": summary
                    }
                ]
            }
        }
        translation_response = make_request(app.config['CALLBACK_URL'], translation_payload)
        translated_summary = translation_response.get("pipelineResponse", [{}])[0].get("output", [{}])[0].get("target", "Translation failed")

        return jsonify({"summary": translated_summary, "original_summary": summary})

    return jsonify({"summary": summary})

@app.route('/ask', methods=['POST'])
def ask():
    """Answers questions based on the summarized article."""
    global article_data
    question = request.json.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400
    if not article_data["text"]:
        return jsonify({"error": "No article loaded. Summarize an article first."}), 400

    # Get answer and confidence score
    result = qa_model(question=question, context=article_data["text"])
    
    # Set confidence threshold
    confidence_threshold = 0.3

    # If confidence is too low, return "No information available"
    if result["score"] < confidence_threshold:
        return jsonify({"answer": "No information available."})
    
    return jsonify({"answer": result["answer"]})

if __name__ == '__main__':
    app.run(debug=True)
