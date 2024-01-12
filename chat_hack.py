import openai
import pdfplumber
import spacy
import networkx as nx
import pytesseract
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Set your OpenAI API key
openai.api_key = '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'

# Load spaCy for NLP tasks with a larger English model
nlp = spacy.load("en_core_web_md")

# Add a Named Entity Recognition (NER) pipeline
ner = nlp.get_pipe("ner")

# Add custom entity labels
ner.add_label("EMPLOYEE")
ner.add_label("SKILL")
ner.add_label("EXPERIENCE")

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize global variables
conversation_context = {
    'job_skills_graph': nx.Graph(),
    'courses_graph': nx.Graph(),
    'other_info': 'whatever_initial_info_you_need'
}

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def extract_text_from_image_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for img_index, img_dict in enumerate(page.images):
                img = pdf.pages[page.page_number - 1].images[img_index]
                img_data = img['stream'].get_data()
                image_text = pytesseract.image_to_string(Image.open(io.BytesIO(img_data)))
                text += image_text
            text += page.extract_text()
    return text

def process_pdf_and_get_context(pdf_path):
    # Initialize the conversational context
    global conversation_context

    # Check if a PDF is uploaded
    if pdf_path:
        # Extract text from the PDF
        with pdfplumber.open(pdf_path) as pdf:
            contains_images = any(page.images for page in pdf.pages)

        if contains_images:
            extracted_text = extract_text_from_image_pdf(pdf_path)
        else:
            extracted_text = extract_text_from_pdf(pdf_path)

        # Extract entities using the custom NER model
        doc = nlp(extracted_text)
        entities = [(ent.text.lower().replace('\n', '').strip(), ent.label_) for ent in doc.ents]

        # Update or create the knowledge graph
        if 'job_skills_graph' not in conversation_context:
            conversation_context['job_skills_graph'] = nx.Graph()
        if 'courses_graph' not in conversation_context:
            conversation_context['courses_graph'] = nx.Graph()

        # Update the graph with new entities and edges
        conversation_context['job_skills_graph'].add_nodes_from(entities)
        for i in range(len(entities) - 1):
            conversation_context['job_skills_graph'].add_edge(entities[i][0], entities[i + 1][0])

        # Print the nodes of the knowledge graph for debugging
        print("Job Skills Graph Nodes:", conversation_context['job_skills_graph'].nodes)

    # Return the updated or existing conversation context
    return conversation_context

def process_user_query(query, context):
    # Use the context knowledge graph to enrich the query
    # Extract relevant entities from the knowledge graph
    graph_entities = list(context['job_skills_graph'].nodes)
    print("Job Skills Graph Entities:", graph_entities)

    # Add extracted entities to the user's query to provide more context
    enriched_query = f"{query} {graph_entities}"
    print("Enriched Query:", enriched_query)

    prompt = f"{context} User Query: {enriched_query}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system",
             "content": f"You are a helpful assistant that analyzes  answers questions based on information"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response['choices'][0]['message']['content']

@app.route('/ask', methods=['POST'])
def ask_question():
    user_query = request.form.get('query')
    file = request.files.getlist("file")
    # Use get() to handle cases where "file" is not present
    file_path = "temp_file.pdf" if file else None

    conversation_context = process_pdf_and_get_context(file_path)

    # Process user query and update conversation context
    response_text = process_user_query(user_query, conversation_context)
    conversation_context['response'] = response_text  # Update response in the context

    # Return AI's response and graph nodes as a list
    return jsonify({'response': response_text})

if __name__ == '__main__':
    app.run()
