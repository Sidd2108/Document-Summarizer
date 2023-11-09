import streamlit as st
from PyPDF2 import PdfReader
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import string

# Function to summarize text using NLTK
def summarize_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize the text into words
    words = word_tokenize(text)

    # Remove punctuation and stop words
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words("english") + list(string.punctuation))
    words = [word for word in words if word not in stop_words]

    # Calculate word frequency
    word_freq = FreqDist(words)

    # Calculate sentence scores based on word frequency
    sentence_scores = {sentence: sum(word_freq[word] for word in word_tokenize(sentence)) for sentence in sentences}

    # Select top 3 sentences as the summary
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:3]

    return ' '.join(summary_sentences)


# Streamlit app
def main():
    st.title("Document Summarizer with NLTK")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            # Read the PDF file
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Summarize the text
            summary = summarize_text(text)

            st.success("Summary generated successfully!")

            st.write("Here is the summarized text:")
            st.write(summary)


if __name__ == "__main__":
    main()
