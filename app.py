import streamlit as st
import time
import PyPDF2
import pandas as pd
import numpy as np

st.set_page_config(page_title="Safe Multi-Agent Gov Policy AI", layout="wide")
st.title("AI-Powered Government Policy Analyzer (Safe, Open-Source, Multi-Agent)")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt"])
if uploaded_file:
    # --- Extract Text ---
    if uploaded_file.type == "application/pdf":
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except:
            text = ""
            st.warning("PDF extraction failed. File may be corrupted or unreadable.")
    else:
        try:
            text = uploaded_file.read().decode("utf-8")
        except:
            text = ""
            st.warning("Text file could not be read.")

    if text:
        st.subheader("Extracted Text")
        st.write(text[:1000] + " ...")

        # --- NLP: spaCy Named Entities ---
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.subheader("Named Entities")
            st.dataframe(pd.DataFrame(entities, columns=["Entity", "Type"]))
        except:
            st.warning("spaCy failed or is too heavy. Named Entity Recognition skipped.")
            doc = None
            entities = []

        # --- Multi-Agent Semantic Search ---
        st.subheader("Multi-Agent Semantic Search")
        try:
            from sentence_transformers import SentenceTransformer, util
            sentences = [sent.text for sent in doc.sents] if doc else text.split(".")
            emb_model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings = emb_model.encode(sentences, convert_to_tensor=True)
            query = st.text_input("Ask a question about the document:")
            if query:
                agent_results = []
                for i in range(1, 4):
                    start_time = time.time()
                    st.write(f"Agent {i} is thinking...")
                    time.sleep(0.5 + i*0.5)
                    query_emb = emb_model.encode(query, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_emb, embeddings)[0]
                    top_idx = cos_scores.argsort(descending=True)[:2]
                    answers = [sentences[idx] for idx in top_idx]
                    elapsed = time.time() - start_time
                    agent_results.append((f"Agent {i}", answers, elapsed))
                for name, answers, elapsed in agent_results:
                    st.write(f"**{name} answers in {elapsed:.2f}s:**")
                    for ans in answers:
                        st.write(f"- {ans}")
        except:
            st.warning("Sentence Transformers too heavy or failed. Semantic search skipped.")

        # --- Text Summarization ---
        st.subheader("Document Summary (Agent 4)")
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            st.write(summary)
        except:
            st.warning("Summarization model is too heavy for free Streamlit. Skipped.")

        # --- KPI / Keyword Extraction ---
        st.subheader("Top Keywords / KPIs")
        try:
            if doc:
                words = [token.text for token in doc if not token.is_stop and token.is_alpha]
            else:
                words = text.split()
            freq = pd.Series(words).value_counts().head(10)
            st.bar_chart(freq)
        except:
            st.warning("Keyword extraction failed. Skipped.")

        # --- Knowledge Graph ---
        st.subheader("Knowledge Graph")
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            G = nx.Graph()
            for ent_text, ent_type in entities:
                G.add_node(ent_text, type=ent_type)
            for i in range(min(len(entities)-1, 15)):
                G.add_edge(entities[i][0], entities[i+1][0])
            fig, ax = plt.subplots(figsize=(8,5))
            nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=10)
            st.pyplot(fig)
        except:
            st.warning("Knowledge Graph visualization failed or too heavy. Skipped.")

        # --- Model Evaluation Demo ---
        st.subheader("Model Evaluation Metrics (Cosine Similarity Demo)")
        try:
            if embeddings is not None and len(sentences) >= 2:
                sim = util.cos_sim(embeddings[0], embeddings[1]).item()
                st.write(f"Cosine similarity between first two sentences: {sim:.3f}")
        except:
            st.warning("Model evaluation skipped due to missing embeddings or failure.")
