import streamlit as st
import time
import PyPDF2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Safe Multi-Agent AI Demo", layout="wide")
st.title("AI-Powered Gov Policy Analyzer (Free Streamlit Compatible)")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    # --- Extract text ---
    if uploaded_file.type == "application/pdf":
        try:
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages])
        except:
            text = ""
            st.warning("PDF extraction failed. Using placeholder text.")
    else:
        try:
            text = uploaded_file.read().decode("utf-8")
        except:
            text = ""
            st.warning("Text file read failed. Using placeholder text.")

    if not text:
        text = "This is placeholder text for demonstration purposes."

    st.subheader("Extracted Text (truncated)")
    st.write(text[:500] + " ...")

    # --- Multi-Agent Simulation ---
    st.subheader("Multi-Agent Responses")
    dummy_agents = ["Agent 1", "Agent 2", "Agent 3"]
    for agent in dummy_agents:
        st.write(f"{agent} is thinking...")
        start_time = time.time()
        time.sleep(0.5)  # simulate processing
        elapsed = time.time() - start_time
        st.write(f"{agent} answer (simulated) in {elapsed:.2f}s: {text[:100]} ...")

    # --- Knowledge Graph ---
    st.subheader("Knowledge Graph (Simulated)")
    G = nx.Graph()
    entities = text.split()[:10]  # simulate entity extraction
    for i, entity in enumerate(entities):
        G.add_node(entity)
        if i > 0:
            G.add_edge(entities[i-1], entity)
    fig, ax = plt.subplots(figsize=(8,5))
    nx.draw(G, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1200, font_size=10)
    st.pyplot(fig)

    # --- KPI / Keywords ---
    st.subheader("Top Keywords / KPIs (Simulated)")
    words = text.split()
    unique, counts = np.unique(words[:20], return_counts=True)
    st.bar_chart(dict(zip(unique, counts)))

    st.success("Demo running fully on Free Streamlit with open-source tools!")
