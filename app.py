# Streamlit Dashboard - Call Center NLP
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keybert import KeyBERT
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Call Center NLP Dashboard",
    page_icon="📞",
    layout="wide"
)

# ── LOAD MODELS (cached so they never reload between tab switches) ─────────────
@st.cache_resource
def load_spacy():
    return spacy.load('en_core_web_sm')

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

@st.cache_resource
def load_summarizer():
    model_name = "facebook/bart-large-cnn"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline(
        task="summarization",
        model=model,
        tokenizer=tokenizer
    )

@st.cache_resource
def load_topic_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

@st.cache_resource
def load_keybert():
    return KeyBERT()

@st.cache_resource
def load_qa_pipeline():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

# ── BUILD RAG IN-MEMORY FROM ANY DATAFRAME ────────────────────────────────────
# We pass a tuple of tuples (hashable) so Streamlit can cache it properly.
@st.cache_resource
def build_rag(_docs_tuple):
    """
    Accepts a tuple of (transcript, name, call_type, sentiment, summary) tuples.
    Builds an in-memory ChromaDB vector store — no persist_directory needed.
    """
    docs = []
    for transcript, name, call_type, sentiment, summary in _docs_tuple:
        docs.append(Document(
            page_content=str(transcript),
            metadata={
                "customer":  str(name),
                "type":      str(call_type),
                "sentiment": str(sentiment),
                "summary":   str(summary)
            }
        ))
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma.from_documents(documents=docs, embedding=embeddings)

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_default_data():
    # Fallback: ship your processed CSV with the repo
    return pd.read_csv('processed_calls.csv')
    
def get_customer_name(transcript):
    match = re.search(r'Customer\s*\(([^)]+)\)', transcript)
    if match:
        return match.group(1)
    return None

def get_dataset_summary(df):
    total      = len(df)
    call_types = df['Type'].value_counts()
    sentiments = df['Sentiment'].value_counts()
    return f"Total calls: {total}\nCall Types: {call_types}\nSentiments: {sentiments}"

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📞 Call Center NLP Dashboard")
st.markdown("**End-to-end NLP pipeline for call center transcript analysis**")
st.markdown("---")

# ── FILE UPLOADER ─────────────────────────────────────────────────────────────
st.sidebar.title("📂 Data Source")
uploaded_file = st.sidebar.file_uploader(
    "Upload your own call CSV",
    type=["csv"],
    help="CSV must have at least a 'Transcript' column. Optional: Name, Type, Sentiment, Summary."
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"✅ Loaded {len(df)} rows from your file")
else:
    df = load_default_data()
    st.sidebar.info("Using default dataset (20 calls). Upload your own CSV above.")

# Ensure required columns exist with safe fallbacks
for col, default in [("Name","Unknown"), ("Type","Unknown"),
                     ("Sentiment","Unknown"), ("Summary",""),
                     ("Predicted_Topic",""), ("Predicted_Intent","")]:
    if col not in df.columns:
        df[col] = default

if "Transcript" not in df.columns:
    st.error("❌ Your CSV must have a 'Transcript' column.")
    st.stop()

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.title("Navigation")
tab_selection = st.sidebar.radio(
    "Choose a section:",
    ["📊 Overview & EDA",
     "🔍 Analyze a Call",
     "🤖 RAG Assistant",
     "🗂️ Cluster Explorer"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════════════════════════════════════
if tab_selection == "📊 Overview & EDA":
    st.header("📊 Dataset Overview")

    if len(df) < 3:
        st.warning("Please upload a CSV with at least 3 call transcripts to view charts.")
        st.stop()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Calls", len(df))
    col2.metric("Call Types", df['Type'].nunique())
    col3.metric("Sentiment Types", df['Sentiment'].nunique())
    col4.metric(
        "Avg Words/Call",
        f"{df['Transcript'].apply(lambda x: len(str(x).split())).mean():.0f}"
    )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['Sentiment'].value_counts()
        fig, ax = plt.subplots()
        colors = plt.cm.Set2.colors[:len(sentiment_counts)]
        ax.bar(sentiment_counts.index, sentiment_counts.values, color=colors)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Call Type Distribution")
        type_counts = df['Type'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.subheader("Raw Data")

    display_cols = [c for c in ['Name','Type','Sentiment',
                                 'Predicted_Topic','Predicted_Intent','Summary']
                    if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYZE A SINGLE CALL
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selection == "🔍 Analyze a Call":
    st.header("🔍 Analyze a Single Transcript")
    st.markdown("Paste any call transcript below to run the full NLP pipeline on it.")

    nlp              = load_spacy()
    sentiment_model  = load_sentiment_model()
    summarizer       = load_summarizer()
    topic_classifier = load_topic_classifier()
    kw_model         = load_keybert()

    user_transcript = st.text_area(
        "Paste transcript here:",
        height=200,
        placeholder="Hello, I'm calling about my order..."
    )

    if st.button("🚀 Analyze", type="primary"):
        if user_transcript.strip() == "":
            st.warning("Please enter a transcript first.")
        else:
            with st.spinner("Running NLP pipeline..."):

                # 1. PII Redaction
                doc      = nlp(user_transcript)
                redacted = user_transcript
                customer_name = get_customer_name(user_transcript)
                for ent in doc.ents:
                    if ent.label_ == 'PERSON' and ent.text == customer_name:
                        redacted = redacted.replace(ent.text, '[CUSTOMER_NAME]')
                redacted = re.sub(r'\b\d{6}\b', '[ORDER_NUMBER]', redacted)
                redacted = re.sub(r'\b\d{4}-\d{3}-[A-Z]\b', '[ACCOUNT_NUMBER]', redacted)
                           
                # 2. Sentiment
                sentiment_result = sentiment_model(user_transcript[:512])[0]

                # 3. Summarization
                if len(user_transcript.split()) > 30:
                    summary_result = summarizer(
                        user_transcript[:1024],
                        max_length=60,
                        min_length=20,
                        do_sample=False
                    )[0]['summary_text']
                else:
                    summary_result = "Transcript too short to summarize."

                # 4. Topic (zero-shot)
                topic_labels = ['product inquiry','complaint',
                                'technical issue','compliment','order placement']
                topic_result = topic_classifier(
                    user_transcript[:512],
                    candidate_labels=topic_labels
                )['labels'][0]

                # 5. Intent (zero-shot)
                intent_labels = ['get information','file a complaint',
                                 'request refund','cancel order',
                                 'get technical support','give positive feedback']
                intent_result = topic_classifier(
                    user_transcript[:512],
                    candidate_labels=intent_labels
                )['labels'][0]

                # 6. Keywords
                keywords     = kw_model.extract_keywords(
                    user_transcript,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=5
                )
                keyword_list = [kw[0] for kw in keywords]

            # ── Store results in session_state so they don't vanish on rerun ──
            st.session_state['analysis'] = {
                'sentiment': sentiment_result,
                'topic':     topic_result,
                'intent':    intent_result,
                'summary':   summary_result,
                'keywords':  keyword_list,
                'redacted':  redacted,
                'entities':  [(ent.text, ent.label_) for ent in doc.ents]
            }

    # ── Display results if they exist in session_state ────────────────────────
    if 'analysis' in st.session_state:
        r = st.session_state['analysis']

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sentiment", r['sentiment']['label'])
        col2.metric("Topic",     r['topic'].title())
        col3.metric("Intent",    r['intent'].title())

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📝 Summary")
            st.info(r['summary'])

            st.subheader("🔑 Keywords")
            # st.badge doesn't exist — use st.markdown with inline code
            kw_html = " ".join(
                f"<code style='background:#e8f4fd;padding:3px 8px;"
                f"border-radius:4px;margin:2px;display:inline-block'>{kw}</code>"
                for kw in r['keywords']
            )
            st.markdown(kw_html, unsafe_allow_html=True)

        with col2:
            st.subheader("🔒 PII Redacted Version")
            st.success(r['redacted'])

            st.subheader("🏷️ Named Entities Found")
            if r['entities']:
                for text, label in r['entities']:
                    st.write(f"**{text}** → `{label}`")
            else:
                st.write("No named entities detected.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RAG ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selection == "🤖 RAG Assistant":
    st.header("🤖 RAG Assistant")
    st.markdown(
        "Ask questions about **your loaded call data**. "
        "The assistant searches actual transcripts and generates an answer."
    )

    # Build RAG from whatever CSV is currently loaded
    with st.spinner("Building vector store from transcripts... (first load only)"):
        docs_tuple = tuple(
            (
                row.get('Transcript', ''),
                row.get('Name', 'Unknown'),
                row.get('Type', 'Unknown'),
                row.get('Sentiment', 'Unknown'),
                row.get('Summary', '')
            )
            for _, row in df.iterrows()
        )
        vectordb   = build_rag(docs_tuple)

    qa_pipeline = load_qa_pipeline()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    st.subheader("💡 Try these questions:")
    col1, col2, col3 = st.columns(3)
    if col1.button("What are common complaints?"):
        st.session_state.current_q = "What are common complaints?"
    if col2.button("Which customers are angry?"):
        st.session_state.current_q = "Which customers are angry?"
    if col3.button("What products are mentioned?"):
        st.session_state.current_q = "What products are mentioned?"

    question = st.text_input(
        "Or type your own question:",
        value=st.session_state.get('current_q', '')
    )

    if st.button("Ask 🔍"):
        if question:
            with st.spinner("Searching transcripts..."):
                relevant_docs = vectordb.similarity_search(question, k=3)
                context       = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )
                summary = get_dataset_summary(df)
                st.write(summary)
                prompt = f"""Based on call center transcripts, answer briefly.
Dataset Summary: {summary}
Context: {context[:800]}
Question: {question}
Answer:"""
                answer = qa_pipeline(
                    prompt, max_length=100, do_sample=False
                )[0]['generated_text']

                st.session_state.chat_history.append({
                    'question': question,
                    'answer':   answer,
                    'sources': [
                        f"{d.metadata['customer']} ({d.metadata['type']})"
                        for d in relevant_docs
                    ]
                })
            # Clear the suggested question after asking
            st.session_state.current_q = ''

    for chat in reversed(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
            st.caption(f"Sources: {', '.join(chat['sources'])}")
            st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif tab_selection == "🗂️ Cluster Explorer":
    st.header("🗂️ Customer Cluster Explorer")
    st.markdown(
        "Transcripts are vectorized with TF-IDF, grouped into clusters "
        "using K-Means, then visualized in 2D via PCA. "
        "Works on whatever CSV you've uploaded."
    )

if len(df) < 10:
    st.warning("Not enough data. Please upload a CSV with at least 10 call transcripts to use the Cluster Explorer.")
else:
    n_clusters = st.slider(
        "Number of clusters", min_value=2,
        max_value=min(10, len(df)), value=min(5, len(df))
    )

    vectorizer    = TfidfVectorizer(stop_words='english', max_features=100)
    X             = vectorizer.fit_transform(df['Transcript'])
    kmeans        = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)
    pca           = PCA(n_components=2)
    X_2d          = pca.fit_transform(X.toarray())

    st.subheader("Call Clusters (PCA Visualization)")
    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(n_clusters):
        mask = df['Cluster'] == cluster
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[colors[cluster % 10]],
            label=f'Cluster {cluster}',
            s=120, alpha=0.8
        )
        for idx in df[mask].index:
            name = str(df['Name'].iloc[idx]).split()[0] if 'Name' in df.columns else str(idx)
            ax.annotate(name, (X_2d[idx, 0], X_2d[idx, 1]), fontsize=8)

    ax.legend()
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    selected_cluster = st.selectbox(
        "Select a cluster to explore:",
        options=list(range(n_clusters)),
        format_func=lambda x: f"Cluster {x}"
    )
    cluster_df = df[df['Cluster'] == selected_cluster]
    st.subheader(f"Cluster {selected_cluster} — {len(cluster_df)} calls")

    display_cols = [c for c in ['Name','Type','Sentiment',
                                 'Predicted_Topic','Summary']
                    if c in cluster_df.columns]
    st.dataframe(cluster_df[display_cols], use_container_width=True)
