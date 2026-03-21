import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("ISOM5240 Project")
# Your app code continues here...


# Page config
st.set_page_config(
    page_title="CAD Drawing Sentiment Analyzer", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4;}
    .metric-card {background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white;}
</style>
""", unsafe_allow_html=True)

def extract_engineering_info(text: str):
    """Parse CAD drawing text for engineering metadata"""
    # Extract scales, dimensions, titles
    scale = re.search(r'1[:\-]\d+', text, re.IGNORECASE)
    dimensions = re.findall(r'\d+[xX×]\s*\d+', text)
    title_words = re.findall(r'\b[A-Z]{4,}\b', text)
    
    return {
        'scale': scale.group() if scale else 'N/A',
        'dimensions': dimensions[:3],
        'title': title_words[0] if title_words else 'N/A',
        'text_count': len(text.split())
    }

def analyze_sentiment_hf(text: str, pipe):
    """Sentiment analysis with engineering context"""
    # Clean technical noise for sentiment
    clean_text = re.sub(r'[\d:\-xX×mm]+', '', text)
    clean_text = re.sub(r'[A-Z]{2,4}\d+', '', clean_text)
    clean_text = ' '.join(clean_text.split())[:512]  # Truncate for model
    
    if len(clean_text.strip()) < 10:
        return {'label': 'NEUTRAL', 'score': 0.5, 'compound': 0.0}
    
    result = pipe(clean_text)[0]
    
    # Engineering flags
    neg_flags = re.findall(r'\b(urgent|critical|reject|fail|error|defect|revise|fix|redesign)\b', text.lower())
    pos_flags = re.findall(r'\b(approved|ok|good|final|complete|acceptable)\b', text.lower())
    
    compound = 1.0 if result['label'] == 'POSITIVE' else -1.0 if result['label'] == 'NEGATIVE' else 0.0
    
    return {
        'label': result['label'],
        'score': result['score'],
        'compound': compound,
        'neg_flags': neg_flags,
        'pos_flags': pos_flags,
        'raw_prediction': result
    }

# Header
st.markdown('<h1 class="main-header">🔍 CAD Drawing Sentiment Analyzer</h1>', unsafe_allow_html=True)
st.markdown("**Extract text from engineering drawings → Analyze engineer sentiment with Hugging Face Transformers**")

# Sidebar
st.sidebar.header("⚙️ Settings")
max_files = st.sidebar.slider("Max files to process", 1, 10, 3)

# Load models
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

text_input = st.text_area("📄 Paste extracted text here:", 
                         value=st.session_state.extracted_text,
                         height=200)

# File uploader
uploaded_files = st.file_uploader(
    "Upload CAD Drawings (JPG/PNG)", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True,
    max_uploaded_files=max_files
)

if uploaded_files:
    # Process all files
    results = []
    
    for uploaded_file in uploaded_files:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # OCR
            image = Image.open(uploaded_file)
            ocr_results = st.session_state.ocr_reader.readtext(uploaded_file, detail=1)
            
            # Filter high-confidence text
            texts = [text for (_, text, conf) in ocr_results if conf > 0.4]
            full_text = ' '.join(texts)
            
            # Engineering metadata
            eng_info = extract_engineering_info(full_text)
            
            # Sentiment analysis
            sentiment = analyze_sentiment_hf(full_text, st.session_state.sentiment_pipe)
            
            results.append({
                'filename': uploaded_file.name,
                'image': image,
                'texts_found': len(ocr_results),
                'confidence_texts': len(texts),
                **eng_info,
                'full_text': full_text,
                **sentiment,
                'processing_time': time.time()
            })
    
    # Results Dashboard
    st.subheader("📊 Analysis Results")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Drawings", len(results))
    with col2:
        avg_compound = sum(r['compound'] for r in results) / len(results)
        st.metric("Avg Sentiment", f"{avg_compound:.3f}")
    with col3:
        total_texts = sum(r['texts_found'] for r in results)
        st.metric("Total Text Elements", total_texts)
    with col4:
        pos_count = sum(1 for r in results if r['label'] == 'POSITIVE')
        st.metric("Positive Drawings", f"{pos_count}/{len(results)}")
    
    # Results table
    df_display = pd.DataFrame([{
        'File': r['filename'][:30],
        'Sentiment': r['label'],
        'Confidence': f"{r['score']:.1%}",
        'Texts': r['texts_found'],
        'Title': r['title'],
        'Scale': r['scale'],
        'Flags': len(r['neg_flags'] + r['pos_flags'])
    } for r in results])
    
    st.dataframe(df_display, use_container_width=True)
    
    # Individual results
    for result in results:
        with st.expander(f"📐 {result['filename']} - {result['label']} (Conf: {result['score']:.1%})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(result['image'], caption="CAD Drawing", use_column_width=True)
                
                st.subheader("📋 Extracted Engineering Data")
                eng_df = pd.DataFrame([{
                    'Texts Found': result['texts_found'],
                    'High Confidence': result['confidence_texts'],
                    'Title': result['title'],
                    'Scale': result['scale'],
                    'Dimensions': ', '.join(result['dimensions']) if result['dimensions'] else 'N/A'
                }]).T
                st.dataframe(eng_df, use_container_width=True)
                
                if result['neg_flags'] or result['pos_flags']:
                    st.warning("🚩 **Engineering Flags Found:**")
                    st.write(f"**Negative**: {', '.join(result['neg_flags'])}")
                    st.write(f"**Positive**: {', '.join(result['pos_flags'])}")
            
            with col2:
                # Sentiment gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=result['score'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence"},
                    delta={'reference': 0.5},
                    gauge={
                        'axis': {'range': [None, 1]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgray"},
                            {'range': [0.5, 0.8], 'color': "yellow"},
                            {'range': [0.8, 1.0], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 0.5
                        }
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📝 Sample Extracted Text")
                st.text_area("", result['full_text'][:300], height=150, disabled=True)
    
    # Download results
    csv = pd.DataFrame(results).to_csv(index=False)
    st.download_button(
        "💾 Download Full Results CSV",
        csv,
        f"cad_sentiment_analysis_{len(results)}_drawings.csv",
        "text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**Built with ❤️ using EasyOCR + Hugging Face Transformers**  
Upload your CAD drawings to analyze engineer sentiment instantly!
""")
