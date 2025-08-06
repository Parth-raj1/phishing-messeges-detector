import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from emotion_analyzer import EmotionAnalyzer
from text_processor import TextProcessor

def main():
    st.set_page_config(
        page_title="Scam Message Emotion Analysis",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Phishing Email Emotion Analysis")
    st.markdown("### Analyze emotional valence and arousal in phishing emails using EmoBank methodology")
    
    # Initialize components
    if 'emotion_analyzer' not in st.session_state:
        with st.spinner("Loading EmoBank lexicon..."):
            st.session_state.emotion_analyzer = EmotionAnalyzer()
            st.session_state.text_processor = TextProcessor()
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # File upload section
    st.header("📁 Upload Phishing Emails CSV")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing phishing emails",
        type=['csv'],
        help="CSV should contain a column with email text messages"
    )
    
    if uploaded_file is not None:
        try:
            # Load the CSV file
            df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(df)} messages from CSV file")
            
            # Display basic info about the dataset
            st.subheader("📊 Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Messages", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                text_cols = df.select_dtypes(include=['object'])
                if len(text_cols.columns) > 0:
                    lengths = text_cols.apply(lambda x: x.astype(str).str.len()).mean()
                    avg_length = lengths.max() if len(lengths) > 0 else 0
                    st.metric("Avg Message Length", f"{avg_length:.0f}" if not pd.isna(avg_length) else "N/A")
                else:
                    st.metric("Avg Message Length", "N/A")
            
            # Column selection
            st.subheader("🎯 Select Message Column")
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                st.error("No text columns found in the CSV file. Please ensure your CSV contains text data.")
                return
            
            selected_column = st.selectbox(
                "Choose the column containing scam messages:",
                text_columns,
                help="Select the column that contains the scam messages to analyze"
            )
            
            # Preview data
            st.subheader("👀 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Analysis configuration
            st.subheader("⚙️ Analysis Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                min_word_length = st.slider("Minimum word length", 1, 10, 3)
                remove_stopwords = st.checkbox("Remove stopwords", value=True)
            
            with col2:
                normalize_scores = st.checkbox("Normalize emotion scores", value=True)
                include_neutral = st.checkbox("Include neutral messages", value=True)
            
            # Analysis button
            if st.button("🚀 Analyze Emotions", type="primary"):
                analyze_emotions(df, selected_column, min_word_length, remove_stopwords, normalize_scores, include_neutral)
        
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            st.info("Please ensure your CSV file is properly formatted and contains text data.")
    
    else:
        # Show example of expected CSV format and provide sample data
        st.subheader("📋 Expected CSV Format")
        example_data = {
            'message': [
                'Congratulations! You have won $1000. Click here to claim your prize now!',
                'URGENT: Your account has been compromised. Verify your details immediately.',
                'Limited time offer! Get rich quick with our investment scheme.'
            ],
            'source': ['email', 'sms', 'social_media']
        }
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        st.info("Your CSV should contain at least one column with text messages to analyze.")
        
        # Provide sample spam dataset
        st.subheader("🔬 Try Sample Dataset")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Load Sample Phishing Emails", type="secondary"):
                try:
                    sample_df = pd.read_csv('data/phishing_emails.csv', encoding='utf-8')
                    # Clean up the dataframe - keep only relevant columns
                    sample_df = sample_df[['Email Text', 'Email Type']].copy()
                    sample_df.columns = ['message', 'label']
                    
                    # Filter to show only phishing emails for analysis
                    phishing_messages = sample_df[sample_df['label'] == 'Phishing Email'].copy()
                    
                    st.success(f"Loaded {len(phishing_messages)} phishing emails from sample dataset")
                    st.session_state['sample_data'] = phishing_messages
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
        
        with col2:
            st.markdown("""
            **Sample Dataset Info:**
            - Real phishing/safe email dataset
            - 18,650 emails total (7,328 phishing)
            - Pre-labeled phishing/safe emails
            - Perfect for testing the emotion analysis
            """)
        
        # Show loaded sample data if available
        if 'sample_data' in st.session_state:
            st.subheader("📊 Sample Phishing Emails Loaded")
            sample_df = st.session_state['sample_data']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Phishing Emails", len(sample_df))
            with col2:
                avg_length = sample_df['message'].astype(str).str.len().mean()
                st.metric("Avg Email Length", f"{avg_length:.0f}")
            with col3:
                st.metric("Ready for Analysis", "✓")
            
            # Preview the data
            st.subheader("👀 Data Preview")
            st.dataframe(sample_df.head(10), use_container_width=True)
            
            # Analysis button for sample data
            st.subheader("⚙️ Analysis Configuration")
            col1, col2 = st.columns(2)
            
            with col1:
                min_word_length = st.slider("Minimum word length", 1, 10, 3, key="sample_min_word")
                remove_stopwords = st.checkbox("Remove stopwords", value=True, key="sample_stopwords")
            
            with col2:
                normalize_scores = st.checkbox("Normalize emotion scores", value=True, key="sample_normalize")
                include_neutral = st.checkbox("Include neutral messages", value=True, key="sample_neutral")
            
            if st.button("🚀 Analyze Sample Phishing Emails", type="primary"):
                analyze_emotions(sample_df, 'message', min_word_length, remove_stopwords, normalize_scores, include_neutral)

def analyze_emotions(df, text_column, min_word_length, remove_stopwords, normalize_scores, include_neutral):
    """Perform emotion analysis on the messages"""
    
    # Validate the selected column
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the dataset.")
        return
    
    # Remove rows with missing text data
    original_count = len(df)
    df_clean = df.dropna(subset=[text_column]).copy()
    df_clean = df_clean[df_clean[text_column].astype(str).str.strip() != '']
    
    if len(df_clean) == 0:
        st.error("No valid messages found in the selected column.")
        return
    
    if len(df_clean) < original_count:
        st.warning(f"Removed {original_count - len(df_clean)} messages with missing or empty text.")
    
    # Process messages
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    emotion_analyzer = st.session_state.emotion_analyzer
    text_processor = st.session_state.text_processor
    
    for idx, message in enumerate(df_clean[text_column]):
        progress = (idx + 1) / len(df_clean)
        progress_bar.progress(progress)
        status_text.text(f"Processing message {idx + 1} of {len(df_clean)}")
        
        try:
            # Preprocess text
            processed_text = text_processor.preprocess_text(
                str(message), 
                min_word_length=min_word_length,
                remove_stopwords=remove_stopwords
            )
            
            # Analyze emotions
            valence, arousal, word_count, coverage = emotion_analyzer.analyze_emotion(processed_text)
            
            results.append({
                'original_message': message,
                'processed_text': processed_text,
                'valence': valence,
                'arousal': arousal,
                'word_count': word_count,
                'coverage': coverage,
                'message_length': len(str(message)),
                'processed_length': len(processed_text)
            })
            
        except Exception as e:
            st.warning(f"Error processing message {idx + 1}: {str(e)}")
            results.append({
                'original_message': message,
                'processed_text': '',
                'valence': np.nan,
                'arousal': np.nan,
                'word_count': 0,
                'coverage': 0,
                'message_length': len(str(message)),
                'processed_length': 0
            })
    
    progress_bar.empty()
    status_text.empty()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Filter out neutral messages if requested
    if not include_neutral:
        # Define neutral as messages with valence close to 5 (middle of 1-9 scale) and low arousal
        neutral_mask = (abs(results_df['valence'] - 5) <= 1) & (results_df['arousal'] <= 3)
        results_df = results_df[~neutral_mask]
        st.info(f"Filtered out {neutral_mask.sum()} neutral messages.")
    
    # Normalize scores if requested  
    if normalize_scores:
        results_df['valence_normalized'] = (results_df['valence'] - 1) / 8  # Scale 1-9 to 0-1
        results_df['arousal_normalized'] = (results_df['arousal'] - 1) / 8   # Scale 1-9 to 0-1
    
    # Display results
    display_results(results_df, normalize_scores)

def display_results(results_df, normalize_scores):
    """Display analysis results with visualizations and statistics"""
    
    st.success(f"✅ Analysis complete! Processed {len(results_df)} messages.")
    
    # Statistical Summary
    st.subheader("📈 Statistical Summary")
    
    valence_col = 'valence_normalized' if normalize_scores else 'valence'
    arousal_col = 'arousal_normalized' if normalize_scores else 'arousal'
    
    # Remove NaN values for statistics
    valid_results = results_df.dropna(subset=[valence_col, arousal_col])
    
    if len(valid_results) == 0:
        st.error("No valid emotion scores calculated. Please check your data and EmoBank lexicon.")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mean Valence", 
            f"{valid_results[valence_col].mean():.3f}",
            f"{valid_results[valence_col].std():.3f} std"
        )
    
    with col2:
        st.metric(
            "Mean Arousal", 
            f"{valid_results[arousal_col].mean():.3f}",
            f"{valid_results[arousal_col].std():.3f} std"
        )
    
    with col3:
        st.metric(
            "Coverage Rate", 
            f"{valid_results['coverage'].mean():.1%}",
            f"Avg words recognized"
        )
    
    with col4:
        st.metric(
            "Valid Analyses", 
            f"{len(valid_results)}/{len(results_df)}",
            f"{len(valid_results)/len(results_df):.1%} success rate"
        )
    
    # Visualizations
    st.subheader("📊 Emotion Distribution")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Valence-Arousal Plot", "Distributions", "Message Analysis", "Detailed Results"])
    
    with tab1:
        # Valence-Arousal scatter plot
        fig = px.scatter(
            valid_results,
            x=valence_col,
            y=arousal_col,
            hover_data=['coverage', 'word_count'],
            title="Valence-Arousal Distribution of Scam Messages",
            labels={
                valence_col: f"Valence {'(Normalized)' if normalize_scores else '(1-9 scale)'}",
                arousal_col: f"Arousal {'(Normalized)' if normalize_scores else '(1-9 scale)'}"
            },
            color='coverage',
            color_continuous_scale='viridis'
        )
        
        # Add quadrant lines for interpretation
        if normalize_scores:
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.5)
        else:
            fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation guide
        st.info("""
        **Quadrant Interpretation:**
        - **Top Right**: High valence (positive) + High arousal (exciting/energetic)
        - **Top Left**: Low valence (negative) + High arousal (alarming/urgent)
        - **Bottom Right**: High valence (positive) + Low arousal (calm/pleasant)
        - **Bottom Left**: Low valence (negative) + Low arousal (sad/depressing)
        """)
    
    with tab2:
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_val = px.histogram(
                valid_results,
                x=valence_col,
                nbins=20,
                title="Valence Distribution",
                labels={valence_col: f"Valence {'(Normalized)' if normalize_scores else '(1-9 scale)'}"}
            )
            st.plotly_chart(fig_val, use_container_width=True)
        
        with col2:
            fig_ar = px.histogram(
                valid_results,
                x=arousal_col,
                nbins=20,
                title="Arousal Distribution",
                labels={arousal_col: f"Arousal {'(Normalized)' if normalize_scores else '(1-9 scale)'}"}
            )
            st.plotly_chart(fig_ar, use_container_width=True)
    
    with tab3:
        # Message length vs emotion analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_len = px.scatter(
                valid_results,
                x='message_length',
                y=valence_col,
                color=arousal_col,
                title="Message Length vs Valence",
                labels={
                    'message_length': 'Original Message Length (characters)',
                    valence_col: f"Valence {'(Normalized)' if normalize_scores else '(1-9 scale)'}"
                }
            )
            st.plotly_chart(fig_len, use_container_width=True)
        
        with col2:
            fig_cov = px.scatter(
                valid_results,
                x='coverage',
                y=valence_col,
                color=arousal_col,
                title="Coverage vs Valence",
                labels={
                    'coverage': 'EmoBank Coverage Rate',
                    valence_col: f"Valence {'(Normalized)' if normalize_scores else '(1-9 scale)'}"
                }
            )
            st.plotly_chart(fig_cov, use_container_width=True)
    
    with tab4:
        # Detailed results table
        st.subheader("Detailed Analysis Results")
        
        # Prepare display dataframe
        display_df = results_df.copy()
        
        # Round numerical columns
        numeric_cols = ['valence', 'arousal', 'coverage']
        if normalize_scores:
            numeric_cols.extend(['valence_normalized', 'arousal_normalized'])
        
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)
        
        # Truncate long messages for display
        display_df['message_preview'] = display_df['original_message'].astype(str).str[:100] + '...'
        
        # Select columns for display
        cols_to_show = ['message_preview', 'valence', 'arousal', 'coverage', 'word_count', 'message_length']
        if normalize_scores:
            cols_to_show.extend(['valence_normalized', 'arousal_normalized'])
        
        st.dataframe(
            display_df[cols_to_show],
            use_container_width=True,
            height=400
        )
        
        # Export functionality
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV export
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv_buffer.getvalue(),
                file_name="scam_emotion_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Summary statistics export
            summary_stats = valid_results[numeric_cols].describe()
            summary_buffer = io.StringIO()
            summary_stats.to_csv(summary_buffer)
            st.download_button(
                label="📊 Download Summary Stats",
                data=summary_buffer.getvalue(),
                file_name="emotion_analysis_summary.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
