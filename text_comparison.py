import streamlit as st
import pandas as pd
import base64
from html import unescape
import re
from difflib import SequenceMatcher

# Set page config with light theme
st.set_page_config(
    layout="wide", 
    page_title="Text Comparison Viewer",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Set light theme
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    /* Style for column names in inputs */
    .column-name {
        display: inline-block;
        background-color: #f2f2f2;
        padding: 2px 6px;
        border-radius: 3px;
        font-weight: bold;
        margin-right: 5px;
    }
    /* Remove red highlighting from multiselect */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #f0f2f6 !important;
        color: #000000 !important;
    }
    /* Make multiselect pills neutral */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #e6e6e6 !important;
    }
    /* Unified section header styling */
    .section-header {
        font-size: 1rem;
        font-weight: bold;
        display: inline-block;
        margin-right: 0.5rem;
    }
    /* Value styling next to header */
    .section-value {
        display: inline-block;
    }
    /* Container for inputs to maintain fixed height */
    .inputs-container {
        margin-bottom: 15px;
    }
    /* Add spacing above stats */
    .stats-container {
        margin-bottom: 28px;
    }
    /* Preference buttons styling */
    .stButton>button {
        width: 100%;
        height: 46px;
        margin-bottom: 10px;
    }
    /* Highlight selected button */
    .selected-preference {
        margin-top: 10px;
        margin-bottom: 15px;
        padding: 5px 10px;
        background-color: #f2f2f2;
        border-radius: 4px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.title("Text Comparison Viewer")
st.write("Upload your CSV file to view all text comparisons sorted by similarity score. The file should contain model_1_output and model_2_output columns.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

def render_html_compact(html_str):
    # Unescape any HTML entities
    html_str = unescape(html_str) if html_str else ""
    return html_str

def identify_language_by_chars(text):
    # Remove spaces and punctuation
    text = ''.join(c for c in text if c.isalpha())
    
    # Character counters
    japanese_chars = 0
    chinese_chars = 0
    thai_chars = 0
    total_chars = len(text)
    
    if total_chars == 0:
        return "No text to analyze"
    
    for char in text:
        # Japanese-specific characters (Hiragana and Katakana)
        if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF':
            japanese_chars += 1
        # Chinese characters (also used in Japanese)
        elif '\u4E00' <= char <= '\u9FFF':
            chinese_chars += 1
        # Thai characters
        elif '\u0E00' <= char <= '\u0E7F':
            thai_chars += 1
    
    # Calculate percentages
    japanese_percent = japanese_chars / total_chars * 100
    chinese_percent = chinese_chars / total_chars * 100
    thai_percent = thai_chars / total_chars * 100
    
    # Less strict detection - any significant presence of these languages makes it unsupported
    # Significantly lower the threshold to catch mixed content
    if japanese_percent > 20 or chinese_percent > 20 or thai_chars > 10:
        if japanese_percent > chinese_percent and japanese_percent > thai_percent:
            return "Japanese"
        elif chinese_percent > japanese_percent and chinese_percent > thai_percent:
            return "Chinese"
        elif thai_percent > japanese_percent and thai_percent > chinese_percent:
            return "Thai"
        else:
            return "Mixed Asian"
    
    return "Other"

def count_words(text):
    # Simple word count by splitting on whitespace
    if not text:
        return 0
    return len(text.split())

def is_unsupported_language(lang):
    return lang in ["Japanese", "Chinese", "Thai", "Mixed Asian"]

def generate_word_comparison_html(text1, text2):
    """Compare texts word by word instead of character by character"""
    if not text1 and not text2:
        return "", ""
    
    # Split texts into words
    words1 = text1.split() if text1 else []
    words2 = text2.split() if text2 else []
    
    # Use SequenceMatcher on word lists
    matcher = SequenceMatcher(None, words1, words2)
    
    result1 = []
    result2 = []
    
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            result1.extend([f"{w} " for w in words1[i1:i2]])
            result2.extend([f"{w} " for w in words2[j1:j2]])
        elif op == 'delete':
            result1.extend([f'<span style="background-color: #ffcccc;">{w} </span>' for w in words1[i1:i2]])
            # Nothing added to result2
        elif op == 'insert':
            # Nothing added to result1
            result2.extend([f'<span style="background-color: #ccffcc;">{w} </span>' for w in words2[j1:j2]])
        elif op == 'replace':
            result1.extend([f'<span style="background-color: #ffcccc;">{w} </span>' for w in words1[i1:i2]])
            result2.extend([f'<span style="background-color: #ccffcc;">{w} </span>' for w in words2[j1:j2]])
    
    return ''.join(result1), ''.join(result2)

if uploaded_file is not None:
    # Load the CSV file
    try:
        # Initialize session state to store preferences
        if 'preferences' not in st.session_state:
            st.session_state.preferences = {}
        
        if 'comments' not in st.session_state:
            st.session_state.comments = {}

        df = pd.read_csv(uploaded_file)
        st.success(f"File loaded successfully! {len(df)} rows found.")
        
        # Check if the expected columns exist
        required_columns = ['model_1_output', 'model_2_output']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
        else:
            # Allow user to select input columns to display
            st.markdown("### Select Input Columns")
            st.write("Choose input columns to display alongside model outputs for better comparison context.")
            
            # Get all columns that are not the model outputs or internal columns
            excluded_cols = ['model_1_output', 'model_2_output', 'Row ID', 'Model Preference', 'Comment',
                            'Similarity Score', 'detected_lang_model_1', 'detected_lang_model_2',
                            'is_unsupported', 'word_count_model_1', 'word_count_model_2', 
                            'word_count_diff', 'model_1_output_html', 'model_2_output_html', 'sort_key']
            available_cols = [col for col in df.columns if col not in excluded_cols]
            
            if available_cols:
                selected_input_cols = st.multiselect(
                    "Select input columns to display",
                    options=available_cols,
                    default=[]
                )
            else:
                selected_input_cols = []
                st.info("No additional input columns found in the CSV file.")
            
            # Process the data
            # Add row numbers if not already present
            if 'Row ID' not in df.columns:
                df['Row ID'] = range(1, len(df) + 1)
            
            # Detect languages
            df['detected_lang_model_1'] = df['model_1_output'].apply(identify_language_by_chars)
            df['detected_lang_model_2'] = df['model_2_output'].apply(identify_language_by_chars)
            
            # Determine if unsupported language
            df['is_unsupported'] = df.apply(
                lambda x: is_unsupported_language(x['detected_lang_model_1']) or 
                          is_unsupported_language(x['detected_lang_model_2']), 
                axis=1
            )
            
            # Count words for supported languages
            df['word_count_model_1'] = df.apply(
                lambda x: count_words(x['model_1_output']) if not x['is_unsupported'] else None, 
                axis=1
            )
            df['word_count_model_2'] = df.apply(
                lambda x: count_words(x['model_2_output']) if not x['is_unsupported'] else None, 
                axis=1
            )
            df['word_count_diff'] = df.apply(
                lambda x: x['word_count_model_2'] - x['word_count_model_1'] 
                if not pd.isna(x['word_count_model_1']) and not pd.isna(x['word_count_model_2']) 
                else None, 
                axis=1
            )
            
            # Calculate similarity score if not present or recalculate based on words
            df['Similarity Score'] = df.apply(
                lambda x: SequenceMatcher(None, x['model_1_output'].split(), x['model_2_output'].split()).ratio()
                if not x['is_unsupported'] else 0, 
                axis=1
            )
            
            # Generate HTML comparison for supported languages - using word-level comparison
            html_results = df.apply(
                lambda x: generate_word_comparison_html(x['model_1_output'], x['model_2_output'])
                if not x['is_unsupported'] else (x['model_1_output'], x['model_2_output']), 
                axis=1
            )
            df['model_1_output_html'] = [html[0] for html in html_results]
            df['model_2_output_html'] = [html[1] for html in html_results]
            
            # Add "Model Preference" column if not present
            if 'Model Preference' not in df.columns:
                df['Model Preference'] = ""
                
            # Add "Comment" column if not present
            if 'Comment' not in df.columns:
                df['Comment'] = ""
                
            # Set "Language is not supported" for unsupported languages
            df.loc[df['is_unsupported'], 'Model Preference'] = "Language is not supported"
            
            # Sort: unsupported languages at bottom, supported from most to least similar
            df['sort_key'] = df.apply(
                lambda x: 2 if x['is_unsupported'] else 1, 
                axis=1
            )
            df = df.sort_values(by=['sort_key', 'Similarity Score'], ascending=[True, False])
            
            # Create a mapping from UI text with emoji to clean values for storage
            PREFERENCE_MAPPING = {
                "👈 Model 1 is better": "Model 1",
                "👉 Model 2 is better": "Model 2",
                "🤝 Tie": "Equally Good",
                "👎 Both are bad": "Neither Preferred"
            }
            
            # Load existing preferences from Model Preference column
            for idx, row in df.iterrows():
                row_id = row['Row ID']
                if row['Model Preference'] and row['Model Preference'] != "Language is not supported":
                    # Store in session state if not already there
                    if row_id not in st.session_state.preferences:
                        st.session_state.preferences[row_id] = row['Model Preference']
                
                if row.get('Comment') and not pd.isna(row.get('Comment')):
                    # Store comment in session state
                    if row_id not in st.session_state.comments:
                        st.session_state.comments[row_id] = row['Comment']
            
            # Function to update model preference and comment
            def update_preference(row_id, preference, comment=None):
                # Store in session state
                clean_preference = PREFERENCE_MAPPING.get(preference, preference)
                st.session_state.preferences[row_id] = clean_preference
                
                # Also update dataframe
                row_idx = df[df['Row ID'] == row_id].index[0]
                if not df.at[row_idx, 'is_unsupported']:
                    df.at[row_idx, 'Model Preference'] = clean_preference
                    
                    # Update comment if provided
                    if comment is not None:
                        st.session_state.comments[row_id] = comment
                        df.at[row_idx, 'Comment'] = comment
            
            # Create containers for supported and unsupported languages
            supported_container = st.container()
            
            # Create a section for supported languages first
            with supported_container:
                st.markdown("## Supported Languages")
                
                # Show only supported language rows
                supported_df = df[~df['is_unsupported']]
                
                if supported_df.empty:
                    st.warning("No entries with supported languages found.")
                else:
                    st.success(f"Showing {len(supported_df)} entries with supported languages sorted from most similar to least similar.")
                    
                    # Apply preferences from session state to dataframe
                    for row_id, pref in st.session_state.preferences.items():
                        # Find the row in dataframe
                        matching_rows = df[df['Row ID'] == row_id]
                        if not matching_rows.empty:
                            row_idx = matching_rows.index[0]
                            # Only update if it's a supported language
                            if not df.at[row_idx, 'is_unsupported']:
                                df.at[row_idx, 'Model Preference'] = pref
                    
                    # Apply comments from session state to dataframe
                    for row_id, comment in st.session_state.comments.items():
                        # Find the row in dataframe
                        matching_rows = df[df['Row ID'] == row_id]
                        if not matching_rows.empty:
                            row_idx = matching_rows.index[0]
                            df.at[row_idx, 'Comment'] = comment
                    
                    for idx, row in supported_df.iterrows():
                        # Create a container with border for each comparison
                        with st.container():
                            st.markdown(f"<hr>", unsafe_allow_html=True)
                            
                            # Display row ID at the top
                            st.markdown(f"**Row {row['Row ID']}**")
                            
                            # Display stats at top area
                            stats_row = st.container()
                            with stats_row:
                                st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown('<div class="section-header">Similarity Score:</div>', unsafe_allow_html=True)
                                    st.markdown(f'<div class="section-value">{round(row["Similarity Score"], 4)}</div>', unsafe_allow_html=True)
                                
                                with col2:
                                    # Word count info on same line as header
                                    word_diff = row['word_count_diff']
                                    if word_diff is not None:
                                        diff_sign = "+" if word_diff > 0 else ""
                                        st.markdown('<div class="section-header">Word Count:</div>', unsafe_allow_html=True)
                                        st.markdown(f'<div class="section-value">{row["word_count_model_1"]} → {row["word_count_model_2"]} ({diff_sign}{word_diff})</div>', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display selected input columns if any were chosen
                            if selected_input_cols:
                                with st.container():
                                    st.markdown('<div class="inputs-container">', unsafe_allow_html=True)
                                    st.markdown('<div class="section-header">Original Inputs:</div>', unsafe_allow_html=True)
                                    for col in selected_input_cols:
                                        st.markdown(f"""<span class="column-name">{col}</span> {row[col]}""", unsafe_allow_html=True)
                                    st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Display models side-by-side
                            model_col1, model_col2 = st.columns(2)
                            
                            with model_col1:
                                st.markdown('<div class="section-header">Model 1:</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                                        height: 350px; overflow-y: auto; background-color: #f9f9f9;">
                                    {render_html_compact(row['model_1_output_html'])}
                                </div>
                                """, unsafe_allow_html=True)
                                
                            with model_col2:
                                st.markdown('<div class="section-header">Model 2:</div>', unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                                        height: 350px; overflow-y: auto; background-color: #f9f9f9;">
                                    {render_html_compact(row['model_2_output_html'])}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display current preference if already selected
                            # Get from session state if available
                            current_value = st.session_state.preferences.get(row['Row ID'], row['Model Preference'])
                            if current_value and current_value != "Language is not supported":
                                st.markdown(f'<div class="selected-preference">Current selection: <strong>{current_value}</strong></div>', unsafe_allow_html=True)
                            
                            # Create button-based preference selection - 2 rows of 2 buttons each for better layout
                            # First row: Model 1 vs Model 2
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                button_text = "👈 Model 1 is better"
                                model1_btn = st.button(button_text, key=f"m1_{row['Row ID']}")
                                if model1_btn:
                                    # Get current comment if any
                                    current_comment = st.session_state.comments.get(row['Row ID'], row.get('Comment', ""))
                                    update_preference(row['Row ID'], button_text, current_comment)
                                    st.rerun()
                                    
                            with col2:
                                button_text = "👉 Model 2 is better"
                                model2_btn = st.button(button_text, key=f"m2_{row['Row ID']}")
                                if model2_btn:
                                    current_comment = st.session_state.comments.get(row['Row ID'], row.get('Comment', ""))
                                    update_preference(row['Row ID'], button_text, current_comment)
                                    st.rerun()
                            
                            # Second row: Tie vs Neither
                            col3, col4 = st.columns(2)
                                    
                            with col3:
                                button_text = "🤝 Tie"
                                tie_btn = st.button(button_text, key=f"tie_{row['Row ID']}")
                                if tie_btn:
                                    current_comment = st.session_state.comments.get(row['Row ID'], row.get('Comment', ""))
                                    update_preference(row['Row ID'], button_text, current_comment)
                                    st.rerun()
                                    
                            with col4:
                                button_text = "👎 Both are bad"
                                neither_btn = st.button(button_text, key=f"neither_{row['Row ID']}")
                                if neither_btn:
                                    current_comment = st.session_state.comments.get(row['Row ID'], row.get('Comment', ""))
                                    update_preference(row['Row ID'], button_text, current_comment)
                                    st.rerun()
                            
                            # Show comment field regardless of whether a preference has been selected
                            # Get current comment from session state if available
                            current_comment = st.session_state.comments.get(row['Row ID'], "")
                            if pd.isna(current_comment):
                                current_comment = ""
                                
                            st.markdown('<div class="comment-section">', unsafe_allow_html=True)
                            st.markdown('<div class="section-header">Additional Comments</div>', unsafe_allow_html=True)
                            comment = st.text_area(
                                "", 
                                value=current_comment,
                                key=f"comment_{row['Row ID']}",
                                height=100
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Update comment if changed
                            if comment != current_comment:
                                # Save in session state
                                st.session_state.comments[row['Row ID']] = comment
                                
                                # Update dataframe
                                current_preference = st.session_state.preferences.get(row['Row ID'], row.get('Model Preference', ""))
                                update_preference(row['Row ID'], current_preference, comment)
                
            # Create a section for unsupported languages at the bottom
            unsupported_df = df[df['is_unsupported']]
            
            if not unsupported_df.empty:
                st.markdown("## Unsupported Languages")
                st.warning(f"There are {len(unsupported_df)} entries with unsupported languages (Japanese, Chinese, Thai, or mixed).")
                
                for idx, row in unsupported_df.iterrows():
                    # Create a container with border for each comparison
                    with st.container():
                        st.markdown(f"<hr>", unsafe_allow_html=True)
                        
                        # Display row ID at the top
                        st.markdown(f"**Row {row['Row ID']}**")
                        
                        # Display language info at top
                        st.markdown('<div class="stats-container">', unsafe_allow_html=True)
                        st.markdown('<div class="section-header">Languages:</div>', unsafe_allow_html=True)
                        st.write(f"Model 1: {row['detected_lang_model_1']}, Model 2: {row['detected_lang_model_2']}")
                        st.error("Language is not supported")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display selected input columns if any were chosen
                        if selected_input_cols:
                            with st.container():
                                st.markdown('<div class="inputs-container">', unsafe_allow_html=True)
                                st.markdown('<div class="section-header">Original Inputs:</div>', unsafe_allow_html=True)
                                for col in selected_input_cols:
                                    st.markdown(f"""<span class="column-name">{col}</span> {row[col]}""", unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display the content as plain text in full width
                        model_col1, model_col2 = st.columns(2)
                        
                        with model_col1:
                            st.markdown('<div class="section-header">Model 1:</div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                                    height: 350px; overflow-y: auto; background-color: #f5f5f5; color: #333333;">
                                <pre style="white-space: pre-wrap; word-break: break-word;">{row['model_1_output']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        with model_col2:
                            st.markdown('<div class="section-header">Model 2:</div>', unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="border: 1px solid #ddd; padding: 10px; border-radius: 5px; 
                                    height: 350px; overflow-y: auto; background-color: #f5f5f5; color: #333333;">
                                <pre style="white-space: pre-wrap; word-break: break-word;">{row['model_2_output']}</pre>
                            </div>
                            """, unsafe_allow_html=True)
                        
                st.markdown(f"<hr>", unsafe_allow_html=True)
            
            # Prepare columns for download in the desired order
            download_columns = [
                'Row ID', 
                'model_1_output', 
                'model_2_output', 
                'Similarity Score',
                'word_count_model_1', 
                'word_count_model_2', 
                'word_count_diff', 
                'detected_lang_model_1', 
                'detected_lang_model_2',
                'Model Preference',
                'Comment'
            ]
            
            # Add selected input columns if any
            for col in selected_input_cols:
                if col not in download_columns:
                    # Insert before Model Preference and Comment
                    download_columns.insert(-2, col)
            
            # Apply all preferences and comments from session state to dataframe before download
            for row_id, pref in st.session_state.preferences.items():
                matching_rows = df[df['Row ID'] == row_id]
                if not matching_rows.empty:
                    row_idx = matching_rows.index[0]
                    if not df.at[row_idx, 'is_unsupported']:
                        df.at[row_idx, 'Model Preference'] = pref
            
            for row_id, comment in st.session_state.comments.items():
                matching_rows = df[df['Row ID'] == row_id]
                if not matching_rows.empty:
                    row_idx = matching_rows.index[0]
                    df.at[row_idx, 'Comment'] = comment
            
            # Keep only the necessary columns for the download in the specified order
            download_df = df[download_columns].copy()
            
            # Create the download button with simpler styling
            csv = download_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            
            # Style the download button with a simple, professional look
            st.markdown(f"""
            <style>
            .simple-download-button {{
                display: inline-block;
                padding: 0.5em 1.5em;
                color: #333333;
                background-color: #e6e6e6;
                border-radius: 6px;
                text-decoration: none;
                font-weight: bold;
                margin: 1.5em 0;
                text-align: center;
                border: 1px solid #cccccc;
            }}
            </style>
            <div style="text-align: center; margin: 20px 0;">
                <a href="data:file/csv;base64,{b64}" download="model_comparison_with_preferences.csv" class="simple-download-button">
                    Download CSV File with Preferences
                </a>
            </div>
            """, unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        st.exception(e)
else:
    st.info("Please upload a CSV file to begin.")

# Add some instructions
st.markdown("""
### How to read the comparisons:
- **Red text** indicates words present in Model 1 output but not in Model 2
- **Green text** indicates words present in Model 2 output but not in Model 1
- Results are sorted by similarity score from most similar to least similar
- Click one of the buttons to record which model output you prefer for each example
- Word counts and word differences are displayed for supported languages
- Japanese, Chinese, Thai, and mixed Asian languages are currently unsupported for word count and HTML diff comparison
""")
