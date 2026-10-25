import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from src.model import predict

st.set_page_config(page_title='SpendSense AI', page_icon='💸', layout='wide')
st.title('💸 SpendSense AI')
st.caption('Classify expense descriptions locally using TF-IDF + Logistic Regression. Educational demo, not financial advice.')
text = st.text_input('Transaction description', 'Monthly Netflix subscription')
if st.button('Categorize'):
    try:
        result = predict(text)[0]
        st.metric('Category', result['category'])
        st.metric('Model confidence', f"{result['confidence']:.1%}")
        if result['needs_review']:
            st.warning('Low confidence: manually review this classification.')
    except ValueError as exc:
        st.error(str(exc))
st.divider()
st.subheader('Batch expense analysis')
file = st.file_uploader('Upload a CSV with description and amount columns', type='csv')
if file:
    try:
        df = pd.read_csv(file)
        if not {'description','amount'}.issubset(df.columns):
            st.error('CSV requires description and amount columns.')
        else:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df.dropna(subset=['amount','description']).copy()
            if df.empty:
                st.warning('No valid records found.')
            else:
                result = predict(df['description'].astype(str).tolist())
                df['category'] = [r['category'] for r in result]
                df['confidence'] = [r['confidence'] for r in result]
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.groupby('category')['amount'].sum())
                st.download_button('Download categorized CSV', df.to_csv(index=False), 'categorized_expenses.csv','text/csv')
    except (ValueError, pd.errors.ParserError) as exc:
        st.error(f'Invalid CSV: {exc}')
