"""
Contract Query Streamlit App
User-friendly interface for querying contract entities using natural language

Features:
- Natural language query input
- Real-time code generation display
- Interactive results visualization
- Entity browser
- Query history
"""

import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# API Configuration
API_BASE_URL = "http://localhost:5000"

# Page configuration
st.set_page_config(
    page_title="Contract Query Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .code-block {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        font-family: 'Courier New', monospace;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200, response.json()
    except:
        return False, None

def get_entities():
    """Get available entities from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/entities", timeout=5)
        if response.status_code == 200:
            return response.json()['entities']
        return None
    except:
        return None

def send_query(query_text):
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query_text},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def execute_custom_code(code):
    """Execute custom pandas code via API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/execute",
            json={"code": code},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_preview_data(limit=10):
    """Get preview of contract data"""
    try:
        response = requests.get(f"{API_BASE_URL}/preview?limit={limit}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def display_result(result_data):
    """Display query results based on type"""
    if not result_data:
        return
    
    result_type = result_data.get('type')
    data = result_data.get('data')
    
    if result_type == 'dataframe':
        st.subheader("📊 Results Table")
        df = pd.DataFrame(data)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Data Type", "DataFrame")
        
        # Display dataframe
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Visualization options
        if len(df) > 0:
            st.subheader("📈 Visualizations")
            
            viz_type = st.selectbox(
                "Select visualization type",
                ["None", "Bar Chart", "Pie Chart", "Line Chart", "Scatter Plot"]
            )
            
            if viz_type != "None":
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                
                if viz_type == "Bar Chart" and categorical_cols and numeric_cols:
                    x_col = st.selectbox("X-axis (Category)", categorical_cols)
                    y_col = st.selectbox("Y-axis (Value)", numeric_cols)
                    fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Pie Chart" and categorical_cols:
                    col = st.selectbox("Select column", categorical_cols)
                    value_counts = df[col].value_counts()
                    fig = px.pie(values=value_counts.values, names=value_counts.index, 
                               title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Line Chart" and numeric_cols:
                    y_cols = st.multiselect("Select columns", numeric_cols)
                    if y_cols:
                        fig = px.line(df, y=y_cols, title="Line Chart")
                        st.plotly_chart(fig, use_container_width=True)
    
    elif result_type == 'series':
        st.subheader("📊 Results Series")
        series_df = pd.DataFrame(list(data.items()), columns=['Index', 'Value'])
        st.dataframe(series_df, use_container_width=True)
    
    elif result_type == 'scalar':
        st.subheader("🔢 Result")
        st.markdown(f'<div class="success-box"><h2>{data}</h2></div>', unsafe_allow_html=True)
    
    elif result_type == 'collection':
        st.subheader("📋 Results")
        st.json(data)
    
    else:
        st.subheader("📝 Result")
        st.code(str(data))

def main():
    # Header
    st.markdown('<div class="main-header">📄 Contract Query Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your contracts in natural language</div>', 
                unsafe_allow_html=True)
    
    # Check API health
    is_healthy, health_info = check_api_health()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if is_healthy:
            st.markdown('<div class="success-box">✅ API Connected</div>', unsafe_allow_html=True)
            if health_info:
                st.metric("Contracts Loaded", health_info.get('rows', 0))
                st.metric("Attributes", health_info.get('columns', 0))
        else:
            st.markdown('<div class="error-box">❌ API Not Connected<br>Please start the Flask API server</div>', 
                       unsafe_allow_html=True)
            st.code("python contract_query_api.py")
            return
        
        st.divider()
        
        # Entity Browser
        st.header("📚 Available Entities")
        entities = get_entities()
        
        if entities:
            search_entity = st.text_input("🔍 Search entities", "")
            
            filtered_entities = {k: v for k, v in entities.items() 
                               if search_entity.lower() in k.lower() or 
                                  search_entity.lower() in v.lower()}
            
            with st.expander(f"View Entities ({len(filtered_entities)})", expanded=False):
                for entity, description in filtered_entities.items():
                    st.markdown(f"**{entity}**")
                    st.caption(description)
                    st.divider()
        
        st.divider()
        
        # Query History
        st.header("📜 Query History")
        if st.session_state.query_history:
            for i, hist_query in enumerate(reversed(st.session_state.query_history[-5:])):
                if st.button(f"🔄 {hist_query[:50]}...", key=f"hist_{i}"):
                    st.session_state.rerun_query = hist_query
        else:
            st.caption("No query history yet")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["💬 Query", "⚙️ Custom Code", "👁️ Data Preview"])
    
    # Tab 1: Natural Language Query
    with tab1:
        st.header("Ask Your Question")
        
        # Example queries
        with st.expander("💡 Example Queries"):
            st.markdown("""
            - Show me all MSA contracts
            - How many contracts expire in 2024?
            - Which contracts have TechNova Solutions as Party A?
            - What's the total contract value by contract type?
            - Show contracts with payment terms Net 30
            - List all contracts with arbitration in California
            - Which contracts have non-compete clauses?
            - Show me contracts sorted by total value
            """)
        
        # Query input
        query_text = st.text_area(
            "Enter your question about contracts:",
            height=100,
            placeholder="e.g., Show me all MSA contracts with total value greater than $100,000"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_query = st.button("🚀 Run Query", type="primary", use_container_width=True)
        
        if submit_query and query_text:
            with st.spinner("🤖 Generating and executing code..."):
                # Send query to API
                result = send_query(query_text)
                
                # Add to history
                if query_text not in st.session_state.query_history:
                    st.session_state.query_history.append(query_text)
                
                # Display results
                if result.get('success'):
                    st.success(f"✅ Query executed successfully in {result.get('execution_time', 0):.2f}s")
                    
                    # Show generated code
                    st.subheader("🔧 Generated Pandas Code")
                    st.code(result.get('generated_code', ''), language='python')
                    
                    # Show results
                    st.divider()
                    display_result(result.get('result'))
                    
                    # Store result
                    st.session_state.current_result = result
                
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
                    if result.get('generated_code'):
                        st.subheader("Generated Code (Failed)")
                        st.code(result.get('generated_code', ''), language='python')
                    
                    if result.get('traceback'):
                        with st.expander("Show Traceback"):
                            st.code(result.get('traceback'))
    
    # Tab 2: Custom Code Execution
    with tab2:
        st.header("Execute Custom Pandas Code")
        st.info("💡 Write pandas code directly. Use 'df' to reference the contracts dataframe. Store result in 'result' variable.")
        
        # Example code
        with st.expander("💡 Example Code"):
            st.code("""# Filter MSA contracts
result = df[df['contract_type'] == 'MSA']

# Count by contract type
result = df['contract_type'].value_counts()

# Group by party and sum values
result = df.groupby('party_a_name')['total_contract_value'].sum().sort_values(ascending=False)
""", language='python')
        
        custom_code = st.text_area(
            "Enter pandas code:",
            height=200,
            value="result = df.head(10)"
        )
        
        if st.button("▶️ Execute Code", type="primary"):
            with st.spinner("Executing code..."):
                result = execute_custom_code(custom_code)
                
                if result.get('success'):
                    st.success("✅ Code executed successfully")
                    display_result(result.get('result'))
                else:
                    st.error(f"❌ Error: {result.get('error', 'Unknown error')}")
                    if result.get('traceback'):
                        with st.expander("Show Traceback"):
                            st.code(result.get('traceback'))
    
    # Tab 3: Data Preview
    with tab3:
        st.header("Contract Data Preview")
        
        limit = st.slider("Number of rows to preview", 5, 50, 10)
        
        if st.button("🔄 Refresh Preview"):
            preview_data = get_preview_data(limit)
            
            if preview_data and preview_data.get('success'):
                st.success(f"Showing {preview_data['preview_rows']} of {preview_data['total_rows']} total contracts")
                
                df_preview = pd.DataFrame(preview_data['data'])
                st.dataframe(df_preview, use_container_width=True, height=500)
                
                # Column info
                with st.expander("📋 Column Information"):
                    for col in preview_data['columns']:
                        st.write(f"- **{col}**")
            else:
                st.error("Failed to load preview data")

if __name__ == "__main__":
    main()