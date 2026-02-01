"""
Contract Query API with LLM-Generated Pandas Code
Uses open-source code generation LLM to write pandas code based on user queries
Executes code safely with restricted imports and returns results via Flask API

Features:
- Natural language queries about contract entities
- LLM generates pandas code
- Safe execution environment (restricted imports)
- Flask API endpoints for Streamlit frontend
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Contract entities dictionary for context
CONTRACT_ENTITIES = {
    'party_a_name': 'Name of the first party (Party A, Provider, Vendor)',
    'party_a_address': 'Complete address of Party A',
    'party_a_contact': 'Contact information of Party A (email, phone)',
    'party_a_signatory': 'Name and title of Party A signatory',
    'party_b_name': 'Name of the second party (Party B, Client, Customer)',
    'party_b_address': 'Complete address of Party B',
    'party_b_contact': 'Contact information of Party B (email, phone)',
    'party_b_signatory': 'Name and title of Party B signatory',
    'contract_type': 'Type of contract (MSA, SOW, NDA, Employment, etc.)',
    'contract_title': 'Official title of the contract',
    'effective_date': 'Effective date or start date of the contract',
    'execution_date': 'Date when contract was signed/executed',
    'expiration_date': 'End date or expiration date of the contract',
    'contract_term': 'Duration or term of the contract',
    'renewal_terms': 'Automatic renewal provisions or renewal terms',
    'total_contract_value': 'Total monetary value of the contract',
    'payment_amount': 'Payment amount(s) specified',
    'payment_schedule': 'Payment schedule or frequency (monthly, quarterly, etc.)',
    'payment_terms': 'Net payment terms (Net 30, Net 60, etc.)',
    'late_payment_fee': 'Late payment penalties or fees',
    'currency': 'Currency used in the contract',
    'billing_contact': 'Billing or invoicing contact information',
    'scope_of_work': 'Description of services or work to be performed',
    'deliverables': 'Specific deliverables mentioned',
    'performance_metrics': 'KPIs, SLAs, or performance standards',
    'milestones': 'Key milestones or deadlines',
    'exclusions': 'Services or items explicitly excluded',
    'governing_law': 'Governing law jurisdiction (state, country)',
    'dispute_resolution': 'Dispute resolution mechanism (arbitration, mediation, litigation)',
    'arbitration_location': 'Location for arbitration if specified',
    'liability_cap': 'Limitation of liability amount or cap',
    'indemnification': 'Indemnification provisions summary',
    'insurance_requirements': 'Insurance coverage requirements',
    'ip_ownership': 'Intellectual property ownership terms',
    'license_grant': 'License grants or permissions',
    'work_for_hire': 'Work for hire provisions',
    'confidentiality_term': 'Duration of confidentiality obligations',
    'data_protection': 'Data protection or privacy requirements',
    'permitted_disclosures': 'Exceptions to confidentiality',
    'termination_notice': 'Notice period required for termination',
    'termination_for_cause': 'Grounds for termination for cause',
    'termination_consequences': 'Post-termination obligations',
    'non_compete': 'Non-compete provisions or restrictions',
    'warranties': 'Warranties provided by either party',
    'force_majeure': 'Force majeure provisions',
    'amendment_process': 'How contract can be amended',
    'notice_address': 'Address for official notices',
    'entire_agreement': 'Entire agreement clause present (yes/no)',
    'severability': 'Severability clause present (yes/no)',
}

class SafePandasExecutor:
    """Execute pandas code in a restricted, safe environment"""
    
    ALLOWED_MODULES = {
        'pd': pd,
        'pandas': pd,
        'np': np,
        'numpy': np,
        'datetime': datetime,
    }
    
    ALLOWED_BUILTINS = {
        'len': len,
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'tuple': tuple,
        'set': set,
        'abs': abs,
        'max': max,
        'min': min,
        'sum': sum,
        'round': round,
        'sorted': sorted,
        'enumerate': enumerate,
        'zip': zip,
        'range': range,
        'print': print,
    }
    
    FORBIDDEN_KEYWORDS = [
        'import', 'exec', 'eval', 'compile', '__import__',
        'open', 'file', 'input', 'raw_input',
        'os.', 'sys.', 'subprocess', 'shutil',
        'pickle', 'socket', 'urllib', 'requests',
    ]
    
    @staticmethod
    def is_code_safe(code: str) -> tuple[bool, str]:
        """Check if code is safe to execute"""
        code_lower = code.lower()
        
        for keyword in SafePandasExecutor.FORBIDDEN_KEYWORDS:
            if keyword in code_lower:
                return False, f"Forbidden keyword detected: {keyword}"
        
        return True, "Code is safe"
    
    @staticmethod
    def execute_pandas_code(code: str, df: pd.DataFrame) -> dict:
        """Execute pandas code safely and return results"""
        # Check if code is safe
        is_safe, message = SafePandasExecutor.is_code_safe(code)
        if not is_safe:
            return {
                'success': False,
                'error': message,
                'result': None
            }
        
        try:
            # Create restricted namespace
            namespace = {
                'df': df.copy(),  # Work on a copy
                **SafePandasExecutor.ALLOWED_MODULES,
            }
            
            # Create restricted builtins
            restricted_builtins = {
                '__builtins__': SafePandasExecutor.ALLOWED_BUILTINS
            }
            
            # Execute code
            exec(code, restricted_builtins, namespace)
            
            # Get result (look for 'result' variable)
            if 'result' in namespace:
                result = namespace['result']
            else:
                # If no 'result' variable, return the modified df
                result = namespace['df']
            
            # Convert result to JSON-serializable format
            if isinstance(result, pd.DataFrame):
                result_data = {
                    'type': 'dataframe',
                    'data': result.to_dict('records'),
                    'columns': list(result.columns),
                    'shape': result.shape
                }
            elif isinstance(result, pd.Series):
                result_data = {
                    'type': 'series',
                    'data': result.to_dict(),
                    'name': result.name
                }
            elif isinstance(result, (int, float, str, bool)):
                result_data = {
                    'type': 'scalar',
                    'data': result
                }
            elif isinstance(result, (list, dict)):
                result_data = {
                    'type': 'collection',
                    'data': result
                }
            else:
                result_data = {
                    'type': 'string',
                    'data': str(result)
                }
            
            return {
                'success': True,
                'error': None,
                'result': result_data
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"{type(e).__name__}: {str(e)}",
                'result': None,
                'traceback': traceback.format_exc()
            }


class CodeGenerationLLM:
    """LLM for generating pandas code from natural language queries"""
    
    def __init__(self, use_gpu=False):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"Initializing Code Generation LLM on {self.device}...")
        
        # Using CodeGen - good for code generation, lightweight
        model_name = "Salesforce/codegen-350M-mono"
        
        print(f"  Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("  ✓ Model loaded successfully!")
    
    def generate_pandas_code(self, query: str, available_columns: list) -> str:
        """Generate pandas code from natural language query"""
        
        # Create prompt with context
        prompt = f"""# Task: Write pandas code to answer the following query about a contracts dataset
# The DataFrame is called 'df' and has these columns: {', '.join(available_columns)}
# Store the final result in a variable called 'result'
# Query: {query}

# Pandas code:
import pandas as pd
import numpy as np

# Solution:
"""
        
        # Generate code
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=300,
            num_beams=3,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the code part (after the prompt)
        code = generated_code[len(prompt):].strip()
        
        # Clean up the code
        code = self.clean_generated_code(code)
        
        return code
    
    def clean_generated_code(self, code: str) -> str:
        """Clean and format generated code"""
        # Remove common LLM artifacts
        lines = code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at start
            if not cleaned_lines and not line.strip():
                continue
            
            # Stop at common end markers
            if line.strip().startswith('#') and any(word in line.lower() for word in ['end', 'output', 'example', 'note']):
                break
            
            cleaned_lines.append(line)
        
        code = '\n'.join(cleaned_lines).strip()
        
        # Ensure 'result' variable exists
        if 'result' not in code.lower():
            # If the last line is an expression, assign it to result
            lines = code.split('\n')
            if lines and not lines[-1].strip().startswith('#'):
                last_line = lines[-1].strip()
                if '=' not in last_line:
                    lines[-1] = f"result = {last_line}"
                    code = '\n'.join(lines)
        
        return code


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
CSV_PATH = "/mnt/user-data/outputs/contract_entities.csv"
df_contracts = None
code_llm = None

def load_data():
    """Load the contracts CSV"""
    global df_contracts
    try:
        df_contracts = pd.read_csv(CSV_PATH)
        print(f"✓ Loaded contracts CSV: {df_contracts.shape}")
        return True
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        return False

def initialize_llm():
    """Initialize the code generation LLM"""
    global code_llm
    try:
        code_llm = CodeGenerationLLM(use_gpu=False)
        return True
    except Exception as e:
        print(f"✗ Error initializing LLM: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'csv_loaded': df_contracts is not None,
        'llm_loaded': code_llm is not None,
        'rows': len(df_contracts) if df_contracts is not None else 0,
        'columns': len(df_contracts.columns) if df_contracts is not None else 0
    })


@app.route('/entities', methods=['GET'])
def get_entities():
    """Get list of available entities"""
    return jsonify({
        'entities': CONTRACT_ENTITIES,
        'count': len(CONTRACT_ENTITIES)
    })


@app.route('/query', methods=['POST'])
def process_query():
    """
    Main endpoint: Process natural language query
    
    Request JSON:
    {
        "query": "Show me all MSA contracts with Party A as TechNova Solutions"
    }
    
    Response JSON:
    {
        "success": true,
        "query": "...",
        "generated_code": "...",
        "result": {...},
        "execution_time": 1.23
    }
    """
    start_time = datetime.now()
    
    try:
        # Get query from request
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'No query provided'
            }), 400
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        # Generate pandas code using LLM
        print("Generating pandas code...")
        available_columns = list(df_contracts.columns)
        generated_code = code_llm.generate_pandas_code(query, available_columns)
        
        print(f"\nGenerated Code:\n{generated_code}\n")
        
        # Execute the code safely
        print("Executing code...")
        execution_result = SafePandasExecutor.execute_pandas_code(generated_code, df_contracts)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if execution_result['success']:
            print(f"✓ Execution successful! ({execution_time:.2f}s)")
            return jsonify({
                'success': True,
                'query': query,
                'generated_code': generated_code,
                'result': execution_result['result'],
                'execution_time': execution_time
            })
        else:
            print(f"✗ Execution failed: {execution_result['error']}")
            return jsonify({
                'success': False,
                'query': query,
                'generated_code': generated_code,
                'error': execution_result['error'],
                'traceback': execution_result.get('traceback'),
                'execution_time': execution_time
            }), 400
    
    except Exception as e:
        print(f"✗ Error processing query: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/execute', methods=['POST'])
def execute_code():
    """
    Execute custom pandas code directly
    
    Request JSON:
    {
        "code": "result = df[df['contract_type'] == 'MSA']"
    }
    """
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code:
            return jsonify({
                'success': False,
                'error': 'No code provided'
            }), 400
        
        print(f"\nExecuting custom code:\n{code}\n")
        
        # Execute the code
        execution_result = SafePandasExecutor.execute_pandas_code(code, df_contracts)
        
        if execution_result['success']:
            return jsonify({
                'success': True,
                'code': code,
                'result': execution_result['result']
            })
        else:
            return jsonify({
                'success': False,
                'code': code,
                'error': execution_result['error'],
                'traceback': execution_result.get('traceback')
            }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/preview', methods=['GET'])
def preview_data():
    """Get a preview of the contracts data"""
    try:
        limit = int(request.args.get('limit', 10))
        
        preview_df = df_contracts.head(limit)
        
        return jsonify({
            'success': True,
            'data': preview_df.to_dict('records'),
            'columns': list(df_contracts.columns),
            'total_rows': len(df_contracts),
            'preview_rows': len(preview_df)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """Main function to start the Flask API"""
    print("=" * 80)
    print("CONTRACT QUERY API")
    print("LLM-Powered Pandas Code Generation")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading data...")
    if not load_data():
        print("Failed to load data. Exiting.")
        return
    
    # Initialize LLM
    print("\nInitializing LLM...")
    if not initialize_llm():
        print("Failed to initialize LLM. Exiting.")
        return
    
    print("\n" + "=" * 80)
    print("API SERVER READY")
    print("=" * 80)
    print("Available endpoints:")
    print("  GET  /health     - Health check")
    print("  GET  /entities   - Get available entities")
    print("  POST /query      - Process natural language query")
    print("  POST /execute    - Execute custom pandas code")
    print("  GET  /preview    - Preview contract data")
    print("=" * 80)
    print()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == "__main__":
    main()