"""
Contract Entity Extraction System with RAG (FAISS + LLM)

Extracts 40+ entities/attributes from contract text files using:
1. Text chunking with overlap
2. FAISS vector index for semantic search
3. LLM-based extraction from relevant chunks
4. CSV output with all attributes per contract

Process:
- For each text file, chunk the content
- For each entity, retrieve most relevant chunks using FAISS
- Use LLM to extract entity value from matched chunks
- Output to CSV: [filename, entity1, entity2, ..., entity40+]
"""

import os
import csv
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import pandas as pd
from datetime import datetime

# Contract entities/attributes to extract (40+ attributes)
CONTRACT_ENTITIES = {
    # Party Information
    'party_a_name': 'Name of the first party (Party A, Provider, Vendor)',
    'party_a_address': 'Complete address of Party A',
    'party_a_contact': 'Contact information of Party A (email, phone)',
    'party_a_signatory': 'Name and title of Party A signatory',
    'party_b_name': 'Name of the second party (Party B, Client, Customer)',
    'party_b_address': 'Complete address of Party B',
    'party_b_contact': 'Contact information of Party B (email, phone)',
    'party_b_signatory': 'Name and title of Party B signatory',
    
    # Contract Basics
    'contract_type': 'Type of contract (MSA, SOW, NDA, Employment, etc.)',
    'contract_title': 'Official title of the contract',
    'effective_date': 'Effective date or start date of the contract',
    'execution_date': 'Date when contract was signed/executed',
    'expiration_date': 'End date or expiration date of the contract',
    'contract_term': 'Duration or term of the contract',
    'renewal_terms': 'Automatic renewal provisions or renewal terms',
    
    # Financial Terms
    'total_contract_value': 'Total monetary value of the contract',
    'payment_amount': 'Payment amount(s) specified',
    'payment_schedule': 'Payment schedule or frequency (monthly, quarterly, etc.)',
    'payment_terms': 'Net payment terms (Net 30, Net 60, etc.)',
    'late_payment_fee': 'Late payment penalties or fees',
    'currency': 'Currency used in the contract',
    'billing_contact': 'Billing or invoicing contact information',
    
    # Scope and Services
    'scope_of_work': 'Description of services or work to be performed',
    'deliverables': 'Specific deliverables mentioned',
    'performance_metrics': 'KPIs, SLAs, or performance standards',
    'milestones': 'Key milestones or deadlines',
    'exclusions': 'Services or items explicitly excluded',
    
    # Legal Terms
    'governing_law': 'Governing law jurisdiction (state, country)',
    'dispute_resolution': 'Dispute resolution mechanism (arbitration, mediation, litigation)',
    'arbitration_location': 'Location for arbitration if specified',
    'liability_cap': 'Limitation of liability amount or cap',
    'indemnification': 'Indemnification provisions summary',
    'insurance_requirements': 'Insurance coverage requirements',
    
    # Intellectual Property
    'ip_ownership': 'Intellectual property ownership terms',
    'license_grant': 'License grants or permissions',
    'work_for_hire': 'Work for hire provisions',
    
    # Confidentiality and Data
    'confidentiality_term': 'Duration of confidentiality obligations',
    'data_protection': 'Data protection or privacy requirements',
    'permitted_disclosures': 'Exceptions to confidentiality',
    
    # Termination
    'termination_notice': 'Notice period required for termination',
    'termination_for_cause': 'Grounds for termination for cause',
    'termination_consequences': 'Post-termination obligations',
    
    # Other Important Terms
    'non_compete': 'Non-compete provisions or restrictions',
    'warranties': 'Warranties provided by either party',
    'force_majeure': 'Force majeure provisions',
    'amendment_process': 'How contract can be amended',
    'notice_address': 'Address for official notices',
    'entire_agreement': 'Entire agreement clause present (yes/no)',
    'severability': 'Severability clause present (yes/no)',
}

class ContractEntityExtractor:
    def __init__(self, text_files_folder, output_csv_path, use_gpu=False):
        self.text_files_folder = Path(text_files_folder)
        self.output_csv_path = Path(output_csv_path)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print("=" * 80)
        print("CONTRACT ENTITY EXTRACTION SYSTEM")
        print("RAG-based extraction with FAISS + LLM")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Entities to extract: {len(CONTRACT_ENTITIES)}")
        print()
        
        # Load models
        self.load_models()
        
        # Results storage
        self.results = []
        
    def load_models(self):
        """Load embedding model and LLM"""
        print("Loading models...")
        
        # Sentence transformer for embeddings (FAISS)
        print("  → Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Dimension of all-MiniLM-L6-v2
        print("    ✓ Embedding model loaded")
        
        # LLM for extraction (Flan-T5)
        print("  → Loading LLM (Flan-T5-base)...")
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        self.llm.to(self.device)
        print("    ✓ LLM loaded")
        print()
        
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def create_faiss_index(self, chunks: List[str]) -> faiss.IndexFlatL2:
        """Create FAISS index from text chunks"""
        # Generate embeddings for all chunks
        embeddings = self.embedding_model.encode(chunks, convert_to_numpy=True)
        
        # Create FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings.astype('float32'))
        
        return index, embeddings
    
    def retrieve_relevant_chunks(self, query: str, chunks: List[str], 
                                index: faiss.IndexFlatL2, top_k: int = 3) -> List[str]:
        """Retrieve top-k most relevant chunks for a query using FAISS"""
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS
        distances, indices = index.search(query_embedding.astype('float32'), top_k)
        
        # Return relevant chunks
        relevant_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
        return relevant_chunks
    
    def extract_entity_with_llm(self, entity_name: str, entity_description: str, 
                               relevant_chunks: List[str]) -> str:
        """Use LLM to extract entity value from relevant chunks"""
        # Combine relevant chunks
        context = "\n\n".join(relevant_chunks)
        
        # Create extraction prompt
        prompt = f"""Extract the following information from the contract text below:

Entity: {entity_name}
Description: {entity_description}

Contract Text:
{context}

If the information is found, provide only the extracted value. If not found, respond with "NOT FOUND".

Answer:"""
        
        # Generate with LLM
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, 
                                  truncation=True).to(self.device)
            outputs = self.llm.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                temperature=0.3,
                do_sample=False,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Clean up result
            if not result or result.lower() in ['not found', 'n/a', 'none', 'unknown']:
                return "NOT FOUND"
            
            return result
            
        except Exception as e:
            print(f"    ✗ LLM extraction error: {e}")
            return "ERROR"
    
    def process_single_file(self, text_file_path: Path) -> Dict[str, Any]:
        """Process a single text file and extract all entities"""
        filename = text_file_path.stem
        print(f"\nProcessing: {text_file_path.name}")
        print("-" * 80)
        
        # Read text file
        try:
            with open(text_file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            return None
        
        # Remove header if present
        if "=" * 70 in text:
            parts = text.split("=" * 70)
            text = parts[-1] if len(parts) > 1 else text
        
        print(f"  Text length: {len(text)} characters")
        
        # Chunk the text
        print("  → Chunking text...")
        chunks = self.chunk_text(text, chunk_size=500, overlap=100)
        print(f"    ✓ Created {len(chunks)} chunks")
        
        # Create FAISS index
        print("  → Creating FAISS index...")
        index, embeddings = self.create_faiss_index(chunks)
        print(f"    ✓ Index created with {len(chunks)} vectors")
        
        # Extract entities
        print(f"  → Extracting {len(CONTRACT_ENTITIES)} entities...")
        extracted_data = {'filename': filename}
        
        for idx, (entity_name, entity_description) in enumerate(CONTRACT_ENTITIES.items(), 1):
            # Create search query for this entity
            query = f"{entity_name}: {entity_description}"
            
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, chunks, index, top_k=3)
            
            # Extract entity value using LLM
            value = self.extract_entity_with_llm(entity_name, entity_description, relevant_chunks)
            extracted_data[entity_name] = value
            
            # Progress indicator
            if idx % 10 == 0:
                print(f"    Progress: {idx}/{len(CONTRACT_ENTITIES)} entities extracted")
        
        print(f"  ✓ Extraction complete for {filename}")
        
        return extracted_data
    
    def process_all_files(self):
        """Process all text files in the folder"""
        # Find all text files
        text_files = list(self.text_files_folder.glob("*.txt"))
        
        if not text_files:
            print(f"No text files found in {self.text_files_folder}")
            return
        
        print(f"Found {len(text_files)} text files to process")
        print()
        
        # Process each file
        for idx, text_file in enumerate(text_files, 1):
            print(f"\n{'=' * 80}")
            print(f"FILE {idx}/{len(text_files)}")
            print(f"{'=' * 80}")
            
            try:
                extracted_data = self.process_single_file(text_file)
                if extracted_data:
                    self.results.append(extracted_data)
            except Exception as e:
                print(f"\n✗ ERROR processing {text_file.name}: {e}")
                # Add error row
                error_data = {'filename': text_file.stem}
                for entity in CONTRACT_ENTITIES.keys():
                    error_data[entity] = "ERROR"
                self.results.append(error_data)
        
        # Save results to CSV
        self.save_to_csv()
    
    def save_to_csv(self):
        """Save extracted entities to CSV"""
        if not self.results:
            print("\nNo results to save.")
            return
        
        print(f"\n{'=' * 80}")
        print("SAVING RESULTS TO CSV")
        print(f"{'=' * 80}")
        
        # Create DataFrame
        df = pd.DataFrame(self.results)
        
        # Reorder columns: filename first, then all entities
        columns = ['filename'] + list(CONTRACT_ENTITIES.keys())
        df = df[columns]
        
        # Save to CSV
        df.to_csv(self.output_csv_path, index=False, encoding='utf-8')
        
        print(f"✓ Saved to: {self.output_csv_path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Show sample
        print(f"\nSample of extracted data:")
        print("-" * 80)
        print(df.head(3).to_string())
        
        # Statistics
        print(f"\n{'=' * 80}")
        print("EXTRACTION STATISTICS")
        print(f"{'=' * 80}")
        
        for entity in CONTRACT_ENTITIES.keys():
            if entity in df.columns:
                found_count = df[entity].apply(lambda x: x not in ['NOT FOUND', 'ERROR', '']).sum()
                found_pct = (found_count / len(df)) * 100
                print(f"  {entity}: {found_count}/{len(df)} ({found_pct:.1f}%)")
        
        print(f"{'=' * 80}")


def main():
    """
    Main function to run contract entity extraction
    
    Usage:
        1. Place contract text files in the input folder
        2. Run this script
        3. Find extracted entities in CSV output
    """
    
    # Configuration
    TEXT_FILES_FOLDER = "/mnt/user-data/outputs/extracted_texts"  # Folder with text files
    OUTPUT_CSV = "/mnt/user-data/outputs/contract_entities.csv"   # Output CSV path
    USE_GPU = False  # Set to True if you have GPU
    
    # Create extractor
    extractor = ContractEntityExtractor(
        text_files_folder=TEXT_FILES_FOLDER,
        output_csv_path=OUTPUT_CSV,
        use_gpu=USE_GPU
    )
    
    # Process all files
    extractor.process_all_files()
    
    print(f"\n{'=' * 80}")
    print("COMPLETE!")
    print(f"{'=' * 80}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()