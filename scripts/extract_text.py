"""
PDF to Text Extractor with Multi-Modal LLM, OCR, and Direct Extraction
Three-tier fallback approach:
1. Multimodal Vision Model (LLaVA or similar) - converts page images to text
2. OCR (Tesseract) - if vision model fails
3. Direct PDF text extraction (PyPDF2) - if OCR fails

Processes all PDFs in a folder and creates corresponding .txt files
"""

import os
import base64
from io import BytesIO
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PDF and Image processing
from pdf2image import convert_from_path
from PIL import Image
import PyPDF2

# OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR fallback disabled.")

# Multimodal LLM
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer, AutoModelForCausalLM
import requests

class PDFTextExtractor:
    def __init__(self, input_folder, output_folder, use_gpu=False):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        print("=" * 70)
        print("PDF TEXT EXTRACTOR - Multi-Modal LLM + OCR + Direct Extraction")
        print("=" * 70)
        print(f"Device: {self.device}")
        print()
        
        # Load multimodal vision model
        self.load_vision_model()
        
    def load_vision_model(self):
        """Load a lightweight multimodal vision model"""
        try:
            print("Loading multimodal vision model (Salesforce BLIP-2)...")
            # Using BLIP-2 - lightweight, reliable, good for document understanding
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            
            self.vision_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.vision_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.vision_model.to(self.device)
            self.vision_model_available = True
            print("✓ Vision model loaded successfully!")
            print()
        except Exception as e:
            print(f"✗ Could not load vision model: {e}")
            print("Will skip vision model and use OCR/direct extraction only.")
            self.vision_model_available = False
            print()
    
    def pdf_to_images_base64(self, pdf_path):
        """Convert PDF pages to base64 encoded images"""
        try:
            # Convert PDF to images (one per page)
            images = convert_from_path(pdf_path, dpi=200)
            
            base64_images = []
            for img in images:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_images.append((img, img_base64))
            
            return base64_images
        except Exception as e:
            print(f"  ✗ Error converting PDF to images: {e}")
            return []
    
    def extract_text_with_vision_model(self, image, page_num):
        """Extract text from image using multimodal vision model"""
        if not self.vision_model_available:
            return None
        
        try:
            # Prepare prompt for document text extraction
            prompt = "Extract all text from this document page. Transcribe everything you see, maintaining the original layout and formatting as much as possible."
            
            # Process image
            inputs = self.vision_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            
            # Generate text
            generated_ids = self.vision_model.generate(
                **inputs,
                max_length=1000,
                num_beams=5,
                temperature=0.7,
                do_sample=False
            )
            
            # Decode
            generated_text = self.vision_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Clean up the output (remove the prompt if it's repeated)
            if generated_text.lower().startswith(prompt.lower()):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text if generated_text else None
            
        except Exception as e:
            print(f"  ✗ Vision model failed for page {page_num}: {e}")
            return None
    
    def extract_text_with_ocr(self, image, page_num):
        """Extract text using OCR (Tesseract)"""
        if not TESSERACT_AVAILABLE:
            return None
        
        try:
            text = pytesseract.image_to_string(image, config='--psm 1')
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"  ✗ OCR failed for page {page_num}: {e}")
            return None
    
    def extract_text_direct(self, pdf_path):
        """Extract text directly from PDF (fallback method)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_pages = []
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_pages.append(text.strip() if text else "")
                
                return text_pages
        except Exception as e:
            print(f"  ✗ Direct extraction failed: {e}")
            return None
    
    def process_single_pdf(self, pdf_path):
        """Process a single PDF with 3-tier fallback approach"""
        pdf_name = pdf_path.stem
        print(f"\nProcessing: {pdf_path.name}")
        print("-" * 70)
        
        # Convert PDF to images
        print("  Converting PDF to images...")
        images_base64 = self.pdf_to_images_base64(pdf_path)
        
        if not images_base64:
            print("  Using direct extraction fallback...")
            text_pages = self.extract_text_direct(pdf_path)
            if text_pages:
                full_text = "\n\n--- PAGE BREAK ---\n\n".join(text_pages)
                method = "Direct Extraction"
            else:
                full_text = "[ERROR: Could not extract text from PDF]"
                method = "FAILED"
        else:
            print(f"  ✓ Converted to {len(images_base64)} pages")
            
            # Process each page
            text_pages = []
            methods_used = []
            
            for page_num, (image, img_base64) in enumerate(images_base64, 1):
                print(f"\n  Page {page_num}/{len(images_base64)}:")
                
                # Tier 1: Try vision model
                text = None
                method = None
                
                if self.vision_model_available:
                    print(f"    → Trying vision model...")
                    text = self.extract_text_with_vision_model(image, page_num)
                    if text and len(text) > 50:  # Minimum text threshold
                        method = "Vision Model"
                        print(f"    ✓ Success with vision model ({len(text)} chars)")
                
                # Tier 2: Try OCR
                if not text and TESSERACT_AVAILABLE:
                    print(f"    → Trying OCR...")
                    text = self.extract_text_with_ocr(image, page_num)
                    if text and len(text) > 20:
                        method = "OCR"
                        print(f"    ✓ Success with OCR ({len(text)} chars)")
                
                # Tier 3: Direct extraction for this page
                if not text:
                    print(f"    → Trying direct extraction...")
                    direct_pages = self.extract_text_direct(pdf_path)
                    if direct_pages and page_num <= len(direct_pages):
                        text = direct_pages[page_num - 1]
                        if text:
                            method = "Direct Extraction"
                            print(f"    ✓ Success with direct extraction ({len(text)} chars)")
                
                # Store result
                if text:
                    text_pages.append(text)
                    methods_used.append(method)
                else:
                    text_pages.append(f"[ERROR: Could not extract text from page {page_num}]")
                    methods_used.append("FAILED")
                    print(f"    ✗ All methods failed for page {page_num}")
            
            # Combine all pages
            full_text = "\n\n--- PAGE BREAK ---\n\n".join(text_pages)
            method = f"Mixed ({', '.join(set(methods_used))})"
        
        # Save to text file
        output_path = self.output_folder / f"{pdf_name}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"PDF: {pdf_path.name}\n")
            f.write(f"Extraction Method: {method}\n")
            f.write(f"Total Pages: {len(images_base64) if images_base64 else 'Unknown'}\n")
            f.write("=" * 70 + "\n\n")
            f.write(full_text)
        
        print(f"\n  ✓ Saved to: {output_path.name}")
        print(f"  Method: {method}")
        
        return output_path
    
    def process_all_pdfs(self):
        """Process all PDFs in the input folder"""
        # Find all PDF files
        pdf_files = list(self.input_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print()
        
        # Process each PDF
        successful = 0
        failed = 0
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n{'=' * 70}")
            print(f"PDF {idx}/{len(pdf_files)}")
            print(f"{'=' * 70}")
            
            try:
                self.process_single_pdf(pdf_path)
                successful += 1
            except Exception as e:
                print(f"\n✗ ERROR processing {pdf_path.name}: {e}")
                failed += 1
        
        # Summary
        print(f"\n\n{'=' * 70}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output folder: {self.output_folder}")
        print(f"{'=' * 70}")


def main():
    """
    Main function to run the PDF text extractor
    
    Usage:
        1. Place PDF files in the input folder
        2. Run this script
        3. Find extracted text files in the output folder
    """
    
    # Configuration
    INPUT_FOLDER = "/mnt/user-data/outputs"  # Folder containing PDFs (or change to your folder)
    OUTPUT_FOLDER = "/mnt/user-data/outputs/extracted_texts"  # Where to save text files
    USE_GPU = False  # Set to True if you have a GPU
    
    # Create extractor
    extractor = PDFTextExtractor(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        use_gpu=USE_GPU
    )
    
    # Process all PDFs
    extractor.process_all_pdfs()


if __name__ == "__main__":
    main()