import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional

# For PDF processing
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from PIL import Image

# For DOCX processing
import docx
from docx.table import Table as DocxTable

# For table detection and extraction
import pandas as pd
from dotenv import load_dotenv
from tabula import read_pdf
import camelot
import mammoth
import groq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
import os
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Client(api_key=GROQ_API_KEY)

class DocumentAgent:
    """Agent responsible for identifying tables in documents."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_extension = os.path.splitext(file_path)[1].lower()
        self.tables_info = []

    def process(self) -> List[Dict[str, Any]]:
        """Process the document and identify tables."""
        if self.file_extension == '.pdf':
            return self._process_pdf()
        elif self.file_extension in ['.docx', '.doc']:
            return self._process_docx()
        elif self.file_extension in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self._process_image()
        else:
            raise ValueError(f"Unsupported file format: {self.file_extension}")

    def _process_pdf(self) -> List[Dict[str, Any]]:
        """Identify tables in PDF documents using multiple methods."""
        logger.info(f"Identifying tables in PDF: {self.file_path}")

        # Method 1: Use tabula-py for initial detection
        tables_tabula = []
        try:
            dfs = read_pdf(self.file_path, pages='all', multiple_tables=True)
            for i, df in enumerate(dfs):
                if not df.empty:
                    tables_tabula.append({
                        'method': 'tabula',
                        'page': i + 1,  # Approximate page number
                        'data': df
                    })
        except Exception as e:
            logger.warning(f"Tabula extraction error: {e}")

        # Method 2: Use camelot for more accurate detection
        tables_camelot = []
        try:
            tables = camelot.read_pdf(self.file_path, pages='all', flavor='lattice')
            tables.extend(camelot.read_pdf(self.file_path, pages='all', flavor='stream'))

            for i, table in enumerate(tables):
                if table.df.empty:
                    continue
                tables_camelot.append({
                    'method': 'camelot',
                    'page': table.page,
                    'accuracy': table.accuracy,
                    'data': table.df
                })
        except Exception as e:
            logger.warning(f"Camelot extraction error: {e}")

        # Method 3: Visual detection using OpenCV
        tables_visual = []
        try:
            doc = fitz.open(self.file_path)
            for page_num, page in enumerate(doc):
                pix = page.get_pixmap()
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

                # Convert to grayscale if it's RGB
                if pix.n >= 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array

                # Apply image processing to detect tables
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)

                # Find contours
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Filter contours that might represent tables
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 10000:  # Minimum area threshold for a table
                        x, y, w, h = cv2.boundingRect(contour)

                        # Check if it has a table-like aspect ratio
                        if 0.5 < w / h < 5:
                            tables_visual.append({
                                'method': 'visual',
                                'page': page_num + 1,
                                'bbox': (x, y, x + w, y + h),
                                'raw_image': img_array[y:y + h, x:x + w]
                            })
            doc.close()
        except Exception as e:
            logger.warning(f"Visual detection error: {e}")

        # Combine and deduplicate results
        all_tables = []
        all_tables.extend(tables_tabula)
        all_tables.extend(tables_camelot)
        all_tables.extend(tables_visual)

        # Sort by page number
        all_tables.sort(key=lambda x: x.get('page', 0))

        return all_tables

    def _process_docx(self) -> List[Dict[str, Any]]:
        """Identify tables in DOCX documents."""
        logger.info(f"Identifying tables in DOCX: {self.file_path}")

        tables = []
        try:
            doc = docx.Document(self.file_path)
            for i, table in enumerate(doc.tables):
                tables.append({
                    'method': 'docx',
                    'index': i,
                    'rows': len(table.rows),
                    'cols': len(table.columns),
                    'table_obj': table
                })

            # Alternative approach using mammoth
            result = mammoth.convert_to_html(self.file_path)
            html = result.value

            # If tables were missed by the docx approach, we'll have them in HTML
            if len(tables) == 0 and "<table" in html:
                logger.info("Found tables using mammoth HTML conversion")
                tables.append({
                    'method': 'mammoth',
                    'html': html
                })

        except Exception as e:
            logger.warning(f"DOCX processing error: {e}")

        return tables

    def _process_image(self) -> List[Dict[str, Any]]:
        """Identify tables in image files."""
        logger.info(f"Identifying tables in image: {self.file_path}")

        tables = []
        try:
            # Read the image
            img = cv2.imread(self.file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply threshold to get binary image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

            # Find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours that might represent tables
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5000:  # Minimum area threshold for a table
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check if it has a table-like aspect ratio
                    if 0.5 < w / h < 5:
                        tables.append({
                            'method': 'image',
                            'bbox': (x, y, x + w, y + h),
                            'raw_image': img[y:y + h, x:x + w]
                        })
        except Exception as e:
            logger.warning(f"Image processing error: {e}")

        return tables


class ExtractionAgent:
    """Agent responsible for extracting data from identified tables."""

    def __init__(self, tables_info: List[Dict[str, Any]], file_path: str):
        self.tables_info = tables_info
        self.file_path = file_path
        self.extracted_tables = []

    def process(self) -> List[Dict[str, Any]]:
        """Extract data from each identified table."""
        logger.info(f"Extracting data from {len(self.tables_info)} tables")

        for i, table_info in enumerate(self.tables_info):
            method = table_info.get('method', '')

            try:
                if method == 'tabula' or method == 'camelot':
                    # Data is already available as DataFrame
                    df = table_info.get('data')
                    table_data = self._clean_dataframe(df)

                elif method == 'docx':
                    # Extract from DOCX table object
                    table_obj = table_info.get('table_obj')
                    table_data = self._extract_from_docx_table(table_obj)

                elif method == 'mammoth':
                    # Parse tables from HTML
                    html = table_info.get('html')
                    table_data = self._extract_from_html(html)

                elif method == 'visual' or method == 'image':
                    # Use OCR on the image
                    img = table_info.get('raw_image')
                    table_data = self._extract_with_ocr(img)

                else:
                    logger.warning(f"Unknown extraction method: {method}")
                    continue

                # Validate and add to results
                if table_data and len(table_data) > 0:
                    extracted_table = {
                        'table_id': i + 1,
                        'method': method,
                        'data': table_data
                    }

                    # Add page number if available
                    if 'page' in table_info:
                        extracted_table['page'] = table_info['page']

                    self.extracted_tables.append(extracted_table)

            except Exception as e:
                logger.error(f"Error extracting table {i + 1}: {e}")

        return self.extracted_tables

    def _clean_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Clean and convert DataFrame to structured data."""
        # Drop rows and columns that are all NaN
        df = df.dropna(how='all').dropna(axis=1, how='all')

        # Fill NaN values with empty strings
        df = df.fillna('')

        # Convert to list of dicts
        records = df.to_dict('records')

        # Clean column headers if they're numeric or unnamed
        if all(isinstance(col, (int, float)) or str(col).startswith('Unnamed:') for col in df.columns):
            # If the first row looks like headers, use it
            first_row = df.iloc[0].tolist()
            if any(str(x).strip() != '' for x in first_row):
                headers = [str(x).strip() if str(x).strip() else f"Column_{i}" for i, x in enumerate(first_row)]
                records = df.iloc[1:].to_dict('records')

                # Create new records with proper headers
                cleaned_records = []
                for record in records:
                    cleaned_record = {}
                    for i, header in enumerate(headers):
                        if i < len(df.columns):
                            col = df.columns[i]
                            cleaned_record[header] = record[col]
                    cleaned_records.append(cleaned_record)
                return cleaned_records

        return records

    def _extract_from_docx_table(self, table: DocxTable) -> List[Dict[str, Any]]:
        """Extract data from a DOCX table object."""
        data = []

        # Get headers from first row
        headers = []
        for cell in table.rows[0].cells:
            headers.append(cell.text.strip() or f"Column_{len(headers) + 1}")

        # Extract data from remaining rows
        for row in table.rows[1:]:
            row_data = {}
            for i, cell in enumerate(row.cells):
                if i < len(headers):
                    row_data[headers[i]] = cell.text.strip()
            data.append(row_data)

        return data

    def _extract_from_html(self, html: str) -> List[Dict[str, Any]]:
        """Extract table data from HTML content."""
        # Use pandas to parse HTML tables
        tables = pd.read_html(html)

        all_data = []
        for df in tables:
            all_data.extend(self._clean_dataframe(df))

        return all_data

    def _extract_with_ocr(self, img) -> List[Dict[str, Any]]:
        """Extract table data using OCR."""
        # Preprocess the image for better OCR results
        if isinstance(img, np.ndarray):
            # Convert to PIL Image
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            img_pil = img

        # Use Tesseract to extract text with table structure preserved
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        ocr_text = pytesseract.image_to_string(img_pil, config=custom_config)

        # Try to parse as CSV-like data
        lines = [line.strip() for line in ocr_text.splitlines() if line.strip()]

        # Attempt to detect columns based on consistent spacing
        if not lines:
            return []

        # Use the first line as header
        header_line = lines[0]

        # Try to split by multiple spaces or tabs
        import re
        pattern = r'\s{2,}|\t'
        headers = [h.strip() for h in re.split(pattern, header_line) if h.strip()]

        # If no headers were found, try another approach
        if len(headers) <= 1:
            # Try to split by other common delimiters
            for delimiter in ['|', ',', ';']:
                if delimiter in header_line:
                    headers = [h.strip() for h in header_line.split(delimiter) if h.strip()]
                    break

        # If still no valid headers, create default ones based on the line with most splits
        if len(headers) <= 1:
            max_splits = 0
            split_pattern = None

            for pattern in [r'\s{2,}|\t', r'\|', r',', r';']:
                for line in lines:
                    splits = len([s for s in re.split(pattern, line) if s.strip()])
                    if splits > max_splits:
                        max_splits = splits
                        split_pattern = pattern

            if max_splits > 1:
                headers = [f"Column_{i + 1}" for i in range(max_splits)]
                pattern = split_pattern
            else:
                # Last resort - just treat each line as a single value
                headers = ["Content"]
                return [{"Content": line} for line in lines]

        # Process data rows
        data = []
        for line in lines[1:]:
            values = [v.strip() for v in re.split(pattern, line) if v.strip()]

            # Skip rows that don't have enough values
            if len(values) <= 1:
                continue

            row_data = {}
            for i, value in enumerate(values):
                if i < len(headers):
                    row_data[headers[i]] = value
                else:
                    # Handle case where there are more values than headers
                    row_data[f"Extra_{i - len(headers) + 1}"] = value

            data.append(row_data)

        return data


class ValidationAgent:
    """Agent responsible for validating and correcting extracted table data."""

    def __init__(self, tables: List[Dict[str, Any]], groq_api_key: Optional[str] = None):
        self.tables = tables
        self.groq_api_key = groq_api_key
        if groq_api_key:
            groq.api_key = groq_api_key

    def process(self) -> List[Dict[str, Any]]:
        """Validate and correct the extracted tables."""
        logger.info(f"Validating and correcting {len(self.tables)} tables")
        validated_tables = []

        for table in self.tables:
            table_id = table.get('table_id', 0)
            data = table.get('data', [])

            if not data:
                logger.warning(f"Table {table_id} has no data to validate")
                continue

            # First, perform basic validation
            validation_issues = self._validate_table(data)

            # If issues are detected, try to correct them
            if validation_issues:
                logger.info(f"Table {table_id} has {len(validation_issues)} issues, attempting correction")
                corrected_data = self._correct_table(data, validation_issues)
                table['data'] = corrected_data
                table['validation'] = 'corrected'
            else:
                table['validation'] = 'valid'

            validated_tables.append(table)

        return validated_tables

    def _validate_table(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform basic validation on table data."""
        issues = []

        if not data:
            return [{'type': 'empty_table', 'description': 'Table has no data'}]

        # Check for consistent columns across all rows
        all_keys = set()
        for row in data:
            all_keys.update(row.keys())

        for i, row in enumerate(data):
            row_keys = set(row.keys())
            missing_keys = all_keys - row_keys

            if missing_keys:
                issues.append({
                    'type': 'missing_fields',
                    'row': i,
                    'fields': list(missing_keys),
                    'description': f"Row {i} is missing fields: {', '.join(missing_keys)}"
                })

        # Check for completely empty cells
        for i, row in enumerate(data):
            for key, value in row.items():
                if value == '' or value is None:
                    issues.append({
                        'type': 'empty_cell',
                        'row': i,
                        'field': key,
                        'description': f"Empty cell at row {i}, column '{key}'"
                    })

        # Check for inconsistent data types
        for key in all_keys:
            types = set()
            for row in data:
                if key in row and row[key]:
                    value = row[key]
                    # Try to determine if it's a number, date, or text
                    try:
                        float(value)
                        types.add('number')
                    except ValueError:
                        # Check for date patterns - simplified approach
                        if '/' in value or '-' in value:
                            if sum(c.isdigit() for c in value) >= 4:
                                types.add('date')
                            else:
                                types.add('text')
                        else:
                            types.add('text')

            if len(types) > 1:
                issues.append({
                    'type': 'inconsistent_types',
                    'field': key,
                    'types': list(types),
                    'description': f"Column '{key}' has inconsistent data types: {', '.join(types)}"
                })

        return issues

    def _correct_table(self, data: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attempt to correct table data issues."""
        corrected_data = data.copy()

        # Process issues by type
        for issue in issues:
            issue_type = issue.get('type')

            if issue_type == 'missing_fields':
                row_idx = issue.get('row')
                fields = issue.get('fields', [])

                for field in fields:
                    # Add missing field with empty value for now
                    corrected_data[row_idx][field] = ''

            elif issue_type == 'empty_cell':
                row_idx = issue.get('row')
                field = issue.get('field')

                # Try to infer a value
                inferred_value = self._infer_value(corrected_data, row_idx, field)

                if inferred_value:
                    corrected_data[row_idx][field] = inferred_value
                elif self.groq_api_key:
                    # If we can't infer, and LLM is available, use it
                    llm_value = self._correct_with_llm(corrected_data, row_idx, field)
                    if llm_value:
                        corrected_data[row_idx][field] = llm_value

        # If OpenAI API is available, use it for final validation
        if self.groq_api_key:
            corrected_data = self._llm_final_validation(corrected_data)

        return corrected_data

    def _infer_value(self, data: List[Dict[str, Any]], row_idx: int, field: str) -> str:
        """Try to infer a missing value based on patterns in the data."""
        # Get values for this field from other rows
        field_values = [row.get(field, '') for i, row in enumerate(data) if i != row_idx and field in row]

        if not field_values:
            return ''

        # If all values are the same, use that value
        if len(set(field_values)) == 1 and field_values[0]:
            return field_values[0]

        # Try to infer numeric values
        numeric_values = []
        for value in field_values:
            try:
                numeric_values.append(float(value))
            except (ValueError, TypeError):
                pass

        if numeric_values and len(numeric_values) == len(field_values):
            # If all values are numeric, use the average
            return str(sum(numeric_values) / len(numeric_values))

        # If most values are the same, use the most common
        from collections import Counter
        value_counts = Counter(field_values)
        most_common = value_counts.most_common(1)
        if most_common and most_common[0][1] > len(field_values) / 2:
            return most_common[0][0]

        return ''

    def _correct_with_llm(self, data: List[Dict[str, Any]], row_idx: int, field: str) -> str:
        """Use a language model to suggest a correction for an empty or problematic value."""
        if not self.groq_api_key:
            return ''

        try:
            # Prepare context for the LLM
            table_context = json.dumps(data[:5] if len(data) > 5 else data, indent=2)

            prompt = f"""
            I have a table with missing or problematic data. Below is part of the table:

            {table_context}

            Row {row_idx} is missing a value for the column '{field}'. Based on the other data in the table, 
            what is the most likely value for this field? Please provide only the value with no explanation.
            """


            chat_completion = client.chat.completions.create(
                messages=[{"role": "system",
                     "content": "You are a data correction assistant that fills in missing values in tables. Your answers should be concise and contain only the corrected value."},
                          {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=50,
                temperature = 0.1
            )

            # Extract and clean the suggested correction
            suggested_value = chat_completion.choices[0].message.content.strip()
            return suggested_value

        except Exception as e:
            logger.warning(f"Error using LLM for correction: {e}")
            return ''

    def _llm_final_validation(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use LLM for a final validation pass on the entire table."""
        if not self.groq_api_key or not data:
            return data

        try:
            # Sample the table if it's large
            sample_size = min(len(data), 5)
            sample_data = data[:sample_size]

            prompt = f"""
            I have extracted a table from a document. Below is a sample of the data:

            {json.dumps(sample_data, indent=2)}

            Please review this table for any inconsistencies, missing values, or obvious errors. 
            Return the corrected version of the ENTIRE input sample in valid JSON format with no extra text.
            """

            chat_completion = client.chat.completions.create(
                messages=[{"role": "system",
                           "content": "You are a data validation assistant that fixes errors in extracted table data. Your answers should contain only valid JSON with the corrected data."},
                          {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=1500,
                temperature=0.1
            )
            # Extract the corrected JSON
            corrected_json_str = chat_completion.choices[0].message.content.strip()

            # Remove any markdown code block syntax if present
            if corrected_json_str.startswith("```") and corrected_json_str.endswith("```"):
                corrected_json_str = corrected_json_str[3:-3].strip()
            if corrected_json_str.startswith("```json") and corrected_json_str.endswith("```"):
                corrected_json_str = corrected_json_str[7:-3].strip()

            try:
                corrected_sample = json.loads(corrected_json_str)

                # Apply corrections from the sample to the full dataset
                if len(corrected_sample) == sample_size:
                    for i in range(sample_size):
                        data[i] = corrected_sample[i]

            except json.JSONDecodeError:
                logger.warning("Could not parse LLM correction as JSON")

        except Exception as e:
            logger.warning(f"Error using LLM for final validation: {e}")

        return data


class TableExtractionPipeline:
    """Main pipeline coordinating the table extraction process."""

    def __init__(self, file_path: str, output_path: str = None, groq_api_key: str = None):
        self.file_path = file_path
        self.output_path = output_path or f"{os.path.splitext(file_path)[0]}_tables.json"
        self.groq_api_key = groq_api_key

    def run(self) -> Dict[str, Any]:
        """Execute the full table extraction pipeline."""
        logger.info(f"Starting table extraction for {self.file_path}")

        # Step 1: Identify tables in the document
        document_agent = DocumentAgent(self.file_path)
        tables_info = document_agent.process()
        logger.info(f"Found {len(tables_info)} potential tables")

        if not tables_info:
            logger.warning("No tables found in the document")
            return {"tables": [], "message": "No tables found"}

        # Step 2: Extract data from identified tables
        extraction_agent = ExtractionAgent(tables_info, self.file_path)
        extracted_tables = extraction_agent.process()
        logger.info(f"Extracted data from {len(extracted_tables)} tables")

        # Step 3: Validate and correct the extracted data
        validation_agent = ValidationAgent(extracted_tables, self.groq_api_key)
        validated_tables = validation_agent.process()

        result = {
            "file": os.path.basename(self.file_path),
            "tables": validated_tables,
            "table_count": len(validated_tables),
            "extraction_timestamp": pd.Timestamp.now().isoformat()
        }

        # Save to JSON file
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved extraction results to {self.output_path}")

        return result


def main():


    file_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\input\ast_sci_data_tables_sample.pdf"
    output_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\output"
    if not os.path.isfile(file_path):
        print(f"Error: Input file path not found.")
        return
    output_path = os.path.join(output_path, "extracted_tables.json")
    # Run the pipeline
    pipeline = TableExtractionPipeline(
        file_path=file_path,
        output_path=output_path,
        groq_api_key=GROQ_API_KEY
    )
    result = pipeline.run()

    print(f"\nExtraction complete!")
    print(f"Found {result['table_count']} tables in {os.path.basename(file_path)}")
    print(f"Results saved to {pipeline.output_path}")


if __name__ == "__main__":
    main()