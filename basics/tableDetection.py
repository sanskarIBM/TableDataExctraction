import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Union, Optional
import pdfplumber
import camelot
import tabula
from PyPDF2 import PdfReader


def extract_tables_from_pdf(pdf_path: str, method: str = "auto") -> List[pd.DataFrame]:
    """
    Extract tables from a PDF file using the specified method or automatically choose the best method.

    Args:
        pdf_path: Path to the PDF file
        method: Extraction method ('camelot', 'tabula', 'pdfplumber', or 'auto')

    Returns:
        List of pandas DataFrames containing the tables
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if method == "auto":
        # Check file size to make a decision
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Size in MB

        # Get number of pages
        with open(pdf_path, 'rb') as f:
            pdf = PdfReader(f)
            num_pages = len(pdf.pages)

        # Choose method based on file characteristics
        if file_size < 5 and num_pages < 10:
            # For smaller files, try camelot first as it's more accurate
            try:
                tables = extract_with_camelot(pdf_path)
                if tables and len(tables) > 0:
                    return tables
            except Exception as e:
                print(f"Camelot failed: {e}. Trying other methods...")

        # Try multiple methods and use the one that gives the best results
        methods = ["tabula", "pdfplumber", "camelot"]
        best_tables = []
        best_score = 0

        for m in methods:
            try:
                if m == "camelot":
                    tables = extract_with_camelot(pdf_path)
                elif m == "tabula":
                    tables = extract_with_tabula(pdf_path)
                elif m == "pdfplumber":
                    tables = extract_with_pdfplumber(pdf_path)

                # Evaluate table quality (non-empty cells and consistent columns)
                score = evaluate_tables(tables)
                if score > best_score:
                    best_tables = tables
                    best_score = score
            except Exception as e:
                print(f"Method {m} failed: {e}")

        return best_tables
    elif method == "camelot":
        return extract_with_camelot(pdf_path)
    elif method == "tabula":
        return extract_with_tabula(pdf_path)
    elif method == "pdfplumber":
        return extract_with_pdfplumber(pdf_path)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'camelot', 'tabula', 'pdfplumber', or 'auto'")


def evaluate_tables(tables: List[pd.DataFrame]) -> float:
    """
    Evaluate the quality of extracted tables.

    Args:
        tables: List of DataFrames to evaluate

    Returns:
        A score indicating table quality (higher is better)
    """
    if not tables or len(tables) == 0:
        return 0

    total_score = 0
    for table in tables:
        if table.empty:
            continue

        # Calculate percentage of non-empty cells
        non_empty_cells = table.count().sum() / (table.shape[0] * table.shape[1])

        # Check for consistent column structure
        col_consistency = 1.0 - (table.columns.duplicated().sum() / len(table.columns))

        # Basic shape score (reward larger tables)
        size_score = min(1.0, (table.shape[0] * table.shape[1]) / 1000)

        # Combined score
        score = (non_empty_cells * 0.5) + (col_consistency * 0.3) + (size_score * 0.2)
        total_score += score

    return total_score / len(tables)


def extract_with_camelot(pdf_path: str) -> List[pd.DataFrame]:
    """Extract tables using camelot."""
    tables = []
    # Try lattice mode first (for tables with lines/borders)
    lattice_tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice", line_scale=40)

    if len(lattice_tables) > 0:
        for table in lattice_tables:
            if table.df.shape[0] > 1 and table.df.shape[1] > 1:
                tables.append(table.df)

    # If no tables found with lattice, try stream mode (for tables without clear borders)
    if len(tables) == 0:
        stream_tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        for table in stream_tables:
            if table.df.shape[0] > 1 and table.df.shape[1] > 1:
                tables.append(table.df)

    # Handle tables that span multiple pages
    return merge_similar_tables(tables)


def extract_with_tabula(pdf_path: str) -> List[pd.DataFrame]:
    """Extract tables using tabula-py."""
    raw_tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True)
    clean_tables = [table for table in raw_tables if not table.empty and table.shape[0] > 1 and table.shape[1] > 1]
    return merge_similar_tables(clean_tables)


def extract_with_pdfplumber(pdf_path: str) -> List[pd.DataFrame]:
    """Extract tables using pdfplumber."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table_data in page_tables:
                if table_data and len(table_data) > 1 and len(table_data[0]) > 1:
                    # Convert to DataFrame
                    df = pd.DataFrame(table_data[1:], columns=table_data[0])
                    tables.append(df)

    return merge_similar_tables(tables)


def merge_similar_tables(tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """
    Merge tables that appear to be continuations of the same table across pages.

    Args:
        tables: List of extracted table DataFrames

    Returns:
        List of merged DataFrames
    """
    if not tables or len(tables) <= 1:
        return tables

    merged_tables = []
    current_table = tables[0]

    for i in range(1, len(tables)):
        next_table = tables[i]

        # Check if tables should be merged
        if are_tables_related(current_table, next_table):
            # Handle header rows that might be repeated
            if has_repeated_header(current_table, next_table):
                # Remove the header row from the next table
                next_table = next_table.iloc[1:]

            # Concatenate the tables
            current_table = pd.concat([current_table, next_table], ignore_index=True)
        else:
            # Store the current table and start a new one
            merged_tables.append(current_table)
            current_table = next_table

    # Add the last table
    merged_tables.append(current_table)

    return merged_tables


def are_tables_related(table1: pd.DataFrame, table2: pd.DataFrame) -> bool:
    """
    Determine if two tables are likely to be related or parts of the same table.

    Args:
        table1: First DataFrame
        table2: Second DataFrame

    Returns:
        True if tables appear to be related, False otherwise
    """
    # Check if the tables have the same number of columns
    if table1.shape[1] != table2.shape[1]:
        return False

    # Check if column names or first row values are similar
    col_similarity = column_similarity(table1, table2)

    # Check if the content structure is similar
    content_similarity = content_type_similarity(table1, table2)

    # If both similarities are high, tables are likely related
    return col_similarity > 0.7 and content_similarity > 0.6


def column_similarity(table1: pd.DataFrame, table2: pd.DataFrame) -> float:
    """
    Calculate similarity between columns of two tables.

    Args:
        table1: First DataFrame
        table2: Second DataFrame

    Returns:
        Similarity score between 0 and 1
    """
    # If different number of columns, they're not similar
    if table1.shape[1] != table2.shape[1]:
        return 0.0

    # Convert column names to strings for comparison
    cols1 = [str(col) for col in table1.columns]
    cols2 = [str(col) for col in table2.columns]

    # Count matching columns
    matches = sum(1 for c1, c2 in zip(cols1, cols2) if c1 == c2 or similar_strings(c1, c2))

    return matches / len(cols1)


def content_type_similarity(table1: pd.DataFrame, table2: pd.DataFrame) -> float:
    """
    Check if the content types in the tables are similar.

    Args:
        table1: First DataFrame
        table2: Second DataFrame

    Returns:
        Similarity score between 0 and 1
    """
    # If different number of columns, they're not similar
    if table1.shape[1] != table2.shape[1]:
        return 0.0

    similarity_count = 0

    for col_idx in range(table1.shape[1]):
        # Get sample values from both tables for this column
        sample1 = table1.iloc[-3:, col_idx].dropna() if table1.shape[0] > 3 else table1.iloc[:, col_idx].dropna()
        sample2 = table2.iloc[:3, col_idx].dropna() if table2.shape[0] > 3 else table2.iloc[:, col_idx].dropna()

        if len(sample1) == 0 or len(sample2) == 0:
            continue

        # Check if the type of content is similar
        type1 = determine_content_type(sample1)
        type2 = determine_content_type(sample2)

        if type1 == type2:
            similarity_count += 1

    # Return similarity ratio
    return similarity_count / table1.shape[1] if table1.shape[1] > 0 else 0


def determine_content_type(series: pd.Series) -> str:
    """
    Determine the type of content in a series.

    Args:
        series: Series of values

    Returns:
        Content type label
    """
    # Convert to strings for analysis
    str_values = series.astype(str)

    # Check if values are numeric
    numeric_pattern = re.compile(r'^[-+]?\d*\.?\d+$')
    numeric_ratio = sum(1 for val in str_values if numeric_pattern.match(val)) / len(str_values)

    # Check if values are dates
    date_pattern = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}')
    date_ratio = sum(1 for val in str_values if date_pattern.search(val)) / len(str_values)

    # Check if values are years
    year_pattern = re.compile(r'^(19|20)\d{2}$')
    year_ratio = sum(1 for val in str_values if year_pattern.match(val)) / len(str_values)

    # Determine type based on the highest ratio
    if numeric_ratio > 0.6:
        return "numeric"
    elif date_ratio > 0.6:
        return "date"
    elif year_ratio > 0.6:
        return "year"
    else:
        return "text"


def has_repeated_header(table1: pd.DataFrame, table2: pd.DataFrame) -> bool:
    """
    Check if the second table starts with the same header as the first table.

    Args:
        table1: First DataFrame
        table2: Second DataFrame

    Returns:
        True if header appears to be repeated, False otherwise
    """
    if table1.shape[1] != table2.shape[1] or table1.empty or table2.empty:
        return False

    # Check if the column names match
    col_match = column_similarity(table1, table2) > 0.8

    # Check if the first row of table2 is similar to column names of table1
    header_match = False
    if table2.shape[0] > 0:
        first_row = [str(val) for val in table2.iloc[0]]
        col_names = [str(col) for col in table1.columns]
        matches = sum(1 for r, c in zip(first_row, col_names) if similar_strings(r, c))
        header_match = matches / len(col_names) > 0.7

    return col_match or header_match


def similar_strings(str1: str, str2: str) -> bool:
    """
    Check if two strings are similar.

    Args:
        str1: First string
        str2: Second string

    Returns:
        True if strings are similar, False otherwise
    """
    # Convert to lowercase and remove common punctuation
    s1 = re.sub(r'[^\w\s]', '', str(str1).lower())
    s2 = re.sub(r'[^\w\s]', '', str(str2).lower())

    # If strings are too short or very different in length, they're not similar
    if len(s1) < 2 or len(s2) < 2 or abs(len(s1) - len(s2)) > max(len(s1), len(s2)) * 0.5:
        return False

    # Calculate similarity using Levenshtein distance or simple containment
    if s1 in s2 or s2 in s1:
        return True

    # Calculate word overlap for multi-word strings
    words1 = set(s1.split())
    words2 = set(s2.split())

    if not words1 or not words2:
        return False

    overlap = words1.intersection(words2)
    overlap_ratio = len(overlap) / min(len(words1), len(words2))

    return overlap_ratio > 0.5


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a table.

    Args:
        df: DataFrame to clean

    Returns:
        Cleaned DataFrame
    """
    # Remove completely empty rows
    df = df.dropna(how='all')

    # Remove completely empty columns
    df = df.dropna(axis=1, how='all')

    # Clean column names
    df.columns = [str(col).strip() if not pd.isna(col) else f"col_{i}" for i, col in enumerate(df.columns)]

    # Replace empty cells with NaN
    df = df.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)

    return df


def save_tables(tables: List[pd.DataFrame], output_path: str, file_format: str = "csv") -> List[str]:
    """
    Save tables to files.

    Args:
        tables: List of table DataFrames
        output_path: Directory to save files
        file_format: Format to save tables ('csv', 'excel', or 'html')

    Returns:
        List of file paths
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    file_paths = []
    for i, table in enumerate(tables):
        # Clean the table
        table = clean_table(table)

        # Skip empty tables
        if table.empty:
            continue

        file_name = f"table_{i + 1}"

        if file_format == "csv":
            file_path = os.path.join(output_path, f"{file_name}.csv")
            table.to_csv(file_path, index=False)
        elif file_format == "excel":
            file_path = os.path.join(output_path, f"{file_name}.xlsx")
            table.to_excel(file_path, index=False)
        elif file_format == "html":
            file_path = os.path.join(output_path, f"{file_name}.html")
            table.to_html(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        file_paths.append(file_path)

    return file_paths


def main():
    """
    Main function to extract tables from a PDF file.
    """
    # import argparse

    # parser = argparse.ArgumentParser(description='Extract tables from PDF files')
    # parser.add_argument('pdf_path', type=str, help='Path to the PDF file')
    # parser.add_argument('--output', type=str, default='output', help='Output directory')
    # parser.add_argument('--method', type=str, default='auto', choices=['auto', 'camelot', 'tabula', 'pdfplumber'],
    #                     help='Extraction method')
    # parser.add_argument('--format', type=str, default='csv', choices=['csv', 'excel', 'html'],
    #                     help='Output file format')
    # args = parser.parse_args()
    file_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\input\data.pdf"
    output_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\output"
    if not os.path.isfile(file_path):
        print(f"Error: Input file path not found.")
        return
    format = 'csv'
    method = 'auto'

    try:
        print(f"Extracting tables from: {file_path}")
        tables = extract_tables_from_pdf(file_path, method=method)

        if not tables:
            print("No tables found in the PDF.")
            return

        print(f"Found {len(tables)} tables.")

        # Save tables
        file_paths = save_tables(tables, output_path, format)
        print(f"Tables saved to: {', '.join(file_paths)}")

        # Display information about tables
        for i, table in enumerate(tables):
            print(f"\nTable {i + 1}:")
            print(f"Shape: {table.shape}")
            print("Preview:")
            print(table.head(3))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
#
# def main():
#
#     file_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\input\small PDF.pdf"
#     output_path = r"C:\Users\SanskarZanwar\PycharmProjects\TableDataExctraction\output"
#     if not os.path.isfile(file_path):
#         print(f"Error: Input file path not found.")
#         return
#     format = 'json'
#     method = 'both'
#     saved_files = detect_and_process_tables(
#         file_path,
#         output_dir=output_path,
#         format=format,
#         method=method
#     )
#
#     if saved_files:
#         print(f"\nSuccessfully extracted {len(saved_files)} tables:")
#         for file in saved_files:
#             print(f"  - {file}")
#     else:
#         print("No tables were extracted.")
#
#
# if __name__ == "__main__":
#     main()