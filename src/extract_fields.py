# src/extract_fields.py
import re

# Keywords for company detection
COMPANY_KEYWORDS = [
    'SDN BHD', 'LTD', 'ENTERPRISE', 'TRADING', 'PLT',
    'SHOP', 'STORE', 'DECO', 'MART', 'MARKET', 'CO'
]

# Lines considered noise (ignore for address/company)
NOISE_KEYWORDS = [
    'ITEM', 'TOTAL', 'CASHIER', 'WWW', 'PAY', 'INVOICE', 'DATE', 'TIME', 'DOCUMENT', 'MEMBER', 'AMOUNT', 'DISC'
]

def clean_line(line):
    """Remove non-printable characters and excessive symbols"""
    line = re.sub(r"[^\x20-\x7E]+", "", line)
    line = re.sub(r'[*_#@\-]{2,}', '', line)  # remove repeating symbols
    line = line.strip()
    return line

def extract_fields(text):
    """
    Extract company, address, date, and total from OCR text.
    Optimized for TrOCR outputs.
    """
    # --- Preprocess OCR text ---
    lines = [clean_line(line) for line in text.splitlines() if clean_line(line)]
    text_combined = " ".join(lines)

    # --- DATE ---
    date_match = re.search(
        r"(\d{2}[/-]\d{2}[/-]\d{4})|(\d{2}[/-]\d{2}[/-]\d{2})|(\d{2}\s[A-Z]{3}\s\d{4})",
        text_combined
    )
    date = date_match.group(0) if date_match else None

    # --- TOTAL ---
    total = None
    total_line_pattern = r"(ROUNDED TOTAL|TOTAL|GRAND TOTAL|AMOUNT)"
    money_pattern = r"(?:RM|\$)?\s*([0-9]{1,6}[.,][0-9]{2})"
    
    # scan bottom-up for lines containing TOTAL keywords
    for line in lines[::-1]:
        if re.search(total_line_pattern, line, re.I):
            match = re.search(money_pattern, line.replace(',', '.'))
            if match:
                total = match.group(1)
                break
    # fallback: last money-like amount in the text
    if total is None:
        matches = re.findall(money_pattern, text_combined.replace(',', '.'))
        if matches:
            total = matches[-1]

    # --- COMPANY ---
    company_candidates = []
    for line in lines[:10]:  # top 10 lines
        if any(kw.upper() in line.upper() for kw in COMPANY_KEYWORDS):
            company_candidates.append(line)
    # fallback: first line with letters ignoring noise
    if not company_candidates:
        for line in lines[:10]:
            if re.search(r'[A-Za-z]{2,}', line) and not any(nk.upper() in line.upper() for nk in NOISE_KEYWORDS):
                company_candidates.append(line)
                break
    company = company_candidates[0] if company_candidates else None

    # --- ADDRESS ---
    address_lines = []
    company_idx = None
    if company:
        for i, line in enumerate(lines):
            if company.upper() in line.upper():
                company_idx = i
                break

    if company_idx is not None:
        for line in lines[company_idx + 1:company_idx + 8]:  # look 7 lines after company
            if line.strip() and not any(nk.upper() in line.upper() for nk in NOISE_KEYWORDS):
                address_lines.append(line.strip())
    # fallback: lines containing numbers and letters
    if not address_lines:
        for line in lines:
            if re.search(r'\d', line) and re.search(r'[A-Za-z]', line):
                address_lines.append(line.strip())
                if len(address_lines) >= 2:  # limit fallback to 2 lines
                    break
    address_text = " ".join(address_lines) if address_lines else None

    return {
        "company": company,
        "address": address_text,
        "date": date,
        "total": total
    }
