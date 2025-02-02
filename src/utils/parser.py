import re

def gsm8k_parser(text):
    pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"

    try:
        match = "".join(re.findall(pattern, text)[-1])
        match = match.strip("$.").replace(",", "")
    except IndexError:
        print(f"IndexError: {text}")
        return 0

    # generated_match = find_last_number(generated_text)
    # target_match = find_last_number(target_text)
    if match == None:
        return 0
    else:
        return match

def wikisql_parser(text):
    pattern = r'(?<=```sql\n)([\s\S]*?)(?=\n```)'  # Regex pattern to capture SQL queries
    matches = re.findall(pattern, text)
    if not matches:
        pattern_alt = r'SELECT[\s\S]*?;'  # Alternative pattern to capture inline SQL queries
        matches = re.findall(pattern_alt, text, re.IGNORECASE)

    if not matches:
        patter_alt2 = r'"(SELECT .*? FROM .*? WHERE .*?)\sAND'
        matches = re.search(pattern, text, re.DOTALL)

    return matches[0] if matches else ""