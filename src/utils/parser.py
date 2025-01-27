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