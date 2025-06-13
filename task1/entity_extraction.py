import re

def extract_entities(text):
    entities = {}
    emails = re.findall(r"[\w\.-]+@[\w\.-]+", text)
    phones = re.findall(r"\+?\d[\d\-\(\) ]{7,}\d", text)
    urls = re.findall(r"https?://\S+", text)
    
    if emails:
        entities['emails'] = emails
    if phones:
        entities['phones'] = phones
    if urls:
        entities['urls'] = urls

    return entities
