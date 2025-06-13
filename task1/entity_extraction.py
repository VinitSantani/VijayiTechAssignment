import re

def extract_entities(text, product_list, complaint_keywords):
    entities = {}
    entities['products'] = [prod for prod in product_list if prod.lower() in text.lower()]
    entities['dates'] = re.findall(r'\b(?:\d{1,2}[/-])?\d{1,2}[/-]\d{2,4}\b', text)
    entities['complaints'] = [word for word in complaint_keywords if word in text.lower()]
    return entities
