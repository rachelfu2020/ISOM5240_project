import easyocr
import re
from pathlib import Path
from typing import Dict, Any

def analyze_cad_drawing(image_path_or_bytes) -> Dict[str, Any]:
    """Extract engineering data from CAD drawings"""
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Read text
    results = reader.readtext(image_path_or_bytes, detail=1, paragraph=False)
    texts = [text for (_, text, conf) in results if conf > 0.5]
    full_text = ' '.join(texts)
    
    # Engineering parsing
    scale = re.search(r'1[:\-]\d+', full_text, re.I)
    dimensions = re.findall(r'\d+[xX×]\d+', full_text)
    title = next((t for t in texts if t.isupper() and len(t) > 3), 'N/A')
    
    return {
        'texts_found': len(results),
        'confidence_texts': len(texts),
        'title': title,
        'scale': scale.group() if scale else 'N/A',
        'dimensions': dimensions[:3],
        'full_text': full_text,
        'sample_text': full_text[:200]
    }
