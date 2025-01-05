from typing import Dict, Any

class Template:
    def convert(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Template must implement the convert method.")

