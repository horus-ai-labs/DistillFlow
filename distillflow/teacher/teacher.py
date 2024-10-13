class TeacherModel:

    def __init__(self):
        pass

    """
    Base class for all teacher models.
    """
    def generate_response(self, prompt, max_tokens=150):
        raise NotImplementedError("Teacher models must implement this method.")