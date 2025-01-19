from pydantic import ValidationError

from . import Config


def print_validation_error(error: ValidationError):
    """
    Print detailed validation errors including field descriptions and examples
    """
    print("Validation Error:")
    print("-" * 50)

    for error in error.errors():
        # Get the field path
        field_path = " -> ".join(str(item) for item in error['loc'])

        # Find the relevant model and field
        current_model = Config
        field_info = None

        for item in error['loc']:
            if isinstance(item, int):
                continue
            if hasattr(current_model, 'model_fields') and item in current_model.model_fields:
                field_info = current_model.model_fields[item]
                # If this field is another model, update current_model for nested fields
                if hasattr(field_info.annotation, 'model_fields'):
                    current_model = field_info.annotation

        print(f"Field: {field_path}")
        print(f"Error: {error['msg']}")

        if field_info:
            if field_info.description:
                print(f"Description: {field_info.description}")
            if field_info.examples:
                print(f"Example: {field_info.examples}")
        print("-" * 50)