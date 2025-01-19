from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from . import ShareGptArgs, AlpacaArgs


class TemplateArgs(BaseModel):
    name: Literal["sharegpt", "alpaca"] = Field(
        default="sharegpt",
        description="Template used by the dataset. Based on the template, the data will be converted to a standard format before training"
    )
    args: Optional[Union[ShareGptArgs, AlpacaArgs]] = Field(
        default=None,
        description="Template args to map the corresponding dataset columns to the ones expected by sharegpt/alpaca"
    )

    # Validate and transform args based on template name
    @model_validator(mode='after')
    def validate_args(self) -> 'TemplateArgs':
        if self.args is None:
            # Set default args based on template
            self.args = {
                "sharegpt": ShareGptArgs(),
                "alpaca": AlpacaArgs()
            }[self.name]
            return self

        # If args is a dict, convert to appropriate type
        if isinstance(self.args, dict):
            template_classes = {
                "sharegpt": ShareGptArgs,
                "alpaca": AlpacaArgs
            }

            template_class = template_classes.get(self.name)
            if template_class is None:
                raise ValueError(f"Unknown template: {self.name}")

            try:
                self.args = template_class(**self.args)
            except Exception as e:
                raise ValueError(f"Invalid args for template '{self.name}': {str(e)}")

        # Validate that args matches template
        expected_type = ShareGptArgs if self.name == "sharegpt" else AlpacaArgs
        if not isinstance(self.args, expected_type):
            raise ValueError(
                f"Template '{self.name}' expects {expected_type.__name__} "
                f"but got {type(self.args).__name__}"
            )
        print("Called validate args")
        return self

    model_config = {
        "extra": "forbid"
    }