import json
from dataclasses import dataclass
from enum import Enum, unique
from typing import Dict, Any, Optional

from .template import Template
from ..dataset_args import Role

@dataclass
class AlpacaArgs:
    prompt: Optional[str] = "instruction"
    query: Optional[str] = "input"
    response: Optional[str] = "output"
    history: Optional[str] = None
    system: Optional[str] = None
    tools: Optional[str] = None

'''
Expected data format
[
  {
    "instruction": "human instruction (required)",
    "input": "human input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["human instruction in the first round (optional)", "model response in the first round (optional)"],
      ["human instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
'''
class Alpaca(Template):
    def __init__(self, args: AlpacaArgs):
        self.args = args

    def convert(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Converts alpaca format dataset to the standard format.
        """
        # prompt = {}
        # if self.args.history and isinstance(example[self.args.history], list):
        #     for old_prompt, old_response in example[self.args.history]:
        #         prompt = {"role": Role.USER.value, "content": old_prompt}
        #         prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.args.prompt and example[self.args.prompt]:
            query.append(example[self.args.prompt])

        if self.args.query and example[self.args.query]:
            query.append(example[self.args.query])

        prompt = {"role": Role.USER.value, "content": "\n".join(query)}  # "prompt\nquery"

        # if args.kto_tag and isinstance(example[args.kto_tag], bool):  # kto example
        #     response = [{"role": Role.ASSISTANT.value, "content": example[args.response]}]
        #     if example[self.args.kto_tag]:
        #         response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        #     else:
        #         response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        # elif (
        #         args.ranking
        #         and isinstance(example[args.chosen], str)
        #         and isinstance(example[args.rejected], str)
        # ):  # pairwise example
        #     response = [
        #         {"role": Role.ASSISTANT.value, "content": example[args.chosen]},
        #         {"role": Role.ASSISTANT.value, "content": example[args.rejected]},
        #     ]
        if self.args.response and isinstance(example[self.args.response], str):  # normal example
            response = {"role": Role.ASSISTANT.value, "content": example[self.args.response]}
        else:  # unsupervised
            response = {}

        system = example[self.args.system] if self.args.system else {"role": Role.SYSTEM.value, "content": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."}

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.args.tools] if self.args.tools else "",
        }
        return output