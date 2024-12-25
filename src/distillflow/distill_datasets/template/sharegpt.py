import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

from .template import Template
from .role import Role
from ...common import get_logger

logger = get_logger(__name__)

@dataclass
class ShareGptArgs:
    messages: Optional[str] = "conversations"
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"

class ShareGpt(Template):

    def __init__(self, args: Optional[ShareGptArgs] = ShareGptArgs()):
        self.args = args

    def convert(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Converts sharegpt format dataset to the standard format.
        """
        tag_mapping = {
            self.args.user_tag: Role.USER.value,
            self.args.assistant_tag: Role.ASSISTANT.value,
            self.args.observation_tag: Role.OBSERVATION.value,
            self.args.function_tag: Role.FUNCTION.value,
            self.args.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.args.user_tag, self.args.observation_tag)
        even_tags = (self.args.assistant_tag, self.args.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.args.messages]
        system = []
        if (
                self.args.system_tag
                and len(messages) != 0
                and messages[0][self.args.role_tag] == self.args.system_tag
        ):
            system.append({"role": Role.SYSTEM.value, "content": messages[0][self.args.content_tag]})
            messages = messages[1:]
        else:
            system.append({"role": Role.SYSTEM.value, "content": example[self.args.system_tag]} if self.args.system_tag in example else {"role": Role.SYSTEM.value,
                                                                         "content": "You are a helpful assistant."})

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.args.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning("Invalid role tag in {}.".format(messages))
                broken_data = True

            aligned_messages.append(
                {"role": tag_mapping[message.get(self.args.role_tag, '')], "content": message.get(self.args.content_tag, '')}
            )


        # if (not self.args.ranking and len(aligned_messages) % 2 != 0) or (
        #         self.args.ranking and len(aligned_messages) % 2 == 0
        # ):
        #     logger.warning("Invalid message count in {}.".format(messages))
        #     broken_data = True

        # if self.args.kto_tag and isinstance(example[self.args.kto_tag], bool):  # kto example
        #     prompt = aligned_messages[:-1]
        #     response = aligned_messages[-1:]
        #     if example[self.args.kto_tag]:
        #         response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        #     else:
        #         response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        # elif (
        #         self.args.ranking
        #         and isinstance(example[self.args.chosen], dict)
        #         and isinstance(example[self.args.rejected], dict)
        # ):  # pairwise example
        #     chosen = example[self.args.chosen]
        #     rejected = example[self.args.rejected]
        #     if (
        #             chosen[self.args.role_tag] not in accept_tags[-1]
        #             or rejected[self.args.role_tag] not in accept_tags[-1]
        #     ):
        #         logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
        #         broken_data = True
        #
        #     prompt = aligned_messages
        #     response = [
        #         {"role": tag_mapping[chosen[self.args.role_tag]], "content": chosen[self.args.content_tag]},
        #         {"role": tag_mapping[rejected[self.args.role_tag]], "content": rejected[self.args.content_tag]},
        #     ]
        # else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in response: {response}") from e
        if broken_data:
            logger.warning("Skipping this abnormal example.")
            prompt, response = [], []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
        }

        return output