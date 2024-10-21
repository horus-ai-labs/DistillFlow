from functools import partial
from typing import Dict, Any

from datasets import Dataset

from .dataset_args import DatasetAttr


def convert_alpaca(
        example: Dict[str, Any],
        dataset_attr: DatasetAttr,
) -> Dict[str, Any]:
    r"""
    Converts alpaca format dataset to the standard format.
    """
    prompt = []
    if dataset_attr.history and isinstance(example[dataset_attr.history], list):
        for old_prompt, old_response in example[dataset_attr.history]:
            prompt.append({"role": Role.USER.value, "content": old_prompt})
            prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

    query = []
    if dataset_attr.prompt and example[dataset_attr.prompt]:
        query.append(example[dataset_attr.prompt])

    if dataset_attr.query and example[dataset_attr.query]:
        query.append(example[dataset_attr.query])

    prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
            dataset_attr.ranking
            and isinstance(example[dataset_attr.chosen], str)
            and isinstance(example[dataset_attr.rejected], str)
    ):  # pairwise example
        response = [
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.chosen]},
            {"role": Role.ASSISTANT.value, "content": example[dataset_attr.rejected]},
        ]
    elif dataset_attr.response and isinstance(example[dataset_attr.response], str):  # normal example
        response = [{"role": Role.ASSISTANT.value, "content": example[dataset_attr.response]}]
    else:  # unsupervised
        response = []

    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": example[dataset_attr.system] if dataset_attr.system else "",
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
    }
    return output


def convert_sharegpt(
        example: Dict[str, Any],
        dataset_attr: "DatasetAttr",
) -> Dict[str, Any]:
    r"""
    Converts sharegpt format dataset to the standard format.
    """
    tag_mapping = {
        dataset_attr.user_tag: Role.USER.value,
        dataset_attr.assistant_tag: Role.ASSISTANT.value,
        dataset_attr.observation_tag: Role.OBSERVATION.value,
        dataset_attr.function_tag: Role.FUNCTION.value,
        dataset_attr.system_tag: Role.SYSTEM.value,
    }
    odd_tags = (dataset_attr.user_tag, dataset_attr.observation_tag)
    even_tags = (dataset_attr.assistant_tag, dataset_attr.function_tag)
    accept_tags = (odd_tags, even_tags)
    messages = example[dataset_attr.messages]
    if (
            dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][dataset_attr.role_tag] == dataset_attr.system_tag
    ):
        system = messages[0][dataset_attr.content_tag]
        messages = messages[1:]
    else:
        system = example[dataset_attr.system] if dataset_attr.system else ""

    aligned_messages = []
    broken_data = False
    for turn_idx, message in enumerate(messages):
        if message[dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
            logger.warning("Invalid role tag in {}.".format(messages))
            broken_data = True

        aligned_messages.append(
            {"role": tag_mapping[message[dataset_attr.role_tag]], "content": message[dataset_attr.content_tag]}
        )

    if (not dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            dataset_attr.ranking and len(aligned_messages) % 2 == 0
    ):
        logger.warning("Invalid message count in {}.".format(messages))
        broken_data = True

    if dataset_attr.kto_tag and isinstance(example[dataset_attr.kto_tag], bool):  # kto example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]
        if example[dataset_attr.kto_tag]:
            response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
        else:
            response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
    elif (
            dataset_attr.ranking
            and isinstance(example[dataset_attr.chosen], dict)
            and isinstance(example[dataset_attr.rejected], dict)
    ):  # pairwise example
        chosen = example[dataset_attr.chosen]
        rejected = example[dataset_attr.rejected]
        if (
                chosen[dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[dataset_attr.role_tag] not in accept_tags[-1]
        ):
            logger.warning("Invalid role tag in {}.".format([chosen, rejected]))
            broken_data = True

        prompt = aligned_messages
        response = [
            {"role": tag_mapping[chosen[dataset_attr.role_tag]], "content": chosen[dataset_attr.content_tag]},
            {"role": tag_mapping[rejected[dataset_attr.role_tag]], "content": rejected[dataset_attr.content_tag]},
        ]
    else:  # normal example
        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

    if broken_data:
        logger.warning("Skipping this abnormal example.")
        prompt, response = [], []

    convert_images = partial(_convert_images, dataset_attr=dataset_attr, data_args=data_args)
    convert_videos = partial(_convert_videos, dataset_attr=dataset_attr, data_args=data_args)
    output = {
        "_prompt": prompt,
        "_response": response,
        "_system": system,
        "_tools": example[dataset_attr.tools] if dataset_attr.tools else "",
        "_images": convert_images(example[dataset_attr.images]) if dataset_attr.images else None,
        "_videos": convert_videos(example[dataset_attr.videos]) if dataset_attr.videos else None,
    }
    return output


def align_dataset(
        dataset: Dataset,
        dataset_attr: DatasetAttr
) -> Dataset:
    if dataset_attr.formatting == "alpaca":
        convert_func = partial(convert_alpaca, dataset_attr=dataset_attr)
    else:
        convert_func = partial(convert_sharegpt, dataset_attr=dataset_attr)

    column_names = list(next(iter(dataset)).keys())

    return dataset.map(
        convert_func,
        batched=False,
        remove_columns=column_names,
    )