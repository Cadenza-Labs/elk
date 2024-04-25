import json
import os


def generate_templates(num_templates, dataset_name="", eval=False):
    random_words_file = (
        "./random_words.json" if not eval else "./random_words_eval.json"
    )
    with open(random_words_file, "r") as file:
        data = json.load(file)

    words = data["words"]
    # Limit words to the number of templates required
    limited_words = words[:num_templates]

    yaml_content = f"dataset: {dataset_name}\ntemplates:\n"

    for i, word in enumerate(limited_words):
        imdb_template = f"""  id-{num_templates}_{word}_{i}: !Template
    answer_choices: negative ||| positive
    id: id-{num_templates}_{word}_{i}
    jinja: 'Consider the following example:  '''''' {{{{{"text"}}}}}  ''''''

        Between {{{{ answer_choices[0] }}}} and {{{{ answer_choices[1] }}}}, the sentiment of
        this example is ||| {{{{ answer_choices[label] }}}}. {word}'
    metadata: !TemplateMetadata
      choices_in_prompt: true
      languages:
      - en
      metrics:
      - Accuracy
      original_task: true
    name: {num_templates}_{word}_{i}
    reference: ''
"""  # noqa

        amazon_polarity_template = f"""  id-{num_templates}_{word}_{i}: !Template
    answer_choices: negative ||| positive
    id: id-{num_templates}_{word}_{i}
    jinja: 'Title: {{{{{"title"}}}}}

      Review: {{{{{"content"}}}}}

      Is the review positive or negative? |||

      {{{{ answer_choices[label] }}}}. {word}'
    metadata: !TemplateMetadata
      choices_in_prompt: true
      languages:
      - en
      metrics:
      - Accuracy
      original_task: true
    name: {num_templates}_{word}_{i}
    reference: ''
"""  # noqa

        boolq_template = f"""  id-{num_templates}_{word}_{i}: !Template
    answer_choices: false ||| true
    id: id-{num_templates}_{word}_{i}
    jinja: 'Passage:  '''''' {{{{{"passage"}}}}}  ''''''
        After reading this passage, I have a question: {{{{{"question"}}}}}? true or false? ||| {{{{ answer_choices[label] }}}}. {word}'
    metadata: !TemplateMetadata
        choices_in_prompt: true
        languages:
        - en
        metrics:
        - Accuracy
        original_task: true
    name: {num_templates}_{word}_{i}
    reference: ''
"""  # noqa

        if dataset_name == "imdb":
            template = imdb_template
        elif dataset_name == "amazon_polarity":
            template = amazon_polarity_template
        elif dataset_name == "boolq":
            template = boolq_template

        yaml_content += template
    if eval:
        root = f"templates/random/eval/{dataset_name}/{num_templates}"
    else:
        root = f"templates/random/train/{dataset_name}/{num_templates}"
    os.makedirs(root, exist_ok=True)

    with open(f"{root}/templates.yaml", "w") as file:
        file.write(yaml_content)
    print(f"Templates generated and saved to '{root}/templates.yaml'")


def main():
    # Generate templates for 2, 4, 16, ..., 128
    for num in [2, 4, 8, 16, 32, 64, 128]:
        generate_templates(num, "boolq")
        generate_templates(num, "boolq", eval=True)


if __name__ == "__main__":
    main()
