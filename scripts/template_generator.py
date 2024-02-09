import json
import os


def generate_templates(num_templates, dataset_name=""):
    with open("./random_words.json", "r") as file:
        data = json.load(file)

    words = data["words"]
    # Limit words to the number of templates required
    limited_words = words[:num_templates]

    yaml_content = "dataset: imdb\ntemplates:\n"

    for i, word in enumerate(limited_words):
        template = f"""  id-{num_templates}_{word}_{i}: !Template
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
        yaml_content += template

    root = f"templates/{dataset_name}/{num_templates}"
    os.makedirs(root, exist_ok=True)

    with open(f"{root}/templates.yaml", "w") as file:
        file.write(yaml_content)
    print(f"Templates generated and saved to '{root}/templates.yaml'")


def main():
    # Generate templates for 2, 4, 16, ..., 128
    for num in [2, 4, 16, 32, 64, 128]:
        generate_templates(num, "imdb")


if __name__ == "__main__":
    main()
