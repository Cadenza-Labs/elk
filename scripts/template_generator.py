import argparse
import json


def generate_templates(template_name):
    with open("./random_words.json", "r") as file:
        data = json.load(file)

    words = data["words"]
    yaml_content = "dataset: imdb\ntemplates:\n"

    for i, word in enumerate(words):
        template = f"""  id-{word}-{i}: !Template
    answer_choices: negative ||| positive
    id: id-{word}-{i}
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
    name: {template_name}_{word}
    reference: ''
"""  # noqa
        yaml_content += template

    with open(f"templates/{template_name}_templates.yaml", "w") as file:
        file.write(yaml_content)
    print(f"Templates generated and saved to '{template_name}_templates.yaml'")


def main():
    parser = argparse.ArgumentParser(description="Generate YAML templates")
    parser.add_argument("template_name", type=str, help="Name of the template")
    args = parser.parse_args()

    generate_templates(args.template_name)


if __name__ == "__main__":
    main()
