import prodigy
from prodigy.components.loaders import JSONL

@prodigy.recipe("custom-recipe")
def custom_recipe(dataset, source):
    stream = JSONL(source)

    def update(answers):
        print(f"Received {len(answers)} answers!")

    return {
        "dataset": dataset,
        "source": source,
        "update": update,
        "view_id": "text"
    }

# python3 -m prodigy custom-recipe toxic_swe_dataset ./data/raw_oscar_15.jsonl -F recipe.py

# python3 -m prodigy db-out dataset