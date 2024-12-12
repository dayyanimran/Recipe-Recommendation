import pandas as pd
import json
from sklearn.metrics import precision_score, recall_score, f1_score

# Loading the dataset
file_path = r"C:\Users\Dayyan\Downloads\finalrecipedataset.csv"
recipes = pd.read_csv(file_path)
recipes = recipes.drop_duplicates()

# Query expansion
expansion_dict = {
    "tomato": ["basil", "olive oil", "cheese"],
    "chicken": ["garlic", "flour", "potato"],
    "pasta": ["cheese", "flour", "olive oil"],
    "lentils": ["tofu", "carrot", "spinach"],
    "quinoa": ["lemon", "avocado", "chickpeas"]
}

# Suggest additional ingredients based on the query
def suggest_ingredients(query):
    suggested = set()
    for ingredient in query:
        if ingredient in expansion_dict:
            suggested.update(expansion_dict[ingredient])
    return list(suggested - set(query))

# Matching function
def calculate_match_percentage(recipe_ingredients, query_ingredients):
    recipe_ingredients_set = set(recipe_ingredients.split(", "))
    query_ingredients_set = set(query_ingredients)
    matches = len(recipe_ingredients_set & query_ingredients_set)
    return round((matches / len(query_ingredients_set) * 100), 2) if query_ingredients_set else 0

def match_recipes(query_ingredients, recipes):
    recipes["Match Score (%)"] = recipes["Ingredients"].apply(
        lambda ingredients: calculate_match_percentage(ingredients, query_ingredients)
    )
    ranked_recipes = recipes.sort_values(by="Match Score (%)", ascending=False).drop_duplicates(subset="Recipe Name")
    return ranked_recipes

# Evaluation metrics
def metricsevaluation(true_relevance, predicted_relevance):
    precision = precision_score(true_relevance, predicted_relevance)
    recall = recall_score(true_relevance, predicted_relevance)
    f1 = f1_score(true_relevance, predicted_relevance)
    return {"Precision": precision, "Recall": recall, "F1-Score": f1}

# Test query
query = ["chicken", "cheese"]
suggested_ingredients = list(set(suggest_ingredients(query)))
print(f"Original Query: {query}")
print(f"Suggested Ingredients: {suggested_ingredients}")

# Get matching recipes
ranked_recipes = match_recipes(query, recipes)

# Display the top 5 results
print("Top Matching Recipes:")
print(ranked_recipes[["Recipe Name", "Match Score (%)", "Ingredients", "Dietary Filters"]].head(5).to_string(index=False))

# Display the top recipe using a suggested ingredient
if suggested_ingredients:
    suggested_query = query + [suggested_ingredients[0]]
    top_recipe_with_suggestion = match_recipes(suggested_query, recipes)
    top_recipe_with_suggestion = top_recipe_with_suggestion[
        top_recipe_with_suggestion["Ingredients"].str.contains(suggested_ingredients[0], na=False)
    ].iloc[0] if not top_recipe_with_suggestion.empty else None
    if top_recipe_with_suggestion is not None:
        print("\nTop Recipe Using Suggested Ingredient:")
        print(top_recipe_with_suggestion[["Recipe Name", "Match Score (%)", "Ingredients", "Dietary Filters"]].to_frame().T.to_string(index=False))

# Evaluate
true_relevance = [1, 1, 0, 0, 0]
predicted_relevance = [1, 1, 1, 0, 0]
metrics = metricsevaluation(true_relevance, predicted_relevance)
print("\nEvaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.2f}")

# Save the ranked results to a CSV file
ranked_recipes_file_path = r"C:\Users\Dayyan\Downloads\ranked_recipes.csv"
ranked_recipes.to_csv(ranked_recipes_file_path, index=False)
print("Ranked recipes saved as 'ranked_recipes.csv'.")

# Testing multiple queries and save results
queries = [
    ["milk", "flour", "ginger"],
    ["lentils", "rice", "avocado"],
    ["pasta", "potato", "beef"],
    ["carrot", "quinoa", "soy sauce"],
    ["chickpeas", "egg", "broccoli"],
    ["tofu", "basil", "sugar"],
    ["bell pepper", "olive oil", "tomato"],
    ["butter", "garlic", "chicken"],
    ["cheese", "onion", "spinach"],
    ["lemon", "cucumber", "milk"],
    ["rice", "basil", "flour"],
    ["ginger", "beef", "avocado"],
    ["quinoa", "chickpeas", "soy sauce"],
    ["broccoli", "egg", "potato"],
    ["tofu", "carrot", "bell pepper"],
    ["basil", "tomato", "butter"],
    ["cheese", "garlic", "cucumber"],
    ["spinach", "sugar", "onion"],
    ["chicken", "lemon", "rice"],
    ["pasta", "milk", "olive oil"],
    ["lentils", "tomato", "broccoli"],
    ["potato", "butter", "cheese"],
    ["avocado", "ginger", "spinach"],
    ["onion", "soy sauce", "basil"],
    ["chickpeas", "bell pepper", "carrot"],
    ["tofu", "egg", "olive oil"],
    ["tomato", "chicken", "garlic"],
    ["flour", "potato", "cucumber"],
    ["lemon", "milk", "pasta"],
    ["quinoa", "butter", "lentils"],
    ["avocado", "tomato", "spinach"],
    ["basil", "sugar", "rice"],
    ["cheese", "soy sauce", "onion"],
    ["chicken", "chickpeas", "carrot"],
    ["potato", "bell pepper", "garlic"],
    ["olive oil", "broccoli", "flour"],
    ["tomato", "lemon", "tofu"],
    ["basil", "spinach", "rice"],
    ["avocado", "ginger", "broccoli"],
    ["butter", "pasta", "chickpeas"],
    ["lentils", "potato", "soy sauce"],
    ["egg", "chicken", "tomato"],
    ["cucumber", "cheese", "garlic"],
    ["milk", "basil", "bell pepper"],
    ["sugar", "avocado", "carrot"],
    ["onion", "flour", "quinoa"],
    ["olive oil", "spinach", "broccoli"],
    ["lemon", "butter", "basil"],
    ["rice", "chicken", "potato"],
    ["cheese", "tomato", "bell pepper"]
]

# Collect results for all queries
all_results = []

for query in queries:
    suggested_ingredients = suggest_ingredients(query)
    ranked_recipes = match_recipes(query, recipes)
    non_relevant_recipes = recipes.copy()
    non_relevant_recipes["Match Score (%)"] = non_relevant_recipes["Ingredients"].apply(
        lambda ingredients: calculate_match_percentage(ingredients, query)
    )
    non_relevant_dish = non_relevant_recipes[non_relevant_recipes["Match Score (%)"] == 0].head(1)
    top_recipe_with_suggestion = None
    if suggested_ingredients:
        suggested_query = query + [suggested_ingredients[0]]
        top_recipe_with_suggestion = match_recipes(suggested_query, recipes)
        top_recipe_with_suggestion = top_recipe_with_suggestion[
            top_recipe_with_suggestion["Ingredients"].str.contains(suggested_ingredients[0], na=False)
        ].iloc[0] if not top_recipe_with_suggestion.empty else None

    # Compile results for each query
    result = {
        "Query": query,
        "Suggested Ingredients": suggested_ingredients,
        "Top Matching Recipes": ranked_recipes[["Recipe Name", "Match Score (%)", "Ingredients", "Dietary Filters"]].head(5).to_dict(orient="records"),
        "Top Recipe Using Suggested Ingredient": top_recipe_with_suggestion[["Recipe Name", "Match Score (%)", "Ingredients", "Dietary Filters"]].to_dict() if top_recipe_with_suggestion is not None else None,
        "Non-Relevant Dish (0% Match)": non_relevant_dish[["Recipe Name", "Ingredients", "Dietary Filters"]].to_dict(orient="records") if not non_relevant_dish.empty else None
    }
    all_results.append(result)

# Save results to a JSON file
query_results_file_path = r"C:\Users\Dayyan\Downloads\query_results.json"
with open(query_results_file_path, "w") as json_file:
    json.dump(all_results, json_file, indent=4)
input_file_path = r"C:\Users\Dayyan\Downloads\query_results.json"
output_file_path = r"C:\Users\Dayyan\Downloads\query_results_fixed.json"

with open(input_file_path, "r") as json_file:
    data = json.load(json_file)

# Convert NaN to null (JSON validation)
def replace_nan_with_null(data):
    if isinstance(data, list):
        return [replace_nan_with_null(item) for item in data]
    elif isinstance(data, dict):
        return {key: replace_nan_with_null(value) for key, value in data.items()}
    elif pd.isna(data):  # Check for NaN or None
        return None
    else:
        return data

fixed_data = replace_nan_with_null(data)
with open(output_file_path, "w") as json_file:
    json.dump(fixed_data, json_file, indent=4)

print(f"Query results saved as '{output_file_path}'.")
