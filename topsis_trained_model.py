import numpy as np
import pandas as pd


def normalize_data(matrix):
    normalized_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
    return normalized_matrix

def determine_ideal_solutions(normalized_matrix, criteria_types):
    ideal_best_solution = np.max(normalized_matrix, axis=0) if criteria_types == 1 else np.min(normalized_matrix, axis=0)
    ideal_worst_solution = np.min(normalized_matrix, axis=0) if criteria_types == 1 else np.max(normalized_matrix, axis=0)
    return ideal_best_solution, ideal_worst_solution

def compute_distances(normalized_matrix, ideal_best_solution, ideal_worst_solution):
    best_solution_distance = np.sqrt(np.sum((normalized_matrix - ideal_best_solution) ** 2, axis=1))
    worst_solution_distance = np.sqrt(np.sum((normalized_matrix - ideal_worst_solution) ** 2, axis=1))
    return best_solution_distance, worst_solution_distance

def calculate_ranking_score(best_solution_distance, worst_solution_distance):
    ranking_score = worst_solution_distance / (best_solution_distance + worst_solution_distance)
    return ranking_score

def execute_topsis(input_csv, output_csv):
    data = pd.read_csv(input_csv)

    matrix_data = data.iloc[:, 1:].values

    normalized_data = normalize_data(matrix_data)

    criteria = [0, 1, 1]  # 0 for cost criteria, 1 for benefit criteria
    ideal_best_solution, ideal_worst_solution = determine_ideal_solutions(normalized_data, criteria)

    best_solution_distance, worst_solution_distance = compute_distances(normalized_data, ideal_best_solution, ideal_worst_solution)

    ranking_scores = calculate_ranking_score(best_solution_distance, worst_solution_distance)

    data['Ranking Score'] = ranking_scores
    data['Position'] = pd.Series(ranking_scores).rank(ascending=False)

    data.to_csv(output_csv, index=False)

    print(data)



    print(f'Results saved to {output_csv}')

input_csv = 'input_text_generation_models.csv'
output_csv = 'topsis_text_generation_results.csv'

execute_topsis(input_csv, output_csv)
