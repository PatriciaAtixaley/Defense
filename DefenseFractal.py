from TechnicalUtils import get_sub_list, save_list
from Utils import aggregate_models, eval_classifier, put_model_to_device
from ModelEvaluation import evaluate_model_and_put
from defenses.AbstractDefense import AbstractDefense
import numpy as np
import torch

class PuzzleDefense(AbstractDefense):

    def __init__(self, noise_level=0):
        self.noise_level = noise_level
        super().__init__(clipping_factor=1, clipping_boundary=None)

    def visualize_model(self, model_state_dict, title="Model Visualization"):
        """
        Visualize the weights of a model layer-by-layer for analysis.

        Args:
            model_state_dict (dict): Weights of the model to visualize.
            title (str): Title for the visualization.
        """
        print(f"\n{title}")
        for i, (layer, weights) in enumerate(model_state_dict.items()):
            print(f"Layer {i} ({layer}): mean={weights.mean():.4f}, std={weights.std():.4f}")

    def filter_poisoned_layers(self, global_model_state_dict, client_model_state_dict):
        """
        Filters out poisoned layers by comparing client models to the global model.

        Args:
            global_model_state_dict (dict): Weights of the current global model.
            client_model_state_dict (dict): Weights of a specific client model.

        Returns:
            dict: Updated client model weights with poisoned layers replaced by global model layers.
        """
        filtered_model = client_model_state_dict.copy()

        for layer_name in global_model_state_dict.keys():
            global_weights = global_model_state_dict[layer_name]
            client_weights = client_model_state_dict[layer_name]

            # Calculate cosine similarity or distance metrics
            similarity = torch.nn.functional.cosine_similarity(
                global_weights.flatten(), client_weights.flatten(), dim=0
            )

            # Threshold to determine if the layer is poisoned
            threshold = 0.9  # Adjust based on experiments

            if similarity < threshold:
                print(f"Replacing poisoned layer: {layer_name}")
                filtered_model[layer_name] = global_weights

        return filtered_model

    def __call__(self, exp_directory, prefix_for_run, results, helper, data_source, poisoned_data_source,
                 vocab_size, formatted_round_index, global_model_state_dict, all_models, seed, poison_this_round,
                 data_sizes, number_of_benign_clients, number_of_malicious_clients):
        """
        Apply the Puzzle Defense mechanism.

        Args:
            (same as earlier implementation)

        Returns:
            dict: Aggregated weights for the global model.
        """
        # Visualize the initial global model
        self.visualize_model(global_model_state_dict, title="Initial Global Model")

        new_models = []
        for client_model in all_models:
            # Filter each client's model by removing poisoned sections
            filtered_model = self.filter_poisoned_layers(global_model_state_dict, client_model)
            new_models.append(filtered_model)

        # Aggregate the filtered models to create the new global model
        aggregated_weights = aggregate_models(helper, new_models, global_model_state_dict)

        # Ensure the aggregated weights are properly moved to the correct device
        aggregated_weights = put_model_to_device(aggregated_weights)

        # Load the new global model weights into the target model
        helper.target_model.load_state_dict(aggregated_weights)

        # Visualize the repaired global model for analysis
        self.visualize_model(aggregated_weights, title="Repaired Global Model")

        # Evaluate the defended model's performance
        evaluate_model_and_put(
            helper=helper,
            round_index=formatted_round_index,
            data_source=data_source,
            poisoned_data_source=poisoned_data_source,
            model=helper.target_model,
            description='Defended',
            results=results,
        )

        # Return the updated global model weights
        return aggregated_weights

# Example Usage
if __name__ == "__main__":
    # Instantiate the defense
    puzzle_defense = PuzzleDefense(noise_level=0.1)

    # Example arguments (placeholders for actual values in a real implementation)
    exp_directory = "path/to/experiment"
    prefix_for_run = "run_01"
    results = {}
    helper = None  # Replace with the actual helper instance
    data_source = None  # Replace with the actual dataset
    poisoned_data_source = None  # Replace with the poisoned dataset
    vocab_size = 5000
    formatted_round_index = 1
    global_model_state_dict = {}  # Replace with the global model's weights
    all_models = [{}]  # Replace with a list of client models' weights
    seed = 42
    poison_this_round = True
    data_sizes = [500, 500, 500]
    number_of_benign_clients = 8
    number_of_malicious_clients = 2

    # Apply the defense
    new_global_weights = puzzle_defense(
        exp_directory,
        prefix_for_run,
        results,
        helper,
        data_source,
        poisoned_data_source,
        vocab_size,
        formatted_round_index,
        global_model_state_dict,
        all_models,
        seed,
        poison_this_round,
        data_sizes,
        number_of_benign_clients,
        number_of_malicious_clients
    )
