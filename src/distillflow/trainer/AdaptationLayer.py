from typing import List, Dict

import torch

class AdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim,
                 teacher_dim,
                 num_student_layers: int,
                 num_teacher_layers: int,
                 strategy="interpolate",
                 dtype=torch.bfloat16,
                 selection_indices: List[int]=None,
                 weights:List[List[int]] =None):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        # self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.layer_mapping = self.map_teacher_to_student_layers(num_student_layers, num_teacher_layers, strategy, selection_indices, weights)
        self.dtype = dtype

    def map_teacher_to_student_layers(self, num_student_layers, num_teacher_layers, strategy="select",
                                      selection_indices:List[int]=None, weights:List[List[int]]=None) -> {}:
        """
        Maps teacher model layers to student model layers based on the specified strategy.

        Args:
            num_student_layers (int): Number of layers in the student model.
            num_teacher_layers (int): Number of layers in the teacher model.
            strategy (str): Layer mapping strategy ("direct", "select", "interpolate", "weighted").
            selection_indices (list): Specific teacher layers to select (used when strategy="select").
            weights (list of lists): Weights for combining teacher layers for each student layer
                                     (used when strategy="weighted").

        Returns:
            list: List of mapping indices or weights from teacher layers to align with student layers.
        """
        if strategy == "direct":
            # Direct one-to-one mapping
            return {
                i: i
                for i in range(num_student_layers)
            }

        elif strategy == "select":
            # Use specific teacher layers for mapping
            if selection_indices is None:
                raise ValueError("selection_indices must be provided for 'select' strategy.")
            if len(selection_indices) != num_student_layers:
                raise ValueError("Number of selection_indices must match num_student_layers.")
            return {
                i: teacher_layer
                for i, teacher_layer in enumerate(selection_indices)
            }

        elif strategy == "interpolate":
            # Interpolate teacher layers to match student layers
            return {
                i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
                for i in range(num_student_layers)
            }
        elif strategy == "weighted":
            # Weighted combination of teacher layers for each student layer
            if weights is None:
                raise ValueError("weights must be provided for 'weighted' strategy.")
            if len(weights) != num_student_layers:
                raise ValueError("Number of weight sets must match num_student_layers.")
            return {
                i: {j: weight for j, weight in enumerate(weight_set) if weight > 0}
                for i, weight_set in enumerate(weights)
            }
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states