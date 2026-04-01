from trainers.supervised import SupervisedModel
from trainers.ssl_byol import SemiSupervisedBYOLModel
from trainers.ssl_graph import GraphBasedModel
from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel

TRAINER_REGISTRY = {
    ('sl', 'mlc', False, False): SupervisedModel,
    ('sl', 'hmlc', False, False): SupervisedModel,
    ('sl', 'hmlc', True, False): GraphBasedModel,
    ('ssl', 'hmlc', True, False): GraphBasedModel,
    ('ssl', 'hmlc', False, True): SemiSupervisedBYOLModel,
    ('ssl', 'hmlc', True, True): SemiSupervisedGraphBYOLModel,
    ('sl', 'hmlc', True, True): SemiSupervisedGraphBYOLModel,
}


def get_trainer_class(training_mode, learning_task, apply_edges, apply_byol):
    """Resolve the trainer class from method config parameters."""
    key = (training_mode, learning_task, apply_edges, apply_byol)
    cls = TRAINER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"No trainer found for: training_mode={training_mode}, "
            f"learning_task={learning_task}, apply_edges={apply_edges}, "
            f"apply_byol={apply_byol}"
        )
    return cls
