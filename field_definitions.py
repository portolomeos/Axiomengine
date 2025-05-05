"""
Field Definitions - Defines the mathematical properties of fields in Collapse Geometry

Implements the fields that govern collapse dynamics, including awareness,
polarity, collapse metric, and grain activation.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class RelationalField:
    """Base class for fields defined over the relational manifold"""
    name: str
    description: str
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> Any:
        """Evaluate the field at a specific node"""
        raise NotImplementedError("Subclasses must implement evaluate method")
    
    def gradient(self, node_id: str, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Calculate field gradient at a specific node"""
        raise NotImplementedError("Subclasses must implement gradient method")


class AwarenessField(RelationalField):
    """
    Awareness field ρ(x,t) - scalar field representing structural presence under constraint
    """
    
    def __init__(self):
        super().__init__(
            name="awareness",
            description="Scalar field representing structural presence under constraint"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> float:
        """Get the awareness value at a specific node"""
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if node:
            return node.get('awareness', 0.0)
        return 0.0
    
    def gradient(self, node_id: str, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate awareness gradient at a specific node
        Returns a dict mapping neighbor_id -> gradient value
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return {}
        
        node_awareness = node.get('awareness', 0.0)
        relations = node.get('relations', {})
        
        gradients = {}
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if neighbor:
                neighbor_awareness = neighbor.get('awareness', 0.0)
                gradient = (neighbor_awareness - node_awareness) * relation_strength
                gradients[neighbor_id] = gradient
        
        return gradients


class PolarityField(RelationalField):
    """
    Polarity field π(x,t) - vector field governing directional interactions
    """
    
    def __init__(self):
        super().__init__(
            name="polarity",
            description="Vector field governing directional interactions between configurations"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Get the polarity vectors at a specific node"""
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if node:
            return node.get('polarity_vectors', {})
        return {}
    
    def gradient(self, node_id: str, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate polarity gradient at a specific node
        For polarity, this is ∇π(x) which represents how directional alignment changes
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return {}
        
        polarity_vectors = node.get('polarity_vectors', {})
        relations = node.get('relations', {})
        
        gradients = {}
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if not neighbor:
                continue
                
            neighbor_polarity = neighbor.get('polarity_vectors', {}).get(node_id, np.zeros(3))
            my_polarity = polarity_vectors.get(neighbor_id, np.zeros(3))
            
            # Calculate directional change
            if np.any(my_polarity) and np.any(neighbor_polarity):
                # Use normalized cross product to measure rotation
                my_norm = my_polarity / (np.linalg.norm(my_polarity) + 1e-10)
                neighbor_norm = neighbor_polarity / (np.linalg.norm(neighbor_polarity) + 1e-10)
                
                cross_product = np.cross(my_norm, neighbor_norm)
                dot_product = np.dot(my_norm, neighbor_norm)
                
                # Weighted by relation strength and alignment
                gradient = cross_product * relation_strength * abs(1 - dot_product)
                gradients[neighbor_id] = gradient
        
        return gradients


class CollapseMetricField(RelationalField):
    """
    Collapse metric field C∞(x,t) - accumulated irreversible structure
    """
    
    def __init__(self):
        super().__init__(
            name="collapse_metric",
            description="Accumulated irreversible structure over time"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> float:
        """Get the collapse metric value at a specific node"""
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if node:
            return node.get('collapse_metric', 0.0)
        return 0.0
    
    def gradient(self, node_id: str, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate collapse metric gradient at a specific node
        This represents -v→collapse(x,t), the negative collapse velocity
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return {}
        
        node_metric = node.get('collapse_metric', 0.0)
        relations = node.get('relations', {})
        
        gradients = {}
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if neighbor:
                neighbor_metric = neighbor.get('collapse_metric', 0.0)
                gradient = (neighbor_metric - node_metric) * relation_strength
                gradients[neighbor_id] = gradient
        
        return gradients


class GrainActivationField(RelationalField):
    """
    Grain activation field Γ(x,t) - measures structural contrast and readiness to collapse
    """
    
    def __init__(self, sigma: float = 0.7, eta: float = 0.3):
        super().__init__(
            name="grain_activation",
            description="Measures structural contrast and readiness to collapse"
        )
        self.sigma = sigma  # Weight for awareness gradient
        self.eta = eta      # Weight for polarity gradient
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> float:
        """Get the grain activation value at a specific node"""
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if node:
            return node.get('grain_activation', 0.0)
        return 0.0
    
    def calculate(self, node_id: str, state: Dict[str, Any]) -> float:
        """
        Calculate grain activation based on awareness and polarity gradients
        Γ(x,t) = σ|∇ρ(x,t)| + η|∇π(x,t)|
        """
        # Get the awareness field gradient
        awareness_field = AwarenessField()
        awareness_gradient = awareness_field.gradient(node_id, state)
        
        # Get the polarity field gradient
        polarity_field = PolarityField()
        polarity_gradient = polarity_field.gradient(node_id, state)
        
        # Calculate magnitudes
        awareness_magnitude = 0.0
        for gradient in awareness_gradient.values():
            awareness_magnitude += abs(gradient)
        
        polarity_magnitude = 0.0
        for gradient in polarity_gradient.values():
            if isinstance(gradient, np.ndarray):
                polarity_magnitude += np.linalg.norm(gradient)
        
        # Normalize by number of neighbors if applicable
        num_neighbors = max(1, len(awareness_gradient))
        awareness_magnitude /= num_neighbors
        polarity_magnitude /= num_neighbors
        
        # Combine for total grain activation
        grain_activation = self.sigma * awareness_magnitude + self.eta * polarity_magnitude
        
        return grain_activation


class GrainSaturationField(RelationalField):
    """
    Grain saturation field Gsat(x,t) - measures how committed a region has become
    """
    
    def __init__(self):
        super().__init__(
            name="grain_saturation",
            description="Measures how much a region has committed to structure"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> float:
        """Get the grain saturation value at a specific node"""
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if node:
            return node.get('grain_saturation', 0.0)
        return 0.0
    
    def gradient(self, node_id: str, state: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate saturation gradient at a specific node
        This represents how saturation changes across the manifold
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return {}
        
        node_saturation = node.get('grain_saturation', 0.0)
        relations = node.get('relations', {})
        
        gradients = {}
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if neighbor:
                neighbor_saturation = neighbor.get('grain_saturation', 0.0)
                gradient = (neighbor_saturation - node_saturation) * relation_strength
                gradients[neighbor_id] = gradient
        
        return gradients


class CurvatureField(RelationalField):
    """
    Curvature field κ(x) - represents the bending of collapse flow
    κ(x) = ∇²C∞(x,t) + ∇⋅(∇π(x))
    """
    
    def __init__(self):
        super().__init__(
            name="curvature",
            description="Represents the curvature of collapse flow and constraint geometry"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> float:
        """
        Calculate curvature at a specific node
        κ(x) = ∇²C∞(x,t) + ∇⋅(∇π(x))
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return 0.0
        
        # Get collapse metric field to calculate Laplacian
        collapse_field = CollapseMetricField()
        collapse_gradient = collapse_field.gradient(node_id, state)
        
        # Approximate Laplacian by looking at neighbors of neighbors
        laplacian = 0.0
        relations = node.get('relations', {})
        
        # First term: ∇²C∞(x,t)
        node_metric = node.get('collapse_metric', 0.0)
        
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if not neighbor:
                continue
                
            neighbor_metric = neighbor.get('collapse_metric', 0.0)
            metric_diff = neighbor_metric - node_metric
            laplacian += metric_diff * relation_strength
        
        if relations:
            laplacian /= len(relations)
        
        # Second term: ∇⋅(∇π(x))
        polarity_field = PolarityField()
        polarity_divergence = 0.0
        
        # For each neighbor, examine how polarity changes
        for neighbor_id, relation_strength in relations.items():
            neighbor = nodes.get(neighbor_id)
            if not neighbor:
                continue
                
            # Get my polarity vector to this neighbor
            my_polarity = node.get('polarity_vectors', {}).get(neighbor_id, np.zeros(3))
            
            # For this neighbor, check how its polarity changes to its neighbors
            neighbor_relations = neighbor.get('relations', {})
            
            for third_id, third_relation_strength in neighbor_relations.items():
                if third_id == node_id:
                    continue
                    
                third_node = nodes.get(third_id)
                if not third_node:
                    continue
                    
                neighbor_polarity = neighbor.get('polarity_vectors', {}).get(third_id, np.zeros(3))
                
                if np.any(my_polarity) and np.any(neighbor_polarity):
                    # Take dot product to measure directional alignment/divergence
                    divergence_contribution = np.dot(my_polarity, neighbor_polarity)
                    polarity_divergence += divergence_contribution * relation_strength * third_relation_strength
        
        if relations:
            num_second_order_relations = max(1, sum(len(nodes.get(nid, {}).get('relations', {})) 
                                                 for nid in relations if nid in nodes))
            polarity_divergence /= num_second_order_relations
        
        # Combine for total curvature
        return laplacian + polarity_divergence


class ElectricField(RelationalField):
    """
    Electric field E→(x,t) = -∇C∞(x,t) - collapse tension gradient
    """
    
    def __init__(self):
        super().__init__(
            name="electric_field",
            description="Collapse tension gradient, analogous to electric field"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Calculate electric field at a specific node, which is the negative
        gradient of the collapse metric, scaled by relation strength
        """
        collapse_field = CollapseMetricField()
        collapse_gradient = collapse_field.gradient(node_id, state)
        
        electric_field = {}
        for neighbor_id, gradient in collapse_gradient.items():
            # E = -∇C∞
            if isinstance(gradient, (int, float)):
                # Convert scalar gradient to 3D vector in the direction of the relation
                nodes = state.get('nodes', {})
                node = nodes.get(node_id)
                if node:
                    polarity = node.get('polarity_vectors', {}).get(neighbor_id, np.zeros(3))
                    if np.any(polarity):
                        direction = polarity / np.linalg.norm(polarity)
                        electric_field[neighbor_id] = -gradient * direction
                    else:
                        # Default direction if no polarity is defined
                        electric_field[neighbor_id] = np.array([-gradient, 0, 0])
            else:
                # Already a vector
                electric_field[neighbor_id] = -gradient
        
        return electric_field


class MagneticField(RelationalField):
    """
    Magnetic field B→(x,t) = ∇ × π→(x,t) - polarity rotation
    """
    
    def __init__(self):
        super().__init__(
            name="magnetic_field",
            description="Polarity rotation, analogous to magnetic field"
        )
    
    def evaluate(self, node_id: str, state: Dict[str, Any]) -> np.ndarray:
        """
        Calculate magnetic field at a specific node, which is the curl of the polarity field
        B→(x,t) = ∇ × π→(x,t)
        """
        nodes = state.get('nodes', {})
        node = nodes.get(node_id)
        if not node:
            return np.zeros(3)
        
        # Get polarity vectors for this node
        polarity_vectors = node.get('polarity_vectors', {})
        relations = node.get('relations', {})
        
        if not polarity_vectors or len(polarity_vectors) < 2:
            return np.zeros(3)
        
        # Approximate curl by looking at how polarity rotates around this node
        # For a true curl, we'd need a proper coordinate system, but we're in a relational manifold
        # So we'll estimate it by looking at the circulation of polarity vectors
        curl = np.zeros(3)
        
        # Get all neighbor pairs to calculate circulation
        neighbor_ids = list(relations.keys())
        
        for i in range(len(neighbor_ids)):
            id1 = neighbor_ids[i]
            for j in range(i+1, len(neighbor_ids)):
                id2 = neighbor_ids[j]
                
                if id1 not in polarity_vectors or id2 not in polarity_vectors:
                    continue
                
                # Get polarity vectors
                pol1 = polarity_vectors[id1]
                pol2 = polarity_vectors[id2]
                
                # Calculate contribution to curl (cross product)
                if np.any(pol1) and np.any(pol2):
                    circulation = np.cross(pol1, pol2)
                    rel_strength = relations[id1] * relations[id2]
                    curl += circulation * rel_strength
        
        # Normalize by number of neighbor pairs
        num_pairs = max(1, len(neighbor_ids) * (len(neighbor_ids) - 1) // 2)
        curl /= num_pairs
        
        return curl


# Dictionary of all field types
FIELD_TYPES = {
    'awareness': AwarenessField,
    'polarity': PolarityField,
    'collapse_metric': CollapseMetricField,
    'grain_activation': GrainActivationField,
    'grain_saturation': GrainSaturationField,
    'curvature': CurvatureField,
    'electric': ElectricField,
    'magnetic': MagneticField
}


def create_field(field_type: str, **kwargs) -> RelationalField:
    """Factory function to create a field of the specified type"""
    if field_type not in FIELD_TYPES:
        raise ValueError(f"Unknown field type: {field_type}")
    
    field_class = FIELD_TYPES[field_type]
    return field_class(**kwargs)