"""
Polarity Space - Core representation of directional alignment and interaction geometry

Implements the polarity space P using the Collapse Epistemology Tensor approach,
representing directional interactions and memory inheritance in the Collapse Geometry framework
without fixed vector coordinates.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
import uuid


class EpistomologyRelation:
    """
    Represents a relational memory connection between configurations using the
    Collapse Epistemology Tensor E(x,t) = (R, F, Φ) approach.
    
    Components:
    - strength: Basic directional memory (replaces vector-based polarity)
    - resolution: How well tension is resolving (R)
    - frustration: How blocked contradiction is (F)
    - fidelity: How aligned collapse is with memory/ancestry (Φ)
    """
    
    def __init__(self, strength: float = 0.0, resolution: float = 0.5, 
               frustration: float = 0.0, fidelity: float = 0.5,
               orientation: float = 0.0):
        """
        Initialize a relation with epistemology components.
        
        Args:
            strength: Basic relation strength and direction, -1.0 to 1.0
            resolution: How well tension resolves, 0.0 to 1.0
            frustration: How blocked contradiction is, 0.0 to 1.0
            fidelity: How aligned collapse is with memory, 0.0 to 1.0
            orientation: Angular orientation in radians, 0.0 to 2π
        """
        self.strength = strength
        self.resolution = self._clamp(resolution)
        self.frustration = self._clamp(frustration)
        self.fidelity = self._clamp(fidelity)
        self.orientation = orientation % (2 * math.pi)  # Ensure 0 to 2π range
        self._flow_history = []  # Track flow history for phase continuity
    
    def _clamp(self, value: float) -> float:
        """Clamp a value to range [0.0, 1.0]"""
        return max(0.0, min(1.0, value))
    
    def update(self, new_strength: float = None, new_resolution: float = None,
             new_frustration: float = None, new_fidelity: float = None,
             new_orientation: float = None, blending_factor: float = 0.2):
        """
        Update the relation with new epistemology values using smooth blending.
        Only updates components that are provided (not None).
        
        Args:
            new_strength: Updated strength value
            new_resolution: Updated resolution value
            new_frustration: Updated frustration value
            new_fidelity: Updated fidelity value
            new_orientation: Updated angular orientation
            blending_factor: How quickly the relation updates (0-1)
        """
        if new_strength is not None:
            self.strength = (1 - blending_factor) * self.strength + blending_factor * new_strength
            
        if new_resolution is not None:
            new_value = (1 - blending_factor) * self.resolution + blending_factor * new_resolution
            self.resolution = self._clamp(new_value)
            
        if new_frustration is not None:
            new_value = (1 - blending_factor) * self.frustration + blending_factor * new_frustration
            self.frustration = self._clamp(new_value)
            
        if new_fidelity is not None:
            new_value = (1 - blending_factor) * self.fidelity + blending_factor * new_fidelity
            self.fidelity = self._clamp(new_value)
            
        if new_orientation is not None:
            # Calculate the smallest angle difference to ensure proper blending around the circle
            angle_diff = ((new_orientation - self.orientation + math.pi) % (2 * math.pi)) - math.pi
            self.orientation = (self.orientation + blending_factor * angle_diff) % (2 * math.pi)
    
    def get_flow_tendency(self) -> float:
        """
        Calculate flow tendency from epistemology components.
        Flow = Resolution * (1-Frustration) * Fidelity * Strength
        
        Returns:
            Flow tendency value
        """
        # Calculate transmissibility from resolution and frustration
        transmissibility = self.resolution * (1.0 - self.frustration)
        
        # Calculate flow as transmissibility * fidelity * strength
        flow = transmissibility * self.fidelity * self.strength
        
        # Save to history for phase tracking
        self._flow_history.append(flow)
        if len(self._flow_history) > 20:  # Keep history limited
            self._flow_history.pop(0)
            
        return flow
    
    def get_backflow_potential(self) -> float:
        """
        Calculate backflow potential from epistemology components.
        High frustration + low resolution = backflow potential
        
        Returns:
            Backflow potential value
        """
        return -self.strength * self.frustration * (1.0 - self.resolution)
    
    def is_aligned_with(self, other: 'EpistomologyRelation', threshold: float = 0.7) -> bool:
        """
        Check if this relation is aligned with another in terms of flow direction.
        
        Args:
            other: Another relation to compare with
            threshold: Alignment threshold
            
        Returns:
            True if relations are aligned, False otherwise
        """
        # Calculate flow tendencies
        flow1 = self.get_flow_tendency()
        flow2 = other.get_flow_tendency()
        
        # Check if flows have same direction (both positive or both negative)
        # and have significant magnitude
        if abs(flow1) > 0.1 and abs(flow2) > 0.1:
            if flow1 * flow2 > 0:  # Same sign = aligned
                # Calculate similarity as normalized dot product equivalent
                similarity = min(abs(flow1), abs(flow2)) / max(abs(flow1), abs(flow2))
                return similarity > threshold
        
        return False
    
    def is_opposed_to(self, other: 'EpistomologyRelation', threshold: float = 0.7) -> bool:
        """
        Check if this relation is opposed to another in terms of flow direction.
        
        Args:
            other: Another relation to compare with
            threshold: Opposition threshold
            
        Returns:
            True if relations are opposed, False otherwise
        """
        # Calculate flow tendencies
        flow1 = self.get_flow_tendency()
        flow2 = other.get_flow_tendency()
        
        # Check if flows have opposite directions (opposite signs)
        # and have significant magnitude
        if abs(flow1) > 0.1 and abs(flow2) > 0.1:
            if flow1 * flow2 < 0:  # Opposite signs = opposed
                # Calculate similarity as normalized dot product equivalent
                similarity = min(abs(flow1), abs(flow2)) / max(abs(flow1), abs(flow2))
                return similarity > threshold
        
        return False
    
    def get_orientation_vector(self) -> np.ndarray:
        """
        Convert orientation angle to a unit vector.
        
        Returns:
            Unit vector [x, y, z] where z is a third dimension allowing for
            greater expressivity in relational rotations
        """
        # For 2D orientation, we use x and y components
        x = math.cos(self.orientation)
        y = math.sin(self.orientation)
        
        # Using strength to modulate the z component allows for richer topological structure
        z = self.strength * 0.5  # Scale to keep within reasonable range
        
        # Normalize to ensure it's a unit vector
        vector = np.array([x, y, z])
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def __repr__(self):
        return (f"EpistomologyRelation(strength={self.strength:.2f}, "
              f"resolution={self.resolution:.2f}, "
              f"frustration={self.frustration:.2f}, "
              f"fidelity={self.fidelity:.2f}, "
              f"orientation={self.orientation:.2f})")


class RelativeRotationTensor:
    """
    Implements the Relative Rotation Tensor ΔΘᵢⱼ(t)
    
    Encodes the angular displacement in polarity orientation between two 
    collapse-linked nodes i and j at time t, modulo 2π. It captures the local 
    rotational curvature of the polarity field and provides a fully relational 
    representation of angular structure across the manifold.
    """
    
    def __init__(self):
        self.relations = {}  # Maps (source_id, target_id) -> rotation angle (radians)
        self.phase_continuity = {}  # Maps node_id -> accumulated phase
    
    def calculate_rotation(self, relation1: EpistomologyRelation, relation2: EpistomologyRelation) -> float:
        """
        Calculate the relative rotation angle between two epistemology relations.
        
        Args:
            relation1: First relation
            relation2: Second relation
            
        Returns:
            Relative rotation angle in radians (0 to 2π)
        """
        # Get orientation vectors
        vector1 = relation1.get_orientation_vector()
        vector2 = relation2.get_orientation_vector()
        
        # Calculate the dot product
        dot_product = np.dot(vector1, vector2)
        dot_product = max(-1.0, min(1.0, dot_product))  # Clamp to [-1, 1]
        
        # Calculate the angle between vectors
        angle = math.acos(dot_product)
        
        # Determine the direction of rotation using cross product
        cross_product = np.cross(vector1, vector2)
        direction = np.sign(cross_product[2]) if len(cross_product) > 2 else 1.0
        
        # Adjust to get the signed angle in the range [0, 2π)
        rotation = (angle * direction) % (2 * math.pi)
        
        return rotation
    
    def update(self, source_id: str, target_id: str, relation1: EpistomologyRelation, 
             relation2: EpistomologyRelation, time_delta: float = 1.0):
        """
        Update the rotation tensor for a pair of nodes and their relations.
        
        Args:
            source_id: First node ID
            target_id: Second node ID
            relation1: Relation from source to target
            relation2: Relation from target to source
            time_delta: Time step for phase accumulation
        """
        # Calculate the relative rotation angle
        rotation = self.calculate_rotation(relation1, relation2)
        
        # Store in the tensor
        self.relations[(source_id, target_id)] = rotation
        self.relations[(target_id, source_id)] = (2 * math.pi - rotation) % (2 * math.pi)
        
        # Update phase continuity for both nodes
        if source_id not in self.phase_continuity:
            self.phase_continuity[source_id] = 0.0
        if target_id not in self.phase_continuity:
            self.phase_continuity[target_id] = 0.0
        
        # Accumulate phase based on rotation and relation strength
        flow1 = relation1.get_flow_tendency()
        flow2 = relation2.get_flow_tendency()
        
        # Weight by flow tendency and time delta
        phase_contribution = rotation * (abs(flow1) + abs(flow2)) * 0.5 * time_delta
        
        # Accumulate phase (modulo larger value to track winding numbers)
        self.phase_continuity[source_id] = (self.phase_continuity[source_id] + phase_contribution) % (4 * math.pi)
        self.phase_continuity[target_id] = (self.phase_continuity[target_id] + phase_contribution) % (4 * math.pi)
    
    def get_rotation(self, source_id: str, target_id: str) -> float:
        """
        Get the rotation angle between two nodes.
        
        Args:
            source_id: First node ID
            target_id: Second node ID
            
        Returns:
            Rotation angle in radians (0 to 2π) or 0.0 if not found
        """
        return self.relations.get((source_id, target_id), 0.0)
    
    def get_phase_continuity(self, node_id: str) -> float:
        """
        Get the accumulated phase continuity for a node.
        
        Args:
            node_id: Node ID
            
        Returns:
            Accumulated phase in radians or 0.0 if not found
        """
        return self.phase_continuity.get(node_id, 0.0)
    
    def calculate_rotational_curvature(self, node_id: str, neighbor_ids: List[str]) -> float:
        """
        Calculate rotational curvature (κ_rot) for a node and its neighbors.
        
        κ_rot(i) = ∑ⱼ∈ₙ(i) ΔΘᵢⱼ
        
        Args:
            node_id: Center node ID
            neighbor_ids: List of neighbor node IDs
            
        Returns:
            Rotational curvature scalar
        """
        curvature = 0.0
        
        for neighbor_id in neighbor_ids:
            rotation = self.get_rotation(node_id, neighbor_id)
            curvature += rotation
        
        # Normalize by the number of neighbors
        if neighbor_ids:
            curvature /= len(neighbor_ids)
        
        return curvature
    
    def detect_vortices(self, nodes: Dict[str, Any], threshold: float = 0.5 * math.pi) -> List[Dict[str, Any]]:
        """
        Detect vortex-like structures based on rotational curvature.
        
        Args:
            nodes: Dictionary mapping node_id -> node
            threshold: Minimum rotational curvature to qualify as a vortex
            
        Returns:
            List of vortex data dictionaries
        """
        vortices = []
        
        for node_id, node in nodes.items():
            # Get neighbors
            neighbor_ids = list(getattr(node, 'relations', {}).keys())
            
            if len(neighbor_ids) >= 3:  # Need at least 3 neighbors for a meaningful vortex
                # Calculate rotational curvature
                curvature = self.calculate_rotational_curvature(node_id, neighbor_ids)
                
                # Check if it exceeds the threshold
                if abs(curvature) > threshold:
                    # Get average rotation
                    total_rotation = 0.0
                    for neighbor_id in neighbor_ids:
                        total_rotation += self.get_rotation(node_id, neighbor_id)
                    
                    avg_rotation = total_rotation / len(neighbor_ids)
                    
                    # Determine rotation direction
                    rotation_direction = "clockwise" if curvature > 0 else "counterclockwise"
                    
                    # Add to vortices list
                    vortices.append({
                        'center_node': node_id,
                        'strength': abs(curvature),
                        'rotation_direction': rotation_direction,
                        'average_rotation': avg_rotation,
                        'neighbor_count': len(neighbor_ids)
                    })
        
        return vortices


class EpistomologyField:
    """
    Represents the polarity field using the Collapse Epistemology Tensor approach.
    Maps relationships between configuration points to directional epistemology relations.
    """
    
    def __init__(self):
        self.relations = {}  # Maps (source_id, target_id) -> EpistomologyRelation
        self.rotation_tensor = RelativeRotationTensor()
        self.time = 0.0
    
    def set_relation(self, source_id: str, target_id: str, 
                   strength: float = 0.0, resolution: float = 0.5,
                   frustration: float = 0.0, fidelity: float = 0.5,
                   orientation: float = None):
        """
        Set the epistemology relation from source to target
        
        Args:
            source_id: Origin point ID
            target_id: Target point ID
            strength: Basic relation strength and direction
            resolution: How well tension resolves
            frustration: How blocked contradiction is
            fidelity: How aligned collapse is with memory
            orientation: Angular orientation in radians (if None, calculated from strength)
        """
        # If orientation is not provided, calculate it from strength
        if orientation is None:
            # Map strength [-1.0, 1.0] to orientation [0, 2π)
            # This creates a basic mapping, but you may want a more sophisticated one
            orientation = (math.atan2(strength, 0.5) + math.pi) % (2 * math.pi)
        
        relation_key = (source_id, target_id)
        self.relations[relation_key] = EpistomologyRelation(
            strength=strength,
            resolution=resolution,
            frustration=frustration,
            fidelity=fidelity,
            orientation=orientation
        )
    
    def get_relation(self, source_id: str, target_id: str) -> Optional[EpistomologyRelation]:
        """Get the epistemology relation from source to target"""
        relation_key = (source_id, target_id)
        return self.relations.get(relation_key)
    
    def update_relation(self, source_id: str, target_id: str,
                      strength: float = None, resolution: float = None,
                      frustration: float = None, fidelity: float = None,
                      orientation: float = None, blending_factor: float = 0.2):
        """
        Update an existing relation with new epistemology values
        
        Args:
            source_id: Origin point ID
            target_id: Target point ID
            strength, resolution, frustration, fidelity, orientation: New values (None = no change)
            blending_factor: How quickly relation updates
        """
        relation = self.get_relation(source_id, target_id)
        if relation is None:
            # Create new relation if it doesn't exist
            self.set_relation(
                source_id, target_id,
                strength=strength or 0.0,
                resolution=resolution or 0.5,
                frustration=frustration or 0.0,
                fidelity=fidelity or 0.5,
                orientation=orientation
            )
        else:
            # Update existing relation
            relation.update(
                new_strength=strength,
                new_resolution=resolution,
                new_frustration=frustration,
                new_fidelity=fidelity,
                new_orientation=orientation,
                blending_factor=blending_factor
            )
    
    def collapse_update(self, source_id: str, target_id: str, 
                      collapse_strength: float,
                      saturation: float,
                      ancestry_similarity: float = 0.0,
                      inheritance_strength: float = 0.8,
                      time_delta: float = 1.0):
        """
        Update relation through continuous collapse inheritance
        
        Args:
            source_id: Origin point of collapse
            target_id: Target point of collapse
            collapse_strength: Strength of collapse relation
            saturation: Grain saturation at source (0-1)
            ancestry_similarity: Similarity of ancestry between nodes (0-1)
            inheritance_strength: Strength of inheritance effect
            time_delta: Time step for phase accumulation and tensor updates
        """
        # Update simulation time
        self.time += time_delta
        
        # Calculate memory-weighted inheritance
        memory_factor = 1.0 - saturation
        inheritance_weight = memory_factor * inheritance_strength
        
        # Calculate epistemology components
        
        # 1. Relation strength - directional memory
        strength = collapse_strength
        
        # 2. Resolution - how well tension resolves
        # Higher when collapse is strong (successful resolution)
        resolution = 0.5 + 0.5 * abs(collapse_strength)
        
        # 3. Frustration - how blocked contradiction is
        # Lower when collapse strength is higher (less frustration)
        # Higher when source is highly saturated (more blocked)
        frustration = 0.5 - 0.3 * abs(collapse_strength) + 0.3 * saturation
        
        # 4. Fidelity - how aligned with memory/ancestry
        # Higher when ancestry is similar (compatible history)
        fidelity = 0.5 + 0.5 * ancestry_similarity
        
        # 5. Orientation - calculate from collapse strength and existing orientation
        # This allows orientation to evolve based on collapse dynamics
        source_target_relation = self.get_relation(source_id, target_id)
        target_source_relation = self.get_relation(target_id, source_id)
        
        # Calculate new orientation based on collapse strength
        # Stronger collapse = more definitive orientation change
        if source_target_relation:
            # Base the new orientation on the current one, but allow it to evolve
            current_orientation = source_target_relation.orientation
            orientation_shift = collapse_strength * math.pi * 0.1  # Small shift based on strength
            new_orientation = (current_orientation + orientation_shift) % (2 * math.pi)
        else:
            # Initialize with a basic orientation if relation doesn't exist
            new_orientation = (math.atan2(collapse_strength, 0.5) + math.pi) % (2 * math.pi)
        
        # Update source->target relation
        self.update_relation(
            source_id, target_id,
            strength=strength,
            resolution=resolution,
            frustration=frustration,
            fidelity=fidelity,
            orientation=new_orientation,
            blending_factor=inheritance_weight
        )
        
        # Create or update target->source relation (opposite direction)
        opposite_strength = -strength
        
        # Resolution is slightly reduced in reverse
        opposite_resolution = resolution * 0.9
        
        # Frustration is slightly increased in reverse
        opposite_frustration = frustration * 1.1
        
        # Fidelity is the same in both directions
        opposite_fidelity = fidelity
        
        # Orientation is opposite (shifted by π)
        opposite_orientation = (new_orientation + math.pi) % (2 * math.pi)
        
        self.update_relation(
            target_id, source_id,
            strength=opposite_strength,
            resolution=opposite_resolution,
            frustration=opposite_frustration,
            fidelity=opposite_fidelity,
            orientation=opposite_orientation,
            blending_factor=inheritance_weight * 0.8  # Slower update in reverse
        )
        
        # Update rotation tensor
        updated_source_target = self.get_relation(source_id, target_id)
        updated_target_source = self.get_relation(target_id, source_id)
        
        if updated_source_target and updated_target_source:
            self.rotation_tensor.update(
                source_id, target_id,
                updated_source_target, updated_target_source,
                time_delta
            )
    
    def calculate_flow_tendency(self, point_id: str, related_ids: List[str]) -> Dict[str, float]:
        """
        Calculate flow tendencies from a point to its related points
        based on epistemology relations.
        
        Args:
            point_id: Center point ID
            related_ids: IDs of related points
            
        Returns:
            Dictionary mapping related_id -> flow_tendency
        """
        flow_tendencies = {}
        
        for related_id in related_ids:
            relation = self.get_relation(point_id, related_id)
            if relation:
                flow_tendencies[related_id] = relation.get_flow_tendency()
            else:
                flow_tendencies[related_id] = 0.0
        
        return flow_tendencies
    
    def calculate_backflow_potential(self, point_id: str, related_ids: List[str]) -> Dict[str, float]:
        """
        Calculate backflow potential from a point to its related points
        
        Args:
            point_id: Center point ID
            related_ids: IDs of related points
            
        Returns:
            Dictionary mapping related_id -> backflow_potential
        """
        backflow_potentials = {}
        
        for related_id in related_ids:
            relation = self.get_relation(point_id, related_id)
            if relation:
                backflow_potentials[related_id] = relation.get_backflow_potential()
            else:
                backflow_potentials[related_id] = 0.0
        
        return backflow_potentials
    
    def calculate_field_contour(self, point_id: str, related_ids: List[str]) -> float:
        """
        Calculate field contour at a point based on epistemology relations.
        Contour measures consistency of relations across neighbors.
        
        Args:
            point_id: Center point ID
            related_ids: IDs of related points
            
        Returns:
            Contour value at the point
        """
        if not related_ids:
            return 0.0
            
        # Collect epistemology components across relations
        resolutions = []
        frustrations = []
        fidelities = []
        flow_tendencies = []
        
        for related_id in related_ids:
            relation = self.get_relation(point_id, related_id)
            if relation:
                resolutions.append(relation.resolution)
                frustrations.append(relation.frustration)
                fidelities.append(relation.fidelity)
                flow_tendencies.append(relation.get_flow_tendency())
        
        if not resolutions:
            return 0.0
            
        # Calculate variance of each component
        def variance(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return sum((v - mean) ** 2 for v in values) / len(values)
        
        resolution_variance = variance(resolutions)
        frustration_variance = variance(frustrations)
        fidelity_variance = variance(fidelities)
        flow_variance = variance(flow_tendencies)
        
        # Contour is inverse of variance (high contour = consistent field)
        # Weighted average of component variances
        total_variance = (
            0.3 * resolution_variance +
            0.3 * frustration_variance +
            0.2 * fidelity_variance +
            0.2 * flow_variance
        )
        
        contour = 1.0 / (1.0 + 5.0 * total_variance)
        
        return contour
    
    def calculate_rotational_curvature(self, point_id: str, related_ids: List[str]) -> float:
        """
        Calculate rotational curvature for a point and its neighbors
        
        Args:
            point_id: Center point ID
            related_ids: IDs of related points
            
        Returns:
            Rotational curvature value
        """
        return self.rotation_tensor.calculate_rotational_curvature(point_id, related_ids)
    
    def get_phase_continuity(self, point_id: str) -> float:
        """
        Get accumulated phase continuity for a point
        
        Args:
            point_id: Point ID
            
        Returns:
            Phase continuity value
        """
        return self.rotation_tensor.get_phase_continuity(point_id)
    
    def detect_vortices(self, nodes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect vortex-like structures based on rotational curvature
        
        Args:
            nodes: Dictionary of nodes
            
        Returns:
            List of vortex information dictionaries
        """
        return self.rotation_tensor.detect_vortices(nodes)
    
    def find_coherent_regions(self, threshold: float = 0.7) -> List[Set[str]]:
        """
        Identify regions with coherent epistemology patterns.
        These represent areas where relations form consistent patterns.
        
        Args:
            threshold: Coherence threshold
            
        Returns:
            List of sets containing coherent point IDs
        """
        # Extract all unique point IDs
        all_points = set()
        for source, target in self.relations:
            all_points.add(source)
            all_points.add(target)
        
        # Start with each point in its own region
        regions = [{point} for point in all_points]
        
        # Merge regions based on relation coherence
        merged = True
        while merged:
            merged = False
            new_regions = []
            
            while regions:
                current = regions.pop(0)
                
                # Check if current region can merge with any remaining region
                merged_region = False
                for i, other in enumerate(regions):
                    # Check coherence between regions
                    coherent = False
                    
                    for point_id in current:
                        for other_id in other:
                            # Check for coherent relations between points
                            relation1 = self.get_relation(point_id, other_id)
                            relation2 = self.get_relation(other_id, point_id)
                            
                            if relation1 and relation2:
                                # Check alignment between relations
                                if relation1.is_aligned_with(relation2, threshold):
                                    coherent = True
                                    break
                        
                        if coherent:
                            break
                    
                    if coherent:
                        # Merge regions
                        merged_region = current.union(other)
                        regions.pop(i)
                        regions.append(merged_region)
                        merged = True
                        break
                
                if not merged_region:
                    new_regions.append(current)
            
            regions = new_regions
        
        return regions
    
    def find_opposition_pairs(self, threshold: float = 0.7) -> List[Tuple[str, str]]:
        """
        Identify pairs of points with opposing epistemology patterns.
        These represent potential duality relationships.
        
        Args:
            threshold: Opposition threshold
            
        Returns:
            List of (point_id1, point_id2) tuples with opposing relations
        """
        opposition_pairs = []
        
        # Get all unique points
        all_points = set()
        for source, target in self.relations:
            all_points.add(source)
            all_points.add(target)
        
        # Check all possible point pairs
        checked_pairs = set()
        
        for point_id1 in all_points:
            for point_id2 in all_points:
                if point_id1 == point_id2:
                    continue
                    
                # Skip if already checked (either direction)
                if (point_id1, point_id2) in checked_pairs or (point_id2, point_id1) in checked_pairs:
                    continue
                    
                checked_pairs.add((point_id1, point_id2))
                
                # Get relations in both directions
                relation1 = self.get_relation(point_id1, point_id2)
                relation2 = self.get_relation(point_id2, point_id1)
                
                if relation1 and relation2:
                    # Check for opposition
                    if relation1.is_opposed_to(relation2, threshold):
                        opposition_pairs.append((point_id1, point_id2))
                        
                        # Check shared relations for pattern confirmation
                        shared_relations = 0
                        opposing_shared = 0
                        
                        for other_id in all_points:
                            if other_id == point_id1 or other_id == point_id2:
                                continue
                                
                            # Check relations from both points to this third point
                            rel1_other = self.get_relation(point_id1, other_id)
                            rel2_other = self.get_relation(point_id2, other_id)
                            
                            if rel1_other and rel2_other:
                                shared_relations += 1
                                
                                # Check if these relations are also opposed
                                if rel1_other.is_opposed_to(rel2_other, threshold):
                                    opposing_shared += 1
                        
                        # Strong opposition pairs have multiple shared oppositions
                        if shared_relations > 0 and opposing_shared > 0:
                            # Upgrade to strong opposition (add again to mark strength)
                            opposition_pairs.append((point_id1, point_id2))
        
        return opposition_pairs
    
    def analyze_toroidal_structure(self, nodes: Dict[str, Any], slice_count: int = 12) -> Dict[str, Any]:
        """
        Analyze emergent toroidal structure by dividing into angular slices
        
        Args:
            nodes: Dictionary of nodes
            slice_count: Number of slices to divide the torus into
            
        Returns:
            Dictionary with toroidal analysis results
        """
        # Initialize slice containers
        theta_slices = [{} for _ in range(slice_count)]  # Slices around the major circle
        phi_slices = [{} for _ in range(slice_count)]    # Slices around the minor circle
        
        # Classify nodes into slices based on phase continuity
        for node_id, node in nodes.items():
            phase = self.get_phase_continuity(node_id)
            
            # Map to theta (0 to 2π) - major circle
            theta = phase % (2 * math.pi)
            theta_index = min(slice_count - 1, int(theta / (2 * math.pi) * slice_count))
            
            # Map to phi (0 to 2π) - minor circle
            # For phi, we'll use the rotational curvature
            related_ids = list(getattr(node, 'relations', {}).keys())
            curvature = self.calculate_rotational_curvature(node_id, related_ids)
            phi = (curvature + math.pi) % (2 * math.pi)  # Normalize to [0, 2π)
            phi_index = min(slice_count - 1, int(phi / (2 * math.pi) * slice_count))
            
            # Add node to slices
            theta_slices[theta_index][node_id] = node
            phi_slices[phi_index][node_id] = node
        
        # Calculate metrics for each slice
        theta_metrics = []
        phi_metrics = []
        
        for slice_nodes in theta_slices:
            if slice_nodes:
                # Calculate average values for this slice
                avg_awareness = sum(getattr(node, 'awareness', 0.0) for node in slice_nodes.values()) / len(slice_nodes)
                avg_saturation = sum(getattr(node, 'grain_saturation', 0.0) for node in slice_nodes.values()) / len(slice_nodes)
                
                theta_metrics.append({
                    'node_count': len(slice_nodes),
                    'avg_awareness': avg_awareness,
                    'avg_saturation': avg_saturation
                })
            else:
                theta_metrics.append({
                    'node_count': 0,
                    'avg_awareness': 0.0,
                    'avg_saturation': 0.0
                })
        
        for slice_nodes in phi_slices:
            if slice_nodes:
                # Calculate average values for this slice
                avg_awareness = sum(getattr(node, 'awareness', 0.0) for node in slice_nodes.values()) / len(slice_nodes)
                avg_saturation = sum(getattr(node, 'grain_saturation', 0.0) for node in slice_nodes.values()) / len(slice_nodes)
                
                phi_metrics.append({
                    'node_count': len(slice_nodes),
                    'avg_awareness': avg_awareness,
                    'avg_saturation': avg_saturation
                })
            else:
                phi_metrics.append({
                    'node_count': 0,
                    'avg_awareness': 0.0,
                    'avg_saturation': 0.0
                })
        
        # Detect mode patterns using FFT
        theta_node_counts = [m['node_count'] for m in theta_metrics]
        phi_node_counts = [m['node_count'] for m in phi_metrics]
        
        # Calculate FFT to identify patterns
        theta_fft = np.abs(np.fft.fft(theta_node_counts))
        phi_fft = np.abs(np.fft.fft(phi_node_counts))
        
        # Find dominant frequencies (modes)
        theta_modes = []
        phi_modes = []
        
        # Skip the DC component (index 0)
        for i in range(1, len(theta_fft) // 2):
            if theta_fft[i] > np.mean(theta_fft) + np.std(theta_fft):
                theta_modes.append((i, float(theta_fft[i])))
        
        for i in range(1, len(phi_fft) // 2):
            if phi_fft[i] > np.mean(phi_fft) + np.std(phi_fft):
                phi_modes.append((i, float(phi_fft[i])))
        
        # Sort modes by strength
        theta_modes.sort(key=lambda x: x[1], reverse=True)
        phi_modes.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'theta_slices': theta_metrics,
            'phi_slices': phi_metrics,
            'theta_modes': theta_modes,
            'phi_modes': phi_modes,
            'has_theta_pattern': len(theta_modes) > 0,
            'has_phi_pattern': len(phi_modes) > 0,
            'dominant_theta_mode': theta_modes[0][0] if theta_modes else 0,
            'dominant_phi_mode': phi_modes[0][0] if phi_modes else 0
        }