"""
Configuration Space - Manages the geometry of configurations and their compatibility on a torus

This module implements the configuration space where geometric structure, compatibility,
and tension between nodes are defined and managed on a toroidal manifold.
This space represents the geometric complement to the relational manifold.

Enhanced with support for void regions, decay particles, and toroidal referencing.
"""

import math
import numpy as np
import uuid
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from collections import defaultdict, deque

# Import dictionary templates from emergent field rules
from axiom7.collapse_rules.emergent_field_rules import (
    VOID_REGION_TEMPLATE,
    DECAY_PARTICLE_TEMPLATE,
    VoidDecayRule
)

def angular_difference(a: float, b: float) -> float:
    """Calculate the minimum angular difference between two angles, respecting wraparound"""
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)

def toroidal_distance(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Calculate distance between two points on a torus"""
    theta_diff = angular_difference(theta1, theta2)
    phi_diff = angular_difference(phi1, phi2)
    return math.sqrt(theta_diff**2 + phi_diff**2)


class ConfigurationPoint:
    """
    Represents a point in configuration space corresponding to a grain.
    Stores geometric information and calculates compatibility with other points.
    Enhanced with toroidal coordinates for proper topological representation.
    """
    
    def __init__(self, point_id: str, dimensions: int = 4, theta: float = None, phi: float = None):
        """
        Initialize a configuration point.
        
        Args:
            point_id: Unique identifier for this point
            dimensions: Number of dimensions in the configuration space
            theta: Major circle coordinate on torus (0 to 2π), randomized if None
            phi: Minor circle coordinate on torus (0 to 2π), randomized if None
        """
        self.id = point_id
        self.dimensions = dimensions
        
        # Toroidal coordinates (primary representation for topological structure)
        self.theta = theta if theta is not None else random.random() * 2 * math.pi
        self.phi = phi if phi is not None else random.random() * 2 * math.pi
        
        # Geometric properties (maintained for compatibility with existing code)
        self.position = np.random.random(dimensions) * 2 - 1  # Range [-1, 1]
        self.orientation = np.random.random(dimensions) * 2 - 1
        self.orientation = self.orientation / (np.linalg.norm(self.orientation) + 1e-10)  # Normalize
        
        # Update position to reflect toroidal coordinates for first two dimensions
        self.position[0] = math.cos(self.theta) * (2 + math.cos(self.phi)) # Major circle + minor circle
        self.position[1] = math.sin(self.theta) * (2 + math.cos(self.phi)) # Major circle + minor circle
        self.position[2] = math.sin(self.phi)  # Height on minor circle
        
        # Field properties
        self.awareness = 0.0
        self.collapse_metric = 0.0
        self.grain_activation = 0.0
        
        # Structural properties
        self.connections = {}  # Maps target_id -> (compatibility, tension)
        self.structural_tension = 0.0  # Overall tension from incompatible connections
        
        # Void presence (for void-decay mechanism)
        self.void_presence = 0.0
        
        # Toroidal field metrics
        self.phase_stability = 0.5  # How stable this point's phase is (0-1)
        self.toroidal_curvature = 0.0  # Local curvature in the toroidal field
    
    def calculate_compatibility(self, other: 'ConfigurationPoint') -> float:
        """
        Calculate compatibility with another configuration point using toroidal metrics.
        
        Higher values indicate better compatibility, range [0, 1].
        
        Args:
            other: The other configuration point
            
        Returns:
            Compatibility value
        """
        # Toroidal compatibility (primary method)
        theta_diff = angular_difference(self.theta, other.theta)
        phi_diff = angular_difference(self.phi, other.phi)
        
        # Angular distance on torus, normalized to [0, 1] range
        # Maximum possible distance is sqrt(2*pi^2), normalize by pi*sqrt(2)
        angular_dist = math.sqrt(theta_diff**2 + phi_diff**2)
        normalized_dist = angular_dist / (math.pi * math.sqrt(2))
        
        # Higher compatibility for points closer on the torus
        toroidal_compatibility = 1.0 - normalized_dist
        
        # Orientation alignment (secondary factor)
        orientation_alignment = np.abs(np.dot(self.orientation, other.orientation))
        
        # Combine both factors with emphasis on toroidal compatibility
        compatibility = 0.7 * toroidal_compatibility + 0.3 * orientation_alignment
        
        return max(0.0, min(1.0, compatibility))
    
    def calculate_tension(self, other: 'ConfigurationPoint') -> float:
        """
        Calculate tension with another configuration point using toroidal metrics.
        
        Higher values indicate more tension (incompatibility), range [0, 1].
        
        Args:
            other: The other configuration point
            
        Returns:
            Tension value
        """
        # Geometric tension based on toroidal incompatibility
        compatibility = self.calculate_compatibility(other)
        base_tension = 1.0 - compatibility
        
        # Phase misalignment increases tension
        phase_diff = abs(self.phase_stability - other.phase_stability)
        phase_tension = 0.4 * phase_diff
        
        # Awareness difference increases tension
        awareness_diff = abs(self.awareness - other.awareness)
        awareness_tension = 0.3 * awareness_diff
        
        # Calculate curvature difference (tension from toroidal curvature mismatch)
        curvature_diff = abs(self.toroidal_curvature - other.toroidal_curvature)
        curvature_tension = 0.2 * min(1.0, curvature_diff / 0.5)
        
        # Combine tension factors with phase and curvature components
        tension = 0.5 * base_tension + 0.2 * phase_tension + 0.2 * awareness_tension + 0.1 * curvature_tension
        
        return max(0.0, min(1.0, tension))
    
    def update_structural_tension(self, increment: float) -> bool:
        """
        Update structural tension and check if it exceeds threshold for void formation.
        
        Args:
            increment: Amount to increase tension by
            
        Returns:
            True if void should form, False otherwise
        """
        # Update tension
        self.structural_tension = min(1.0, self.structural_tension + increment)
        
        # Get void-decay rule to check threshold
        void_decay_rule = VoidDecayRule()
        
        # Check if void should form
        return void_decay_rule.should_form_void(self.structural_tension)
    
    def adjust_position_toroidal(self, other: 'ConfigurationPoint', compatibility: float, 
                               learning_rate: float = 0.1) -> None:
        """
        Adjust toroidal position based on compatibility with another point.
        
        Args:
            other: The other configuration point
            compatibility: Current compatibility value
            learning_rate: How quickly position adapts
        """
        # Calculate desired compatibility based on connections
        target_compatibility = self.connections.get(other.id, (0.5, 0.0))[0]
        
        # Determine adjustment direction on the torus
        if compatibility < target_compatibility:
            # Move closer along the torus
            # Calculate shortest angular paths for both coordinates
            theta_diff = ((other.theta - self.theta + math.pi) % (2 * math.pi)) - math.pi
            phi_diff = ((other.phi - self.phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Adjust by a fraction towards the target
            adjustment_factor = learning_rate * (target_compatibility - compatibility)
            
            # Apply adjustments with phase stability influence
            self.theta = (self.theta + theta_diff * adjustment_factor) % (2 * math.pi)
            self.phi = (self.phi + phi_diff * adjustment_factor) % (2 * math.pi)
        else:
            # Random small adjustment to maintain dynamic equilibrium
            # Use smaller adjustments for more stable points
            stability_factor = 1.0 - 0.5 * self.phase_stability
            self.theta = (self.theta + (random.random() - 0.5) * 0.05 * learning_rate * stability_factor) % (2 * math.pi)
            self.phi = (self.phi + (random.random() - 0.5) * 0.05 * learning_rate * stability_factor) % (2 * math.pi)
        
        # Update cartesian position to reflect new toroidal coordinates
        self.update_cartesian_from_toroidal()
    
    def update_cartesian_from_toroidal(self) -> None:
        """Update Cartesian coordinates based on toroidal position"""
        # Standard torus parametric equations
        r_major = 2.0  # Major radius
        r_minor = 1.0  # Minor radius
        
        # Update position based on torus parametric equations
        self.position[0] = (r_major + r_minor * math.cos(self.phi)) * math.cos(self.theta)
        self.position[1] = (r_major + r_minor * math.cos(self.phi)) * math.sin(self.theta)
        self.position[2] = r_minor * math.sin(self.phi)
        
        # Remaining dimensions remain random
    
    def adjust_orientation(self, other: 'ConfigurationPoint', compatibility: float,
                         learning_rate: float = 0.05) -> None:
        """
        Adjust orientation based on compatibility with another point.
        
        Args:
            other: The other configuration point
            compatibility: Current compatibility value
            learning_rate: How quickly orientation adapts
        """
        # Calculate desired compatibility
        target_compatibility = self.connections.get(other.id, (0.5, 0.0))[0]
        
        # Determine adjustment direction
        if compatibility < target_compatibility:
            # Align orientations while respecting toroidal angle
            # Project orientation onto toroidal tangent space
            # (This is a simplification - full implementation would use torus tangent space)
            
            # Create tangent vector from toroidal coordinates
            tangent_theta = np.array([
                -math.sin(self.theta) * (2 + math.cos(self.phi)),
                math.cos(self.theta) * (2 + math.cos(self.phi)),
                0
            ])
            
            tangent_phi = np.array([
                -math.cos(self.theta) * math.sin(self.phi),
                -math.sin(self.theta) * math.sin(self.phi),
                math.cos(self.phi)
            ])
            
            # Normalize tangent vectors
            tangent_theta = tangent_theta / (np.linalg.norm(tangent_theta) + 1e-10)
            tangent_phi = tangent_phi / (np.linalg.norm(tangent_phi) + 1e-10)
            
            # Project other's orientation into our tangent space
            projection = np.zeros(self.dimensions)
            projection[:3] = (
                np.dot(other.orientation[:3], tangent_theta) * tangent_theta +
                np.dot(other.orientation[:3], tangent_phi) * tangent_phi
            )
            
            if self.dimensions > 3:
                projection[3:] = other.orientation[3:]
            
            # Adjust orientation towards projection
            adjustment = learning_rate * (projection - self.orientation)
            self.orientation += adjustment
        else:
            # Random small adjustment (maintains dynamic equilibrium)
            random_adjustment = np.random.random(self.dimensions) * 0.01 * learning_rate
            self.orientation += random_adjustment
        
        # Renormalize
        norm = np.linalg.norm(self.orientation)
        if norm > 0:
            self.orientation = self.orientation / norm
    
    def update_void_presence(self, amount: float) -> None:
        """
        Update void presence for this point.
        
        Args:
            amount: Amount to adjust void presence
        """
        self.void_presence = max(0.0, min(1.0, self.void_presence + amount))
    
    def set_toroidal_curvature(self, curvature: float) -> None:
        """
        Set the local toroidal curvature value.
        
        Args:
            curvature: Curvature value
        """
        self.toroidal_curvature = curvature
    
    def update_phase_stability(self, stability_delta: float) -> None:
        """
        Update phase stability.
        
        Args:
            stability_delta: Change in stability
        """
        self.phase_stability = max(0.0, min(1.0, self.phase_stability + stability_delta))
    
    def get_toroidal_coordinates(self) -> Tuple[float, float]:
        """
        Get toroidal coordinates.
        
        Returns:
            Tuple of (theta, phi)
        """
        return (self.theta, self.phi)
    
    def set_toroidal_coordinates(self, theta: float, phi: float) -> None:
        """
        Set toroidal coordinates and update Cartesian position.
        
        Args:
            theta: New theta value (0 to 2π)
            phi: New phi value (0 to 2π)
        """
        self.theta = theta % (2 * math.pi)
        self.phi = phi % (2 * math.pi)
        self.update_cartesian_from_toroidal()


class ConfigurationSpace:
    """
    Represents the space of possible configurations and their relationships.
    Manages the geometric embedding of nodes and their compatibility.
    Enhanced with void regions, decay particles, and toroidal referencing.
    """
    
    def __init__(self, dimensions: int = 4):
        """
        Initialize the configuration space.
        
        Args:
            dimensions: Number of dimensions in the space
        """
        self.dimensions = dimensions
        self.points = {}  # Maps point_id -> ConfigurationPoint
        self.time = 0.0
        
        # Enhanced with void-decay mechanism
        self.void_regions = {}  # Maps void_id -> void region dictionary
        self.decay_particles = []  # List of decay particle dictionaries
        
        # Get void-decay rule
        self.void_decay_rule = VoidDecayRule()
        
        # Toroidal field properties
        self.toroidal_field_metrics = {
            'global_phase_coherence': 0.0,
            'major_mode_strength': 0.0,
            'minor_mode_strength': 0.0,
            'toroidal_flux': 0.0
        }
        
        # Track toroidal neighborhoods
        self.neighborhoods = {}  # Maps point_id -> list of nearby point_ids
        self.neighborhood_radius = 0.5  # Default neighborhood radius in toroidal space
    
    def add_point(self, point: ConfigurationPoint) -> None:
        """
        Add a configuration point to the space.
        
        Args:
            point: The configuration point to add
        """
        self.points[point.id] = point
        
        # Update toroidal neighborhoods
        self.update_toroidal_neighborhood(point.id)
    
    def remove_point(self, point_id: str) -> None:
        """
        Remove a configuration point from the space.
        
        Args:
            point_id: ID of the point to remove
        """
        if point_id in self.points:
            # Remove connections to this point
            for other_id, other_point in self.points.items():
                if point_id in other_point.connections:
                    del other_point.connections[point_id]
            
            # Remove from neighborhoods
            if point_id in self.neighborhoods:
                del self.neighborhoods[point_id]
            
            for other_id in self.neighborhoods:
                if point_id in self.neighborhoods[other_id]:
                    self.neighborhoods[other_id].remove(point_id)
            
            # Remove the point
            del self.points[point_id]
    
    def get_point(self, point_id: str) -> Optional[ConfigurationPoint]:
        """
        Get a configuration point by ID.
        
        Args:
            point_id: ID of the point to get
            
        Returns:
            ConfigurationPoint or None if not found
        """
        return self.points.get(point_id)
    
    def connect_points(self, point1_id: str, point2_id: str, 
                     compatibility: float = 0.5, tension: float = 0.0) -> None:
        """
        Connect two configuration points with specified compatibility.
        
        Args:
            point1_id: ID of the first point
            point2_id: ID of the second point
            compatibility: Compatibility value between points
            tension: Initial tension between points
        """
        point1 = self.get_point(point1_id)
        point2 = self.get_point(point2_id)
        
        if point1 and point2:
            # Create bidirectional connection
            point1.connections[point2_id] = (compatibility, tension)
            point2.connections[point1_id] = (compatibility, tension)
    
    def update_connection(self, point1_id: str, point2_id: str,
                        compatibility: Optional[float] = None,
                        tension: Optional[float] = None) -> None:
        """
        Update connection properties between two points.
        
        Args:
            point1_id: ID of the first point
            point2_id: ID of the second point
            compatibility: New compatibility value (or None to keep current)
            tension: New tension value (or None to keep current)
        """
        point1 = self.get_point(point1_id)
        point2 = self.get_point(point2_id)
        
        if point1 and point2:
            # Get current values
            current1 = point1.connections.get(point2_id, (0.5, 0.0))
            current2 = point2.connections.get(point1_id, (0.5, 0.0))
            
            # Update values
            new_compatibility = compatibility if compatibility is not None else current1[0]
            new_tension = tension if tension is not None else current1[1]
            
            # Set updated values
            point1.connections[point2_id] = (new_compatibility, new_tension)
            point2.connections[point1_id] = (new_compatibility, new_tension)
    
    def update_from_collapse(self, source_id: str, target_id: str, collapse_strength: float,
                           ancestry_similarity: float = 0.0, check_alignment: bool = True) -> None:
        """
        Update configuration space from a collapse event.
        Enhanced with toroidal alignments and phase stability.
        
        Args:
            source_id: ID of source point
            target_id: ID of target point
            collapse_strength: Strength of the collapse
            ancestry_similarity: Similarity of ancestry between nodes
            check_alignment: Whether to check for structural alignment
        """
        source = self.get_point(source_id)
        target = self.get_point(target_id)
        
        if not source or not target:
            return
        
        # Calculate current compatibility
        current_compatibility = source.calculate_compatibility(target)
        
        # Calculate target compatibility based on collapse
        # Higher collapse strength and ancestry similarity increase compatibility
        target_compatibility = 0.5 + 0.3 * collapse_strength + 0.2 * ancestry_similarity
        
        # Calculate adjustment strength based on difference
        adjustment = target_compatibility - current_compatibility
        
        # Limit adjustment
        adjustment = max(-0.3, min(0.3, adjustment))
        
        # Update connection in configuration space
        if adjustment != 0:
            new_compatibility = current_compatibility + adjustment
            current_tension = source.connections.get(target_id, (0.5, 0.0))[1]
            
            # Tension decreases with successful alignment
            new_tension = max(0.0, current_tension - 0.1 * collapse_strength)
            
            self.update_connection(source_id, target_id, new_compatibility, new_tension)
        
        # Update phase stability based on collapse strength and ancestry
        phase_stability_boost = 0.05 * collapse_strength * (1.0 + ancestry_similarity)
        source.update_phase_stability(phase_stability_boost)
        target.update_phase_stability(phase_stability_boost * 0.8)  # Slightly lower for target
        
        # Check for structural alignment if requested
        if check_alignment:
            self.attempt_structural_alignment(source_id, target_id)
        
        # Update toroidal neighborhoods
        self.update_toroidal_neighborhood(source_id)
        self.update_toroidal_neighborhood(target_id)
    
    def attempt_structural_alignment(self, point1_id: str, point2_id: str, 
                                   threshold: float = 0.7) -> bool:
        """
        Attempt to align two points structurally using toroidal alignment.
        
        Args:
            point1_id: ID of the first point
            point2_id: ID of the second point
            threshold: Compatibility threshold for successful alignment
            
        Returns:
            True if alignment succeeded, False if failed
        """
        point1 = self.get_point(point1_id)
        point2 = self.get_point(point2_id)
        
        if not point1 or not point2:
            return False
        
        # Calculate current compatibility
        current_compatibility = point1.calculate_compatibility(point2)
        
        # Check if compatibility meets threshold
        if current_compatibility >= threshold:
            # Successful alignment
            return True
        
        # Try to improve compatibility through toroidal position adjustment
        for _ in range(5):  # Limited attempts
            # Adjust position and orientation
            point1.adjust_position_toroidal(point2, current_compatibility)
            point1.adjust_orientation(point2, current_compatibility)
            
            point2.adjust_position_toroidal(point1, current_compatibility)
            point2.adjust_orientation(point1, current_compatibility)
            
            # Recalculate compatibility
            new_compatibility = point1.calculate_compatibility(point2)
            
            # Check if improved enough
            if new_compatibility >= threshold:
                # Update neighborhoods
                self.update_toroidal_neighborhood(point1_id)
                self.update_toroidal_neighborhood(point2_id)
                return True
            
            # Update for next iteration
            current_compatibility = new_compatibility
        
        # Failed to align sufficiently
        # Handle incompatible structure
        self._handle_incompatible_structure(point1_id, point2_id, current_compatibility)
        
        return False
    
    def _handle_incompatible_structure(self, point1_id: str, point2_id: str, 
                                     compatibility: float) -> None:
        """
        Handle incompatible structure by increasing tension and potentially forming voids.
        
        Args:
            point1_id: ID of the first point
            point2_id: ID of the second point
            compatibility: Current compatibility between points
        """
        point1 = self.get_point(point1_id)
        point2 = self.get_point(point2_id)
        
        if not point1 or not point2:
            return
        
        # Calculate tension increase (inversely proportional to compatibility)
        tension_increase = 0.2 * (1.0 - compatibility)
        
        # Update structural tension for both points
        void_formed1 = point1.update_structural_tension(tension_increase)
        void_formed2 = point2.update_structural_tension(tension_increase)
        
        # Form voids if threshold exceeded
        if void_formed1:
            self._form_void_region(point1_id)
        
        if void_formed2:
            self._form_void_region(point2_id)
    
    def _form_void_region(self, center_point_id: str) -> str:
        """
        Form a void region around a center point.
        
        Args:
            center_point_id: ID of the center point
            
        Returns:
            ID of the created void region
        """
        center_point = self.get_point(center_point_id)
        if not center_point:
            return ""
        
        # Generate unique ID for void
        void_id = f"void_{uuid.uuid4().hex[:8]}"
        
        # Calculate void strength based on structural tension
        void_strength = self.void_decay_rule.calculate_void_strength(center_point.structural_tension)
        
        # Create void region dictionary using template
        void_region = self.void_decay_rule.create_void_region_dict(
            void_id, center_point_id, void_strength, self.time
        )
        
        # Store void region
        self.void_regions[void_id] = void_region
        
        # Apply initial void effect to center point
        center_point.void_presence = 0.3 * void_strength
        
        # Update affected points based on toroidal neighborhood
        if center_point_id in self.neighborhoods:
            void_region['affected_points'].update(self.neighborhoods[center_point_id])
        
        return void_id
    
    def update_toroidal_neighborhood(self, point_id: str) -> List[str]:
        """
        Update the toroidal neighborhood for a point.
        
        Args:
            point_id: ID of the point
            
        Returns:
            List of point IDs in the neighborhood
        """
        point = self.get_point(point_id)
        if not point:
            self.neighborhoods[point_id] = []
            return []
        
        # Get toroidal coordinates
        theta, phi = point.get_toroidal_coordinates()
        
        # Find neighbors within radius on torus
        neighbors = []
        for other_id, other in self.points.items():
            if other_id == point_id:
                continue
                
            other_theta, other_phi = other.get_toroidal_coordinates()
            
            # Calculate toroidal distance
            distance = toroidal_distance(theta, phi, other_theta, other_phi)
            
            if distance < self.neighborhood_radius:
                neighbors.append(other_id)
        
        # Update neighborhood
        self.neighborhoods[point_id] = neighbors
        
        return neighbors
    
    def get_toroidal_neighborhood(self, point_id: str) -> List[str]:
        """
        Get the toroidal neighborhood for a point.
        
        Args:
            point_id: ID of the point
            
        Returns:
            List of point IDs in the neighborhood
        """
        # Return cached neighborhood if available
        if point_id in self.neighborhoods:
            return self.neighborhoods[point_id]
        
        # Otherwise, update and return
        return self.update_toroidal_neighborhood(point_id)
    
    def process_void_regions(self, dt: float = 1.0) -> None:
        """
        Process all void regions, updating their state and effects.
        Enhanced with toroidal propagation patterns.
        
        Args:
            dt: Time delta for this update
        """
        # Process each void region
        void_ids_to_remove = []
        
        for void_id, void in self.void_regions.items():
            # Check if void should emit decay particle
            should_emit = self.void_decay_rule.should_emit_decay(void, self.time)
            
            if should_emit:
                # Create decay particle
                self._emit_decay_particle(void)
                
                # Update void region
                void['decay_emissions'] += 1
                void['last_emission_time'] = self.time
            
            # Update void presence in affected points
            center_id = void['center_point']
            center_point = self.get_point(center_id)
            
            if not center_point:
                # Center point no longer exists, remove void
                void_ids_to_remove.append(void_id)
                continue
            
            # Update center point void presence
            center_point.void_presence = 0.3 * void['strength']
            
            # Get toroidal coordinates of center
            center_theta, center_phi = center_point.get_toroidal_coordinates()
            
            # Propagate void presence to nearby points using toroidal distance
            radius = void['radius']
            
            # Use toroidal neighborhood for more efficient propagation
            neighborhood = self.get_toroidal_neighborhood(center_id)
            neighborhood.append(center_id)  # Include center point
            
            for point_id in neighborhood:
                point = self.get_point(point_id)
                if not point or point_id == center_id:
                    continue
                
                # Get toroidal coordinates
                point_theta, point_phi = point.get_toroidal_coordinates()
                
                # Calculate toroidal distance
                distance = toroidal_distance(center_theta, center_phi, point_theta, point_phi)
                
                # Check if within radius
                if distance <= radius:
                    # Add to affected points
                    void['affected_points'].add(point_id)
                    
                    # Calculate void effect based on distance
                    distance_factor = 1.0 - distance / radius
                    
                    # Toroidal effects - void propagates differently along major vs minor axis
                    theta_diff = angular_difference(center_theta, point_theta)
                    phi_diff = angular_difference(center_phi, point_phi)
                    
                    # Void propagates more easily along the major circle
                    propagation_factor = 1.0 + 0.2 * (phi_diff / (theta_diff + 0.01))
                    void_effect = void['strength'] * distance_factor * 0.2 * dt * propagation_factor
                    
                    # Update point void presence
                    point.update_void_presence(void_effect)
            
            # Gradually reduce void strength
            void['strength'] *= (1.0 - 0.05 * dt)
            
            # Remove void if strength too low
            if void['strength'] < 0.05:
                void_ids_to_remove.append(void_id)
        
        # Remove expired voids
        for void_id in void_ids_to_remove:
            del self.void_regions[void_id]
    
    def _emit_decay_particle(self, void: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emit a decay particle from a void region.
        Enhanced with toroidal trajectory.
        
        Args:
            void: Void region dictionary
            
        Returns:
            New decay particle dictionary
        """
        # Create decay particle using void-decay rule
        particle = self.void_decay_rule.create_decay_particle_dict(
            void['center_point'],
            void['strength'] * 0.5,
            self.time
        )
        
        # Create dedicated impact_score field
        particle['impact_score'] = 0.0
        
        # Add toroidal coordinates for the particle
        center_point = self.get_point(void['center_point'])
        if center_point:
            theta, phi = center_point.get_toroidal_coordinates()
            
            # Add toroidal position and movement direction
            particle['toroidal_position'] = {
                'theta': theta,
                'phi': phi
            }
            
            # Random direction of propagation along the torus
            # Bias towards major circle (theta) for most particles
            if random.random() < 0.7:
                # Major circle propagation
                particle['toroidal_direction'] = {
                    'theta': random.choice([-1, 1]) * (0.1 + random.random() * 0.2),
                    'phi': random.choice([-1, 1]) * random.random() * 0.1
                }
            else:
                # Minor circle propagation
                particle['toroidal_direction'] = {
                    'theta': random.choice([-1, 1]) * random.random() * 0.1,
                    'phi': random.choice([-1, 1]) * (0.1 + random.random() * 0.2)
                }
        
        # Add to decay particles
        self.decay_particles.append(particle)
        
        return particle
    
    def process_decay_particles(self, dt: float = 1.0) -> None:
        """
        Process all decay particles, updating their state and effects.
        Enhanced with toroidal propagation.
        
        Args:
            dt: Time delta for this update
        """
        # Particles to remove
        particles_to_remove = []
        
        # Process each particle
        for particle in self.decay_particles:
            # Update lifetime
            particle['lifetime'] = particle.get('lifetime', 0.0) + dt
            
            # Move particle along toroidal path if it has toroidal properties
            if 'toroidal_position' in particle and 'toroidal_direction' in particle:
                # Update position
                particle['toroidal_position']['theta'] = (
                    particle['toroidal_position']['theta'] + 
                    particle['toroidal_direction']['theta'] * dt
                ) % (2 * math.pi)
                
                particle['toroidal_position']['phi'] = (
                    particle['toroidal_position']['phi'] + 
                    particle['toroidal_direction']['phi'] * dt
                ) % (2 * math.pi)
            
            # Apply decay particle effects
            self._propagate_decay_effects(particle, dt)
            
            # Mark as processed
            particle['processed'] = True
            
            # Reduce strength over time
            particle['strength'] *= max(0.0, 1.0 - 0.1 * dt)
            
            # Remove if strength too low or lifetime too long
            if particle['strength'] < 0.05 or particle['lifetime'] > 20.0:
                particles_to_remove.append(particle)
        
        # Remove expired particles
        for particle in particles_to_remove:
            if particle in self.decay_particles:
                self.decay_particles.remove(particle)
    
    def _propagate_decay_effects(self, particle: Dict[str, Any], dt: float) -> None:
        """
        Propagate effects of a decay particle to points.
        Enhanced with toroidal propagation patterns.
        
        Args:
            particle: Decay particle dictionary
            dt: Time delta
        """
        # Handle particles with toroidal position
        if 'toroidal_position' in particle:
            theta = particle['toroidal_position']['theta']
            phi = particle['toroidal_position']['phi']
            
            # Find points near the particle's current position
            affected_points = []
            effect_radius = 0.3  # Effect radius on torus
            
            for point_id, point in self.points.items():
                point_theta, point_phi = point.get_toroidal_coordinates()
                
                # Calculate toroidal distance
                distance = toroidal_distance(theta, phi, point_theta, point_phi)
                
                if distance < effect_radius:
                    affected_points.append((point_id, distance))
            
            # Affect nearby points
            for point_id, distance in affected_points:
                point = self.get_point(point_id)
                if not point:
                    continue
                
                # Calculate effect strength (stronger for closer points)
                distance_factor = 1.0 - distance / effect_radius
                effect_strength = particle['strength'] * distance_factor * dt
                
                # Apply effects
                if random.random() < 0.4:
                    # Affect void presence
                    void_effect = -effect_strength * 0.3  # Decay reduces void presence
                    point.update_void_presence(void_effect)
                
                if random.random() < 0.6:
                    # Affect awareness
                    if point.awareness > 0.7:
                        # Reduce high awareness
                        point.awareness = max(0.0, point.awareness - effect_strength * 0.1)
                    else:
                        # Potentially increase low awareness
                        point.awareness = min(1.0, point.awareness + effect_strength * 0.05)
                
                # Phase stability effects
                phase_effect = (random.random() - 0.5) * effect_strength * 0.2
                point.update_phase_stability(phase_effect)
                
                # Update impact score
                particle['impact_score'] = particle.get('impact_score', 0.0) + effect_strength
                
                # Record effect in memory trace
                if 'memory_trace' not in particle:
                    particle['memory_trace'] = []
                    
                particle['memory_trace'].append({
                    'target': point_id,
                    'type': 'awareness',
                    'strength': effect_strength,
                    'time': particle['lifetime'],
                    'theta': point_theta,
                    'phi': point_phi
                })
                
                # Add to affected nodes
                if 'affected_nodes' not in particle:
                    particle['affected_nodes'] = set()
                    
                particle['affected_nodes'].add(point_id)
            
            return
        
        # Fall back to original implementation for particles without toroidal position
        origin_id = particle['origin_id']
        origin_point = self.get_point(origin_id)
        
        if not origin_point:
            return
        
        # Get neighboring points
        neighbor_ids = list(origin_point.connections.keys())
        random.shuffle(neighbor_ids)  # Randomize effect order
        
        # Limit affected neighbors
        max_affected = min(len(neighbor_ids), 3)
        affected_ids = neighbor_ids[:max_affected]
        
        for point_id in affected_ids:
            point = self.get_point(point_id)
            if not point:
                continue
            
            # Calculate effect strength
            effect_strength = particle['strength'] * random.uniform(0.5, 1.0)
            
            # Apply effects
            if random.random() < 0.4:
                # Affect void presence
                void_effect = -effect_strength * 0.3  # Decay reduces void presence
                point.update_void_presence(void_effect)
            
            if random.random() < 0.6:
                # Affect awareness
                if point.awareness > 0.7:
                    # Reduce high awareness
                    point.awareness = max(0.0, point.awareness - effect_strength * 0.1)
                else:
                    # Potentially increase low awareness
                    point.awareness = min(1.0, point.awareness + effect_strength * 0.05)
            
            # Update impact score
            particle['impact_score'] = particle.get('impact_score', 0.0) + effect_strength
            
            # Record effect in memory trace
            if 'memory_trace' not in particle:
                particle['memory_trace'] = []
                
            particle['memory_trace'].append({
                'target': point_id,
                'type': 'awareness',
                'strength': effect_strength,
                'time': particle['lifetime']
            })
            
            # Add to affected nodes
            if 'affected_nodes' not in particle:
                particle['affected_nodes'] = set()
                
            particle['affected_nodes'].add(point_id)
    
    def set_awareness(self, point_id: str, awareness: float) -> None:
        """
        Set awareness for a configuration point.
        
        Args:
            point_id: ID of the point
            awareness: New awareness value
        """
        point = self.get_point(point_id)
        if point:
            point.awareness = max(0.0, min(1.0, awareness))
    
    def set_collapse_metric(self, point_id: str, collapse_metric: float) -> None:
        """
        Set collapse metric for a configuration point.
        
        Args:
            point_id: ID of the point
            collapse_metric: New collapse metric value
        """
        point = self.get_point(point_id)
        if point:
            point.collapse_metric = max(0.0, min(1.0, collapse_metric))
    
    def set_grain_activation(self, point_id: str, activation: float) -> None:
        """
        Set grain activation for a configuration point.
        
        Args:
            point_id: ID of the point
            activation: New activation value
        """
        point = self.get_point(point_id)
        if point:
            point.grain_activation = max(0.0, min(1.0, activation))
    
    def optimize_configuration(self, learning_rate: float = 0.05, steps: int = 5) -> None:
        """
        Optimize the configuration space to better reflect desired relationships.
        Enhanced with toroidal optimization.
        
        Args:
            learning_rate: How quickly points adjust
            steps: Number of optimization steps
        """
        # Perform multiple optimization steps
        for _ in range(steps):
            # Process each point
            for point_id, point in self.points.items():
                # Process connections
                for other_id, (desired_compatibility, _) in point.connections.items():
                    other = self.get_point(other_id)
                    if not other:
                        continue
                    
                    # Calculate current compatibility
                    current_compatibility = point.calculate_compatibility(other)
                    
                    # Adjust position and orientation using toroidal metrics
                    if abs(current_compatibility - desired_compatibility) > 0.05:
                        point.adjust_position_toroidal(other, current_compatibility, learning_rate)
                        point.adjust_orientation(other, current_compatibility, learning_rate)
            
            # Update all neighborhoods after optimization step
            for point_id in self.points:
                self.update_toroidal_neighborhood(point_id)
    
    def calculate_toroidal_metrics(self) -> Dict[str, Any]:
        """
        Calculate global toroidal metrics for the configuration space.
        
        Returns:
            Dictionary with toroidal metrics
        """
        if not self.points:
            return {
                'global_phase_coherence': 0.0,
                'major_modes': [],
                'minor_modes': [],
                'toroidal_flux': 0.0
            }
        
        # Analyze distribution of points on the torus
        # Create histogram bins for theta and phi
        bin_count = 12
        theta_bins = [0] * bin_count
        phi_bins = [0] * bin_count
        
        # Populate bins
        for point in self.points.values():
            theta, phi = point.get_toroidal_coordinates()
            
            theta_bin = int((theta / (2 * math.pi)) * bin_count) % bin_count
            phi_bin = int((phi / (2 * math.pi)) * bin_count) % bin_count
            
            theta_bins[theta_bin] += 1
            phi_bins[phi_bin] += 1
        
        # Calculate phase coherence
        # High coherence means points are clustered in specific regions
        max_theta = max(theta_bins)
        max_phi = max(phi_bins)
        
        if sum(theta_bins) > 0 and sum(phi_bins) > 0:
            theta_coherence = max_theta / (sum(theta_bins) / bin_count)
            phi_coherence = max_phi / (sum(phi_bins) / bin_count)
            global_coherence = (theta_coherence + phi_coherence) / 2
        else:
            global_coherence = 0.0
        
        # Find modes (peaks) in the distributions
        theta_modes = []
        phi_modes = []
        
        for i in range(bin_count):
            prev_idx = (i - 1) % bin_count
            next_idx = (i + 1) % bin_count
            
            # Check if this bin is a local maximum
            if theta_bins[i] > theta_bins[prev_idx] and theta_bins[i] > theta_bins[next_idx]:
                theta_modes.append({
                    'bin': i,
                    'angle': (i + 0.5) * (2 * math.pi / bin_count),
                    'strength': theta_bins[i] / (sum(theta_bins) + 0.001)
                })
            
            if phi_bins[i] > phi_bins[prev_idx] and phi_bins[i] > phi_bins[next_idx]:
                phi_modes.append({
                    'bin': i,
                    'angle': (i + 0.5) * (2 * math.pi / bin_count),
                    'strength': phi_bins[i] / (sum(phi_bins) + 0.001)
                })
        
        # Sort modes by strength
        theta_modes.sort(key=lambda x: x['strength'], reverse=True)
        phi_modes.sort(key=lambda x: x['strength'], reverse=True)
        
        # Calculate toroidal flux (flow around the torus)
        # Analyze average flow directions
        theta_flow = 0.0
        phi_flow = 0.0
        flow_count = 0
        
        for p1_id, p1 in self.points.items():
            for p2_id in self.neighborhoods.get(p1_id, []):
                p2 = self.get_point(p2_id)
                if not p2:
                    continue
                
                # Calculate angular differences
                p1_theta, p1_phi = p1.get_toroidal_coordinates()
                p2_theta, p2_phi = p2.get_toroidal_coordinates()
                
                theta_diff = ((p2_theta - p1_theta + math.pi) % (2 * math.pi)) - math.pi
                phi_diff = ((p2_phi - p1_phi + math.pi) % (2 * math.pi)) - math.pi
                
                # Weight by awareness difference (flow direction)
                awareness_diff = p2.awareness - p1.awareness
                
                if abs(awareness_diff) > 0.1:  # Only count significant flows
                    flow_direction = 1 if awareness_diff > 0 else -1
                    
                    theta_flow += theta_diff * flow_direction
                    phi_flow += phi_diff * flow_direction
                    flow_count += 1
        
        # Normalize flows
        toroidal_flux = 0.0
        if flow_count > 0:
            theta_flow /= flow_count
            phi_flow /= flow_count
            
            # Combine into single metric
            toroidal_flux = math.sqrt(theta_flow**2 + phi_flow**2)
        
        # Update stored metrics
        self.toroidal_field_metrics = {
            'global_phase_coherence': global_coherence,
            'major_mode_strength': theta_modes[0]['strength'] if theta_modes else 0.0,
            'minor_mode_strength': phi_modes[0]['strength'] if phi_modes else 0.0,
            'toroidal_flux': toroidal_flux
        }
        
        return {
            'global_phase_coherence': global_coherence,
            'major_modes': theta_modes,
            'minor_modes': phi_modes,
            'toroidal_flux': toroidal_flux,
            'theta_bins': theta_bins,
            'phi_bins': phi_bins
        }
    
    def calculate_alignment_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics about structural alignment in the configuration space.
        Enhanced with toroidal statistics.
        
        Returns:
            Dictionary with alignment statistics
        """
        total_connections = 0
        compatible_connections = 0
        total_tension = 0.0
        
        # Process all points
        for point in self.points.values():
            for other_id, (desired_compatibility, tension) in point.connections.items():
                other = self.get_point(other_id)
                if not other:
                    continue
                
                total_connections += 1
                total_tension += tension
                
                # Check compatibility
                current_compatibility = point.calculate_compatibility(other)
                if current_compatibility >= 0.7:  # Threshold for "compatible"
                    compatible_connections += 1
        
        # Calculate statistics
        if total_connections > 0:
            compatibility_rate = compatible_connections / total_connections
            average_tension = total_tension / total_connections
            failure_rate = 1.0 - compatibility_rate
        else:
            compatibility_rate = 0.0
            average_tension = 0.0
            failure_rate = 0.0
        
        # Calculate void and decay statistics
        void_count = len(self.void_regions)
        decay_count = len(self.decay_particles)
        
        # Calculate toroidal metrics
        toroidal_metrics = self.calculate_toroidal_metrics()
        
        # Find incompatible pairs
        incompatible_pairs = []
        for point1_id, point1 in self.points.items():
            for point2_id in point1.connections:
                point2 = self.get_point(point2_id)
                if not point2:
                    continue
                
                compatibility = point1.calculate_compatibility(point2)
                if compatibility < 0.5:  # Threshold for "incompatible"
                    incompatible_pairs.append((point1_id, point2_id))
        
        # Calculate neighborhood statistics
        neighborhood_sizes = [len(neighbors) for neighbors in self.neighborhoods.values()]
        avg_neighborhood_size = sum(neighborhood_sizes) / len(neighborhood_sizes) if neighborhood_sizes else 0
        
        return {
            'compatibility_rate': compatibility_rate,
            'average_tension': average_tension,
            'failure_rate': failure_rate,
            'void_count': void_count,
            'decay_count': decay_count,
            'incompatible_pairs': incompatible_pairs,
            'toroidal_metrics': toroidal_metrics,
            'avg_neighborhood_size': avg_neighborhood_size,
            'max_neighborhood_size': max(neighborhood_sizes) if neighborhood_sizes else 0
        }
    
    def find_void_regions(self) -> List[Dict[str, Any]]:
        """
        Find all void regions in the configuration space.
        Enhanced with toroidal positioning.
        
        Returns:
            List of void region information dictionaries
        """
        void_info = []
        
        for void_id, void in self.void_regions.items():
            # Get affected points
            affected_points = []
            for point_id in void['affected_points']:
                point = self.get_point(point_id)
                if point:
                    theta, phi = point.get_toroidal_coordinates()
                    affected_points.append({
                        'id': point_id,
                        'void_presence': point.void_presence,
                        'position': point.position.tolist(),
                        'theta': theta,
                        'phi': phi
                    })
            
            # Get center point's toroidal position
            center_point = self.get_point(void['center_point'])
            center_theta = 0
            center_phi = 0
            
            if center_point:
                center_theta, center_phi = center_point.get_toroidal_coordinates()
            
            # Create void info
            info = {
                'void_id': void_id,
                'center_point': void['center_point'],
                'center_theta': center_theta,
                'center_phi': center_phi,
                'strength': void['strength'],
                'radius': void['radius'],
                'formation_time': void['formation_time'],
                'decay_emissions': void['decay_emissions'],
                'affected_points': affected_points
            }
            
            void_info.append(info)
        
        return void_info
    
    def find_toroidal_clusters(self, angular_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Find clusters of points based on toroidal proximity.
        
        Args:
            angular_threshold: Maximum angular distance for points to be in same cluster
            
        Returns:
            List of cluster information dictionaries
        """
        # Initialize visited set
        visited = set()
        clusters = []
        
        # Process each point
        for point_id, point in self.points.items():
            if point_id in visited:
                continue
            
            # Start new cluster
            cluster = {'points': [], 'center_theta': 0, 'center_phi': 0}
            to_visit = [point_id]
            
            # BFS to find connected points
            while to_visit:
                current_id = to_visit.pop(0)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                current = self.get_point(current_id)
                if not current:
                    continue
                    
                # Add to cluster
                current_theta, current_phi = current.get_toroidal_coordinates()
                cluster['points'].append({
                    'id': current_id,
                    'theta': current_theta,
                    'phi': current_phi,
                    'awareness': current.awareness,
                    'grain_activation': current.grain_activation
                })
                
                # Add neighboring points based on toroidal distance
                for other_id, other in self.points.items():
                    if other_id in visited or other_id == current_id:
                        continue
                        
                    other_theta, other_phi = other.get_toroidal_coordinates()
                    
                    # Calculate toroidal distance
                    distance = toroidal_distance(current_theta, current_phi, other_theta, other_phi)
                    
                    if distance < angular_threshold:
                        to_visit.append(other_id)
            
            # Calculate cluster properties if it's not empty
            if cluster['points']:
                # Calculate center coordinates (circular mean)
                sum_sin_theta = sum(math.sin(p['theta']) for p in cluster['points'])
                sum_cos_theta = sum(math.cos(p['theta']) for p in cluster['points'])
                sum_sin_phi = sum(math.sin(p['phi']) for p in cluster['points'])
                sum_cos_phi = sum(math.cos(p['phi']) for p in cluster['points'])
                
                cluster['center_theta'] = math.atan2(sum_sin_theta, sum_cos_theta) % (2 * math.pi)
                cluster['center_phi'] = math.atan2(sum_sin_phi, sum_cos_phi) % (2 * math.pi)
                
                # Calculate spread
                cluster['spread'] = sum(
                    toroidal_distance(p['theta'], p['phi'], cluster['center_theta'], cluster['center_phi'])
                    for p in cluster['points']
                ) / len(cluster['points'])
                
                # Calculate average awareness
                cluster['avg_awareness'] = sum(p['awareness'] for p in cluster['points']) / len(cluster['points'])
                
                # Calculate average activation
                cluster['avg_activation'] = sum(p['grain_activation'] for p in cluster['points']) / len(cluster['points'])
                
                clusters.append(cluster)
        
        return clusters
    
    def find_configuration_clusters(self, threshold: float = 0.7) -> List[Set[str]]:
        """
        Find clusters of compatible configuration points.
        Enhanced to consider toroidal neighborhoods.
        
        Args:
            threshold: Compatibility threshold for points to be in same cluster
            
        Returns:
            List of sets containing point IDs in each cluster
        """
        # Initialize visited set
        visited = set()
        clusters = []
        
        # Process each point
        for point_id in self.points:
            if point_id in visited:
                continue
            
            # Start new cluster
            cluster = set()
            to_visit = [point_id]
            
            # BFS to find connected points
            while to_visit:
                current_id = to_visit.pop(0)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                cluster.add(current_id)
                
                # Add compatible neighbors
                current = self.get_point(current_id)
                if not current:
                    continue
                
                # Consider both direct connections and toroidal neighbors
                potential_neighbors = set(current.connections.keys()) | set(self.get_toroidal_neighborhood(current_id))
                
                for neighbor_id in potential_neighbors:
                    if neighbor_id in visited:
                        continue
                    
                    neighbor = self.get_point(neighbor_id)
                    if not neighbor:
                        continue
                    
                    # Check compatibility
                    compatibility = current.calculate_compatibility(neighbor)
                    if compatibility >= threshold:
                        to_visit.append(neighbor_id)
            
            # Add cluster if not empty
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def find_void_clusters(self) -> List[Set[str]]:
        """
        Find clusters of points affected by void regions.
        
        Returns:
            List of sets containing point IDs in each void cluster
        """
        # Initialize void clusters
        void_clusters = []
        
        for void_id, void in self.void_regions.items():
            # Get affected points
            affected_points = set(
                point_id for point_id in void['affected_points']
                if self.get_point(point_id) is not None
            )
            
            # Add cluster if not empty
            if affected_points:
                void_clusters.append(affected_points)
        
        return void_clusters
    
    def advance_time(self, dt: float = 1.0) -> None:
        """
        Advance time in the configuration space.
        
        Args:
            dt: Time delta
        """
        self.time += dt


def create_configuration_space(dimensions: int = 4) -> ConfigurationSpace:
    """
    Create a new configuration space.
    
    Args:
        dimensions: Number of dimensions
        
    Returns:
        New ConfigurationSpace instance
    """
    return ConfigurationSpace(dimensions)