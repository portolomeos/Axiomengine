"""
Relational Manifold - Core operator for the Collapse Geometry framework

Acts as a "smart but dumb operator" that integrates grain dynamics with field rules,
configuration space, and epistemology to evolve the system through emergent properties.
Enhanced with support for Void-Decay principle for handling incompatible structures
and toroidal referencing for richer topological dynamics.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, TYPE_CHECKING

# Standard library imports
from collections import defaultdict
import random

# Direct imports that don't cause circular dependencies
from axiom7.collapse_rules.grain_dynamics import (
    Grain, ToroidalGrainSystem, 
    angular_difference, toroidal_distance, circular_mean
)
from axiom7.collapse_rules.config_space import ConfigurationSpace, ConfigurationPoint
from axiom7.collapse_rules.polarity_space import EpistomologyField, EpistomologyRelation, RelativeRotationTensor

# Forward references for circular imports
if TYPE_CHECKING:
    from axiom7.collapse_rules.emergent_field_rules import EmergentFieldRule


class RelationalManifold:
    """
    Core operator for the Collapse Geometry framework that integrates grain dynamics
    with field rules, configuration space, and epistemology to evolve the system.
    
    The manifold does not store redundant derived properties but calculates them
    on-demand from the fundamental state of the grains and their relationships.
    
    Enhanced with:
    - Void-Decay handling for incompatible structures
    - Toroidal referencing for proper topological dynamics
    """
    
    def __init__(self, neighborhood_radius: float = 0.5):
        """
        Initialize manifold with minimal state and connections to other components
        
        Args:
            neighborhood_radius: Radius for toroidal neighborhood detection
        """
        # Core state
        self.grains = {}                # Maps grain_id -> Grain
        self.time = 0.0                 # System evolution parameter
        self.collapse_history = []      # Record of collapse events
        self.opposite_pairs = set()     # Pairs of opposite grains (grain_id1, grain_id2)
        
        # Integration with configuration and polarity spaces
        self.config_space = ConfigurationSpace()
        self.epistemology_field = EpistomologyField()
        
        # Minimal state for tracking emergent patterns
        self.phase_memory = {}          # Maps grain_id -> accumulated phase (for rotation tracking)
        
        # Field parameters
        self.field_diffusion_rate = 0.15        # Controls awareness propagation
        self.field_gradient_sensitivity = 0.25  # Sensitivity to field gradients
        self.activation_threshold = 0.5         # Threshold for grain activation
        
        # New fields for void and decay tracking
        self.void_formation_events = []         # History of void formations
        self.decay_emission_events = []         # History of decay emissions
        self.incompatible_structure_events = [] # History of incompatible structure events
        
        # Configuration for void-decay mechanism
        self.void_decay_config = {
            'alignment_threshold': 0.7,       # Threshold for successful alignment
            'void_formation_threshold': 0.8,  # Tension threshold for void formation
            'decay_emission_rate': 0.2,       # Base rate of decay particle emission
            'void_propagation_rate': 0.1,     # How quickly voids spread
            'check_structural_alignment': True, # Whether to enforce structural alignment
            'decay_impact_factor': 0.3        # How strongly decay affects the system
        }
        
        # NEW: Toroidal system for topology-aware grain management
        self.toroidal_system = ToroidalGrainSystem(neighborhood_radius=neighborhood_radius)
        
        # NEW: Toroidal phase tracking
        self.toroidal_phase = {}       # Maps grain_id -> (theta, phi)
        self.phase_stability = {}      # Maps grain_id -> stability value
        self.toroidal_flux = 0.0       # Global toroidal flux
        self.rotation_tensor = RelativeRotationTensor()  # For tracking rotational dynamics
        
        # NEW: Toroidal field metrics
        self.toroidal_metrics = {
            'global_phase_coherence': 0.0,
            'major_circle_flow': 0.0,
            'minor_circle_flow': 0.0,
            'toroidal_vortex_count': 0
        }
    
    def add_grain(self, grain_id: str, theta: float = None, phi: float = None) -> Grain:
        """
        Add a new grain to the manifold with optional toroidal coordinates
        
        Args:
            grain_id: Unique ID for the grain
            theta: Major circle position on torus (0 to 2π), randomized if None
            phi: Minor circle position on torus (0 to 2π), randomized if None
            
        Returns:
            New or existing Grain instance
        """
        if grain_id in self.grains:
            return self.grains[grain_id]
        
        # Create grain in relational layer with toroidal properties
        grain = Grain(grain_id=grain_id, theta=theta, phi=phi)
        self.grains[grain_id] = grain
        
        # Add to toroidal system
        self.toroidal_system.add_grain(grain)
        
        # Create corresponding point in configuration space with matching coordinates
        if theta is None and phi is None and hasattr(grain, 'theta') and hasattr(grain, 'phi'):
            # Use coordinates from grain if available
            theta = grain.theta
            phi = grain.phi
            
        config_point = ConfigurationPoint(point_id=grain_id, theta=theta, phi=phi)
        self.config_space.add_point(config_point)
        
        # Initialize toroidal phase
        if theta is None:
            theta = grain.theta if hasattr(grain, 'theta') else random.random() * 2 * math.pi
        if phi is None:
            phi = grain.phi if hasattr(grain, 'phi') else random.random() * 2 * math.pi
            
        self.toroidal_phase[grain_id] = (theta, phi)
        self.phase_stability[grain_id] = 0.5  # Default medium stability
        
        return grain
    
    def remove_grain(self, grain_id: str):
        """Remove a grain from the manifold"""
        if grain_id not in self.grains:
            return
        
        # Clean up relations in other grains
        for other_grain in self.grains.values():
            if grain_id in other_grain.relations:
                del other_grain.relations[grain_id]
            if grain_id in other_grain.relation_memory:
                del other_grain.relation_memory[grain_id]
            if grain_id in other_grain.negation_memory:
                del other_grain.negation_memory[grain_id]
            if grain_id in other_grain.opposition_memory:
                del other_grain.opposition_memory[grain_id]
            if grain_id in other_grain.field_gradients:
                del other_grain.field_gradients[grain_id]
            
            # Update opposite state references
            if other_grain.opposite_state == grain_id:
                other_grain.opposite_state = None
            
            # Remove from negated options
            if grain_id in other_grain.negated_options:
                other_grain.negated_options.remove(grain_id)
        
        # Remove opposite pair references
        pairs_to_remove = []
        for pair in self.opposite_pairs:
            if grain_id in pair:
                pairs_to_remove.append(pair)
        
        for pair in pairs_to_remove:
            self.opposite_pairs.remove(pair)
        
        # Remove from phase memory
        if grain_id in self.phase_memory:
            del self.phase_memory[grain_id]
        
        # Remove from toroidal tracking
        if grain_id in self.toroidal_phase:
            del self.toroidal_phase[grain_id]
        if grain_id in self.phase_stability:
            del self.phase_stability[grain_id]
        
        # Remove from toroidal system
        self.toroidal_system.remove_grain(grain_id)
        
        # Remove from epistemology field
        for related_id in self.grains:
            relation = self.epistemology_field.get_relation(grain_id, related_id)
            if relation:
                # Relations would need to be removed if epistemology field stored them persistently
                pass
        
        # Remove from configuration space
        self.config_space.remove_point(grain_id)
        
        # Remove the grain itself
        del self.grains[grain_id]
    
    def connect_grains(self, grain_id1: str, grain_id2: str, relation_strength: float = 1.0):
        """Create a bidirectional relation between two grains"""
        # Ensure both grains exist
        grain1 = self.get_or_create_grain(grain_id1)
        grain2 = self.get_or_create_grain(grain_id2)
        
        # Create bidirectional relation
        grain1.update_relation(grain_id2, relation_strength)
        grain2.update_relation(grain_id1, relation_strength)
        
        # Initialize relational memory
        if grain_id2 not in grain1.relation_memory:
            grain1.update_relation_memory(grain_id2, 0.0)
        if grain_id1 not in grain2.relation_memory:
            grain2.update_relation_memory(grain_id1, 0.0)
        
        # Initialize field gradients
        awareness_diff = grain2.awareness - grain1.awareness
        grain1.field_gradients[grain_id2] = awareness_diff
        grain2.field_gradients[grain_id1] = -awareness_diff
        
        # Connect in configuration space
        self.config_space.connect_points(grain_id1, grain_id2, compatibility=relation_strength)
        
        # Initialize epistemology relation
        self._initialize_epistemology_relation(grain_id1, grain_id2, relation_strength)
        
        # NEW: Update toroidal neighborhoods
        self.toroidal_system.update_neighborhoods([grain_id1, grain_id2])
    
    def set_opposite_grains(self, grain_id1: str, grain_id2: str):
        """Define two grains as representing opposite states"""
        grain1 = self.get_or_create_grain(grain_id1)
        grain2 = self.get_or_create_grain(grain_id2)
        
        # Set bidirectional opposite references
        grain1.set_opposite_state(grain_id2)
        grain2.set_opposite_state(grain_id1)
        
        # Add to opposite pairs set
        self.opposite_pairs.add((grain_id1, grain_id2))
        
        # Initialize opposition memory
        if grain_id2 not in grain1.opposition_memory:
            grain1.update_opposition_memory(grain_id2, -0.5)  # Default negative
        if grain_id1 not in grain2.opposition_memory:
            grain2.update_opposition_memory(grain_id1, -0.5)  # Default negative
        
        # Set opposition in configuration space
        config_point1 = self.config_space.get_point(grain_id1)
        config_point2 = self.config_space.get_point(grain_id2)
        
        if config_point1 and config_point2:
            # Set high tension between opposite points
            self.config_space.connect_points(grain_id1, grain_id2, compatibility=0.3, tension=0.7)
        
        # NEW: Set up toroidal opposition (opposite phase)
        if grain_id1 in self.toroidal_phase and grain_id2 in self.toroidal_phase:
            theta1, phi1 = self.toroidal_phase[grain_id1]
            # Opposite position on torus (shift by π)
            theta2 = (theta1 + math.pi) % (2 * math.pi)
            phi2 = (phi1 + math.pi) % (2 * math.pi)
            
            # Update phase of second grain to be opposite
            self.toroidal_phase[grain_id2] = (theta2, phi2)
            
            # Update configuration space
            config_point2 = self.config_space.get_point(grain_id2)
            if config_point2:
                config_point2.set_toroidal_coordinates(theta2, phi2)
    
    def record_negation(self, source_id: str, negated_id: str):
        """Record that a grain was negated (not chosen) during collapse"""
        source = self.get_or_create_grain(source_id)
        negated = self.get_or_create_grain(negated_id)
        
        # Add to negated options
        source.add_negated_option(negated_id)
        
        # Initialize negation memory
        if negated_id not in source.negation_memory:
            source.update_negation_memory(negated_id, 0.5)  # Default positive
        if source_id not in negated.negation_memory:
            negated.update_negation_memory(source_id, -0.5)  # Default negative
        
        # NEW: Apply toroidal effect for negation (subtle phase adjustment)
        if source_id in self.toroidal_phase and negated_id in self.toroidal_phase:
            source_theta, source_phi = self.toroidal_phase[source_id]
            negated_theta, negated_phi = self.toroidal_phase[negated_id]
            
            # Calculate phase difference for negation (smaller than for opposition)
            theta_diff = ((negated_theta - source_theta + math.pi/2) % (2 * math.pi)) - math.pi/2
            phi_diff = ((negated_phi - source_phi + math.pi/2) % (2 * math.pi)) - math.pi/2
            
            # Apply a small phase shift to negated grain (push slightly away)
            phase_shift = 0.1
            new_theta = (negated_theta + theta_diff * phase_shift) % (2 * math.pi)
            new_phi = (negated_phi + phi_diff * phase_shift) % (2 * math.pi)
            
            # Update phase
            self.toroidal_phase[negated_id] = (new_theta, new_phi)
            
            # Update configuration point
            negated_point = self.config_space.get_point(negated_id)
            if negated_point:
                negated_point.set_toroidal_coordinates(new_theta, new_phi)
    
    def get_or_create_grain(self, grain_id: str) -> Grain:
        """Get a grain by ID or create it if it doesn't exist"""
        if grain_id not in self.grains:
            return self.add_grain(grain_id)
        return self.grains[grain_id]
    
    def get_grain(self, grain_id: str) -> Optional[Grain]:
        """Get a grain by ID"""
        return self.grains.get(grain_id)
    
    def get_related_grains(self, grain_id: str) -> List[Grain]:
        """Get all grains directly related to the given grain"""
        grain = self.get_grain(grain_id)
        if not grain:
            return []
        
        return [self.grains[related_id] for related_id in grain.relations 
                if related_id in self.grains]
    
    def get_toroidal_neighbors(self, grain_id: str) -> List[Grain]:
        """
        Get all grains in the toroidal neighborhood of the given grain.
        This includes grains that are close in toroidal space but may not have
        direct relations.
        
        Args:
            grain_id: ID of the grain
            
        Returns:
            List of neighbor Grain instances
        """
        # Get neighborhood from toroidal system
        neighbor_ids = self.toroidal_system.get_toroidal_neighborhood(grain_id)
        
        # Convert to grain objects
        neighbors = [self.grains[n_id] for n_id in neighbor_ids if n_id in self.grains]
        
        return neighbors
    
    def get_opposite_grain(self, grain_id: str) -> Optional[Grain]:
        """Get the opposite grain of the given grain, if defined"""
        grain = self.get_grain(grain_id)
        if not grain or not grain.opposite_state:
            return None
        
        return self.get_grain(grain.opposite_state)
    
    def get_negated_options(self, grain_id: str) -> List[Grain]:
        """Get all grains that were negated during collapse from this grain"""
        grain = self.get_grain(grain_id)
        if not grain:
            return []
        
        return [self.grains[negated_id] for negated_id in grain.negated_options 
                if negated_id in self.grains]
    
    def get_toroidal_phase(self, grain_id: str) -> Tuple[float, float]:
        """
        Get the toroidal phase coordinates for a grain.
        
        Args:
            grain_id: ID of the grain
            
        Returns:
            Tuple of (theta, phi) coordinates
        """
        if grain_id in self.toroidal_phase:
            return self.toroidal_phase[grain_id]
        
        # Try to get from grain
        grain = self.get_grain(grain_id)
        if grain and hasattr(grain, 'theta') and hasattr(grain, 'phi'):
            return (grain.theta, grain.phi)
        
        # Try to get from configuration space
        config_point = self.config_space.get_point(grain_id)
        if config_point and hasattr(config_point, 'get_toroidal_coordinates'):
            coords = config_point.get_toroidal_coordinates()
            self.toroidal_phase[grain_id] = coords  # Cache for future use
            return coords
        
        # Default to random phase if not found
        theta = random.random() * 2 * math.pi
        phi = random.random() * 2 * math.pi
        self.toroidal_phase[grain_id] = (theta, phi)
        return (theta, phi)
    
    def set_toroidal_phase(self, grain_id: str, theta: float, phi: float):
        """
        Set the toroidal phase coordinates for a grain.
        Updates both the internal tracking and the configuration space.
        
        Args:
            grain_id: ID of the grain
            theta: Major circle position (0 to 2π)
            phi: Minor circle position (0 to 2π)
        """
        # Normalize to [0, 2π)
        theta = theta % (2 * math.pi)
        phi = phi % (2 * math.pi)
        
        # Update internal tracking
        self.toroidal_phase[grain_id] = (theta, phi)
        
        # Update grain if available
        grain = self.get_grain(grain_id)
        if grain:
            if hasattr(grain, 'theta'):
                grain.theta = theta
            if hasattr(grain, 'phi'):
                grain.phi = phi
            if hasattr(grain, 'update_toroidal_position'):
                grain.update_toroidal_position(theta, phi)
        
        # Update configuration space
        config_point = self.config_space.get_point(grain_id)
        if config_point and hasattr(config_point, 'set_toroidal_coordinates'):
            config_point.set_toroidal_coordinates(theta, phi)
    
    def _initialize_epistemology_relation(self, source_id: str, target_id: str, relation_strength: float):
        """Initialize epistemology relation between grains"""
        source = self.get_grain(source_id)
        target = self.get_grain(target_id)
        
        if not source or not target:
            return
        
        # Create initial epistemology components
        memory_strength = source.relation_memory.get(target_id, 0.0)
        awareness_diff = target.awareness - source.awareness
        
        # Set default components
        strength = 0.0
        resolution = 0.5
        frustration = 0.0
        fidelity = 0.5
        
        # Orientation based on current time
        orientation = (math.atan2(relation_strength, 0.5) + self.time * 0.1) % (2 * math.pi)
        
        # Set the epistemology relation
        self.epistemology_field.set_relation(
            source_id, target_id,
            strength=strength,
            resolution=resolution,
            frustration=frustration,
            fidelity=fidelity,
            orientation=orientation
        )
    
    def create_relation_from_collapse(self, source_id: str, target_id: str, 
                                    collapse_strength: float,
                                    negated_ids: List[str] = None):
        """
        Create or update a relation based on a collapse event.
        Uses relational memory instead of vector-based polarity.
        Enhanced to handle incompatible structures through Void-Decay principle
        and update toroidal positions.
        """
        # Check structural alignment in configuration space
        if self.void_decay_config['check_structural_alignment']:
            alignment_threshold = self.void_decay_config['alignment_threshold']
            
            # Attempt alignment in configuration space
            alignment_succeeded = self.config_space.attempt_structural_alignment(
                source_id, target_id, alignment_threshold)
            
            if not alignment_succeeded:
                # Handle incompatible structure
                self._handle_incompatible_structure(source_id, target_id, collapse_strength)
                
                # Log incompatible structure event
                event = {
                    'time': self.time,
                    'source': source_id,
                    'target': target_id,
                    'collapse_strength': collapse_strength,
                    'resolution': 'void_formation'
                }
                
                self.incompatible_structure_events.append(event)
                
                # Return empty event for incompatible structure
                return {
                    'time': self.time,
                    'source': source_id,
                    'target': target_id,
                    'strength': 0.0,
                    'incompatible_structure': True,
                    'void_formation': True
                }
        
        source = self.get_or_create_grain(source_id)
        target = self.get_or_create_grain(target_id)
        
        # Update relations - with smoothing for continuity
        existing_relation = source.relations.get(target_id, 0.0)
        new_relation = 1.0 - target.grain_saturation
        relation_strength = existing_relation * 0.7 + new_relation * 0.3
        
        source.update_relation(target_id, relation_strength)
        target.update_relation(source_id, relation_strength)
        
        # Update relational memory based on collapse strength
        source.update_relation_memory(target_id, collapse_strength)
        target.update_relation_memory(source_id, -collapse_strength)  # Opposite direction
        
        # Record ancestry - target inherits source as ancestor
        target.ancestry.add(source_id)
        
        # Record negation for unused grains
        if negated_ids:
            for negated_id in negated_ids:
                self.record_negation(source_id, negated_id)
        
        # Check opposite grains - create opposition link
        opposite_source = self.get_opposite_grain(source_id)
        if opposite_source:
            # Update opposition memory
            source.update_opposition_memory(target_id, -collapse_strength)
            
            # If the opposite doesn't have a relation to target, create one
            if target_id not in opposite_source.relations:
                opposite_rel_strength = 0.5 * relation_strength  # Weaker relation
                opposite_source.update_relation(target_id, opposite_rel_strength)
                target.update_relation(opposite_source.id, opposite_rel_strength)
                
                # Set opposition memory
                opposite_source.update_opposition_memory(target_id, -collapse_strength)
                target.update_opposition_memory(opposite_source.id, collapse_strength)
        
        # Update configuration space from collapse
        # Calculate ancestry similarity for richer configuration update
        ancestry_similarity = 0.0
        if source.ancestry and target.ancestry:
            shared = source.ancestry.intersection(target.ancestry)
            combined = source.ancestry.union(target.ancestry)
            if combined:
                ancestry_similarity = len(shared) / len(combined)
        
        # Update configuration space
        self.config_space.update_from_collapse(
            source_id, target_id, 
            collapse_strength, 
            ancestry_similarity,
            check_alignment=False  # Already checked above
        )
        
        # NEW: Update toroidal phase based on collapse
        source_theta, source_phi = self.get_toroidal_phase(source_id)
        target_theta, target_phi = self.get_toroidal_phase(target_id)
        
        # Calculate phase attraction factor (weighted by collapse strength)
        attr_factor = collapse_strength * 0.2
        
        # Calculate circular differences
        theta_diff = ((source_theta - target_theta + math.pi) % (2 * math.pi)) - math.pi
        phi_diff = ((source_phi - target_phi + math.pi) % (2 * math.pi)) - math.pi
        
        # Move target phase toward source (slightly)
        new_target_theta = (target_theta + theta_diff * attr_factor) % (2 * math.pi)
        new_target_phi = (target_phi + phi_diff * attr_factor) % (2 * math.pi)
        
        # Update target's phase
        self.set_toroidal_phase(target_id, new_target_theta, new_target_phi)
        
        # Increase phase stability based on collapse strength
        if target_id in self.phase_stability:
            stability_increase = collapse_strength * 0.1
            self.phase_stability[target_id] = min(1.0, self.phase_stability[target_id] + stability_increase)
        
        # Log collapse event
        event = {
            'time': self.time,
            'source': source_id,
            'target': target_id,
            'strength': collapse_strength,
            'awareness_change': source.awareness - target.awareness,
            'grain_activation_change': target.grain_activation - source.grain_activation,
            'negated_options': negated_ids if negated_ids else [],
            'has_opposite': opposite_source is not None,
            'ancestry_similarity': ancestry_similarity,
            # NEW: Add toroidal information
            'toroidal_phase_change': (theta_diff, phi_diff),
            'source_phase': (source_theta, source_phi),
            'target_phase': (new_target_theta, new_target_phi)
        }
        
        self.collapse_history.append(event)
        
        # Update grain activation and saturation
        self._update_grain_activation_from_collapse(target, relation_strength, collapse_strength)
        
        # Update field gradients
        awareness_diff = target.awareness - source.awareness
        source.field_gradients[target_id] = awareness_diff
        target.field_gradients[source_id] = -awareness_diff
        
        # Update field histories
        source.update_field_history(self.time)
        target.update_field_history(self.time)
        
        # Update epistemology field
        self._update_epistemology_from_collapse(source_id, target_id, collapse_strength, target.grain_saturation)
        
        # NEW: Update toroidal neighborhoods
        self.toroidal_system.update_neighborhoods([source_id, target_id])
        
        return event
    
    def _handle_incompatible_structure(self, source_id: str, target_id: str, collapse_strength: float):
        """
        Handle incompatible structure between grains.
        Implements the Void-Decay principle when collapse fails due to structure mismatch.
        Enhanced with toroidal disruption effects.
        """
        source = self.get_grain(source_id)
        target = self.get_grain(target_id)
        
        if not source or not target:
            return
        
        # Increase structural tension for both grains
        config_source = self.config_space.get_point(source_id)
        config_target = self.config_space.get_point(target_id)
        
        if config_source and config_target:
            # Calculate tension increment based on collapse strength
            tension_increment = collapse_strength * 0.3
            
            # Update tension
            void_formed_source = config_source.update_structural_tension(tension_increment)
            void_formed_target = config_target.update_structural_tension(tension_increment)
            
            # Handle void formation
            if void_formed_source:
                self._register_void_formation(source_id)
            
            if void_formed_target:
                self._register_void_formation(target_id)
        
        # Modify awareness based on incompatible structure
        # Incompatible collapse causes awareness dispersion
        dispersion = source.awareness * collapse_strength * 0.1
        source.awareness = max(0.0, source.awareness - dispersion)
        
        # Process failed collapse effects on epistemology
        source_target_relation = self.epistemology_field.get_relation(source_id, target_id)
        target_source_relation = self.epistemology_field.get_relation(target_id, source_id)
        
        if source_target_relation and target_source_relation:
            # Update frustration and resolution
            self.epistemology_field.update_relation(
                source_id, target_id,
                frustration=0.8,  # High frustration
                resolution=0.3,   # Low resolution
                blending_factor=0.3
            )
            
            self.epistemology_field.update_relation(
                target_id, source_id,
                frustration=0.8,  # High frustration
                resolution=0.3,   # Low resolution
                blending_factor=0.3
            )
        
        # NEW: Toroidal disruption from incompatible structure
        if source_id in self.toroidal_phase and target_id in self.toroidal_phase:
            # Incompatible structures push each other away in toroidal space
            source_theta, source_phi = self.toroidal_phase[source_id]
            target_theta, target_phi = self.toroidal_phase[target_id]
            
            # Calculate displacement needed to increase distance
            theta_diff = ((target_theta - source_theta + math.pi) % (2 * math.pi)) - math.pi
            phi_diff = ((target_phi - source_phi + math.pi) % (2 * math.pi)) - math.pi
            
            # Calculate repulsion factor
            repulsion = collapse_strength * 0.2
            
            # Apply repulsion to move points apart
            new_source_theta = (source_theta - theta_diff * repulsion) % (2 * math.pi)
            new_source_phi = (source_phi - phi_diff * repulsion) % (2 * math.pi)
            
            new_target_theta = (target_theta + theta_diff * repulsion) % (2 * math.pi)
            new_target_phi = (target_phi + phi_diff * repulsion) % (2 * math.pi)
            
            # Update phases
            self.set_toroidal_phase(source_id, new_source_theta, new_source_phi)
            self.set_toroidal_phase(target_id, new_target_theta, new_target_phi)
            
            # Decrease phase stability
            if source_id in self.phase_stability:
                self.phase_stability[source_id] = max(0.1, self.phase_stability[source_id] - 0.2)
            if target_id in self.phase_stability:
                self.phase_stability[target_id] = max(0.1, self.phase_stability[target_id] - 0.2)
    
    def _register_void_formation(self, center_id: str):
        """
        Register a void formation event.
        Tracks void formation history and handles associated effects.
        Enhanced with toroidal void positioning.
        """
        # Get toroidal coordinates for void
        theta, phi = self.get_toroidal_phase(center_id)
        
        # Log void formation event
        event = {
            'time': self.time,
            'center_id': center_id,
            'related_ids': list(self.grains[center_id].relations.keys() if center_id in self.grains else []),
            'strength': 0.3,  # Initial void strength
            'theta': theta,
            'phi': phi
        }
        
        self.void_formation_events.append(event)
        
        # Process void formation effects
        grain = self.get_grain(center_id)
        if grain:
            # Void formation affects awareness
            awareness_reduction = 0.1
            grain.awareness = max(0.0, grain.awareness - awareness_reduction)
            
            # Void formation affects activation
            activation_reduction = 0.2
            grain.grain_activation = max(0.0, grain.grain_activation - activation_reduction)
            
            # Update configuration space awareness
            self.config_space.set_awareness(center_id, grain.awareness)
            
            # NEW: Void formation affects phase stability
            if center_id in self.phase_stability:
                self.phase_stability[center_id] = max(0.1, self.phase_stability[center_id] - 0.3)
    
    def process_void_and_decay(self, dt: float = 1.0):
        """
        Process void regions and decay particles in the system.
        Updates void presence, handles decay emissions, and propagates effects.
        Enhanced with toroidal propagation patterns.
        """
        # Process void regions in configuration space
        self.config_space.process_void_regions(dt)
        
        # Process decay particles
        self.config_space.process_decay_particles(dt)
        
        # Sync void presence from config space to grains
        for grain_id, grain in self.grains.items():
            config_point = self.config_space.get_point(grain_id)
            if config_point and hasattr(config_point, 'void_presence'):
                # Apply void effects to grain
                void_presence = config_point.void_presence
                if void_presence > 0.1:
                    # Void reduces awareness and activation
                    awareness_reduction = void_presence * 0.05 * dt
                    activation_reduction = void_presence * 0.1 * dt
                    
                    grain.awareness = max(0.0, grain.awareness - awareness_reduction)
                    grain.grain_activation = max(0.0, grain.grain_activation - activation_reduction)
                    
                    # NEW: Void affects phase stability
                    if grain_id in self.phase_stability and void_presence > 0.2:
                        stability_reduction = void_presence * 0.05 * dt
                        self.phase_stability[grain_id] = max(0.1, self.phase_stability[grain_id] - stability_reduction)
                    
                    # NEW: Void can cause phase drift in toroidal space
                    if grain_id in self.toroidal_phase and void_presence > 0.3 and random.random() < 0.3:
                        theta, phi = self.toroidal_phase[grain_id]
                        
                        # Random drift proportional to void presence
                        drift_amount = void_presence * 0.1 * dt
                        theta_drift = random.uniform(-drift_amount, drift_amount)
                        phi_drift = random.uniform(-drift_amount, drift_amount)
                        
                        # Apply drift
                        new_theta = (theta + theta_drift) % (2 * math.pi)
                        new_phi = (phi + phi_drift) % (2 * math.pi)
                        
                        # Update phase
                        self.set_toroidal_phase(grain_id, new_theta, new_phi)
        
        # Check for new decay emissions
        self._process_decay_emissions()
    
    def _process_decay_emissions(self):
        """
        Process decay particle emissions and their effects.
        Decay particles affect the system in various ways.
        Enhanced with toroidal propagation patterns.
        """
        # Check decay particles in configuration space
        for particle in self.config_space.decay_particles:
            # Only process new particles not yet registered
            if 'processed' not in particle or not particle['processed']:
                # Mark as processed
                particle['processed'] = True
                
                # Get toroidal position if available
                theta = None
                phi = None
                if 'toroidal_position' in particle:
                    theta = particle['toroidal_position'].get('theta')
                    phi = particle['toroidal_position'].get('phi')
                
                # Log decay emission
                event = {
                    'time': self.time,
                    'origin_id': particle['origin_id'],
                    'strength': particle['strength'],
                    'memory_trace': particle.get('memory_trace', []),
                    'theta': theta,
                    'phi': phi
                }
                
                self.decay_emission_events.append(event)
                
                # Apply decay effects
                self._apply_decay_effects(particle)
    
    def _apply_decay_effects(self, particle):
        """
        Apply effects of a decay particle to the system.
        Decay affects memory, relations, and field properties.
        Enhanced with toroidal propagation patterns.
        """
        # Check for toroidal propagation
        if 'toroidal_position' in particle:
            # Get particle position on torus
            theta = particle['toroidal_position'].get('theta', 0.0)
            phi = particle['toroidal_position'].get('phi', 0.0)
            effect_radius = 0.3  # Effect radius on torus
            
            # Find grains within effect radius
            affected_grains = []
            for grain_id, grain in self.grains.items():
                grain_theta, grain_phi = self.get_toroidal_phase(grain_id)
                
                # Calculate toroidal distance
                distance = toroidal_distance(theta, phi, grain_theta, grain_phi)
                
                if distance < effect_radius:
                    # Include distance for falloff calculations
                    affected_grains.append((grain_id, distance))
            
            # Apply effects to affected grains
            for grain_id, distance in affected_grains:
                grain = self.get_grain(grain_id)
                if not grain:
                    continue
                
                # Calculate effect strength based on distance
                distance_factor = 1.0 - distance / effect_radius
                effect_strength = particle['strength'] * distance_factor
                
                # Apply various effects
                self._apply_decay_to_grain(grain_id, particle['origin_id'], effect_strength)
                
                # Record impact
                if 'affected_nodes' not in particle:
                    particle['affected_nodes'] = set()
                particle['affected_nodes'].add(grain_id)
            
            return
        
        # Fallback to non-toroidal behavior
        origin_id = particle['origin_id']
        origin_grain = self.get_grain(origin_id)
        
        if not origin_grain:
            return
        
        # Get neighboring grains
        neighbors = self.get_related_grains(origin_id)
        random.shuffle(neighbors)  # Randomize affected neighbors
        
        # Limit propagation
        max_affected = min(len(neighbors), 3)
        affected_grains = neighbors[:max_affected]
        
        for grain in affected_grains:
            # Calculate effect strength
            effect_strength = particle['strength'] * random.uniform(0.5, 1.0)
            
            # Apply effects
            self._apply_decay_to_grain(grain.id, origin_id, effect_strength)
            
            # Record impact
            if 'affected_nodes' not in particle:
                particle['affected_nodes'] = set()
            particle['affected_nodes'].add(grain.id)
    
    def _apply_decay_to_grain(self, grain_id: str, origin_id: str, effect_strength: float):
        """
        Apply decay effects to a specific grain.
        Helper method for _apply_decay_effects.
        """
        grain = self.get_grain(grain_id)
        if not grain:
            return
        
        # 1. Affect void presence
        if random.random() < 0.4:
            # Decay reduces void presence
            config_point = self.config_space.get_point(grain_id)
            if config_point and hasattr(config_point, 'void_presence'):
                void_effect = -effect_strength * 0.3
                config_point.update_void_presence(void_effect)
        
        # 2. Affect awareness
        if random.random() < 0.6:
            if grain.awareness > 0.7:
                # Reduce high awareness
                grain.awareness = max(0.0, grain.awareness - effect_strength * 0.1)
            else:
                # Potentially increase low awareness
                grain.awareness = min(1.0, grain.awareness + effect_strength * 0.05)
        
        # 3. Affect epistemology
        self._apply_decay_to_epistemology(grain_id, origin_id, effect_strength)
        
        # 4. NEW: Affect toroidal phase (can cause phase fluctuations)
        if grain_id in self.toroidal_phase and random.random() < 0.4:
            theta, phi = self.toroidal_phase[grain_id]
            
            # Calculate phase shifts (can be positive or negative)
            theta_shift = (random.random() - 0.5) * effect_strength * 0.2
            phi_shift = (random.random() - 0.5) * effect_strength * 0.2
            
            # Apply shifts
            new_theta = (theta + theta_shift) % (2 * math.pi)
            new_phi = (phi + phi_shift) % (2 * math.pi)
            
            # Update phase
            self.set_toroidal_phase(grain_id, new_theta, new_phi)
        
        # 5. NEW: Affect phase stability
        if grain_id in self.phase_stability and random.random() < 0.3:
            # Decay can either increase or decrease stability
            if random.random() < 0.7:  # More likely to decrease
                stability_change = -effect_strength * 0.15
            else:
                stability_change = effect_strength * 0.1
                
            self.phase_stability[grain_id] = max(0.1, min(1.0, 
                                              self.phase_stability[grain_id] + stability_change))
    
    def _apply_decay_to_epistemology(self, grain_id: str, origin_id: str, effect_strength: float):
        """
        Apply decay effects to the epistemology relations.
        Decay can affect resolution, frustration, and fidelity components.
        """
        # Get the epistemology relation
        relation = self.epistemology_field.get_relation(grain_id, origin_id)
        if not relation:
            return
        
        # Probabilistic effects to different components
        if random.random() < 0.4:
            # Affect frustration - decay can reduce frustration
            frustration_change = -effect_strength * 0.2
            new_frustration = max(0.0, relation.frustration + frustration_change)
            
            # Update epistemology relation
            self.epistemology_field.update_relation(
                grain_id, origin_id,
                frustration=new_frustration,
                blending_factor=0.2
            )
        
        if random.random() < 0.3:
            # Affect resolution - decay can affect resolution in either direction
            if relation.resolution > 0.7:
                # Reduce high resolution
                resolution_change = -effect_strength * 0.15
            else:
                # Increase low resolution
                resolution_change = effect_strength * 0.1
            
            new_resolution = max(0.0, min(1.0, relation.resolution + resolution_change))
            
            # Update epistemology relation
            self.epistemology_field.update_relation(
                grain_id, origin_id,
                resolution=new_resolution,
                blending_factor=0.2
            )
    
    def _update_grain_activation_from_collapse(self, grain: Grain, relation_strength: float, collapse_strength: float):
        """Update grain activation and saturation from a collapse event"""
        # Update activation with smooth transition
        current_activation = grain.grain_activation
        activation_increment = 0.2 * relation_strength
        target_activation = min(1.0, current_activation + activation_increment)
        grain.grain_activation = current_activation * 0.7 + target_activation * 0.3
        
        # Update saturation based on collapse metric and activation
        current_saturation = grain.grain_saturation
        saturation_increment = 0.1 * grain.collapse_metric * grain.grain_activation
        target_saturation = min(1.0, current_saturation + saturation_increment)
        grain.grain_saturation = current_saturation * 0.8 + target_saturation * 0.2
        
        # Sync with configuration space
        self.config_space.set_grain_activation(grain.id, grain.grain_activation)
    
    def _update_epistemology_from_collapse(self, source_id: str, target_id: str, 
                                         collapse_strength: float, saturation: float):
        """Update epistemology field from a collapse event"""
        # Calculate ancestry similarity
        source = self.get_grain(source_id)
        target = self.get_grain(target_id)
        
        if not source or not target:
            return
        
        # Ancestry similarity is the proportion of shared ancestors
        if source.ancestry and target.ancestry:
            shared_ancestors = source.ancestry.intersection(target.ancestry)
            total_ancestors = source.ancestry.union(target.ancestry)
            ancestry_similarity = len(shared_ancestors) / len(total_ancestors) if total_ancestors else 0.0
        else:
            ancestry_similarity = 0.0
        
        # Update epistemology field
        self.epistemology_field.collapse_update(
            source_id, target_id,
            collapse_strength=collapse_strength,
            saturation=saturation,
            ancestry_similarity=ancestry_similarity
        )
        
        # Update phase memory based on rotation
        # Get updated relations
        source_target_rel = self.epistemology_field.get_relation(source_id, target_id)
        target_source_rel = self.epistemology_field.get_relation(target_id, source_id)
        
        if source_target_rel and target_source_rel:
            # Calculate rotation angle
            orientation1 = source_target_rel.orientation
            orientation2 = target_source_rel.orientation
            
            # Calculate the smallest angle difference
            angle_diff = ((orientation2 - orientation1 + math.pi) % (2 * math.pi)) - math.pi
            rotation = abs(angle_diff)
            
            # Initialize phase memory if needed
            if source_id not in self.phase_memory:
                self.phase_memory[source_id] = 0.0
            if target_id not in self.phase_memory:
                self.phase_memory[target_id] = 0.0
            
            # Update phase memory
            phase_contribution = rotation * collapse_strength * 0.1
            self.phase_memory[source_id] = (self.phase_memory[source_id] + phase_contribution) % (4 * math.pi)
            self.phase_memory[target_id] = (self.phase_memory[target_id] + phase_contribution) % (4 * math.pi)
            
            # NEW: Update rotation tensor
            self.rotation_tensor.update(
                source_id, target_id, 
                source_target_rel, 
                target_source_rel, 
                self.time
            )
    
    def update_field_gradients(self):
        """Update field gradients (awareness differences) for all grains"""
        for grain_id, grain in self.grains.items():
            related_grains = self.get_related_grains(grain_id)
            
            for related_grain in related_grains:
                # Calculate gradient (awareness difference)
                gradient = related_grain.awareness - grain.awareness
                grain.field_gradients[related_grain.id] = gradient
                
                # Update configuration space awareness
                self.config_space.set_awareness(grain.id, grain.awareness)
                self.config_space.set_awareness(related_grain.id, related_grain.awareness)
    
    def update_continuous_fields(self):
        """
        Update field dynamics as continuous processes.
        This implements field diffusion, gradient effects, and grain activation.
        Enhanced with void-decay handling and toroidal dynamics.
        """
        # First update field gradients
        self.update_field_gradients()
        
        # Create buffers for updates to avoid sequential update artifacts
        awareness_updates = {grain_id: 0.0 for grain_id in self.grains}
        collapse_updates = {grain_id: 0.0 for grain_id in self.grains}
        grain_activation_updates = {grain_id: 0.0 for grain_id in self.grains}
        
        # Calculate field diffusion
        for grain_id, grain in self.grains.items():
            # Calculate diffusion from related neighbors
            related_grains = self.get_related_grains(grain_id)
            
            for related_grain in related_grains:
                # Get relation strength
                relation_strength = grain.relations[related_grain.id]
                
                # Calculate awareness diffusion
                awareness_diff = related_grain.awareness - grain.awareness
                awareness_flow = awareness_diff * relation_strength * self.field_diffusion_rate
                
                # Void presence reduces flow
                config_grain = self.config_space.get_point(grain_id)
                if config_grain and hasattr(config_grain, 'void_presence') and config_grain.void_presence > 0.1:
                    # Reduce flow based on void presence
                    void_factor = 1.0 - config_grain.void_presence * 0.5
                    awareness_flow *= void_factor
                
                # Apply to buffers
                awareness_updates[grain_id] += awareness_flow
                awareness_updates[related_grain.id] -= awareness_flow
                
                # Calculate gradient influence on collapse metric
                if abs(awareness_diff) > self.field_gradient_sensitivity:
                    collapse_increment = abs(awareness_diff) * relation_strength * 0.1
                    collapse_updates[grain_id] += collapse_increment
                    collapse_updates[related_grain.id] += collapse_increment * 0.7
                
                # Calculate gradient influence on grain activation
                if abs(awareness_diff) > 0.3:  # Significant gradient
                    activation_increment = abs(awareness_diff) * relation_strength * 0.1
                    grain_activation_updates[grain_id] += activation_increment
                    grain_activation_updates[related_grain.id] += activation_increment * 0.8
            
            # NEW: Calculate additional diffusion from toroidal neighbors
            toroidal_neighbors = self.get_toroidal_neighbors(grain_id)
            
            for neighbor in toroidal_neighbors:
                # Skip if already processed as a direct relation
                if neighbor.id in grain.relations:
                    continue
                
                # Calculate toroidal proximity factor (inverse of distance)
                grain_theta, grain_phi = self.get_toroidal_phase(grain_id)
                neighbor_theta, neighbor_phi = self.get_toroidal_phase(neighbor.id)
                
                distance = toroidal_distance(grain_theta, grain_phi, neighbor_theta, neighbor_phi)
                proximity = max(0.0, 1.0 - distance / self.toroidal_system.neighborhood_radius)
                
                # Calculate phase stability influence
                grain_stability = self.phase_stability.get(grain_id, 0.5)
                neighbor_stability = self.phase_stability.get(neighbor.id, 0.5)
                
                # More stable phases have stronger influence
                stability_factor = (grain_stability + neighbor_stability) / 2
                
                # Calculate awareness diffusion based on proximity
                awareness_diff = neighbor.awareness - grain.awareness
                awareness_flow = awareness_diff * proximity * stability_factor * self.field_diffusion_rate * 0.5
                
                # Apply to buffers (reduced effect compared to direct relations)
                awareness_updates[grain_id] += awareness_flow
                awareness_updates[neighbor.id] -= awareness_flow * 0.5  # Asymmetric effect
        
        # Apply all updates
        for grain_id, grain in self.grains.items():
            # Check for void presence which can affect updates
            config_grain = self.config_space.get_point(grain_id)
            void_presence = getattr(config_grain, 'void_presence', 0.0) if config_grain else 0.0
            
            # Get phase stability
            phase_stability = self.phase_stability.get(grain_id, 0.5)
            
            # Apply awareness diffusion
            awareness_update = awareness_updates[grain_id]
            
            # Void presence can create turbulence in awareness field
            if void_presence > 0.3 and random.random() < void_presence * 0.3:
                # Random turbulence in void region
                awareness_update += random.uniform(-0.1, 0.1) * void_presence
            
            # NEW: Phase stability affects field updates
            # More stable phases resist random fluctuations
            if phase_stability < 0.5 and random.random() < (0.5 - phase_stability) * 0.5:
                # Add noise for unstable phases
                awareness_update += random.uniform(-0.1, 0.1) * (0.5 - phase_stability)
            
            grain.awareness += awareness_update
            grain.awareness = max(0.0, min(1.0, grain.awareness))  # Clamp to valid range
            
            # Apply collapse metric updates
            collapse_update = collapse_updates[grain_id]
            
            # Void presence can affect collapse metric updates
            if void_presence > 0.3:
                # Reduce collapse updates in void regions
                collapse_update *= (1.0 - void_presence * 0.5)
            
            grain.collapse_metric = min(1.0, grain.collapse_metric + collapse_update)
            
            # Apply grain activation updates
            current_activation = grain.grain_activation
            activation_update = grain_activation_updates[grain_id]
            
            # Void presence reduces activation updates
            if void_presence > 0.2:
                activation_update *= (1.0 - void_presence * 0.7)
            
            # Only apply activation if above threshold or already active
            if current_activation > 0.3 or activation_update > 0.2:
                target_activation = min(1.0, current_activation + activation_update)
                grain.grain_activation = current_activation * 0.9 + target_activation * 0.1
            else:
                # Slight decay for inactive grains
                grain.grain_activation = max(0.0, current_activation * 0.99)
            
            # Update saturation continuously based on activation and collapse
            if grain.grain_activation > 0.5:
                saturation_increment = grain.grain_activation * grain.collapse_metric * 0.01
                
                # Void presence slows saturation
                if void_presence > 0.2:
                    saturation_increment *= (1.0 - void_presence * 0.8)
                
                grain.grain_saturation = min(1.0, grain.grain_saturation + saturation_increment)
            
            # Sync updated values with configuration space
            self.config_space.set_awareness(grain_id, grain.awareness)
            self.config_space.set_collapse_metric(grain_id, grain.collapse_metric)
            self.config_space.set_grain_activation(grain_id, grain.grain_activation)
            
            # NEW: Update toroidal phase based on field dynamics
            # Only apply small changes to maintain continuity
            if grain_id in self.toroidal_phase:
                theta, phi = self.toroidal_phase[grain_id]
                
                # Calculate phase drift based on awareness and activation
                # High awareness can drive phase movement
                if grain.awareness > 0.7 and grain.grain_activation > 0.5:
                    # Calculate drift in the direction of the toroidal field
                    field_drift = 0.01
                    
                    # Determine preferred drift direction based on neighbors
                    theta_tendency = 0.0
                    phi_tendency = 0.0
                    
                    for neighbor in self.get_toroidal_neighbors(grain_id):
                        if neighbor.awareness > grain.awareness:
                            # Drift toward high-awareness neighbors
                            n_theta, n_phi = self.get_toroidal_phase(neighbor.id)
                            theta_diff = ((n_theta - theta + math.pi) % (2 * math.pi)) - math.pi
                            phi_diff = ((n_phi - phi + math.pi) % (2 * math.pi)) - math.pi
                            
                            # Weight by awareness difference
                            weight = (neighbor.awareness - grain.awareness) * 2.0
                            
                            theta_tendency += theta_diff * weight
                            phi_tendency += phi_diff * weight
                    
                    # Normalize tendencies
                    if abs(theta_tendency) > 0 or abs(phi_tendency) > 0:
                        magnitude = math.sqrt(theta_tendency**2 + phi_tendency**2)
                        if magnitude > 0:
                            theta_tendency /= magnitude
                            phi_tendency /= magnitude
                    
                    # Apply drift weighted by phase stability
                    # Stable phases move more predictably
                    theta_drift = theta_tendency * field_drift * phase_stability
                    phi_drift = phi_tendency * field_drift * phase_stability
                    
                    # Apply drift
                    new_theta = (theta + theta_drift) % (2 * math.pi)
                    new_phi = (phi + phi_drift) % (2 * math.pi)
                    
                    # Update phase
                    self.set_toroidal_phase(grain_id, new_theta, new_phi)
        
        # Process void and decay after field updates
        dt = 1.0  # Default time delta
        self.process_void_and_decay(dt)
        
        # NEW: Update toroidal metrics after field updates
        self.update_toroidal_metrics()
    
    def update_toroidal_metrics(self):
        """
        Update global toroidal metrics based on current state.
        This includes phase coherence, flow patterns, and vortex detection.
        """
        # Calculate global phase coherence
        self.toroidal_metrics['global_phase_coherence'] = self.calculate_phase_coherence()
        
        # Calculate major and minor circle flows
        major_flow, minor_flow = self.calculate_toroidal_flows()
        self.toroidal_metrics['major_circle_flow'] = major_flow
        self.toroidal_metrics['minor_circle_flow'] = minor_flow
        
        # Update toroidal flux (combined flow magnitude)
        self.toroidal_flux = math.sqrt(major_flow**2 + minor_flow**2)
        
        # Detect vortices
        vortices = self.detect_vortices()
        self.toroidal_metrics['toroidal_vortex_count'] = len(vortices)
    
    def calculate_phase_coherence(self) -> float:
        """
        Calculate global phase coherence across all grains.
        Higher values indicate more coherent phase patterns.
        
        Returns:
            Phase coherence value (0.0 to 1.0)
        """
        # Get all phase values
        thetas = []
        phis = []
        
        for grain_id in self.grains:
            theta, phi = self.get_toroidal_phase(grain_id)
            thetas.append(theta)
            phis.append(phi)
        
        if not thetas:
            return 0.0
            
        # Calculate circular variance for theta and phi components
        # For theta component
        sin_theta = sum(math.sin(t) for t in thetas)
        cos_theta = sum(math.cos(t) for t in thetas)
        r_theta = math.sqrt(sin_theta**2 + cos_theta**2) / len(thetas)
        var_theta = 1.0 - r_theta
        
        # For phi component
        sin_phi = sum(math.sin(p) for p in phis)
        cos_phi = sum(math.cos(p) for p in phis)
        r_phi = math.sqrt(sin_phi**2 + cos_phi**2) / len(phis)
        var_phi = 1.0 - r_phi
        
        # Combine components (lower variance = higher coherence)
        theta_coherence = 1.0 - var_theta
        phi_coherence = 1.0 - var_phi
        
        # Weight theta more heavily (major circle dominance)
        combined_coherence = 0.7 * theta_coherence + 0.3 * phi_coherence
        
        return combined_coherence

    def calculate_toroidal_flows(self) -> Tuple[float, float]:
        """
        Calculate the average flow around the major and minor circles.
        Positive values indicate counterclockwise flow, negative indicate clockwise.
    
        Returns:
            Tuple of (major_circle_flow, minor_circle_flow)
        """
        # Initialize flow accumulations
        major_flow = 0.0
        minor_flow = 0.0
        relation_count = 0
    
        # Check all grain relations
        for grain_id, grain in self.grains.items():
            grain_theta, grain_phi = self.get_toroidal_phase(grain_id)
        
            for related_id in grain.relations:
                related = self.get_grain(related_id)
                if not related:
                    continue
            
                related_theta, related_phi = self.get_toroidal_phase(related_id)
            
                # Calculate angular differences
                theta_diff = ((related_theta - grain_theta + math.pi) % (2 * math.pi)) - math.pi
                phi_diff = ((related_phi - grain_phi + math.pi) % (2 * math.pi)) - math.pi
            
                # Weight by awareness gradient (flow direction)
                gradient = related.awareness - grain.awareness
            
                if abs(gradient) > 0.1:  # Only count significant gradients
                    flow_weight = gradient * grain.relations[related_id]
                
                    # Contribute to flow components
                    major_flow += theta_diff * flow_weight
                    minor_flow += phi_diff * flow_weight
                    relation_count += 1
    
        # Normalize flows
        if relation_count > 0:
            major_flow /= relation_count
            minor_flow /= relation_count
    
        return (major_flow, minor_flow)

    def get_toroidal_flow(self, grain_id: str, neighbor_ids: List[str]) -> Tuple[float, float]:
            """
            Calculate the toroidal flow vector for a grain relative to its neighbors.
    
            Args:
                grain_id: ID of the center grain
                neighbor_ids: List of neighbor grain IDs
        
            Returns:
                Tuple of (theta_flow, phi_flow) representing flow along major and minor circles
            """
            # Default values
            theta_flow = 0.0
            phi_flow = 0.0
    
            grain = self.get_grain(grain_id)
            if not grain or not neighbor_ids:
                return (theta_flow, phi_flow)
    
            # Try to get flow from toroidal system if available
            if hasattr(self, 'toroidal_system') and hasattr(self.toroidal_system, 'calculate_flow_field'):
                # Get flow field for all grains
                flow_field = self.toroidal_system.calculate_flow_field()
                if grain_id in flow_field:
                    return flow_field[grain_id]
    
            # Manual calculation if toroidal system not available
            count = 0
    
            # Get center grain's toroidal position
            center_theta, center_phi = self.get_toroidal_phase(grain_id)
    
            for neighbor_id in neighbor_ids:
                neighbor = self.get_grain(neighbor_id)
                if not neighbor:
                    continue
        
                # Get neighbor's toroidal position
                neighbor_theta, neighbor_phi = self.get_toroidal_phase(neighbor_id)
        
                # Calculate angular differences
                theta_diff = ((neighbor_theta - center_theta + math.pi) % (2 * math.pi)) - math.pi
                phi_diff = ((neighbor_phi - center_phi + math.pi) % (2 * math.pi)) - math.pi
        
                # Weight by awareness gradient
                awareness_diff = neighbor.awareness - grain.awareness
        
                # Only consider significant gradients
                if abs(awareness_diff) > 0.1:
                    # Flow direction determined by awareness gradient
                    flow_weight = awareness_diff * grain.relations.get(neighbor_id, 0.5)
            
                    # Add to flow components
                    theta_flow += theta_diff * flow_weight
                    phi_flow += phi_diff * flow_weight
                    count += 1
    
            # Normalize by neighbor count
            if count > 0:
                theta_flow /= count
                phi_flow /= count
    
            return (theta_flow, phi_flow)

    def detect_vortices(self) -> List[Dict[str, Any]]:
        """
        Detect vortex-like structures in the manifold based on rotational curvature.
    
        Returns:
            List of vortex data dictionaries
        """
        # Check if using the epistemology field approach
        if hasattr(self, 'epistemology_field') and hasattr(self.epistemology_field, 'detect_vortices'):
            # Delegate to the epistemology field's implementation
            return self.epistemology_field.detect_vortices(self.grains)
    
        # Fallback implementation if epistemology field not available
        vortices = []
    
        for grain_id, grain in self.grains.items():
            # Get neighbors
            neighbor_ids = list(grain.relations.keys())
        
            if len(neighbor_ids) >= 3:  # Need at least 3 neighbors for a meaningful vortex
                # Get toroidal position if available
                theta = phi = 0.0
                if hasattr(self, 'get_toroidal_phase'):
                    theta, phi = self.get_toroidal_phase(grain_id)
            
                # Calculate flow circulation if available
                circulation = 0.0
                if hasattr(grain, 'flow_circulation'):
                    circulation = grain.flow_circulation
                elif hasattr(self, 'toroidal_system') and hasattr(self.toroidal_system, 'calculate_circulation'):
                    circulations = self.toroidal_system.calculate_circulation()
                    circulation = circulations.get(grain_id, 0.0)
            
                # Significant circulation might indicate a vortex
                if abs(circulation) > 0.5:
                    # Determine rotation direction
                    rotation_direction = "clockwise" if circulation > 0 else "counterclockwise"
                
                    # Determine vortex pattern type
                    pattern_type = "mixed"
                    if hasattr(self, 'get_toroidal_flow'):
                        theta_flow, phi_flow = self.get_toroidal_flow(grain_id, neighbor_ids)
                        if abs(theta_flow) > abs(phi_flow) * 2:
                            pattern_type = "major_circle"
                        elif abs(phi_flow) > abs(theta_flow) * 2:
                            pattern_type = "minor_circle"
                
                    # Add to vortices list
                    vortices.append({
                        'center_node': grain_id,
                        'strength': abs(circulation),
                        'rotation_direction': rotation_direction,
                        'neighbor_count': len(neighbor_ids),
                        'theta': theta,
                        'phi': phi,
                        'pattern_type': pattern_type
                    })
    
        return vortices
        
    def find_collapse_targets(self) -> List[Dict[str, Any]]:
        """
        Find potential collapse target grains based on collapse metrics and grain activation.
        
        Returns:
            List of dictionaries with grain_id and score for potential targets
        """
        targets = []
        
        # Iterate through all grains
        for grain_id, grain in self.grains.items():
            # Skip grains with too high saturation (already saturated)
            if grain.grain_saturation > 0.9:
                continue
            
            # Calculate collapse potential score
            # Higher collapse metric and lower saturation means better target
            collapse_potential = grain.collapse_metric * (1.0 - grain.grain_saturation)
            
            # Awareness and activation both influence collapse potential
            awareness_factor = grain.awareness * 0.5
            activation_factor = grain.grain_activation * 0.5
            
            # Skip if activation too low
            if grain.grain_activation < self.activation_threshold:
                continue
            
            # Combined score
            score = collapse_potential * (awareness_factor + activation_factor)
            
            # Add void presence penalty if applicable
            if hasattr(self.config_space, 'get_void_presence'):
                void_presence = self.config_space.get_void_presence(grain_id)
                if void_presence > 0.2:
                    # Reduce score in void regions
                    score *= (1.0 - void_presence * 0.5)
            
            # Add to targets
            targets.append({
                'grain_id': grain_id,
                'score': score,
                'collapse_potential': collapse_potential,
                'awareness': grain.awareness,
                'activation': grain.grain_activation,
                'saturation': grain.grain_saturation
            })
        
        # Sort by score (descending)
        targets.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top targets (limited to prevent excessive computation)
        return targets[:5]  # Limit to top 5 targets

    def find_collapse_source_candidates(self, target_id: str) -> List[Dict[str, Any]]:
        """
        Find potential source grains that could collapse into the target.
        Sources influence target through collapse events.
        
        Args:
            target_id: ID of the target grain
            
        Returns:
            List of dictionaries with grain_id and score for potential sources
        """
        target = self.get_grain(target_id)
        if not target:
            return []
        
        candidates = []
        
        # First check direct relations
        for related_id, relation_strength in target.relations.items():
            related_grain = self.get_grain(related_id)
            if not related_grain:
                continue
            
            # Calculate source suitability score
            # Higher awareness and relation strength means better source
            awareness_factor = related_grain.awareness * 0.7
            relation_factor = relation_strength * 0.3
            
            # Memory influences source selection
            memory_factor = 0.0
            if related_id in target.relation_memory:
                memory = target.relation_memory[related_id]
                memory_factor = abs(memory) * 0.2  # Memory strength matters more than direction
            
            # Combined score
            score = awareness_factor + relation_factor + memory_factor
            
            # Add toroidal phase influence if available
            if hasattr(self, 'get_toroidal_phase') and hasattr(self, 'phase_stability'):
                source_theta, source_phi = self.get_toroidal_phase(related_id)
                target_theta, target_phi = self.get_toroidal_phase(target_id)
                
                # Phase stability influence
                source_stability = self.phase_stability.get(related_id, 0.5)
                
                # Calculate phase alignment
                theta_diff = ((source_theta - target_theta + math.pi) % (2 * math.pi)) - math.pi
                phi_diff = ((source_phi - target_phi + math.pi) % (2 * math.pi)) - math.pi
                
                # Smaller difference means better alignment
                phase_alignment = 1.0 - (abs(theta_diff) / math.pi + abs(phi_diff) / math.pi) / 2.0
                
                # Add stability factor to score
                phase_factor = phase_alignment * source_stability * 0.2
                score += phase_factor
            
            # Check void presence which can reduce source suitability
            if hasattr(self.config_space, 'get_void_presence'):
                void_presence = self.config_space.get_void_presence(related_id)
                if void_presence > 0.2:
                    # Reduce score in void regions
                    score *= (1.0 - void_presence * 0.6)
            
            # Add to candidates
            candidates.append({
                'grain_id': related_id,
                'score': score,
                'relation_strength': relation_strength,
                'awareness': related_grain.awareness,
                'activation': related_grain.grain_activation
            })
        
        # Next check toroidal neighbors that aren't direct relations
        # This allows collapse across toroidal space without direct connections
        if hasattr(self, 'get_toroidal_neighbors'):
            neighbors = self.get_toroidal_neighbors(target_id)
            
            for neighbor in neighbors:
                # Skip if already processed as direct relation
                if neighbor.id in target.relations:
                    continue
                
                # Calculate toroidal proximity
                target_theta, target_phi = self.get_toroidal_phase(target_id)
                neighbor_theta, neighbor_phi = self.get_toroidal_phase(neighbor.id)
                
                distance = toroidal_distance(target_theta, target_phi, neighbor_theta, neighbor_phi)
                
                # Skip if too far away
                if distance > 0.5:  # Higher threshold means less long-range influence
                    continue
                
                # Calculate proximity factor (inverse of distance)
                proximity = max(0.0, 1.0 - distance * 2.0)
                
                # Calculate source suitability score
                awareness_factor = neighbor.awareness * 0.5
                proximity_factor = proximity * 0.5
                
                # Calculate score with proximity as a substitute for relation strength
                score = awareness_factor * proximity_factor
                
                # Add to candidates with lower priority than direct relations
                candidates.append({
                    'grain_id': neighbor.id,
                    'score': score * 0.7,  # Reduce score for non-direct neighbors
                    'relation_strength': proximity,
                    'awareness': neighbor.awareness,
                    'activation': neighbor.grain_activation
                })
        
        # Sort by score (descending)
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top candidates (limited to prevent excessive computation)
        return candidates[:3]  # Limit to top 3 candidates

    def initiate_collapse(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """
        Initiate a collapse event between source and target grains.
        This is the core mechanism of the Collapse Geometry framework.
        
        Args:
            source_id: ID of the source grain
            target_id: ID of the target grain
            
        Returns:
            Dictionary with collapse event data
        """
        source = self.get_grain(source_id)
        target = self.get_grain(target_id)
        
        if not source or not target:
            return {}
        
        # Calculate collapse strength based on source awareness and activation
        base_strength = source.awareness * 0.7 + source.grain_activation * 0.3
        
        # Adjust for relational factors
        # Direct relation increases strength
        relation_factor = 1.0
        if target_id in source.relations:
            relation_factor = 1.0 + source.relations[target_id] * 0.5
        
        # Memory influences collapse strength
        memory_factor = 1.0
        if target_id in source.relation_memory:
            memory = source.relation_memory[target_id]
            # Positive memory strengthens, negative weakens
            memory_factor = 1.0 + memory * 0.3
        
        # Toroidal phase alignment influences collapse
        phase_factor = 1.0
        if hasattr(self, 'get_toroidal_phase'):
            source_theta, source_phi = self.get_toroidal_phase(source_id)
            target_theta, target_phi = self.get_toroidal_phase(target_id)
            
            # Calculate phase alignment (closer alignment = stronger collapse)
            theta_diff = abs(((source_theta - target_theta + math.pi) % (2 * math.pi)) - math.pi)
            phi_diff = abs(((source_phi - target_phi + math.pi) % (2 * math.pi)) - math.pi)
            
            # Convert to alignment factor (higher = better aligned)
            theta_alignment = 1.0 - theta_diff / math.pi
            phi_alignment = 1.0 - phi_diff / math.pi
            
            # Combined alignment
            combined_alignment = (theta_alignment * 0.7 + phi_alignment * 0.3)
            
            # Phase factor increases with alignment
            phase_factor = 1.0 + combined_alignment * 0.4
        
        # Final collapse strength with random variation
        collapse_strength = base_strength * relation_factor * memory_factor * phase_factor
        collapse_strength *= random.uniform(0.85, 1.15)  # Add some randomness
        collapse_strength = min(1.0, max(0.1, collapse_strength))  # Clamp between 0.1 and 1.0
        
        # Find grains that should be negated (not chosen) during this collapse
        # These are typically grains connected to the target that aren't the source
        negated_ids = []
        
        for related_id in target.relations:
            if related_id != source_id and related_id in self.grains:
                # Higher chance of negation for weaker relations
                relation_strength = target.relations[related_id]
                if random.random() > relation_strength:
                    negated_ids.append(related_id)
        
        # Create the collapse relation
        collapse_event = self.create_relation_from_collapse(
            source_id, target_id, 
            collapse_strength, 
            negated_ids
        )
        
        return collapse_event

    def step(self, dt: float = 1.0):
        """
        Advance the manifold by one time step.
        Updates internal time and allows for time-based effects.
        
        Args:
            dt: Time step size
        """
        # Update internal time
        self.time += dt
        
        # Process time-dependent effects
        # This is a placeholder for any time-dependent processes
        # that aren't handled by the engine
        
        # Decay collapse metric slightly over time
        for grain in self.grains.values():
            # Gradual decay of collapse metric
            grain.collapse_metric = max(0.0, grain.collapse_metric * 0.99)
        
        # Process field dynamics
        # Note: Most field dynamics are handled by the engine
        # via update_continuous_fields(), but we can add simple effects here
        
        # Time-dependent void decay processing
        if hasattr(self, 'process_void_and_decay'):
            # Process voids with smaller dt to avoid instability
            self.process_void_and_decay(dt=dt*0.5)
        
        # Simple evolution of toroidal phase for active grains
        if hasattr(self, 'toroidal_phase'):
            for grain_id, grain in self.grains.items():
                if grain.grain_activation > 0.7:
                    # Active grains may have slight phase drift
                    theta, phi = self.toroidal_phase.get(grain_id, (0, 0))
                    
                    # Calculate phase drift proportional to activation
                    drift_factor = grain.grain_activation * 0.05 * dt
                    theta_drift = random.uniform(-drift_factor, drift_factor)
                    phi_drift = random.uniform(-drift_factor, drift_factor)
                    
                    # Apply drift
                    new_theta = (theta + theta_drift) % (2 * math.pi)
                    new_phi = (phi + phi_drift) % (2 * math.pi)
                    
                    # Update phase
                    self.toroidal_phase[grain_id] = (new_theta, new_phi)

    def get_void_presence(self, grain_id: str) -> float:
        """
        Get the void presence value for a grain.
        Delegates to configuration space.
        
        Args:
            grain_id: ID of the grain
            
        Returns:
            Void presence value (0.0 to 1.0)
        """
        # Get void presence from configuration space
        config_point = self.config_space.get_point(grain_id)
        if config_point and hasattr(config_point, 'void_presence'):
            return config_point.void_presence
        
        return 0.0

    def find_void_regions(self) -> List[Dict[str, Any]]:
        """
        Find void regions in the manifold.
        Delegates to configuration space.
        
        Returns:
            List of void region data dictionaries
        """
        # Check if config space has the method
        if hasattr(self.config_space, 'find_void_regions'):
            return self.config_space.find_void_regions()
        
        # Fallback implementation if config space doesn't provide it
        void_regions = []
        
        # Check all grains for void presence
        for grain_id, grain in self.grains.items():
            void_presence = self.get_void_presence(grain_id)
            
            # Consider significant void presence
            if void_presence > 0.3:
                # Get toroidal coordinates
                theta, phi = self.get_toroidal_phase(grain_id)
                
                # Get affected grains (neighbors)
                affected_grains = [n.id for n in self.get_toroidal_neighbors(grain_id)
                                  if self.get_void_presence(n.id) > 0.1]
                
                # Add void region
                void_regions.append({
                    'id': f"void_{grain_id}",
                    'center': grain_id,
                    'strength': void_presence,
                    'radius': 0.3,
                    'theta': theta,
                    'phi': phi,
                    'affected_grains': affected_grains
                })
        
        return void_regions

    def get_incompatible_structure_stats(self) -> Dict[str, Any]:
        """
        Get statistics on incompatible structures and void-decay events.
        
        Returns:
            Dictionary with statistics
        """
        # Collect stats
        stats = {
            'void_count': 0,
            'decay_count': 0,
            'incompatible_pairs': [],
            'failure_rate': 0.0
        }
        
        # Get void regions
        void_regions = self.find_void_regions()
        stats['void_count'] = len(void_regions)
        
        # Get decay particles from config space
        if hasattr(self.config_space, 'decay_particles'):
            stats['decay_count'] = len(self.config_space.decay_particles)
        
        # Get incompatible structure pairs from events
        incompatible_pairs = set()
        for event in self.incompatible_structure_events:
            if 'source' in event and 'target' in event:
                # Add as sorted tuple to prevent duplicates with reversed order
                pair = tuple(sorted([event['source'], event['target']]))
                incompatible_pairs.add(pair)
        
        stats['incompatible_pairs'] = list(incompatible_pairs)
        
        # Calculate failure rate (incompatible / total collapse attempts)
        total_attempts = len(self.collapse_history) + len(self.incompatible_structure_events)
        if total_attempts > 0:
            stats['failure_rate'] = len(self.incompatible_structure_events) / total_attempts
        
        return stats

    def get_flow_tendency(self, grain_id: str) -> Dict[str, float]:
        """
        Get the flow tendency of a grain relative to its neighbors.
        This represents how awareness tends to flow between the grain and its neighbors.
        
        Args:
            grain_id: ID of the grain
            
        Returns:
            Dictionary mapping neighbor_id -> flow value
        """
        grain = self.get_grain(grain_id)
        if not grain:
            return {}
        
        flow_tendencies = {}
        
        # Calculate flow tendencies for each relation
        for neighbor_id, relation_strength in grain.relations.items():
            neighbor = self.get_grain(neighbor_id)
            if not neighbor:
                continue
            
            # Calculate awareness gradient
            awareness_diff = neighbor.awareness - grain.awareness
            
            # Calculate flow tendency (positive = outward, negative = inward)
            flow = awareness_diff * relation_strength
            
            # Store flow tendency
            flow_tendencies[neighbor_id] = flow
        
        return flow_tendencies