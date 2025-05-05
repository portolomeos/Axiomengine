"""
Emergent Field Rules - Rules for how higher-level phenomena emerge from base fields

Implements the rules that govern the emergence of complex behavior, phase transitions,
and field-like properties from the fundamental collapse dynamics.
Includes templates for emergent structures like decay particles and void regions.
Enhanced with toroidal referencing for richer topological dynamics.
"""

import math
import random
import numpy as np
import warnings
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular imports at runtime
if TYPE_CHECKING:
    from axiom6.core.relational_manifold import RelationalManifold
    from axiom6.collapse_rules.grain_dynamics import Grain


# Utility functions for toroidal coordinates
def angular_difference(a: float, b: float) -> float:
    """Calculate the minimum angular difference between two angles, respecting wraparound"""
    diff = abs(a - b) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)


def toroidal_distance(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Calculate distance between two points on a torus"""
    theta_diff = angular_difference(theta1, theta2)
    phi_diff = angular_difference(phi1, phi2)
    return math.sqrt(theta_diff**2 + phi_diff**2)


def circular_mean(angles: List[float]) -> float:
    """Calculate the circular mean of a set of angles"""
    if not angles:
        return 0.0
    x = sum(math.cos(angle) for angle in angles)
    y = sum(math.sin(angle) for angle in angles)
    return math.atan2(y, x) % (2 * math.pi)


# Dictionary templates for emergent structures
DECAY_PARTICLE_TEMPLATE = {
    'origin_id': None,          # Source node ID
    'strength': 0.0,            # Particle strength
    'creation_time': 0.0,       # When the particle was created
    'impact_score': 0.0,        # Track particle's impact
    'processed': False,         # Whether it's been processed
    'memory_trace': [],         # Track memory effects
    'lifetime': 0.0,            # How long it's existed
    'affected_nodes': set(),    # Nodes affected by this particle
    'direction_vector': None,   # Optional direction of movement
    'position': None,           # Optional position in configuration space
    'toroidal_position': None,  # Position on torus (theta, phi)
    'toroidal_direction': None  # Direction on torus (theta_flow, phi_flow)
}

VOID_REGION_TEMPLATE = {
    'void_id': None,            # Unique identifier
    'center_point': None,       # Central node ID
    'strength': 0.0,            # Void strength
    'formation_time': 0.0,      # When the void formed
    'radius': 0.3,              # Influence radius
    'affected_points': None,    # Set of affected points
    'decay_emissions': 0,       # Count of emitted decay particles
    'last_emission_time': 0.0,  # When last emission occurred
    'properties': {},           # Additional properties
    'theta': None,              # Major circle position on torus
    'phi': None,                # Minor circle position on torus
    'toroidal_radius': 0.4      # Radius in toroidal space
}

# Enhanced with toroidal structure templates
TOROIDAL_VORTEX_TEMPLATE = {
    'vortex_id': None,          # Unique identifier
    'center_point': None,       # Central node ID
    'strength': 0.0,            # Vortex strength
    'formation_time': 0.0,      # When the vortex formed
    'affected_points': None,    # Set of affected points
    'rotation_direction': None, # Clockwise or counterclockwise
    'theta': None,              # Major circle position on torus
    'phi': None,                # Minor circle position on torus
    'winding_number': 0,        # Topological winding number
    'pattern_type': None        # Major circle, minor circle, or mixed
}

PHASE_DOMAIN_TEMPLATE = {
    'domain_id': None,          # Unique identifier
    'phase_type': None,         # Phase type (solid, liquid, gas, etc.)
    'points': None,             # Set of points in domain
    'formation_time': 0.0,      # When the domain formed
    'stability': 0.0,           # Domain stability
    'boundary_points': None,    # Points at phase boundary
    'theta_center': None,       # Domain center on major circle
    'phi_center': None,         # Domain center on minor circle
    'toroidal_extent': None     # Angular extent on torus
}


class EmergentFieldRule:
    """Base class for rules that identify and compute emergent field behavior"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """Evaluate the emergent field rule on the current manifold state"""
        raise NotImplementedError("Subclasses must implement evaluate method")


class VoidDecayRule(EmergentFieldRule):
    """
    Rule for handling void formation and decay particle behavior.
    Defines how voids and decay particles interact with the system.
    Enhanced with toroidal propagation patterns.
    """
    
    def __init__(self):
        super().__init__(
            name="void_decay_rules",
            description="Rules for void formation and decay particle behavior with toroidal dynamics"
        )
        
        # Configuration for void and decay behavior
        self.config = {
            'void_formation_threshold': 0.8,    # Threshold for void formation
            'void_diffusion_rate': 0.15,        # How quickly voids spread
            'decay_emission_rate': 0.2,         # Base rate of decay emission
            'decay_effect_strength': 0.3,       # How strongly decay affects nodes
            'void_impact_factor': 0.4,          # How strongly voids impact dynamics
            'toroidal_propagation_bias': 0.7,   # Bias toward major circle propagation
            'major_circle_speed': 0.15,         # Flow speed along major circle
            'minor_circle_speed': 0.08          # Flow speed along minor circle
        }
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """
        Evaluate void and decay dynamics with toroidal patterns.
        
        Args:
            manifold: The relational manifold
            
        Returns:
            Dictionary with void and decay information
        """
        # Get active void regions and decay particles
        void_regions = []
        decay_particles = []
        
        if hasattr(manifold, 'config_space') and hasattr(manifold.config_space, 'void_regions'):
            void_regions = list(manifold.config_space.void_regions.values())
        
        if hasattr(manifold, 'config_space') and hasattr(manifold.config_space, 'decay_particles'):
            decay_particles = manifold.config_space.decay_particles
        
        # Calculate void-decay metrics
        void_count = len(void_regions)
        decay_count = len(decay_particles)
        
        # Calculate affected nodes ratio
        affected_nodes = set()
        for void in void_regions:
            affected_nodes.update(void.get('affected_points', []))
        
        total_nodes = len(manifold.nodes)
        affected_ratio = len(affected_nodes) / total_nodes if total_nodes > 0 else 0.0
        
        # Calculate mean void strength and radius
        void_strengths = [void.get('strength', 0.0) for void in void_regions]
        void_radii = [void.get('radius', 0.0) for void in void_regions]
        
        mean_void_strength = sum(void_strengths) / len(void_strengths) if void_strengths else 0.0
        mean_void_radius = sum(void_radii) / len(void_radii) if void_radii else 0.0
        
        # Calculate decay emission rate (emissions per time unit)
        recent_time = max(0.0, manifold.time - 10.0)  # Last 10 time units
        recent_emissions = sum(1 for void in void_regions 
                               if void.get('last_emission_time', 0.0) > recent_time)
        emission_rate = recent_emissions / 10.0
        
        # Calculate toroidal metrics for voids
        void_positions = []
        for void in void_regions:
            if 'theta' in void and 'phi' in void:
                void_positions.append((void['theta'], void['phi']))
        
        # Calculate void clustering on torus
        void_clustering = 0.0
        if len(void_positions) >= 2:
            # Calculate average distance between voids
            distances = []
            for i in range(len(void_positions)):
                for j in range(i+1, len(void_positions)):
                    theta1, phi1 = void_positions[i]
                    theta2, phi2 = void_positions[j]
                    distances.append(toroidal_distance(theta1, phi1, theta2, phi2))
            
            mean_distance = sum(distances) / len(distances) if distances else 0.0
            # Normalize to [0,1] scale, where 1 means highly clustered
            void_clustering = 1.0 - min(1.0, mean_distance / math.pi)
        
        # Analyze decay particle trajectories
        major_circle_bias = 0.0
        if decay_particles:
            # Count particles moving primarily along major vs minor circle
            major_circle_count = 0
            for particle in decay_particles:
                if 'toroidal_direction' in particle:
                    theta_flow = abs(particle['toroidal_direction'].get('theta', 0.0))
                    phi_flow = abs(particle['toroidal_direction'].get('phi', 0.0))
                    if theta_flow > phi_flow:
                        major_circle_count += 1
            
            major_circle_bias = major_circle_count / len(decay_particles) if decay_particles else 0.0
        
        return {
            'void_count': void_count,
            'decay_count': decay_count,
            'affected_ratio': affected_ratio,
            'mean_void_strength': mean_void_strength,
            'mean_void_radius': mean_void_radius,
            'emission_rate': emission_rate,
            'void_clustering': void_clustering,
            'major_circle_bias': major_circle_bias,
            'config': self.config
        }
    
    def should_form_void(self, tension: float) -> bool:
        """Check if a void should form based on tension"""
        return tension >= self.config['void_formation_threshold']
    
    def calculate_void_strength(self, tension: float) -> float:
        """Calculate initial void strength based on tension"""
        return (tension - self.config['void_formation_threshold']) / (1.0 - self.config['void_formation_threshold'])
    
    def should_emit_decay(self, void: Dict[str, Any], current_time: float) -> bool:
        """
        Check if a void should emit a decay particle
        
        Args:
            void: Void region dictionary
            current_time: Current simulation time
            
        Returns:
            True if should emit, False otherwise
        """
        # Calculate emission threshold based on void strength
        void_strength = void.get('strength', 0.0)
        emission_threshold = 1.0 / (self.config['decay_emission_rate'] * void_strength)
        
        # Check time since last emission
        last_emission = void.get('last_emission_time', void.get('formation_time', 0.0))
        time_since_emission = current_time - last_emission
        
        return time_since_emission > emission_threshold
    
    def create_decay_particle_dict(self, origin_id: str, strength: float, 
                                  current_time: float) -> Dict[str, Any]:
        """
        Create a new decay particle dictionary with toroidal properties
        
        Args:
            origin_id: Origin node ID
            strength: Initial strength
            current_time: Current simulation time
            
        Returns:
            Decay particle dictionary
        """
        particle = DECAY_PARTICLE_TEMPLATE.copy()
        particle['origin_id'] = origin_id
        particle['strength'] = strength
        particle['creation_time'] = current_time
        particle['impact_score'] = 0.0
        particle['affected_nodes'] = set()
        
        return particle
    
    def create_void_region_dict(self, void_id: str, center_point: str, 
                              strength: float, current_time: float, 
                              theta: float = None, phi: float = None) -> Dict[str, Any]:
        """
        Create a new void region dictionary with toroidal coordinates
        
        Args:
            void_id: Unique void ID
            center_point: Center node ID
            strength: Initial strength
            current_time: Current simulation time
            theta: Major circle position (0 to 2π), randomized if None
            phi: Minor circle position (0 to 2π), randomized if None
            
        Returns:
            Void region dictionary
        """
        void = VOID_REGION_TEMPLATE.copy()
        void['void_id'] = void_id
        void['center_point'] = center_point
        void['strength'] = strength
        void['formation_time'] = current_time
        void['last_emission_time'] = current_time
        void['affected_points'] = {center_point}
        
        # Set toroidal position
        void['theta'] = theta if theta is not None else random.random() * 2 * math.pi
        void['phi'] = phi if phi is not None else random.random() * 2 * math.pi
        
        # Set toroidal radius based on strength
        void['toroidal_radius'] = 0.2 + 0.3 * strength
        
        return void


class EnhancedPhaseClassificationRule(EmergentFieldRule):
    """
    Enhanced rule for classifying regions into different phases of matter.
    Optimized for on-demand calculation of derived properties.
    Now with toroidal phase domains.
    """
    
    def __init__(self):
        super().__init__(
            name="enhanced_phase_classification",
            description="Classifies regions of the manifold into different phases of matter with toroidal domains"
        )
        
        # Define phase classification thresholds
        self.thresholds = {
            'radiant': {
                'collapse_velocity_min': 0.8,
                'grain_saturation_max': 0.3,
                'contour_max': 0.2,
                'tension_max': 0.3
            },
            'solid': {
                'grain_saturation_min': 0.9,
                'collapse_velocity_max': 0.1,
                'tension_max': 0.2
            },
            'liquid': {
                'grain_saturation_min': 0.3,
                'grain_saturation_max': 0.7,
                'collapse_velocity_min': 0.3,
                'flow_alignment_min': 0.5
            },
            'gas': {
                'grain_saturation_max': 0.3,
                'awareness_min': 0.5,
                'tension_min': 0.3
            },
            'viscous': {
                'grain_saturation_min': 0.2,
                'grain_saturation_max': 0.8,
                'contour_abs_min': 0.5,
                'flow_opposition_min': 0.4
            },
            'frozen': {
                'grain_saturation_min': 0.95,
                'collapse_velocity_max': 0.05,
                'awareness_max': 0.1,
                'tension_max': 0.1
            }
        }
        
        # Toroidal phase domain tracking
        self.phase_domains = []  # List of phase domains
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """
        Evaluate phase classification for all grains in the manifold.
        Uses on-demand calculation of field properties.
        Enhanced with toroidal phase domains.
        
        Returns:
            Dictionary with phase classifications and related metadata
        """
        # Initialize phase categories
        phases = {
            'radiant': [],
            'solid': [],
            'liquid': [],
            'gas': [],
            'viscous': [],
            'frozen': []
        }
        
        # Track phase boundaries and transition zones
        phase_boundaries = []
        phase_transitions = []
        
        # Process all grains
        for grain_id, grain in manifold.grains.items():
            # Get fundamental properties
            awareness = grain.awareness
            grain_saturation = grain.grain_saturation
            
            # Calculate derived properties on-demand
            tension = manifold.get_field_tension(grain_id)
            contour = abs(manifold.get_field_contour(grain_id))
            
            # Calculate collapse velocity from gradients
            related_grains = manifold.get_related_grains(grain_id)
            collapse_velocity = 0.0
            
            for related in related_grains:
                # Use gradient as approximation of collapse velocity
                gradient = abs(related.awareness - grain.awareness)
                collapse_velocity += gradient
            
            if related_grains:
                collapse_velocity /= len(related_grains)
            
            # Calculate flow alignment and opposition
            flow_tendencies = manifold.get_flow_tendency(grain_id)
            flow_alignment = 0.0
            flow_opposition = 0.0
            
            if flow_tendencies:
                tendencies = list(flow_tendencies.values())
                
                # Count positive and negative tendencies
                positive_count = sum(1 for t in tendencies if t > 0)
                negative_count = sum(1 for t in tendencies if t < 0)
                total_count = len(tendencies)
                
                if total_count > 0:
                    # Alignment is how uniform the flow directions are
                    flow_alignment = max(positive_count, negative_count) / total_count
                    
                    # Opposition is when flows cancel each other out
                    flow_opposition = min(positive_count, negative_count) / total_count
            
            # Check toroidal factors if available
            toroidal_contour = 0.0
            if hasattr(manifold, 'calculate_toroidal_contour'):
                toroidal_contour = manifold.calculate_toroidal_contour(grain_id, [g.id for g in related_grains])
            
            # Get toroidal phase if available
            theta_phase = 0.0
            phi_phase = 0.0
            if hasattr(manifold, 'get_toroidal_phase'):
                theta_phase, phi_phase = manifold.get_toroidal_phase(grain_id)
            
            # Check duality factors
            has_opposite = manifold.get_opposite_grain(grain_id) is not None
            negated_options = manifold.get_negated_options(grain_id)
            negated_count = len(negated_options)
            
            # Determine phase based on properties
            # 1. Radiant Phase (Light-like)
            if (collapse_velocity >= self.thresholds['radiant']['collapse_velocity_min'] and
                grain_saturation <= self.thresholds['radiant']['grain_saturation_max'] and
                contour <= self.thresholds['radiant']['contour_max'] and
                tension <= self.thresholds['radiant']['tension_max']):
                phases['radiant'].append(grain_id)
            
            # 2. Solid Phase (Rigid)
            elif (grain_saturation >= self.thresholds['solid']['grain_saturation_min'] and
                 collapse_velocity <= self.thresholds['solid']['collapse_velocity_max'] and
                 tension <= self.thresholds['solid']['tension_max']):
                phases['solid'].append(grain_id)
            
            # 3. Frozen Phase (Static Structure)
            elif (grain_saturation >= self.thresholds['frozen']['grain_saturation_min'] and
                 collapse_velocity <= self.thresholds['frozen']['collapse_velocity_max'] and
                 awareness <= self.thresholds['frozen']['awareness_max'] and
                 tension <= self.thresholds['frozen']['tension_max']):
                phases['frozen'].append(grain_id)
            
            # 4. Liquid Phase (Flowing Memory)
            elif (self.thresholds['liquid']['grain_saturation_min'] <= grain_saturation <= 
                 self.thresholds['liquid']['grain_saturation_max'] and
                 collapse_velocity >= self.thresholds['liquid']['collapse_velocity_min'] and
                 (flow_alignment >= self.thresholds['liquid']['flow_alignment_min'] or
                  toroidal_contour >= 0.6)):  # Enhanced with toroidal contour check
                phases['liquid'].append(grain_id)
            
            # 5. Gas Phase (Free Collapse)
            elif (grain_saturation <= self.thresholds['gas']['grain_saturation_max'] and
                 awareness >= self.thresholds['gas']['awareness_min'] and
                 tension >= self.thresholds['gas']['tension_min']):
                phases['gas'].append(grain_id)
            
            # 6. Viscous Phase (Resisted Collapse)
            elif (self.thresholds['viscous']['grain_saturation_min'] <= grain_saturation <= 
                 self.thresholds['viscous']['grain_saturation_max'] and
                 (contour >= self.thresholds['viscous']['contour_abs_min'] or
                  toroidal_contour >= 0.5) and  # Enhanced with toroidal contour
                 flow_opposition >= self.thresholds['viscous']['flow_opposition_min']):
                phases['viscous'].append(grain_id)
            
            # Default classification based on weighted approach
            else:
                # Calculate scores for each phase
                scores = {
                    'solid': grain_saturation * (1 - min(1, collapse_velocity/0.2)),
                    'liquid': (0.7 - abs(0.5 - grain_saturation)) * min(1, collapse_velocity/0.4) * max(0.1, flow_alignment),
                    'gas': (1 - grain_saturation) * awareness * max(0.1, tension),
                    'viscous': (0.8 - abs(0.5 - grain_saturation)) * min(1, contour/0.6) * max(0.1, flow_opposition),
                    'radiant': min(1, collapse_velocity/0.9) * (1 - grain_saturation) * (1 - min(1, contour/0.3)),
                    'frozen': grain_saturation * (1 - min(1, collapse_velocity/0.1)) * (1 - min(1, awareness/0.2))
                }
                
                # Enhance scores with toroidal factors
                if toroidal_contour > 0:
                    # Boost liquid score for high toroidal contour
                    scores['liquid'] += toroidal_contour * 0.2
                    
                    # Boost viscous score for moderate toroidal contour with opposition
                    if flow_opposition > 0.2:
                        scores['viscous'] += toroidal_contour * flow_opposition * 0.3
                
                # Assign to phase with highest score
                best_phase = max(scores, key=scores.get)
                phases[best_phase].append(grain_id)
                
                # Check for phase transition
                best_score = scores[best_phase]
                
                # Find second-best phase
                scores_copy = scores.copy()
                scores_copy.pop(best_phase)
                second_best = max(scores_copy, key=scores_copy.get)
                second_score = scores_copy[second_best]
                
                # If scores are close, this is a transition zone
                if best_score - second_score < 0.2:
                    phase_transitions.append({
                        'grain_id': grain_id,
                        'primary_phase': best_phase,
                        'secondary_phase': second_best,
                        'transition_strength': 1 - (best_score - second_score),
                        'theta_phase': theta_phase,
                        'phi_phase': phi_phase
                    })
            
            # Check for phase boundaries by looking at neighbors
            grain_phase = None
            for phase, grains in phases.items():
                if grain_id in grains:
                    grain_phase = phase
                    break
            
            if grain_phase and related_grains:
                for related_grain in related_grains:
                    # Find the phase of related grain
                    related_phase = None
                    for phase, grains in phases.items():
                        if related_grain.id in grains:
                            related_phase = phase
                            break
                    
                    # If phases differ, this is a boundary
                    if related_phase and related_phase != grain_phase:
                        # Get toroidal positions if available
                        related_theta = 0.0
                        related_phi = 0.0
                        if hasattr(manifold, 'get_toroidal_phase'):
                            related_theta, related_phi = manifold.get_toroidal_phase(related_grain.id)
                        
                        phase_boundaries.append({
                            'grain1': grain_id,
                            'grain2': related_grain.id,
                            'phase1': grain_phase,
                            'phase2': related_phase,
                            'relation_strength': grain.relations.get(related_grain.id, 0),
                            'theta1': theta_phase,
                            'phi1': phi_phase,
                            'theta2': related_theta,
                            'phi2': related_phi
                        })
        
        # Enhance with duality information
        duality_enhancements = {}
        
        # Check for phase opposition in opposite grains
        for pair in manifold.opposite_pairs:
            grain1_id, grain2_id = pair
            
            # Find phases for both grains
            grain1_phase = None
            grain2_phase = None
            
            for phase, grains in phases.items():
                if grain1_id in grains:
                    grain1_phase = phase
                if grain2_id in grains:
                    grain2_phase = phase
            
            if grain1_phase and grain2_phase and grain1_phase != grain2_phase:
                # Get toroidal positions if available
                grain1_theta = 0.0
                grain1_phi = 0.0
                grain2_theta = 0.0
                grain2_phi = 0.0
                if hasattr(manifold, 'get_toroidal_phase'):
                    grain1_theta, grain1_phi = manifold.get_toroidal_phase(grain1_id)
                    grain2_theta, grain2_phi = manifold.get_toroidal_phase(grain2_id)
                
                duality_enhancements[grain1_id] = {
                    'phase': grain1_phase,
                    'opposite_phase': grain2_phase,
                    'enhancement_type': 'phase_opposition',
                    'theta': grain1_theta,
                    'phi': grain1_phi,
                    'opposite_theta': grain2_theta,
                    'opposite_phi': grain2_phi
                }
                
                duality_enhancements[grain2_id] = {
                    'phase': grain2_phase,
                    'opposite_phase': grain1_phase,
                    'enhancement_type': 'phase_opposition',
                    'theta': grain2_theta,
                    'phi': grain2_phi,
                    'opposite_theta': grain1_theta,
                    'opposite_phi': grain1_phi
                }
        
        # Identify phase domains based on toroidal proximity
        # Only do this if toroidal position is available
        phase_domains = []
        if hasattr(manifold, 'get_toroidal_phase'):
            # Process each phase
            for phase_type, grain_ids in phases.items():
                if len(grain_ids) < 3:  # Skip phases with too few grains
                    continue
                
                # Group by toroidal proximity
                visited = set()
                
                for start_id in grain_ids:
                    if start_id in visited:
                        continue
                    
                    # Start new domain
                    domain_grains = set()
                    to_visit = [start_id]
                    
                    # Get start point's toroidal phase
                    start_theta, start_phi = manifold.get_toroidal_phase(start_id)
                    
                    # BFS to find connected points
                    while to_visit:
                        current_id = to_visit.pop(0)
                        
                        if current_id in visited:
                            continue
                            
                        visited.add(current_id)
                        domain_grains.add(current_id)
                        
                        # Get current point's related grains
                        current_grain = manifold.get_grain(current_id)
                        if not current_grain:
                            continue
                            
                        related_ids = list(current_grain.relations.keys())
                        
                        # Also add points that are close in toroidal space
                        current_theta, current_phi = manifold.get_toroidal_phase(current_id)
                        
                        for other_id in grain_ids:
                            if other_id in visited or other_id in to_visit:
                                continue
                                
                            # Check if same phase
                            if other_id not in grain_ids:
                                continue
                                
                            # Get toroidal distance
                            other_theta, other_phi = manifold.get_toroidal_phase(other_id)
                            distance = toroidal_distance(current_theta, current_phi, other_theta, other_phi)
                            
                            # If close enough, add to domain
                            if distance < 0.5:  # Adjust threshold as needed
                                to_visit.append(other_id)
                    
                    # Add domain if it has enough grains
                    if len(domain_grains) >= 3:
                        # Calculate domain center (circular mean of toroidal phases)
                        theta_phases = []
                        phi_phases = []
                        
                        for grain_id in domain_grains:
                            theta, phi = manifold.get_toroidal_phase(grain_id)
                            theta_phases.append(theta)
                            phi_phases.append(phi)
                        
                        # Calculate circular means
                        center_theta = circular_mean(theta_phases)
                        center_phi = circular_mean(phi_phases)
                        
                        # Create domain dictionary
                        domain = PHASE_DOMAIN_TEMPLATE.copy()
                        domain['domain_id'] = f"{phase_type}_{len(phase_domains)}"
                        domain['phase_type'] = phase_type
                        domain['points'] = domain_grains
                        domain['formation_time'] = manifold.time
                        domain['theta_center'] = center_theta
                        domain['phi_center'] = center_phi
                        
                        # Calculate angular extent
                        theta_diffs = [angular_difference(theta, center_theta) for theta in theta_phases]
                        phi_diffs = [angular_difference(phi, center_phi) for phi in phi_phases]
                        
                        theta_extent = max(theta_diffs) if theta_diffs else 0.0
                        phi_extent = max(phi_diffs) if phi_diffs else 0.0
                        
                        domain['toroidal_extent'] = {
                            'theta': theta_extent,
                            'phi': phi_extent
                        }
                        
                        # Find boundary points
                        boundary_points = set()
                        for grain_id in domain_grains:
                            grain = manifold.get_grain(grain_id)
                            if not grain:
                                continue
                                
                            # Check neighbors
                            for neighbor_id in grain.relations:
                                if neighbor_id not in domain_grains:
                                    boundary_points.add(grain_id)
                                    break
                        
                        domain['boundary_points'] = boundary_points
                        
                        # Calculate stability based on phase characteristics
                        if phase_type == 'solid' or phase_type == 'frozen':
                            domain['stability'] = 0.8
                        elif phase_type == 'liquid':
                            domain['stability'] = 0.6
                        elif phase_type == 'viscous':
                            domain['stability'] = 0.5
                        elif phase_type == 'gas':
                            domain['stability'] = 0.3
                        elif phase_type == 'radiant':
                            domain['stability'] = 0.2
                        
                        # Add to domains
                        phase_domains.append(domain)
            
            # Update class-level domain tracking
            self.phase_domains = phase_domains
        
        return {
            'phases': phases,
            'counts': {phase: len(grains) for phase, grains in phases.items()},
            'boundaries': phase_boundaries,
            'transitions': phase_transitions,
            'duality_enhancements': duality_enhancements,
            'phase_domains': phase_domains,
            'domain_count': len(phase_domains)
        }
    
    def get_phase_domains(self) -> List[Dict[str, Any]]:
        """
        Get the current phase domains.
        
        Returns:
            List of phase domain dictionaries
        """
        return self.phase_domains


class RecurrencePatternRule(EmergentFieldRule):
    """
    Rule for identifying recurring patterns in collapse behavior.
    Uses on-demand calculations enhanced with toroidal dynamics.
    """
    
    def __init__(self):
        super().__init__(
            name="recurrence_patterns",
            description="Identifies recurring patterns of collapse behavior with toroidal dynamics"
        )
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """
        Identify recurrence patterns by analyzing collapse history and resonance.
        Enhanced with toroidal pattern detection.
        
        Returns:
            Dictionary with pattern information
        """
        patterns = []
        
        # Find resonant grains (high field resonance)
        resonant_grains = []
        
        for grain_id, grain in manifold.grains.items():
            resonance = manifold.get_field_resonance(grain_id)
            if resonance > 0.7:  # Threshold for high resonance
                resonant_grains.append(grain_id)
        
        # Group connected resonant grains into patterns
        visited = set()
        
        for start_grain_id in resonant_grains:
            if start_grain_id in visited:
                continue
                
            # Find connected resonant grains
            current_pattern = []
            to_visit = [start_grain_id]
            
            while to_visit:
                current_id = to_visit.pop(0)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                current_pattern.append(current_id)
                
                # Add connected resonant neighbors
                grain = manifold.get_grain(current_id)
                if grain:
                    for related_id in grain.relations:
                        if related_id in resonant_grains and related_id not in visited and related_id not in to_visit:
                            to_visit.append(related_id)
            
            # If found a pattern with at least 2 grains, record it
            if len(current_pattern) >= 2:
                # Calculate pattern strength (average resonance)
                avg_resonance = sum(manifold.get_field_resonance(grain_id) for grain_id in current_pattern) / len(current_pattern)
                
                # Calculate toroidal characteristics if available
                pattern_theta = 0.0
                pattern_phi = 0.0
                if hasattr(manifold, 'get_toroidal_phase'):
                    # Calculate average phase position
                    theta_phases = []
                    phi_phases = []
                    
                    for grain_id in current_pattern:
                        theta, phi = manifold.get_toroidal_phase(grain_id)
                        theta_phases.append(theta)
                        phi_phases.append(phi)
                    
                    # Calculate circular means
                    pattern_theta = circular_mean(theta_phases)
                    pattern_phi = circular_mean(phi_phases)
                
                # Add to patterns
                patterns.append({
                    'type': 'resonance_pattern',
                    'grains': current_pattern,
                    'strength': avg_resonance,
                    'size': len(current_pattern),
                    'theta': pattern_theta,
                    'phi': pattern_phi
                })
        
        # Find patterns in collapse history
        history = manifold.collapse_history
        if history:
            # Look for recurring collapse paths
            grain_events = {}
            
            # Use recent history window (last 20 events)
            recent_history = history[-min(20, len(history)):]
            
            # Build event map
            for event in recent_history:
                source_id = event['source']
                target_id = event['target']
                
                for grain_id in [source_id, target_id]:
                    if grain_id not in grain_events:
                        grain_events[grain_id] = []
                    
                    grain_events[grain_id].append(event)
            
            # Find grains with recurring collapse events
            cycle_grains = []
            
            for grain_id, events in grain_events.items():
                if len(events) >= 3:  # Threshold for recurrence
                    grain = manifold.get_grain(grain_id)
                    if grain and grain.grain_saturation < 0.9:  # Not fully saturated
                        cycle_grains.append(grain_id)
            
            # Check for repeating patterns
            for grain_id in cycle_grains:
                events = grain_events[grain_id]
                
                # Need at least 4 events to detect pattern
                if len(events) < 4:
                    continue
                
                # Check for alternating source/target patterns
                role_sequence = []
                
                for event in events:
                    if event['source'] == grain_id:
                        role_sequence.append('source')
                    else:
                        role_sequence.append('target')
                
                # Check for repeating subsequences
                for pattern_length in range(2, 4):
                    if len(role_sequence) < pattern_length * 2:
                        continue
                    
                    for start in range(len(role_sequence) - pattern_length):
                        pattern = role_sequence[start:start+pattern_length]
                        next_seq = role_sequence[start+pattern_length:start+pattern_length*2]
                        
                        if pattern == next_seq:
                            # Get toroidal position if available
                            pattern_theta = 0.0
                            pattern_phi = 0.0
                            if hasattr(manifold, 'get_toroidal_phase'):
                                pattern_theta, pattern_phi = manifold.get_toroidal_phase(grain_id)
                            
                            patterns.append({
                                'type': 'collapse_cycle',
                                'grain_id': grain_id,
                                'pattern': pattern,
                                'length': pattern_length,
                                'theta': pattern_theta,
                                'phi': pattern_phi
                            })
                            break
        
        # Find vortex patterns with enhanced toroidal detection
        vortices = []
        if hasattr(manifold, 'detect_vortices'):
            vortices = manifold.detect_vortices()
        
        # Add vortices as patterns
        for vortex in vortices:
            center_id = vortex['center_node']
            
            # Get neighborhood
            grain = manifold.get_grain(center_id)
            if not grain:
                continue
                
            neighbor_ids = list(grain.relations.keys())
            
            patterns.append({
                'type': 'vortex',
                'center_grain': center_id,
                'grains': [center_id] + neighbor_ids,
                'strength': vortex['strength'],
                'rotation_direction': vortex['rotation_direction'],
                'pattern_type': vortex.get('pattern_type', 'mixed'),
                'theta_curvature': vortex.get('theta_curvature', 0.0),
                'phi_curvature': vortex.get('phi_curvature', 0.0)
            })
        
        # Find toroidal flow patterns if available
        toroidal_patterns = []
        if hasattr(manifold.polarity_field, 'find_toroidal_flow_regions'):
            toroidal_patterns = manifold.polarity_field.find_toroidal_flow_regions()
            
            # Add these as patterns
            for flow_region in toroidal_patterns:
                patterns.append({
                    'type': 'toroidal_flow',
                    'grains': flow_region['points'],
                    'strength': flow_region['flow_magnitude'],
                    'flow_direction': flow_region['flow_direction'],
                    'flow_type': flow_region['flow_type'],
                    'size': flow_region['size']
                })
        
        return {
            'patterns': patterns,
            'resonant_grain_count': len(resonant_grains),
            'vortex_count': len(vortices),
            'flow_pattern_count': len(toroidal_patterns)
        }


class EmergentStructureRule(EmergentFieldRule):
    """
    Rule for identifying emergent structural elements from collapse behavior.
    Optimized for the on-demand calculation of properties with toroidal enhancements.
    """
    
    def __init__(self):
        super().__init__(
            name="emergent_structures",
            description="Identifies stable structures that emerge from collapse with toroidal properties"
        )
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """
        Identify emergent structures in the manifold with toroidal dynamics.
        
        Returns:
            Dictionary with various types of emergent structures
        """
        # Find confinement zones
        confinement_zones = manifold.find_structural_confinement_zones()
        
        # Find collapse attractors
        attractors = manifold.find_collapse_attractors()
        
        # Find collapse resistance zones (high saturation, low contour)
        resistance_zones = []
        
        for grain_id, grain in manifold.grains.items():
            if grain.grain_saturation > 0.8:
                contour = abs(manifold.get_field_contour(grain_id))
                if contour < 0.1:  # Low contour indicates resistance
                    resistance_zones.append(grain_id)
        
        # Find structure boundaries (zones between high and low saturation)
        boundaries = []
        
        for grain_id, grain in manifold.grains.items():
            if 0.4 <= grain.grain_saturation <= 0.6:
                # Check if neighbors have significantly different saturation
                related_grains = manifold.get_related_grains(grain_id)
                
                has_high = False
                has_low = False
                
                for related_grain in related_grains:
                    if related_grain.grain_saturation > 0.7:
                        has_high = True
                    if related_grain.grain_saturation < 0.3:
                        has_low = True
                
                if has_high and has_low:
                    boundaries.append(grain_id)
        
        # Define coherent structures (clusters of aligned flow tendency)
        coherent_structures = []
        
        # Start with attractors and confinement zones as seed points
        seed_points = set(attractors).union(set(confinement_zones))
        for seed_id in seed_points:
            seed_grain = manifold.get_grain(seed_id)
            if not seed_grain:
                continue
            
            # Get neighborhood
            neighbors = []
            for related_id in seed_grain.relations:
                related_grain = manifold.get_grain(related_id)
                if related_grain:
                    neighbors.append(related_grain)
            
            # Need at least 3 grains
            if len(neighbors) < 2:  # 2 neighbors + seed = 3 grains
                continue
            
            # Check flow alignment in neighborhood
            aligned_count = 0
            
            # Get seed flow tendency
            seed_flow = manifold.get_flow_tendency(seed_id)
            
            # Check for toroidal flow if available
            seed_toroidal_flow = None
            if hasattr(manifold, 'calculate_toroidal_flow'):
                seed_toroidal_flow = manifold.calculate_toroidal_flow(seed_id, [n.id for n in neighbors])
            
            for neighbor in neighbors:
                # Get neighbor flow tendency
                neighbor_flow = manifold.get_flow_tendency(neighbor.id)
                
                # Check alignment between flow tendencies
                alignment = 0
                total = 0
                
                for rel_id in seed_flow:
                    if rel_id in neighbor_flow:
                        seed_val = seed_flow[rel_id]
                        neighbor_val = neighbor_flow[rel_id]
                        
                        # Aligned if same sign
                        if seed_val * neighbor_val > 0:
                            alignment += 1
                        
                        total += 1
                
                # Check toroidal alignment if available
                if seed_toroidal_flow and hasattr(manifold, 'calculate_toroidal_flow'):
                    neighbor_toroidal_flow = manifold.calculate_toroidal_flow(neighbor.id, [n.id for n in neighbors])
                    
                    # Compare toroidal flow components
                    for rel_id in seed_toroidal_flow:
                        if rel_id in neighbor_toroidal_flow:
                            seed_theta, seed_phi = seed_toroidal_flow[rel_id]
                            neighbor_theta, neighbor_phi = neighbor_toroidal_flow[rel_id]
                            
                            # Check angular alignment
                            theta_aligned = (seed_theta * neighbor_theta > 0)
                            phi_aligned = (seed_phi * neighbor_phi > 0)
                            
                            if theta_aligned and phi_aligned:
                                alignment += 1
                            
                            total += 1
                
                # Consider aligned if majority of flows align
                if total > 0 and alignment / total > 0.5:
                    aligned_count += 1
            
            # If enough aligned neighbors, this is a coherent structure
            if aligned_count >= len(neighbors) // 2:
                # Get toroidal position if available
                structure_theta = 0.0
                structure_phi = 0.0
                if hasattr(manifold, 'get_toroidal_phase'):
                    # Calculate average phase position
                    theta_phases = []
                    phi_phases = []
                    
                    theta_phases.append(manifold.get_toroidal_phase(seed_id)[0])
                    phi_phases.append(manifold.get_toroidal_phase(seed_id)[1])
                    
                    for neighbor in neighbors:
                        theta, phi = manifold.get_toroidal_phase(neighbor.id)
                        theta_phases.append(theta)
                        phi_phases.append(phi)
                    
                    # Calculate circular means
                    structure_theta = circular_mean(theta_phases)
                    structure_phi = circular_mean(phi_phases)
                
                coherent_structures.append({
                    'seed_id': seed_id,
                    'grains': [seed_id] + [n.id for n in neighbors],
                    'alignment_ratio': aligned_count / len(neighbors),
                    'theta': structure_theta,
                    'phi': structure_phi
                })
        
        # Find toroidal structures if available
        toroidal_structures = []
        if hasattr(manifold, 'find_toroidal_clusters'):
            toroidal_structures = manifold.find_toroidal_clusters()
        
        return {
            'confinement_zones': confinement_zones,
            'attractors': attractors,
            'resistance_zones': resistance_zones,
            'boundaries': boundaries,
            'coherent_structures': coherent_structures,
            'toroidal_structures': toroidal_structures
        }


class ToroidalStructureRule(EmergentFieldRule):
    """
    Rule for analyzing toroidal structure in the manifold.
    Identifies patterns in phase distribution that suggest toroidal organization.
    """
    
    def __init__(self):
        super().__init__(
            name="toroidal_structure",
            description="Analyzes toroidal structure in phase distribution"
        )
    
    def evaluate(self, manifold: 'RelationalManifold', slice_count: int = 12, **kwargs) -> Dict[str, Any]:
        """
        Analyze toroidal structure by dividing into angular slices.
        
        Args:
            manifold: The relational manifold
            slice_count: Number of slices to divide the torus into
            
        Returns:
            Dictionary with toroidal analysis results
        """
        # Initialize slice containers
        theta_slices = [{} for _ in range(slice_count)]  # Slices around major circle
        phi_slices = [{} for _ in range(slice_count)]    # Slices around minor circle
        
        # Check if toroidal phase is available
        has_toroidal_phase = hasattr(manifold, 'get_toroidal_phase')
        
        if not has_toroidal_phase:
            return {
                'has_toroidal_structure': False,
                'error': 'Toroidal phase not available'
            }
        
        # Classify grains into slices based on phase continuity
        for grain_id, grain in manifold.grains.items():
            # Get toroidal phase
            theta_phase, phi_phase = manifold.get_toroidal_phase(grain_id)
            
            # Map to theta (0 to 2π) - major circle
            theta = theta_phase % (2 * math.pi)
            theta_index = min(slice_count - 1, int(theta / (2*math.pi) * slice_count))
            
            # Map to phi (0 to 2π) - minor circle
            phi = phi_phase % (2 * math.pi)
            phi_index = min(slice_count - 1, int(phi / (2*math.pi) * slice_count))
            
            # Add grain to slices
            theta_slices[theta_index][grain_id] = grain
            phi_slices[phi_index][grain_id] = grain
        
        # Calculate metrics for each slice
        theta_metrics = []
        phi_metrics = []
        
        for slice_grains in theta_slices:
            if slice_grains:
                # Calculate average values
                avg_awareness = sum(grain.awareness for grain in slice_grains.values()) / len(slice_grains)
                avg_saturation = sum(grain.grain_saturation for grain in slice_grains.values()) / len(slice_grains)
                
                theta_metrics.append({
                    'grain_count': len(slice_grains),
                    'avg_awareness': avg_awareness,
                    'avg_saturation': avg_saturation
                })
            else:
                theta_metrics.append({
                    'grain_count': 0,
                    'avg_awareness': 0.0,
                    'avg_saturation': 0.0
                })
        
        for slice_grains in phi_slices:
            if slice_grains:
                # Calculate average values
                avg_awareness = sum(grain.awareness for grain in slice_grains.values()) / len(slice_grains)
                avg_saturation = sum(grain.grain_saturation for grain in slice_grains.values()) / len(slice_grains)
                
                phi_metrics.append({
                    'grain_count': len(slice_grains),
                    'avg_awareness': avg_awareness,
                    'avg_saturation': avg_saturation
                })
            else:
                phi_metrics.append({
                    'grain_count': 0,
                    'avg_awareness': 0.0,
                    'avg_saturation': 0.0
                })
        
        # Detect mode patterns using FFT
        theta_grain_counts = [m['grain_count'] for m in theta_metrics]
        phi_grain_counts = [m['grain_count'] for m in phi_metrics]
        
        # Calculate FFT to identify patterns
        theta_fft = np.abs(np.fft.fft(theta_grain_counts))
        phi_fft = np.abs(np.fft.fft(phi_grain_counts))
        
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
        
        # Calculate pattern significance
        theta_significance = 0.0
        if theta_modes:
            # Calculate relative amplitude of dominant mode compared to mean
            theta_mean = np.mean(theta_fft[1:])
            theta_significance = theta_modes[0][1] / theta_mean if theta_mean > 0 else 0.0
        
        phi_significance = 0.0
        if phi_modes:
            # Calculate relative amplitude of dominant mode compared to mean
            phi_mean = np.mean(phi_fft[1:])
            phi_significance = phi_modes[0][1] / phi_mean if phi_mean > 0 else 0.0
        
        # Calculate toroidal flow metrics
        major_circle_flow = 0.0
        minor_circle_flow = 0.0
        
        if hasattr(manifold, 'polarity_field') and hasattr(manifold.polarity_field, 'toroidal_metrics'):
            major_circle_flow = manifold.polarity_field.toroidal_metrics.get('major_circle_flow', 0.0)
            minor_circle_flow = manifold.polarity_field.toroidal_metrics.get('minor_circle_flow', 0.0)
        
        # Determine if toroidal structure is present
        has_toroidal_structure = (
            (theta_significance > 2.0 or phi_significance > 2.0) and
            (abs(major_circle_flow) > 0.1 or abs(minor_circle_flow) > 0.1)
        )
        
        # Calculate phase correlations between major and minor circles
        phase_correlations = []
        for i in range(slice_count):
            theta_count = theta_metrics[i]['grain_count']
            phi_count = phi_metrics[i]['grain_count']
            
            # Calculate correlation
            if theta_count > 0 and phi_count > 0:
                correlation = {
                    'theta_index': i,
                    'phi_index': i,
                    'theta_count': theta_count,
                    'phi_count': phi_count,
                    'theta_awareness': theta_metrics[i]['avg_awareness'],
                    'phi_awareness': phi_metrics[i]['avg_awareness'],
                    'theta_saturation': theta_metrics[i]['avg_saturation'],
                    'phi_saturation': phi_metrics[i]['avg_saturation']
                }
                phase_correlations.append(correlation)
        
        # Find cross-phase structures (areas where theta and phi interact)
        cross_phase_structures = []
        
        # Check each significant mode pair
        if theta_modes and phi_modes:
            # Focus on dominant modes
            theta_mode = theta_modes[0][0]
            phi_mode = phi_modes[0][0]
            
            # If modes are resonant (related by integer ratio)
            if theta_mode > 0 and phi_mode > 0:
                ratio = theta_mode / phi_mode
                ratio_error = abs(ratio - round(ratio))
                
                if ratio_error < 0.2:  # Close to integer ratio
                    cross_phase_structures.append({
                        'theta_mode': theta_mode,
                        'phi_mode': phi_mode,
                        'ratio': ratio,
                        'type': 'resonant_modes',
                        'strength': (theta_significance + phi_significance) / 2
                    })
        
        # Calculate toroidal density (how much of the torus is populated)
        theta_occupied = sum(1 for m in theta_metrics if m['grain_count'] > 0)
        phi_occupied = sum(1 for m in phi_metrics if m['grain_count'] > 0)
        
        if slice_count > 0:
            theta_coverage = theta_occupied / slice_count
            phi_coverage = phi_occupied / slice_count
            toroidal_density = (theta_coverage + phi_coverage) / 2
        else:
            toroidal_density = 0.0
        
        # Find potential knot structures on the torus
        potential_knots = []
        if hasattr(manifold, 'rotation_tensor') and hasattr(manifold.rotation_tensor, 'calculate_winding_number'):
            # Check significant grains for non-zero winding numbers
            significant_grains = attractors + list(confinement_zones)
            
            for grain_id in significant_grains:
                grain = manifold.get_grain(grain_id)
                if not grain:
                    continue
                
                # Get toroidal phase
                theta_phase, phi_phase = manifold.get_toroidal_phase(grain_id)
                
                # Get related grains
                related_ids = list(grain.relations.keys())
                
                # Calculate winding number
                winding = manifold.rotation_tensor.calculate_winding_number(grain_id, related_ids)
                
                # If non-zero winding, this might be a knot
                if winding != 0:
                    potential_knots.append({
                        'grain_id': grain_id,
                        'winding_number': winding,
                        'theta': theta_phase,
                        'phi': phi_phase,
                        'related_count': len(related_ids)
                    })
        
        return {
            'theta_slices': theta_metrics,
            'phi_slices': phi_metrics,
            'theta_modes': theta_modes,
            'phi_modes': phi_modes,
            'has_theta_pattern': len(theta_modes) > 0,
            'has_phi_pattern': len(phi_modes) > 0,
            'dominant_theta_mode': theta_modes[0][0] if theta_modes else 0,
            'dominant_phi_mode': phi_modes[0][0] if phi_modes else 0,
            'theta_significance': theta_significance,
            'phi_significance': phi_significance,
            'major_circle_flow': major_circle_flow,
            'minor_circle_flow': minor_circle_flow,
            'has_toroidal_structure': has_toroidal_structure,
            'phase_correlations': phase_correlations,
            'cross_phase_structures': cross_phase_structures,
            'toroidal_density': toroidal_density,
            'potential_knots': potential_knots
        }


class AncestryEntanglementRule(EmergentFieldRule):
    """
    Rule for analyzing and enhancing ancestry-driven entanglement in the manifold.
    This rule identifies and reinforces emergent structures based on shared ancestry.
    Enhanced with toroidal entanglement correlations.
    """
    
    def __init__(self):
        super().__init__(
            name="ancestry_entanglement",
            description="Analyzes and reinforces ancestry-driven entanglement dynamics with toroidal aspects"
        )
        
        # Configuration for ancestry entanglement
        self.config = {
            'entanglement_threshold': 0.6,    # Threshold for significant entanglement
            'ancestry_weight': 0.7,           # Weight of ancestry in entanglement calculations
            'memory_influence': 0.5,          # How strongly memory affects entanglement
            'coherence_boost': 0.3,           # Boost to phase coherence from entanglement
            'ancestry_decay_rate': 0.1,       # Rate at which ancestry influence decays
            'toroidal_coupling': 0.4          # How strongly entanglement affects toroidal structure
        }
    
    def evaluate(self, manifold: 'RelationalManifold', **kwargs) -> Dict[str, Any]:
        """
        Evaluate ancestry-driven entanglement in the manifold with toroidal correlations.
        
        Args:
            manifold: The relational manifold
            
        Returns:
            Dictionary with entanglement information
        """
        # Track entangled grain pairs
        entangled_pairs = []
        
        # Track entanglement clusters
        entanglement_clusters = []
        
        # Entanglement strength by grain
        entanglement_strength = {}
        
        # Check for toroidal capabilities
        has_toroidal = hasattr(manifold, 'get_toroidal_phase')
        
        # Calculate entanglement for all grains
        for grain_id, grain in manifold.grains.items():
            # Skip if grain has no ancestry
            if not hasattr(grain, 'ancestry') or not grain.ancestry:
                continue
                
            # Get ancestry set
            ancestry = grain.ancestry
            
            # Calculate grain's base entanglement strength
            base_strength = grain.awareness * (1.0 - grain.grain_saturation)
            entanglement_strength[grain_id] = base_strength
            
            # Find potential entanglement partners
            for other_id, other_grain in manifold.grains.items():
                if grain_id == other_id:
                    continue
                    
                if not hasattr(other_grain, 'ancestry') or not other_grain.ancestry:
                    continue
                
                # Calculate shared ancestry
                shared = ancestry.intersection(other_grain.ancestry)
                if not shared:
                    continue
                
                # Calculate ancestry overlap ratio
                shared_count = len(shared)
                max_count = max(len(ancestry), len(other_grain.ancestry))
                overlap_ratio = shared_count / max_count
                
                # Check if overlap exceeds threshold
                if overlap_ratio >= self.config['entanglement_threshold']:
                    # Calculate memory influence
                    memory_factor = 1.0
                    
                    # Check relation memory
                    if hasattr(grain, 'relation_memory') and other_id in grain.relation_memory:
                        mem_val = grain.relation_memory[other_id]
                        # Positive memory enhances entanglement
                        if mem_val > 0:
                            memory_factor += mem_val * self.config['memory_influence']
                    
                    # Calculate toroidal distance if available
                    toroidal_factor = 1.0
                    toroidal_distance = float('inf')
                    
                    if has_toroidal:
                        # Get toroidal phases
                        grain_theta, grain_phi = manifold.get_toroidal_phase(grain_id)
                        other_theta, other_phi = manifold.get_toroidal_phase(other_id)
                        
                        # Calculate toroidal distance
                        toroidal_distance = toroidal_distance(grain_theta, grain_phi, other_theta, other_phi)
                        
                        # Closer points on torus are more strongly entangled
                        # Normalize distance to [0,1] range (max distance ~= π*sqrt(2))
                        normalized_distance = min(1.0, toroidal_distance / (math.pi * math.sqrt(2)))
                        toroidal_factor = 1.0 + self.config['toroidal_coupling'] * (1.0 - normalized_distance)
                    
                    # Calculate entanglement strength
                    e_strength = (
                        overlap_ratio * 
                        self.config['ancestry_weight'] * 
                        memory_factor * 
                        base_strength * 
                        toroidal_factor
                    )
                    
                    # Create entangled pair info
                    pair_info = {
                        'grain1': grain_id,
                        'grain2': other_id,
                        'shared_ancestry': list(shared),
                        'strength': e_strength,
                        'overlap_ratio': overlap_ratio
                    }
                    
                    # Add toroidal info if available
                    if has_toroidal:
                        pair_info.update({
                            'grain1_theta': grain_theta,
                            'grain1_phi': grain_phi,
                            'grain2_theta': other_theta,
                            'grain2_phi': other_phi,
                            'toroidal_distance': toroidal_distance
                        })
                    
                    # Add to entangled pairs
                    entangled_pairs.append(pair_info)
        
        # Find entanglement clusters
        visited = set()
        
        for pair in entangled_pairs:
            grain1 = pair['grain1']
            grain2 = pair['grain2']
            
            if grain1 in visited and grain2 in visited:
                continue
                
            # Find or create cluster containing these grains
            matching_cluster = None
            
            for cluster in entanglement_clusters:
                if grain1 in cluster['grains'] or grain2 in cluster['grains']:
                    matching_cluster = cluster
                    break
            
            if matching_cluster is None:
                # Create new cluster
                matching_cluster = {
                    'grains': set(),
                    'shared_ancestry': set(pair['shared_ancestry']),
                    'total_strength': 0.0
                }
                entanglement_clusters.append(matching_cluster)
            
            # Add grains to cluster
            matching_cluster['grains'].add(grain1)
            matching_cluster['grains'].add(grain2)
            
            # Update shared ancestry (intersection)
            if matching_cluster['shared_ancestry']:
                matching_cluster['shared_ancestry'] = matching_cluster['shared_ancestry'].intersection(pair['shared_ancestry'])
            
            # Add strength
            matching_cluster['total_strength'] += pair['strength']
            
            # Mark as visited
            visited.add(grain1)
            visited.add(grain2)
        
        # Calculate phase coherence enhancement
        phase_coherence_boost = {}
        
        for pair in entangled_pairs:
            grain1 = pair['grain1']
            grain2 = pair['grain2']
            
            # Calculate phase coherence boost
            boost = pair['strength'] * self.config['coherence_boost']
            
            # Store for each grain
            if grain1 not in phase_coherence_boost:
                phase_coherence_boost[grain1] = 0.0
            phase_coherence_boost[grain1] += boost
            
            if grain2 not in phase_coherence_boost:
                phase_coherence_boost[grain2] = 0.0
            phase_coherence_boost[grain2] += boost
        
        # Enhance clusters with toroidal information
        if has_toroidal:
            for cluster in entanglement_clusters:
                if len(cluster['grains']) < 2:
                    continue
                
                # Calculate toroidal center
                thetas = []
                phis = []
                
                for grain_id in cluster['grains']:
                    theta, phi = manifold.get_toroidal_phase(grain_id)
                    thetas.append(theta)
                    phis.append(phi)
                
                # Calculate circular means
                cluster['center_theta'] = circular_mean(thetas)
                cluster['center_phi'] = circular_mean(phis)
                
                # Calculate spread
                distances = []
                for grain_id in cluster['grains']:
                    theta, phi = manifold.get_toroidal_phase(grain_id)
                    distance = toroidal_distance(theta, phi, cluster['center_theta'], cluster['center_phi'])
                    distances.append(distance)
                
                cluster['toroidal_spread'] = sum(distances) / len(distances) if distances else 0.0
                
                # Check if cluster spans the torus (large spread in either dimension)
                max_theta_diff = max(angular_difference(thetas[i], thetas[j]) 
                                 for i in range(len(thetas)) 
                                 for j in range(i+1, len(thetas))) if len(thetas) > 1 else 0.0
                
                max_phi_diff = max(angular_difference(phis[i], phis[j]) 
                               for i in range(len(phis)) 
                               for j in range(i+1, len(phis))) if len(phis) > 1 else 0.0
                
                cluster['spans_major_circle'] = max_theta_diff > math.pi
                cluster['spans_minor_circle'] = max_phi_diff > math.pi
        
        # Convert clusters to serializable format
        serializable_clusters = []
        for cluster in entanglement_clusters:
            cluster_data = {
                'grains': list(cluster['grains']),
                'shared_ancestry': list(cluster['shared_ancestry']),
                'total_strength': cluster['total_strength'],
                'size': len(cluster['grains'])
            }
            
            # Add toroidal data if available
            if has_toroidal and 'center_theta' in cluster:
                cluster_data.update({
                    'center_theta': cluster['center_theta'],
                    'center_phi': cluster['center_phi'],
                    'toroidal_spread': cluster['toroidal_spread'],
                    'spans_major_circle': cluster['spans_major_circle'],
                    'spans_minor_circle': cluster['spans_minor_circle']
                })
            
            serializable_clusters.append(cluster_data)
        
        return {
            'entangled_pairs': entangled_pairs,
            'entanglement_clusters': serializable_clusters,
            'entanglement_strength': entanglement_strength,
            'phase_coherence_boost': phase_coherence_boost,
            'pair_count': len(entangled_pairs),
            'cluster_count': len(entanglement_clusters),
            'config': self.config,
            'has_toroidal': has_toroidal
        }
    
    def apply_entanglement_effects(self, manifold: 'RelationalManifold') -> None:
        """
        Apply the effects of entanglement to the manifold.
        Enhanced with toroidal coupling effects.
        
        Args:
            manifold: The relational manifold
        """
        # Get entanglement data
        result = self.evaluate(manifold)
        
        # Apply phase coherence boosts
        coherence_boosts = result['phase_coherence_boost']
        
        for grain_id, boost in coherence_boosts.items():
            grain = manifold.get_grain(grain_id)
            if grain:
                # Apply boost to phase continuity
                current_phase = manifold.get_phase_continuity(grain_id)
                
                # Enhance phase stability rather than changing the phase itself
                # Create phase_stability dict if it doesn't exist
                if not hasattr(manifold, 'phase_stability'):
                    manifold.phase_stability = {}
                    
                manifold.phase_stability[grain_id] = manifold.phase_stability.get(grain_id, 0.5) + boost
                
                # Cap at 1.0
                manifold.phase_stability[grain_id] = min(1.0, manifold.phase_stability[grain_id])
        
        # Apply entanglement to memory persistence
        for pair in result['entangled_pairs']:
            grain1 = pair['grain1']
            grain2 = pair['grain2']
            strength = pair['strength']
            
            # Increase memory persistence between entangled grains
            g1 = manifold.get_grain(grain1)
            g2 = manifold.get_grain(grain2)
            
            if g1 and g2:
                # Only enhance existing memory
                if hasattr(g1, 'relation_memory') and grain2 in g1.relation_memory:
                    mem_val = g1.relation_memory[grain2]
                    # Positive memory gets enhanced by entanglement
                    if mem_val > 0:
                        g1.relation_memory[grain2] = min(1.0, mem_val + (strength * 0.2))
                
                if hasattr(g2, 'relation_memory') and grain1 in g2.relation_memory:
                    mem_val = g2.relation_memory[grain1]
                    # Positive memory gets enhanced by entanglement
                    if mem_val > 0:
                        g2.relation_memory[grain1] = min(1.0, mem_val + (strength * 0.2))
        
        # Apply toroidal coupling effects
        if result['has_toroidal']:
            # Apply cluster-level effects
            for cluster in result['entanglement_clusters']:
                if len(cluster['grains']) < 3:  # Only for significant clusters
                    continue
                    
                if 'center_theta' not in cluster:
                    continue
                
                # Get toroidal center
                center_theta = cluster['center_theta']
                center_phi = cluster['center_phi']
                
                # Apply gentle influence toward cluster center
                influence = 0.1 * (cluster['total_strength'] / len(cluster['grains']))
                
                for grain_id in cluster['grains']:
                    current_theta, current_phi = manifold.get_toroidal_phase(grain_id)
                    
                    # Calculate phase differences (circular)
                    theta_diff = ((center_theta - current_theta + math.pi) % (2*math.pi)) - math.pi
                    phi_diff = ((center_phi - current_phi + math.pi) % (2*math.pi)) - math.pi
                    
                    # Shift phase slightly
                    new_theta = current_theta + (theta_diff * influence)
                    new_phi = current_phi + (phi_diff * influence)
                    
                    # Normalize to [0, 2π)
                    new_theta = new_theta % (2*math.pi)
                    new_phi = new_phi % (2*math.pi)
                    
                    # Update phase
                    if hasattr(manifold, 'phase_continuity') and grain_id in manifold.phase_continuity:
                        manifold.phase_continuity[grain_id] = new_theta
                    
                    if hasattr(manifold, 'toroidal_phase') and grain_id in manifold.toroidal_phase:
                        manifold.toroidal_phase[grain_id] = [new_theta, new_phi]


# Dictionary of all field rule types
FIELD_RULE_TYPES = {
    'phase_classification': EnhancedPhaseClassificationRule,
    'recurrence': RecurrencePatternRule,
    'structures': EmergentStructureRule,
    'toroidal': ToroidalStructureRule,
    'void_decay': VoidDecayRule,
    'ancestry_entanglement': AncestryEntanglementRule
}


def create_field_rule(rule_type: str, **kwargs) -> EmergentFieldRule:
    """
    Factory function to create a field rule of the specified type
    
    Args:
        rule_type: Type of rule to create
        **kwargs: Additional parameters to pass to the rule constructor
        
    Returns:
        An instance of the requested field rule
    """
    if rule_type not in FIELD_RULE_TYPES:
        raise ValueError(f"Unknown field rule type: {rule_type}")
    
    rule_class = FIELD_RULE_TYPES[rule_type]
    return rule_class(**kwargs)