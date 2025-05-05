"""
Grain Dynamics - Core implementation of the fundamental grain elements with toroidal referencing

Implements the fundamental "grain" concept in the Collapse Geometry framework,
representing elements that can activate, saturate, and retain relational memory
with toroidal coordinates for proper topological representation.
"""

from typing import Dict, List, Set, Optional, Any, Tuple
import random
import math


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


class Grain:
    """
    Fundamental grain element that stores core state properties with minimal redundancy.
    Represents a point in the manifold where collapse can occur, memory can persist,
    and relations can form, now with toroidal positioning.
    """
    
    def __init__(self, grain_id: str, theta: float = None, phi: float = None):
        """
        Initialize a grain with fundamental properties.
        
        Args:
            grain_id: Unique identifier for this grain
            theta: Major circle position on torus (0 to 2π), randomized if None
            phi: Minor circle position on torus (0 to 2π), randomized if None
        """
        # Core identity
        self.id = grain_id
        
        # === FUNDAMENTAL STATE PROPERTIES ===
        # Basic Field Properties
        self.awareness = random.uniform(0.1, 0.3)      # Awareness value ρ(x,t)
        self.collapse_metric = 0.0                     # Accumulated collapse C∞(x,t)
        self.grain_activation = 0.0                    # Activation level Γ(x,t)
        self.grain_saturation = 0.0                    # Saturation level S(x,t)
        
        # Relational Structures - Define manifold topology
        self.relations = {}                            # Maps related_id -> relation_strength
        self.relation_memory = {}                      # Maps related_id -> memory_strength
        self.negation_memory = {}                      # Maps negated_id -> negation_strength
        self.opposition_memory = {}                    # Maps opposite_id -> opposition_strength
        self.ancestry = set()                          # Set of ancestor grain_ids
        self.negated_options = set()                   # Set of negated option grain_ids
        self.opposite_state = None                     # ID of opposite grain (if any)
        
        # Field Gradients - Raw differences in awareness
        self.field_gradients = {}                      # Maps related_id -> awareness difference
        
        # Field History - Required for pattern detection
        self.field_history = []                        # Records of field states (limited)
        self.activation_history = []                   # Records of activation events
        
        # Toroidal coordinates - Define position on manifold
        self.theta = theta if theta is not None else random.random() * 2 * math.pi
        self.phi = phi if phi is not None else random.random() * 2 * math.pi
        
        # Toroidal field properties
        self.phase_stability = 0.5                     # Phase stability (0.0 to 1.0)
        self.toroidal_curvature = 0.0                  # Local curvature in the field
        self.flow_circulation = 0.0                    # Flow circulation around this point
        self.toroidal_neighborhood = set()             # Neighborhood based on toroidal distance
    
    def update_relation(self, related_id: str, strength: float):
        """Update relation strength to another grain"""
        self.relations[related_id] = strength
    
    def update_relation_memory(self, related_id: str, memory_update: float, blending_factor: float = 0.2):
        """Update relation memory with blending"""
        current_memory = self.relation_memory.get(related_id, 0.0)
        self.relation_memory[related_id] = current_memory * (1 - blending_factor) + memory_update * blending_factor
    
    def update_negation_memory(self, negated_id: str, memory_update: float, blending_factor: float = 0.2):
        """Update negation memory with blending"""
        current_memory = self.negation_memory.get(negated_id, 0.0)
        self.negation_memory[negated_id] = current_memory * (1 - blending_factor) + memory_update * blending_factor
    
    def update_opposition_memory(self, opposite_id: str, memory_update: float, blending_factor: float = 0.2):
        """Update opposition memory with blending"""
        current_memory = self.opposition_memory.get(opposite_id, 0.0)
        self.opposition_memory[opposite_id] = current_memory * (1 - blending_factor) + memory_update * blending_factor
    
    def add_negated_option(self, negated_id: str):
        """Add a negated option"""
        self.negated_options.add(negated_id)
    
    def set_opposite_state(self, opposite_id: str):
        """Set opposite state reference"""
        self.opposite_state = opposite_id
    
    def get_relational_distance(self, other_grain) -> float:
        """
        Calculate relational distance to another grain.
        Based on relation strength rather than geometric distance.
        
        Args:
            other_grain: The other grain
            
        Returns:
            Relational distance value
        """
        if other_grain.id in self.relations:
            # Inverse of relation strength (stronger = closer)
            return 1.0 - min(1.0, self.relations[other_grain.id])
        else:
            return float('inf')  # Not directly related
    
    def get_toroidal_distance(self, other_grain) -> float:
        """
        Calculate toroidal distance to another grain.
        Based on angular coordinates on the torus.
        
        Args:
            other_grain: The other grain
            
        Returns:
            Toroidal distance value
        """
        return toroidal_distance(self.theta, self.phi, other_grain.theta, other_grain.phi)
    
    def update_toroidal_position(self, new_theta: float = None, new_phi: float = None, 
                             blending_factor: float = 0.2):
        """
        Update toroidal position with smooth blending.
        
        Args:
            new_theta: New theta coordinate (None = no change)
            new_phi: New phi coordinate (None = no change)
            blending_factor: How quickly position updates (0-1)
        """
        if new_theta is not None:
            # Circular blending for theta
            theta_diff = ((new_theta - self.theta + math.pi) % (2 * math.pi)) - math.pi
            self.theta = (self.theta + blending_factor * theta_diff) % (2 * math.pi)
            
        if new_phi is not None:
            # Circular blending for phi
            phi_diff = ((new_phi - self.phi + math.pi) % (2 * math.pi)) - math.pi
            self.phi = (self.phi + blending_factor * phi_diff) % (2 * math.pi)
    
    def update_field_history(self, time: float, relations_snapshot: Optional[Dict[str, Any]] = None):
        """
        Record current state in field history.
        Keeps history limited to prevent memory bloat.
        
        Args:
            time: Current simulation time
            relations_snapshot: Optional pre-computed relation snapshot
        """
        # Limit history length
        max_history = 10
        if len(self.field_history) >= max_history:
            self.field_history = self.field_history[1:]
        
        # Capture relationships and their values if not provided
        if relations_snapshot is None:
            relations_snapshot = {}
            for rel_id in self.relations:
                relations_snapshot[rel_id] = {
                    'relation_strength': self.relations[rel_id],
                    'memory_strength': self.relation_memory.get(rel_id, 0.0),
                    'awareness_diff': self.field_gradients.get(rel_id, 0.0)
                }
        
        # Store minimal field state - only essential values
        field_state = {
            'time': time,
            'awareness': self.awareness,
            'collapse_metric': self.collapse_metric,
            'grain_activation': self.grain_activation,
            'grain_saturation': self.grain_saturation,
            'relations': relations_snapshot,
            'theta': self.theta,
            'phi': self.phi,
            'phase_stability': getattr(self, 'phase_stability', 0.5)
        }
        
        # Add to history
        self.field_history.append(field_state)
    
    def record_activation_event(self, time: float, gradient: float, contour: float, tension: float,
                             toroidal_contour: float = None):
        """
        Record an activation event with toroidal properties.
        
        Args:
            time: Current simulation time
            gradient: Awareness gradient at activation
            contour: Field contour at activation
            tension: Field tension at activation
            toroidal_contour: Toroidal contour at activation (optional)
        """
        event = {
            'time': time,
            'gradient': gradient,
            'contour': contour,
            'tension': tension,
            'awareness': self.awareness,
            'collapse_metric': self.collapse_metric,
            'theta': self.theta,
            'phi': self.phi
        }
        
        # Add toroidal contour if provided
        if toroidal_contour is not None:
            event['toroidal_contour'] = toroidal_contour
        
        self.activation_history.append(event)
    
    def set_phase_stability(self, stability: float):
        """
        Set the phase stability for this grain.
        
        Args:
            stability: Phase stability value (0.0 to 1.0)
        """
        self.phase_stability = max(0.0, min(1.0, stability))
    
    def adjust_phase_stability(self, adjustment: float):
        """
        Adjust the phase stability for this grain.
        
        Args:
            adjustment: Amount to adjust stability by
        """
        self.phase_stability = max(0.0, min(1.0, self.phase_stability + adjustment))
    
    def set_toroidal_curvature(self, curvature: float):
        """
        Set the toroidal curvature for this grain.
        
        Args:
            curvature: Curvature value
        """
        self.toroidal_curvature = curvature
    
    def set_flow_circulation(self, circulation: float):
        """
        Set the flow circulation value for this grain.
        
        Args:
            circulation: Circulation value
        """
        self.flow_circulation = circulation
    
    def update_toroidal_neighborhood(self, grain_ids: Set[str], radius: float = 0.5):
        """
        Update the toroidal neighborhood for this grain.
        
        Args:
            grain_ids: Set of grain IDs in the neighborhood
            radius: Neighborhood radius used
        """
        self.toroidal_neighborhood = grain_ids
        self.neighborhood_radius = radius
    
    def get_toroidal_position(self) -> Tuple[float, float]:
        """
        Get the toroidal position of this grain.
        
        Returns:
            Tuple of (theta, phi) coordinates
        """
        return (self.theta, self.phi)
    
    def get_toroidal_displacement(self, other_grain) -> Tuple[float, float]:
        """
        Calculate the toroidal displacement vector to another grain.
        
        Args:
            other_grain: The other grain
            
        Returns:
            Tuple of (theta_displacement, phi_displacement)
        """
        # Calculate smallest angular differences
        theta_diff = ((other_grain.theta - self.theta + math.pi) % (2 * math.pi)) - math.pi
        phi_diff = ((other_grain.phi - self.phi + math.pi) % (2 * math.pi)) - math.pi
        
        return (theta_diff, phi_diff)
    
    def adjust_position_toward(self, other_grain, factor: float = 0.1):
        """
        Adjust toroidal position toward another grain.
        
        Args:
            other_grain: The grain to move toward
            factor: How much to move (0.0 to 1.0)
        """
        # Get displacement vector
        theta_diff, phi_diff = self.get_toroidal_displacement(other_grain)
        
        # Move toward other grain
        self.update_toroidal_position(
            self.theta + theta_diff * factor,
            self.phi + phi_diff * factor
        )
    
    def calculate_toroidal_resonance(self, other_grain, field_tension: float = 0.0) -> float:
        """
        Calculate toroidal resonance with another grain.
        Higher values indicate stronger resonance.
        
        Args:
            other_grain: The other grain
            field_tension: Field tension between grains
            
        Returns:
            Resonance value (0.0 to 1.0)
        """
        # Calculate toroidal distance
        distance = self.get_toroidal_distance(other_grain)
        
        # Calculate phase stability similarity
        stability_diff = abs(self.phase_stability - getattr(other_grain, 'phase_stability', 0.5))
        
        # Calculate awareness correlation
        awareness_factor = 1.0 - abs(self.awareness - other_grain.awareness)
        
        # Calculate grain saturation correlation
        saturation_factor = 1.0 - abs(self.grain_saturation - other_grain.grain_saturation)
        
        # Calculate relation factor
        relation_strength = self.relations.get(other_grain.id, 0.0)
        other_relation_strength = other_grain.relations.get(self.id, 0.0)
        relation_factor = (relation_strength + other_relation_strength) / 2
        
        # Combine factors weighted by importance
        resonance = (
            (1.0 - min(1.0, distance / math.pi)) * 0.3 +  # Distance factor
            (1.0 - stability_diff) * 0.2 +                # Phase stability
            awareness_factor * 0.2 +                      # Awareness
            saturation_factor * 0.1 +                     # Saturation
            relation_factor * 0.2                         # Relation strength
        )
        
        # Reduce resonance based on field tension
        resonance *= (1.0 - field_tension * 0.5)
        
        return max(0.0, min(1.0, resonance))


class ToroidalGrainSystem:
    """
    A system of grains with toroidal neighborhood detection and phase tracking.
    Provides utilities for analyzing grain interactions on the torus.
    """
    
    def __init__(self, neighborhood_radius: float = 0.5):
        """
        Initialize the toroidal grain system.
        
        Args:
            neighborhood_radius: Radius for toroidal neighborhood detection
        """
        self.grains = {}  # Maps grain_id -> Grain
        self.neighborhood_radius = neighborhood_radius
        self.time = 0.0
    
    def add_grain(self, grain: Grain):
        """
        Add a grain to the system.
        
        Args:
            grain: The grain to add
        """
        self.grains[grain.id] = grain
        self.update_neighborhoods([grain.id])
    
    def remove_grain(self, grain_id: str):
        """
        Remove a grain from the system.
        
        Args:
            grain_id: ID of the grain to remove
        """
        if grain_id in self.grains:
            # Remove from neighborhoods
            for other_id, other_grain in self.grains.items():
                if hasattr(other_grain, 'toroidal_neighborhood'):
                    other_grain.toroidal_neighborhood.discard(grain_id)
            
            # Remove the grain
            del self.grains[grain_id]
    
    def get_grain(self, grain_id: str) -> Optional[Grain]:
        """
        Get a grain by ID.
        
        Args:
            grain_id: ID of the grain to get
            
        Returns:
            Grain object or None if not found
        """
        return self.grains.get(grain_id)
    
    def get_toroidal_neighborhood(self, grain_id: str) -> Set[str]:
        """
        Get the toroidal neighborhood for a grain.
        
        Args:
            grain_id: ID of the center grain
            
        Returns:
            Set of grain IDs in the neighborhood
        """
        grain = self.get_grain(grain_id)
        if grain and hasattr(grain, 'toroidal_neighborhood'):
            return grain.toroidal_neighborhood
        return set()
    
    def update_neighborhoods(self, grain_ids: List[str] = None):
        """
        Update toroidal neighborhoods for specified grains or all grains.
        
        Args:
            grain_ids: List of grain IDs to update, or None for all
        """
        if grain_ids is None:
            grain_ids = list(self.grains.keys())
        
        for grain_id in grain_ids:
            grain = self.get_grain(grain_id)
            if not grain:
                continue
            
            # Find grains within neighborhood radius
            neighbors = set()
            theta, phi = grain.get_toroidal_position()
            
            for other_id, other_grain in self.grains.items():
                if other_id == grain_id:
                    continue
                
                other_theta, other_phi = other_grain.get_toroidal_position()
                distance = toroidal_distance(theta, phi, other_theta, other_phi)
                
                if distance <= self.neighborhood_radius:
                    neighbors.add(other_id)
            
            # Update grain's neighborhood
            grain.update_toroidal_neighborhood(neighbors, self.neighborhood_radius)
    
    def calculate_phase_coherence(self, grain_ids: List[str] = None) -> float:
        """
        Calculate overall phase coherence for the grain system or a subset.
        
        Args:
            grain_ids: List of grain IDs to calculate for, or None for all
            
        Returns:
            Phase coherence value (0.0 to 1.0)
        """
        if not grain_ids:
            grain_ids = list(self.grains.keys())
        
        if len(grain_ids) < 2:
            return 1.0  # Perfect coherence for a single grain
        
        # Collect theta values
        thetas = []
        phis = []
        
        for grain_id in grain_ids:
            grain = self.get_grain(grain_id)
            if grain:
                thetas.append(grain.theta)
                phis.append(grain.phi)
        
        if not thetas:
            return 0.0
        
        # Calculate circular mean and variance
        # For theta component
        theta_mean = circular_mean(thetas)
        theta_var = sum(1.0 - math.cos(angular_difference(t, theta_mean)) for t in thetas) / len(thetas)
        
        # For phi component
        phi_mean = circular_mean(phis)
        phi_var = sum(1.0 - math.cos(angular_difference(p, phi_mean)) for p in phis) / len(phis)
        
        # Combine components, normalize to [0,1]
        # Lower variance = higher coherence
        theta_coherence = 1.0 - min(1.0, theta_var)
        phi_coherence = 1.0 - min(1.0, phi_var)
        
        # Weight theta more heavily (major circle dominance)
        combined_coherence = 0.7 * theta_coherence + 0.3 * phi_coherence
        
        return combined_coherence
    
    def find_phase_domains(self, coherence_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find coherent phase domains on the torus.
        
        Args:
            coherence_threshold: Minimum coherence to be considered a domain
            
        Returns:
            List of domain dictionaries
        """
        # Start with each grain in its own group
        visited = set()
        domains = []
        
        for start_id in self.grains:
            if start_id in visited:
                continue
            
            # Start new domain
            domain_grains = []
            to_visit = [start_id]
            
            # BFS to find connected coherent grains
            while to_visit:
                current_id = to_visit.pop(0)
                
                if current_id in visited:
                    continue
                
                visited.add(current_id)
                domain_grains.append(current_id)
                
                # Get toroidal neighborhood
                current_grain = self.get_grain(current_id)
                if not current_grain or not hasattr(current_grain, 'toroidal_neighborhood'):
                    continue
                
                # Check each neighbor
                for neighbor_id in current_grain.toroidal_neighborhood:
                    if neighbor_id in visited or neighbor_id in to_visit:
                        continue
                    
                    neighbor_grain = self.get_grain(neighbor_id)
                    if not neighbor_grain:
                        continue
                    
                    # Check phase stability similarity
                    current_stability = getattr(current_grain, 'phase_stability', 0.5)
                    neighbor_stability = getattr(neighbor_grain, 'phase_stability', 0.5)
                    
                    # Similar stability is required for coherent domain
                    stability_diff = abs(current_stability - neighbor_stability)
                    if stability_diff < 0.3:  # Adjust threshold as needed
                        to_visit.append(neighbor_id)
            
            # Check if domain is significant
            if len(domain_grains) >= 3:
                # Calculate domain properties
                coherence = self.calculate_phase_coherence(domain_grains)
                
                if coherence >= coherence_threshold:
                    # Calculate domain center (circular mean of positions)
                    thetas = []
                    phis = []
                    
                    for grain_id in domain_grains:
                        grain = self.get_grain(grain_id)
                        if grain:
                            thetas.append(grain.theta)
                            phis.append(grain.phi)
                    
                    theta_center = circular_mean(thetas)
                    phi_center = circular_mean(phis)
                    
                    # Calculate domain radius
                    max_distance = 0.0
                    for grain_id in domain_grains:
                        grain = self.get_grain(grain_id)
                        if grain:
                            distance = toroidal_distance(grain.theta, grain.phi, theta_center, phi_center)
                            max_distance = max(max_distance, distance)
                    
                    # Calculate average phase stability
                    avg_stability = 0.0
                    for grain_id in domain_grains:
                        grain = self.get_grain(grain_id)
                        if grain:
                            avg_stability += getattr(grain, 'phase_stability', 0.5)
                    avg_stability /= len(domain_grains) if domain_grains else 1.0
                    
                    # Add domain
                    domains.append({
                        'grains': domain_grains,
                        'size': len(domain_grains),
                        'coherence': coherence,
                        'theta_center': theta_center,
                        'phi_center': phi_center,
                        'radius': max_distance,
                        'avg_stability': avg_stability
                    })
        
        return domains
    
    def calculate_flow_field(self) -> Dict[str, Tuple[float, float]]:
        """
        Calculate the flow field across the torus.
        
        Returns:
            Dictionary mapping grain_id -> (theta_flow, phi_flow)
        """
        flow_field = {}
        
        for grain_id, grain in self.grains.items():
            # Calculate flow based on neighborhood
            theta_flow = 0.0
            phi_flow = 0.0
            
            # Get toroidal neighborhood
            neighbors = list(self.get_toroidal_neighborhood(grain_id))
            
            if not neighbors:
                flow_field[grain_id] = (0.0, 0.0)
                continue
            
            # Calculate flow based on:
            # 1. Awareness gradients
            # 2. Relation strengths
            # 3. Phase stability differences
            
            for neighbor_id in neighbors:
                neighbor = self.get_grain(neighbor_id)
                if not neighbor:
                    continue
                
                # Calculate theta and phi displacements
                theta_diff, phi_diff = grain.get_toroidal_displacement(neighbor)
                
                # Calculate awareness gradient
                awareness_diff = neighbor.awareness - grain.awareness
                
                # Calculate relation contribution
                relation_strength = grain.relations.get(neighbor_id, 0.0)
                
                # Calculate flow contribution
                # Flow is stronger for higher awareness gradients and relation strengths
                flow_strength = awareness_diff * relation_strength
                
                # Scale by distance (inverse square law)
                distance = math.sqrt(theta_diff**2 + phi_diff**2)
                if distance > 0:
                    # Direction components
                    theta_direction = theta_diff / distance
                    phi_direction = phi_diff / distance
                    
                    # Add to total flow
                    theta_flow += flow_strength * theta_direction
                    phi_flow += flow_strength * phi_direction
            
            # Normalize by number of neighbors
            if neighbors:
                theta_flow /= len(neighbors)
                phi_flow /= len(neighbors)
            
            # Store flow vector
            flow_field[grain_id] = (theta_flow, phi_flow)
        
        return flow_field
    
    def calculate_circulation(self) -> Dict[str, float]:
        """
        Calculate flow circulation around each grain.
        Positive values indicate clockwise circulation, negative indicate counterclockwise.
        
        Returns:
            Dictionary mapping grain_id -> circulation value
        """
        circulation = {}
        flow_field = self.calculate_flow_field()
        
        for grain_id, grain in self.grains.items():
            # Get neighborhood in sorted angular order
            neighbors = list(self.get_toroidal_neighborhood(grain_id))
            
            if len(neighbors) < 3:
                circulation[grain_id] = 0.0
                continue
            
            # Sort neighbors by angular position around center grain
            def get_angle(neighbor_id):
                neighbor = self.get_grain(neighbor_id)
                if neighbor:
                    theta_diff, phi_diff = grain.get_toroidal_displacement(neighbor)
                    return math.atan2(phi_diff, theta_diff) % (2 * math.pi)
                return 0.0
            
            neighbors.sort(key=get_angle)
            
            # Calculate circulation as line integral around loop
            total_circulation = 0.0
            
            for i in range(len(neighbors)):
                # Get current and next neighbor
                current_id = neighbors[i]
                next_id = neighbors[(i + 1) % len(neighbors)]
                
                current = self.get_grain(current_id)
                next_neighbor = self.get_grain(next_id)
                
                if not current or not next_neighbor:
                    continue
                
                # Get flow vectors
                current_flow = flow_field.get(current_id, (0.0, 0.0))
                next_flow = flow_field.get(next_id, (0.0, 0.0))
                
                # Calculate average flow along segment
                avg_theta_flow = (current_flow[0] + next_flow[0]) / 2
                avg_phi_flow = (current_flow[1] + next_flow[1]) / 2
                
                # Calculate segment vector
                segment_theta = next_neighbor.theta - current.theta
                segment_phi = next_neighbor.phi - current.phi
                
                # Adjust for wraparound
                if segment_theta > math.pi:
                    segment_theta -= 2 * math.pi
                elif segment_theta < -math.pi:
                    segment_theta += 2 * math.pi
                    
                if segment_phi > math.pi:
                    segment_phi -= 2 * math.pi
                elif segment_phi < -math.pi:
                    segment_phi += 2 * math.pi
                
                # Calculate flow contribution along segment (dot product)
                contribution = avg_theta_flow * segment_theta + avg_phi_flow * segment_phi
                
                # Add to total circulation
                total_circulation += contribution
            
            # Set circulation value
            circulation[grain_id] = total_circulation
            
            # Update grain's circulation value
            grain.set_flow_circulation(total_circulation)
        
        return circulation
    
    def advance_time(self, dt: float = 1.0):
        """
        Advance time in the grain system.
        
        Args:
            dt: Time delta
        """
        self.time += dt


def create_random_grain(grain_id: str = None) -> Grain:
    """
    Create a new grain with random properties.
    
    Args:
        grain_id: Optional ID for the grain (random UUID if not provided)
        
    Returns:
        New Grain instance
    """
    if grain_id is None:
        import uuid
        grain_id = str(uuid.uuid4())
    
    return Grain(grain_id=grain_id)