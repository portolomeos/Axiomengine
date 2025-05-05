import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import random

# Import core components
from axiom7.core.relational_manifold import RelationalManifold
from axiom7.core.state import SimulationState
from axiom7.core.engine import EnhancedContinuousDualityEngine

# Import visualization components
from axiom7.visualizer.torus_simulation_visualizer import TorusSimulationVisualizer
from axiom7.visualizer.torus_unwrapping_visualizer import TorusUnwrappingVisualizer


def setup_simulation():
    """Set up the manifold, state, and engine for simulation"""
    # Create the manifold with smaller neighborhood radius to encourage local interactions
    manifold = RelationalManifold(neighborhood_radius=0.3)
    
    # Create the simulation state
    state = SimulationState()
    
    # Configure the engine to promote emergence
    engine_config = {
        'field_continuity': 0.9,        # Higher continuity
        'awareness_diffusion': 0.3,     # Stronger diffusion
        'collapse_rate': 0.25,          # Slightly lower collapse rate for stability
        'field_memory': 0.8,            # Strong memory effects
        'ancestry_coupling': 0.7,       # Stronger ancestry effects
        'toroidal_coupling': 0.8,       # Strong toroidal coupling
        'phase_stability_factor': 0.6,  # Moderate phase stability
        'void_impact_factor': 0.5,      # Stronger void effects for structural tension
        'coherence_threshold': 0.5,     # Lower threshold for coherence effects
        'duality_emergence': 0.7        # Stronger duality emergence
    }
    
    # Create the engine
    engine = EnhancedContinuousDualityEngine(
        manifold=manifold,
        state=state,
        config=engine_config
    )
    
    return manifold, state, engine


def initialize_emergent_structure(manifold, count=30):
    """Initialize grains with a structure that promotes emergent toroidal dynamics"""
    grains = []
    
    # Create initial seed grains
    for i in range(count):
        # Create grain without imposed toroidal coordinates (let them emerge)
        grain_id = f"grain_{i}"
        grain = manifold.add_grain(grain_id)
        
        # Set initial values with some patterns to seed emergence
        angle = (i / count) * 2 * np.pi
        
        # Set awareness in a wave pattern
        grain.awareness = 0.5 + 0.4 * np.cos(angle)
        
        # Vary collapse metric to create tension gradients
        grain.collapse_metric = 0.3 + 0.3 * np.sin(angle * 2)
        
        # Set high activation for some grains to promote collapse
        grain.grain_activation = 0.5 + 0.5 * (np.cos(angle * 3) > 0.3)
        
        # Low initial saturation to allow evolution
        grain.grain_saturation = 0.1 + 0.1 * np.sin(angle)
        
        grains.append(grain)
    
    # Create relationships with pattern that encourages circular/toroidal structure
    for i in range(count):
        # Each grain connects to nearby grains in index space
        for offset in range(1, 5):  # Connect to a few neighbors
            j = (i + offset) % count  # Wrap around to create circular topology
            
            # Calculate relationship strength with some variation
            strength = 0.7 + 0.3 * np.cos(offset * np.pi / 5)
            
            # Connect grains
            manifold.connect_grains(grains[i].id, grains[j].id, relation_strength=strength)
    
    # Create some opposite pairs to seed duality
    for i in range(3):  # Create a few opposites
        idx1 = i * 5  # Spaced out around the circle
        idx2 = (idx1 + count // 2) % count  # Opposite side
        
        manifold.set_opposite_grains(grains[idx1].id, grains[idx2].id)
    
    # Add some "field resonance" and "unresolved tension" to selected grains
    # to promote structural dynamics
    for i in range(count):
        grain = grains[i]
        if i % 7 == 0:  # Select some grains for additional properties
            grain.field_resonance = 0.8
            grain.unresolved_tension = 0.7
    
    # Create a few special "attractor" grains with high awareness
    for i in range(3):
        idx = i * 10
        grains[idx].awareness = 0.9
        grains[idx].grain_activation = 0.9
    
    # Set up some initial field memory to seed circular flow
    for i in range(count):
        grain = grains[i]
        next_idx = (i + 1) % count
        prev_idx = (i - 1) % count
        
        # Set memory polarity to encourage circular flow
        if next_idx in grain.relations:
            grain.update_relation_memory(grains[next_idx].id, 0.6)  # Positive flow forward
        
        if prev_idx in grain.relations:
            grain.update_relation_memory(grains[prev_idx].id, -0.3)  # Negative flow backward
    
    return grains


def run_simulation(engine, steps=200, visualize_every=20):
    """Run the simulation for the specified number of steps"""
    # Create output directories if they don't exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("snapshots", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create visualizers - corrected parameters based on class definitions
    torus_vis = TorusSimulationVisualizer(
        resolution=50,
        adaptation_rate=0.6,  # Higher adaptation rate to capture emergent structure
        memory_weight=0.7     # Stronger memory influence
    )
    
    unwrap_vis = TorusUnwrappingVisualizer(
        field_resolution=100,
        adaptation_rate=0.5,
        memory_weight=0.7,
        vector_density=17
    )
    
    # Run simulation steps
    print(f"Starting simulation with {steps} steps...")
    start_time = time.time()
    
    # Inject some variability for emergence
    noise_steps = [int(steps * 0.25), int(steps * 0.5), int(steps * 0.75)]
    
    for step in range(steps + 1):
        if step > 0:  # Skip step for initial state
            # Run multiple steps between visualizations for faster evolution
            steps_to_run = 5 if step % visualize_every == 0 else 1
            
            for _ in range(steps_to_run):
                engine.step()
            
            # Inject some noise/perturbations at specific points
            if step in noise_steps:
                print(f"Step {step}: Injecting perturbation...")
                inject_perturbation(engine.manifold)
        
        # Visualize at intervals
        if step % visualize_every == 0:
            print(f"Step {step}/{steps} - Time: {engine.manifold.time:.2f}")
            
            # Create 3D visualization
            fig3d, _ = torus_vis.render_3d_torus(
                engine.manifold,
                color_by='awareness',
                show_relations=True,
                show_vortices=True
            )
            fig3d.savefig(f"output/torus_visualization_step_{step}.png", dpi=150)
            plt.close(fig3d)
            
            # Create 2D unwrapped visualization
            fig2d, _ = unwrap_vis.create_standard_unwrapping(
                engine.manifold,
                color_by='awareness',
                show_relations=True
            )
            fig2d.savefig(f"output/unwrapped_torus_step_{step}.png", dpi=150)
            plt.close(fig2d)
            
            # Try different visualization types
            try:
                fig_phase, _ = unwrap_vis.create_phase_domain_visualization(engine.manifold)
                fig_phase.savefig(f"output/phase_domains_step_{step}.png", dpi=150)
                plt.close(fig_phase)
            except Exception as e:
                print(f"Could not create phase domain visualization: {e}")
            
            # Save state snapshot
            with open(f"snapshots/state_step_{step}.txt", "w") as f:
                f.write(f"Simulation Time: {engine.manifold.time}\n")
                f.write(f"Step: {step}\n\n")
                
                # Write state metrics
                f.write("System Metrics:\n")
                for key, value in engine.state.metrics.items():
                    f.write(f"  {key}: {value}\n")
                
                # Write toroidal metrics
                f.write("\nToroidal Metrics:\n")
                if hasattr(engine.manifold, 'toroidal_metrics'):
                    for key, value in engine.manifold.toroidal_metrics.items():
                        f.write(f"  {key}: {value}\n")
                
                # Write phase coherence
                if hasattr(engine.manifold, 'calculate_phase_coherence'):
                    coherence = engine.manifold.calculate_phase_coherence()
                    f.write(f"  Phase Coherence: {coherence}\n")
                
                # Write grain data
                f.write("\nGrain States:\n")
                for grain_id, grain in engine.manifold.grains.items():
                    theta, phi = engine.manifold.get_toroidal_phase(grain_id)
                    f.write(f"  {grain_id}: awareness={grain.awareness:.2f}, "
                          f"activation={grain.grain_activation:.2f}, "
                          f"saturation={grain.grain_saturation:.2f}, "
                          f"theta={theta:.2f}, phi={phi:.2f}\n")
    
    # Save final results
    elapsed_time = time.time() - start_time
    print(f"Simulation completed in {elapsed_time:.2f} seconds")
    
    # Save collapse history
    with open("results/collapse_history.txt", "w") as f:
        f.write(f"Collapse History ({len(engine.manifold.collapse_history)} events):\n\n")
        for i, event in enumerate(engine.manifold.collapse_history):
            f.write(f"Event {i}:\n")
            for key, value in event.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
    
    # Save grain states
    with open("results/grain_states.txt", "w") as f:
        f.write(f"Final Grain States (Time: {engine.manifold.time}):\n\n")
        
        # Sort grains by awareness
        sorted_grains = sorted(
            engine.manifold.grains.items(),
            key=lambda x: x[1].awareness,
            reverse=True
        )
        
        for grain_id, grain in sorted_grains:
            f.write(f"Grain {grain_id}:\n")
            f.write(f"  Awareness: {grain.awareness:.4f}\n")
            f.write(f"  Activation: {grain.grain_activation:.4f}\n")
            f.write(f"  Saturation: {grain.grain_saturation:.4f}\n")
            f.write(f"  Collapse Metric: {grain.collapse_metric:.4f}\n")
            
            # Get toroidal position
            theta, phi = engine.manifold.get_toroidal_phase(grain_id)
            f.write(f"  Toroidal Position: theta={theta:.4f}, phi={phi:.4f}\n")
            
            # Relations
            f.write(f"  Relations: {len(grain.relations)}\n")
            for related_id, strength in grain.relations.items():
                f.write(f"    -> {related_id}: {strength:.4f}\n")
            
            # Void presence if available
            if hasattr(engine.manifold, 'get_void_presence'):
                void = engine.manifold.get_void_presence(grain_id)
                f.write(f"  Void Presence: {void:.4f}\n")
            
            f.write("\n")


def inject_perturbation(manifold):
    """Inject a perturbation into the system to encourage emergence"""
    # Select random grains for perturbation
    grain_ids = list(manifold.grains.keys())
    perturbation_count = min(5, len(grain_ids))
    selected_grains = random.sample(grain_ids, perturbation_count)
    
    for grain_id in selected_grains:
        grain = manifold.grains[grain_id]
        
        # Random awareness spike
        grain.awareness = min(1.0, grain.awareness + random.uniform(0.3, 0.6))
        
        # Increase collapse metric
        grain.collapse_metric = min(1.0, grain.collapse_metric + random.uniform(0.2, 0.5))
        
        # High grain activation to trigger collapse
        grain.grain_activation = min(1.0, grain.grain_activation + random.uniform(0.3, 0.7))
        
        # Create some new connections to promote structural change
        other_grains = random.sample(grain_ids, min(3, len(grain_ids)))
        for other_id in other_grains:
            if other_id != grain_id:
                # Create or strengthen relation
                strength = random.uniform(0.6, 0.9)
                manifold.connect_grains(grain_id, other_id, relation_strength=strength)
                
                # Add relational memory
                grain.update_relation_memory(other_id, random.uniform(-0.7, 0.7))


def analyze_results(state):
    """Analyze the simulation results"""
    print("\nSimulation Analysis:")
    
    # Get summary from state
    summary = state.get_summary()
    
    print(f"Total simulation time: {summary['time']:.2f}")
    print(f"Elapsed real time: {summary['elapsed_real_time']:.2f} seconds")
    print(f"Steps: {summary['step_count']}")
    
    # Display metrics
    print("\nSystem Metrics:")
    for key, value in summary['metrics'].items():
        print(f"  {key}: {value}")
    
    # Display toroidal metrics if available
    if 'toroidal_metrics' in summary:
        print("\nToroidal Metrics:")
        for key, value in summary['toroidal_metrics'].items():
            print(f"  {key}: {value}")
    
    # Count of emergent structures
    print("\nEmergent Structures:")
    for struct_type, count in summary['structures'].items():
        print(f"  {struct_type}: {count}")
    
    # Look for evidence of emergence
    print("\nEvidence of Emergent Topology:")
    
    # Check phase coherence
    phase_coherence = summary['metrics'].get('phase_coherence', 0.0)
    if phase_coherence > 0.4:
        print(f"  Phase coherence is {phase_coherence:.2f}, suggesting emergent coherence")
    
    # Check for vortices
    vortex_count = summary['metrics'].get('vortex_count', 0)
    if vortex_count > 0:
        print(f"  System has {vortex_count} vortices, indicating rotational structure")
    
    # Check for flow patterns
    major_flow = summary['metrics'].get('major_circle_flow', 0.0)
    minor_flow = summary['metrics'].get('minor_circle_flow', 0.0)
    if abs(major_flow) > 0.2 or abs(minor_flow) > 0.2:
        print(f"  Flow patterns detected: major={major_flow:.2f}, minor={minor_flow:.2f}")
    
    # Check collapse events 
    collapse_count = len(state.history.get('events', []))
    if collapse_count > 0:
        print(f"  {collapse_count} collapse events recorded, driving structural evolution")


def main():
    """Main test function"""
    print("Initializing Collapse Geometry simulation for emergent toroidal dynamics...")
    
    # Setup components
    manifold, state, engine = setup_simulation()
    
    # Initialize grains with structure promoting emergence
    print("Creating relational structure to seed emergence...")
    grains = initialize_emergent_structure(manifold, count=40)
    
    # Register event handlers for tracking
    def on_collapse(event):
        source = event.get('source', 'unknown')
        target = event.get('target', 'unknown')
        strength = event.get('strength', 0.0)
        print(f"Collapse: {source} -> {target} (strength: {strength:.2f})")
    
    def on_void_formation(event):
        center = event.get('center_id', 'unknown')
        strength = event.get('strength', 0.0)
        print(f"Void Formation: center={center}, strength={strength:.2f}")
    
    def on_toroidal_vortex(event):
        center = event.get('center_grain', 'unknown')
        strength = event.get('strength', 0.0)
        print(f"Toroidal Vortex: center={center}, strength={strength:.2f}")
    
    def on_field_resonance(event):
        nodes = event.get('pattern_nodes', [])
        resonance = event.get('resonance', 0.0)
        print(f"Field Resonance: strength={resonance:.2f}, nodes={len(nodes)}")
    
    # Register event handlers
    engine.register_event_handler('on_collapse', on_collapse)
    engine.register_event_handler('on_void_formation', on_void_formation)
    engine.register_event_handler('on_toroidal_vortex', on_toroidal_vortex)
    engine.register_event_handler('on_field_resonance', on_field_resonance)
    
    # Run the simulation
    run_simulation(engine, steps=200, visualize_every=20)
    
    # Analyze results
    analyze_results(state)
    
    print("\nSimulation completed. Results are saved in the output, snapshots, and results directories.")


if __name__ == "__main__":
    main()