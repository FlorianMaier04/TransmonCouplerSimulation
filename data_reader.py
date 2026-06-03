"""
Data Reader for Transmon Coupler Simulation Results

Loads previously saved simulation results (JSON + PNG) and reproduces
the same graphical output as the original computation.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class SimulationDataReader:
    """
    Load and visualize previously saved simulation results.
    """
    RES_PATH = "/Users/florianmaier/Library/CloudStorage/OneDrive-TUM/ba/res/"
    def __init__(self, data_file_path):
        """
        Initialize reader with data file.
        
        Args:
            data_file_path: Path to {filename}_data.json file
        """
        path = SimulationDataReader.RES_PATH + data_file_path + "_data.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        
        self.data_file_path = path
        self.base_dir = os.path.dirname(path)
        
        # Load data
        with open(path, 'r') as f:
            self.data = json.load(f)
        
        self.filename = self.data.get('filename', 'simulation')
        self.config = self.data['configuration']
        self.simulator_params = self.data['simulator_parameters_GHz']
        self.sweep_data = self.data['sweep_results']
        
        print(f"\n✓ Loaded simulation data: {self.filename}")
        print(f"  Element: ({self.config['element_indices'][0]}, {self.config['element_indices'][1]})")
        print(f"  Extraction mode: {self.config['extraction_mode']}")
        print(f"  Time: {self.config['time_ns']} ns\n")
    
    def get_element_label(self):
        """Get label based on extraction mode."""
        i, j = self.config['element_indices']
        if self.config['extraction_mode'] == 'diag_diff':
            return rf'$\delta_{{{i}}}$-$\delta_{{{j}}}$'
        else:
            return rf'Ω_{{{i},{j}}}'
    
    def reconstruct_plot(self, show_figure=True):
        """
        Reconstruct the plot from saved data.
        
        Args:
            show_figure: Whether to display the plot (default: True)
        
        Returns:
            fig: Matplotlib figure object
        """
        config = self.config
        sweep_data = self.sweep_data
        i, j = config['element_indices']
        
        # Determine which sweeps were performed
        has_amplitude = 'amplitude' in sweep_data
        has_frequency = 'frequency' in sweep_data
        
        # Create figure
        fig, axes = plt.subplots(1, 2 if has_amplitude and has_frequency else 1, figsize=(14, 5))
        if has_frequency and has_amplitude:
            ax1, ax2 = axes
        else:
            ax2 = axes
            ax1 = axes
        
        element_label = self.get_element_label()
        fig.suptitle(f'{element_label} Parameter Sweeps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Plot amplitude sweep
        if has_amplitude:
            amp_data = sweep_data['amplitude']
            parameters_amp = np.array(amp_data['parameters'])
            values_amp = np.array(amp_data['values'])
            
            ax1.plot(parameters_amp, values_amp, 'o-', linewidth=2, markersize=6, color="#217dbf")
            ax1.set_xlabel('Amplitude (GHz)', fontsize=11)
            ax1.set_ylabel(element_label, fontsize=11)
            ax1.set_title(f'{element_label} vs Amplitude', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Plot frequency sweep
        if has_frequency:
            fre_data = sweep_data['frequency']
            parameters_fre = np.array(fre_data['parameters'])
            values_fre = np.array(fre_data['values'])
            
            ax2.plot(parameters_fre, values_fre, 's-', linewidth=2, markersize=6, color='#ff7f0e')
            ax2.set_xlabel('Frequency (GHz)', fontsize=11)
            ax2.set_ylabel(element_label, fontsize=11)
            ax2.set_title(f'{element_label} vs Frequency', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Add parameter info box
        base_amp = config['base_amplitude_GHz']
        base_fre = config['base_frequency_GHz']
        pulse_mod = config['pulse_modulation']
        
        pulse_text = "cos"
        if 'gaussian' in pulse_mod:
            sigma = pulse_mod.split()[1]
            pulse_text = f"gaussian, σ={sigma}·t_g"
        
        info_text = (
            f"Base Parameters: A={base_amp:.4f} GHz | f={base_fre:.4f} GHz | "
            f"$t_g$={config['time_ns']} ns | Δt={config['timestep_ns']} ns | "
            f"Pulse-Shape: {pulse_text} | Element: ({i},{j})"
        )
        fig.text(0.5, 0.05, info_text, ha='center', va='top', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 transform=fig.transFigure)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if show_figure:
            print("\n" + "="*70)
            print("✓ Plot reconstructed from saved data!")
            print("="*70 + "\n")
        
        return fig
    
    def export_csv(self, output_path=None):
        """
        Export sweep data to CSV files for further analysis.
        
        Args:
            output_path: Directory to save CSV files (default: same as data file)
        
        Returns:
            List of created file paths
        """
        if output_path is None:
            output_path = self.base_dir
        
        os.makedirs(output_path, exist_ok=True)
        created_files = []
        
        sweep_data = self.sweep_data
        filename = self.filename
        
        # Export amplitude sweep
        if 'amplitude' in sweep_data:
            amp_data = sweep_data['amplitude']
            csv_path = os.path.join(output_path, f'{filename}_amplitude.csv')
            
            with open(csv_path, 'w') as f:
                f.write('Amplitude_GHz,Value\n')
                for amp, val in zip(amp_data['parameters'], amp_data['values']):
                    f.write(f'{amp},{val}\n')
            
            print(f"📄 Amplitude data exported: {csv_path}")
            created_files.append(csv_path)
        
        # Export frequency sweep
        if 'frequency' in sweep_data:
            fre_data = sweep_data['frequency']
            csv_path = os.path.join(output_path, f'{filename}_frequency.csv')
            
            with open(csv_path, 'w') as f:
                f.write('Frequency_GHz,Value\n')
                for fre, val in zip(fre_data['parameters'], fre_data['values']):
                    f.write(f'{fre},{val}\n')
            
            print(f"📄 Frequency data exported: {csv_path}")
            created_files.append(csv_path)
        
        return created_files
    
    def print_summary(self):
        """Print summary of loaded simulation."""
        print("\n" + "="*70)
        print("SIMULATION DATA SUMMARY")
        print("="*70)
        print(f"Filename: {self.filename}")
        print(f"Timestamp: {self.data.get('timestamp', 'N/A')}")
        print(f"\nConfiguration:")
        print(f"  Time: {self.config['time_ns']} ns")
        print(f"  Timestep: {self.config['timestep_ns']} ns")
        print(f"  Accuracy: {self.config['accuracy']}")
        print(f"  Element: ({self.config['element_indices'][0]}, {self.config['element_indices'][1]})")
        print(f"  Extraction Mode: {self.config['extraction_mode']}")
        print(f"  Parameter Preset: {self.config['parameter_preset']}")
        print(f"\nSweep Results:")
        for sweep_type, data in self.sweep_data.items():
            print(f"  {sweep_type.capitalize()}: {len(data['parameters'])} points")
        print("="*70 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Load from command-line argument
        data_file = sys.argv[1]
    else:
        # Ask user for file path
        print("\n" + "="*70)
        print("TRANSMON COUPLER SIMULATION - DATA READER")
        print("="*70)
        data_file = input("\n📂 Enter path to data file (*_data.json): ").strip()
    
    try:
        # Create reader and load data
        reader = SimulationDataReader(data_file)
        
        # Print summary
        reader.print_summary()
        
        # Reconstruct and show plot
        fig = reader.reconstruct_plot(show_figure=True)
        
        # Optionally export to CSV
        export_choice = input("💾 Export sweep data to CSV? (y/n): ").strip().lower()
        if export_choice == 'y':
            csv_files = reader.export_csv()
            print(f"✓ Exported {len(csv_files)} CSV file(s)\n")
        
        plt.show()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON file format")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
