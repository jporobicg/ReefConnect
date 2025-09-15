#!/usr/bin/env python3
"""
Test Step 2 & 3: Verify io_utils.py and config functionality
"""

import numpy as np
import tempfile
import os
from pathlib import Path

# Test the new io_utils and config
from io_utils import load_config, extract_reef_id_from_filename, create_netcdf_output, verify_output_structure


def test_config_loading():
    """Test loading configuration from YAML file."""
    print("="*50)
    print("TESTING CONFIG LOADING")
    print("="*50)
    
    config = load_config("config/connectivity_parameters.yaml")
    
    # Check that all required sections exist
    required_sections = ['decay', 'competence', 'bootstrap', 'file_patterns', 'default_paths']
    for section in required_sections:
        assert section in config, f"Missing section: {section}"
        print(f"✅ Section '{section}' found")
    
    # Check some specific parameters
    assert config['decay']['Tcp_decay'] == 2.583, "Wrong Tcp_decay value"
    assert config['competence']['tc'] == 3.333, "Wrong tc value"
    assert config['bootstrap']['sample_size'] == 100, "Wrong sample_size value"
    assert config['bootstrap']['n_repetitions'] == 50, "Wrong n_repetitions value"
    
    print(f"✅ Configuration contains {len(config)} sections")
    print(f"✅ Bootstrap parameters: {config['bootstrap']['sample_size']} particles, {config['bootstrap']['n_repetitions']} repetitions")
    
    return True


def test_filename_parsing():
    """Test extracting reef ID from filenames."""
    print("\n" + "="*50)
    print("TESTING FILENAME PARSING")
    print("="*50)
    
    # Test various filename formats
    test_files = [
        ("GBR1_H2p0_Coral_Release_365_Polygon_1234_Wind_3_percent_displacement_field.nc", 1234),
        ("GBR1_H2p0_Coral_Release_1_Polygon_0_Wind_3_percent_displacement_field.nc", 0),
        ("GBR1_H2p0_Coral_Release_180_Polygon_3805_Wind_3_percent_displacement_field.nc", 3805),
    ]
    
    for filename, expected_id in test_files:
        extracted_id = extract_reef_id_from_filename(filename)
        assert extracted_id == expected_id, f"Wrong ID extracted: {extracted_id} != {expected_id}"
        print(f"✅ Filename '{filename}' -> reef ID: {extracted_id}")
    
    return True


def test_netcdf_creation():
    """Test NetCDF output creation and verification."""
    print("\n" + "="*50)
    print("TESTING NETCDF CREATION")
    print("="*50)
    
    # Create test data
    num_sources, num_sinks, num_samples = 10, 10, 5
    
    angle_data = np.random.uniform(0, 360, (num_sources, num_sinks))
    distance_data = np.random.uniform(1, 1000, (num_sources, num_sinks))
    direction_data = np.random.randint(0, 36, (num_sources, num_sinks))
    connectivity_data = np.random.exponential(0.1, (num_sources, num_sinks, num_samples))
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
        output_path = tmp_file.name
    
    try:
        # Create NetCDF file
        create_netcdf_output(
            output_path, num_sources, num_sinks, num_samples,
            angle_data, distance_data, direction_data, connectivity_data
        )
        
        print(f"✅ NetCDF file created: {output_path}")
        
        # Verify structure
        verification_result = verify_output_structure(
            output_path, num_sources, num_sinks, num_samples
        )
        
        assert verification_result, "NetCDF verification failed"
        print(f"✅ NetCDF structure verified successfully")
        
        # Check file size
        file_size = os.path.getsize(output_path)
        print(f"✅ Output file size: {file_size} bytes")
        
    finally:
        # Cleanup
        if os.path.exists(output_path):
            os.unlink(output_path)
    
    return True


def test_config_directory_structure():
    """Test that config directory structure is correct."""
    print("\n" + "="*50)
    print("TESTING CONFIG DIRECTORY STRUCTURE")
    print("="*50)
    
    # Check config directory exists
    config_dir = Path("config")
    assert config_dir.exists(), "Config directory does not exist"
    print(f"✅ Config directory exists: {config_dir}")
    
    # Check YAML file exists
    yaml_file = config_dir / "connectivity_parameters.yaml"
    assert yaml_file.exists(), "YAML config file does not exist"
    print(f"✅ YAML config file exists: {yaml_file}")
    
    # Check file is readable
    assert yaml_file.is_file(), "YAML file is not a regular file"
    print(f"✅ YAML file is readable")
    
    return True


def test_bash_script():
    """Test that bash script exists and is executable."""
    print("\n" + "="*50)
    print("TESTING BASH SCRIPT")
    print("="*50)
    
    script_path = Path("run_connectivity.sh")
    assert script_path.exists(), "Bash script does not exist"
    print(f"✅ Bash script exists: {script_path}")
    
    # Check if executable
    assert os.access(script_path, os.X_OK), "Bash script is not executable"
    print(f"✅ Bash script is executable")
    
    return True


def main():
    """Run all tests."""
    print("STEP 2 & 3 TESTING - IO_UTILS AND CONFIG")
    print("="*60)
    
    tests = [
        ("Configuration loading", test_config_loading),
        ("Filename parsing", test_filename_parsing),
        ("NetCDF creation", test_netcdf_creation),
        ("Config directory structure", test_config_directory_structure),
        ("Bash script", test_bash_script),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name} test passed!")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("STEP 2 & 3 TESTING COMPLETE")
    print("="*60)
    print(f"✅ Tests passed: {passed}")
    print(f"❌ Tests failed: {failed}")
    
    if failed == 0:
        print("\n✅ All tests passed! IO utilities and configuration are working correctly!")
        print("Ready to proceed to next steps.")
    else:
        print(f"\n❌ {failed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main() 