#!/usr/bin/env python3
"""
Test script to verify RAM filesystem performance on the server.
Run this directly on the RunPod server to test write speeds.
"""

import time
import os
import subprocess
from pathlib import Path

def test_write_speed(path: Path, size_mb: int = 100):
    """Test write speed to a given path."""
    test_file = path / "test_speed.bin"
    
    # Create test data (100MB)
    data = os.urandom(size_mb * 1024 * 1024)
    
    # Test write speed
    start = time.time()
    with open(test_file, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())  # Ensure data is written
    write_time = time.time() - start
    
    # Test read speed
    start = time.time()
    with open(test_file, 'rb') as f:
        _ = f.read()
    read_time = time.time() - start
    
    # Clean up
    test_file.unlink()
    
    write_speed = size_mb / write_time
    read_speed = size_mb / read_time
    
    return write_speed, read_speed, write_time, read_time

def check_filesystem_type(path: Path):
    """Check what type of filesystem a path is on."""
    try:
        result = subprocess.run(
            ["df", "-T", str(path)],
            capture_output=True,
            text=True,
            timeout=5
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            # Parse the filesystem type from df output
            parts = lines[1].split()
            if len(parts) > 1:
                return parts[1]  # Filesystem type
    except:
        pass
    return "unknown"

def main():
    print("🔍 RAM Filesystem Performance Test")
    print("=" * 50)
    
    # Test paths
    paths = [
        ("/dev/shm/transcription/uploads", "RAM (dev/shm)"),
        ("/workspace/uploads", "Disk (workspace)"),
        ("/tmp", "Temp directory"),
    ]
    
    results = []
    
    for path_str, description in paths:
        path = Path(path_str)
        
        # Create directory if needed
        path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📁 Testing: {description}")
        print(f"   Path: {path}")
        
        # Check filesystem type
        fs_type = check_filesystem_type(path)
        print(f"   Filesystem: {fs_type}")
        
        # Check available space
        try:
            statvfs = os.statvfs(str(path))
            available_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
            print(f"   Available: {available_gb:.1f} GB")
        except:
            print(f"   Available: Unable to determine")
        
        # Test performance
        try:
            write_speed, read_speed, write_time, read_time = test_write_speed(path, 100)
            print(f"   ✅ Write: {write_speed:.1f} MB/s ({write_time:.2f}s for 100MB)")
            print(f"   ✅ Read:  {read_speed:.1f} MB/s ({read_time:.2f}s for 100MB)")
            results.append((description, write_speed, read_speed))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results.append((description, 0, 0))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Performance Summary")
    print("=" * 50)
    
    if results:
        # Find baseline (disk)
        disk_write = next((r[1] for r in results if "workspace" in r[0].lower()), 100)
        disk_read = next((r[2] for r in results if "workspace" in r[0].lower()), 100)
        
        for desc, write_speed, read_speed in results:
            if write_speed > 0:
                write_improvement = write_speed / disk_write if disk_write > 0 else 1
                read_improvement = read_speed / disk_read if disk_read > 0 else 1
                print(f"\n{desc}:")
                print(f"  Write: {write_speed:.1f} MB/s ({write_improvement:.1f}x vs disk)")
                print(f"  Read:  {read_speed:.1f} MB/s ({read_improvement:.1f}x vs disk)")
    
    # Check if server is using RAM filesystem
    print("\n" + "=" * 50)
    print("🔍 Server Configuration Check")
    print("=" * 50)
    
    shm_transcription = Path("/dev/shm/transcription")
    if shm_transcription.exists():
        print("✅ /dev/shm/transcription exists")
        for subdir in ["uploads", "outputs", "jobs"]:
            subpath = shm_transcription / subdir
            if subpath.exists():
                print(f"   ✅ {subdir}/ directory present")
            else:
                print(f"   ❌ {subdir}/ directory missing")
    else:
        print("❌ /dev/shm/transcription not found - server may be using disk")
    
    print("\n💡 To verify server is using RAM:")
    print("   1. Upload a file through the API")
    print("   2. Check if it appears in /dev/shm/transcription/uploads/")
    print("   3. Or check server logs for 'RAM filesystem configured'")

if __name__ == "__main__":
    main()