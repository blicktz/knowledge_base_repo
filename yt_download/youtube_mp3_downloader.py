#!/usr/bin/env python3
"""
YouTube MP3 Downloader using Metube API

This script reads YouTube URLs from a file and downloads them as MP3 files
using a local Metube instance running on http://localhost:8081

Usage:
    python youtube_mp3_downloader.py urls.txt [output_dir]
"""

import sys
import time
import requests
import json
from pathlib import Path
from typing import List, Optional


class MetubeDownloader:
    def __init__(self, metube_url: str = "http://localhost:8081"):
        self.metube_url = metube_url.rstrip('/')
        self.session = requests.Session()
        
    def test_connection(self) -> bool:
        """Test if Metube is running and accessible"""
        try:
            response = self.session.get(f"{self.metube_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def add_download(self, url: str, quality: str = "audio", format_type: str = "mp3") -> bool:
        """Add an audio-only MP3 download to Metube queue"""
        try:
            # Use the verified working payload for audio-only MP3 downloads
            payload = {
                "url": url,
                "quality": quality,  # Use "audio" for audio-only downloads
                "format": format_type  # Specify MP3 format
            }
            
            response = self.session.post(
                f"{self.metube_url}/add",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✓ Added to queue (audio-only MP3): {url}")
                return True
            else:
                print(f"✗ Failed to add {url}: HTTP {response.status_code}")
                print(f"   Response: {response.text if response.text else 'No response body'}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Network error adding {url}: {e}")
            return False
    
    def get_history(self) -> Optional[List]:
        """Get download history from Metube"""
        try:
            response = self.session.get(f"{self.metube_url}/history", timeout=5)
            if response.status_code == 200:
                return response.json()
            return None
        except requests.exceptions.RequestException:
            return None


def read_urls_file(file_path: str) -> List[str]:
    """Read YouTube URLs from a text file, one per line"""
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Basic YouTube URL validation
                    if 'youtube.com/watch' in line or 'youtu.be/' in line:
                        urls.append(line)
                    else:
                        print(f"Warning: Line {line_num} doesn't look like a YouTube URL: {line}")
        return urls
    except FileNotFoundError:
        print(f"Error: URLs file not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading URLs file: {e}")
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python youtube_mp3_downloader.py <urls_file> [output_dir]")
        print("\nExample:")
        print("  python youtube_mp3_downloader.py youtube_urls.txt ./downloads")
        sys.exit(1)
    
    urls_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    
    print("YouTube MP3 Downloader")
    print("=====================")
    print("🎵 Configured for audio-only MP3 downloads")
    
    # Initialize downloader
    downloader = MetubeDownloader()
    
    # Test connection to Metube
    print("Testing connection to Metube...")
    if not downloader.test_connection():
        print("✗ Cannot connect to Metube at http://localhost:8081")
        print("  Make sure Metube is running and accessible")
        sys.exit(1)
    print("✓ Connected to Metube")
    
    # Read URLs
    print(f"\nReading URLs from: {urls_file}")
    urls = read_urls_file(urls_file)
    
    if not urls:
        print("No valid YouTube URLs found")
        sys.exit(1)
    
    print(f"Found {len(urls)} YouTube URLs")
    
    # Process each URL
    successful = 0
    failed = 0
    
    print(f"\nStarting downloads...")
    print("-" * 50)
    
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] Processing: {url}")
        
        if downloader.add_download(url):
            successful += 1
            # Small delay between requests to be nice to the server
            time.sleep(1)
        else:
            failed += 1
    
    print("-" * 50)
    print(f"\nDownload Summary:")
    print(f"✓ Successfully queued: {successful}")
    print(f"✗ Failed: {failed}")
    
    if successful > 0:
        print(f"\n📁 MP3 files will be saved to Metube's configured audio output directory")
        if output_dir:
            print(f"   (Note: You configured {output_dir}, but Metube uses its own settings)")
        print(f"🌐 Monitor progress at: http://localhost:8081")
        print(f"⏳ Audio extraction may take some time depending on video length")
        print(f"🎵 Files will be downloaded as MP3 audio-only format")


if __name__ == "__main__":
    main()