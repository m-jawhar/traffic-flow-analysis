#!/usr/bin/env python3
"""
Quick setup: Download sample traffic video for testing
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("🎬 Traffic Video Setup")
    print("=" * 50)
    print()

    # Create directory
    video_dir = Path("data/videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory: {video_dir}")
    print()

    # Sample video from YouTube (Creative Commons)
    print("📥 Downloading sample traffic video...")
    print("   Using YouTube with yt-dlp...")
    print()

    # Install yt-dlp if needed
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        print("✓ yt-dlp is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("📦 Installing yt-dlp...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "yt-dlp"])
        print("✓ yt-dlp installed")

    print()

    # Download sample traffic video
    video_path = video_dir / "sample_traffic.mp4"

    if video_path.exists():
        print(f"✓ Video already exists: {video_path}")
        print(f"   Size: {video_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("⬇️  Downloading traffic footage...")
        print("   (This may take 1-2 minutes)")
        print()

        # Use a Creative Commons traffic video
        video_url = (
            "https://www.youtube.com/watch?v=MNn9qKG2UFI"  # Traffic camera footage
        )

        cmd = [
            "yt-dlp",
            "-f",
            "best[height<=480]",  # Lower quality for faster download
            "--no-playlist",
            "-o",
            str(video_path),
            video_url,
        ]

        try:
            result = subprocess.run(cmd, check=True)
            print()
            print(f"✓ Downloaded: {video_path}")
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ Download failed: {e}")
            print()
            print("📝 Manual alternatives:")
            print("   1. Download any traffic video from YouTube using yt-dlp:")
            print(
                "      yt-dlp -f 'best[height<=720]' -o data/videos/sample_traffic.mp4 <URL>"
            )
            print()
            print("   2. Or search 'traffic camera' on YouTube and use above command")
            print()
            print("   3. Or use your own traffic video file")
            return

    print()
    print("=" * 50)
    print("🎉 Setup complete!")
    print()
    print("📊 Run traffic analysis:")
    print("   python examples/simple_analysis.py")
    print()


if __name__ == "__main__":
    main()
