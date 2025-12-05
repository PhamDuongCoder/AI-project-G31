#!/usr/bin/env python3
"""
Installation script for the Interactive Chess Game.
This script will install the required dependencies.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install the required packages from requirements.txt"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_piece_images():
    """Check if all required piece images are present"""
    required_images = [
        'wp.png', 'bp.png',  # pawns
        'wr.png', 'br.png',  # rooks
        'wn.png', 'bn.png',  # knights
        'wb.png', 'bb.png',  # bishops
        'wq.png', 'bq.png',  # queens
        'wk.png', 'bk.png'   # kings
    ]
    
    missing_images = []
    for image in required_images:
        if not os.path.exists(os.path.join('img', image)):
            missing_images.append(image)
    
    if missing_images:
        print("⚠️  Warning: Missing piece images:")
        for image in missing_images:
            print(f"   - {image}")
        print("The game will still run but some pieces may not display correctly.")
    else:
        print("✅ All piece images found!")
    
    return len(missing_images) == 0

def main():
    """Main installation function"""
    print("🎮 Interactive Chess Game - Installation")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('chess_game.py'):
        print("❌ Error: chess_game.py not found!")
        print("Please run this script from the project directory.")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check piece images
    check_piece_images()
    
    print("\n🎉 Installation complete!")
    print("Run 'python chess_game.py' to start the game.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
