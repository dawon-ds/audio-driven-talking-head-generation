import os
import subprocess
import sys


def test_ffmpeg(ffmpeg_path):
    print(f"Testing ffmpeg path: {ffmpeg_path}")

    path_separator = ';' if sys.platform == 'win32' else ':'
    os.environ["PATH"] = f"{ffmpeg_path}{path_separator}{os.environ['PATH']}"

    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        print("FFmpeg test successful!")
        print("FFmpeg version information:")
        print(result.stdout)
        return True
    except Exception as e:
        print("FFmpeg test failed!")
        print(f"Error message: {str(e)}")
        return False


if __name__ == "__main__":
    default_path = r"ffmpeg-master-latest-win64-gpl-shared\bin"
    ffmpeg_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    test_ffmpeg(ffmpeg_path)
