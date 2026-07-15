import subprocess
import sys


def main():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Colab environment ready.")


if __name__ == "__main__":
    main()
