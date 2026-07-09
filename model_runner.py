import os
import shutil
import subprocess


def run_dehaze(input_image_path):

    repo_path = os.path.dirname(
    os.path.abspath(__file__)
    )

    input_folder = os.path.join(
        repo_path,
        "results/dehaze/input"
    )

    output_folder = os.path.join(
        repo_path,
        "results/dehaze/output"
    )

    # Clear old input images
    for f in os.listdir(input_folder):
        os.remove(
            os.path.join(input_folder, f)
        )

    filename = os.path.basename(
        input_image_path
    )

    shutil.copy(
        input_image_path,
        os.path.join(
            input_folder,
            filename
        )
    )

    subprocess.run(
        [
            "python3",
            "main_test.py",
            "--dataset",
            "dehaze",
            "--datasetpath",
            "results/dehaze/input"
        ],
        cwd=repo_path
    )

    output_path = os.path.join(
        output_folder,
        filename
    )

    return output_path