import click
import os
from src.utils.configs import correctness_evaluation, technique_detection, calculate_creativity

@click.command()
@click.option("--task", type=click.Choice(["correctness", "detection", "creativity"]), help="Task to perform.")
@click.option("--inference-result-path", type=click.Path(exists=True), help="File Path of the inference result of dp dataset.")
@click.option("--human-solution-path", type=click.Path(exists=True), help="File Path of the human solutions", default=None)
@click.option("--test-case-path", type=click.Path(exists=True), help="File Path of the test case of dp dataset.", default=None)
@click.option("--save-folder", type=click.Path(), help="Folder to save the evaluation result.", default=None)

def main(
    task,
    inference_result_path,
    human_solution_path,
    test_case_path,
    save_folder
):
    if task == "detection":
        assert human_solution_path is not None, "Please provide human solution path."

        technique_detection(human_solution_path=human_solution_path,
                            inference_result_path=inference_result_path)
    elif task == "correctness":
        assert test_case_path is not None, "Please provide test case path."
        correctness_evaluation(inference_result_path=inference_result_path,
                               test_case_path=test_case_path,
                               save_folder=save_folder)
    elif task == "creativity":
        convergent_thinking, divergent_thinking, creativity = calculate_creativity(
            human_solution_path=human_solution_path,
            inference_result_path=inference_result_path,
            save_folder=save_folder
            )
        print(f"Convergent Thinking: {convergent_thinking}")
        print(f"Divergent Thinking: {divergent_thinking}")
        print(f"Creativity: {creativity}")
    else:
        raise ValueError("Invalid task.")


if __name__ == "__main__":
    main()
