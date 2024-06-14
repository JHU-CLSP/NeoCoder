"""Perform single-thread inference using specificed model
"""
import click
import os
from torch.utils.data import DataLoader

from src.utils.configs import get_dp_generate_params
from src.dp.generator import CodeGenerator

@click.command()
@click.option("--problem-set-dir", type=click.Path(exists=True), required=True, help="The path of the problem set.")
@click.option("--model-name", type=str, required=True, help="The name of the model.")
@click.option("--num-sample", type=int, default=100, help="The number of samples to run DP.")
@click.option("--dp-rounds", type=int, default=3, help="The number of DP rounds.")
@click.option("--output-dir", type=click.Path(exists=True), required=True, help="The directory to save the generated DP.")
@click.option("--overwrite", is_flag=True, help="Whether to overwrite the existing DP generation file.")

def main(
    problem_set_dir,
    model_name,
    num_sample,
    dp_rounds,
    output_dir,
    overwrite,
):
    params = get_dp_generate_params(problem_set_dir, 
                                    model_name,
                                    num_sample,
                                    dp_rounds)

    generator: CodeGenerator = params['generator']
    dataloader: DataLoader = params['dataloader']

    file_name = os.path.basename(problem_set_dir)
    num_sample = len(dataloader.dataset)
    diff = file_name.split('diff=')[1].split('_')[0] if 'diff=' in file_name else 'None'
    save_path = os.path.join(output_dir, f'{model_name}_diff={diff}_sample={num_sample}_dp={dp_rounds}.json') if diff != 'None' else \
                os.path.join(output_dir, f'{model_name}_sample={num_sample}_dp={dp_rounds}.json')

    generator.inference(dataloader, 
                        save_path, 
                        overwrite)


if __name__ == '__main__':
    main()
