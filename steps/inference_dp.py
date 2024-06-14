"""Perform parallel-thread inference on generated DP dataset.
"""
import click
import os
from torch.utils.data import DataLoader

from src.dp.generator import CodeGenerator
from src.utils.configs import get_dp_inference_params

@click.command()
@click.option("--dataset-path", type=click.Path(exists=True), required=True, help="The path of the DP dataset.")
@click.option("--model-name", type=str, required=True, help="The name of the model.")
@click.option("--dp-rounds", type=int, default=3, help="The number of DP rounds.")
@click.option("--batch-size", type=int, default=2, help="Note the real batch size feeding to hf model is batch_size * dp_rounds.")
@click.option("--output-dir", type=click.Path(exists=True), required=True, help="The directory to save the generated codes.")
@click.option("--overwrite", is_flag=True, help="Overwrite the existing file.")

def main(
    dataset_path,
    model_name,
    dp_rounds,
    batch_size,
    output_dir,
    overwrite
):

    params = get_dp_inference_params(dataset_path,
                                     model_name,
                                     dp_rounds,
                                     batch_size)
    
    generator: CodeGenerator = params['generator']
    dataloader: DataLoader = params['dataloader']

    file_name = os.path.basename(dataset_path)
    diff = file_name.split('diff=')[1].split('_')[0] if 'diff=' in file_name else 'None'
    num_sample = len(dataloader.dataset)
    model_name = model_name.split('/')[1] if '/' in model_name else model_name
    save_path = os.path.join(output_dir, f'{model_name}_diff={diff}_sample={num_sample}_dp={dp_rounds}.json') if diff != 'None' else \
                os.path.join(output_dir, f'{model_name}_sample={num_sample}_dp={dp_rounds}.json')

    generator.inference(dataloader,
                        save_path,
                        overwrite=overwrite)

if __name__ == "__main__":
    main()

