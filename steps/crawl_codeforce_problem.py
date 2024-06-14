import click
import os
from src.crawl.codeforce_crawler import CodeforceCrawler

@click.command()
@click.option("--raw-data-dir", type=click.Path(exists=True), help="File Path of the raw spreadsheet of CodeForce problems.")
@click.option("--save-dir", type=click.Path(exists=True), help="Folder directory to save the crawled data.")
@click.option("--num-sample", type=click.INT, default=300, help="The number of samples to be crawled.")
@click.option("--difficulty", type=click.INT, default=800, help="The difficulty of the problems to be crawled.")
def main(
    raw_data_dir,
    save_dir,
    num_sample,
    difficulty,
):
    
    crawler = CodeforceCrawler(raw_data_dir,
                               num_sample, 
                               difficulty)
    # get problem info
    crawler.get_info()

    # crawling problem statements
    crawler.crawl()

    save_path = os.path.join(save_dir, 'problem_set_diff={diff}_sample={sample}.jsonl'.format(
        diff=difficulty,
        sample=num_sample
    ))

    crawler.save_to_dir(path=save_path)

if __name__ == "__main__":
    main()