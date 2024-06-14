import click
import os

from src.crawl.codeforce_crawler import CodeforceSolutionCrawler

@click.command()
@click.option("--crawled-problem-path", type=click.Path(exists=True), help="File Path of the crawled CodeForce problems.")
@click.option("--num-sample", type=click.INT, default=100, help="Number of samples to crawl.")
@click.option("--save-dir", type=click.Path(exists=True), help="Folder directory to save the crawled data.")
@click.option("--max-solution-num", type=int, default=30, help="Maximum number of solutions to crawl for each problem.")

def main(
    crawled_problem_path,
    num_sample,
    save_dir,
    max_solution_num,
):
    
    
    save_file_name = os.path.basename(crawled_problem_path).split('sample')[0] + f'sample={num_sample}' + f'_sol={max_solution_num}.json'
    save_path = os.path.join(save_dir, save_file_name)
    
    crawler = CodeforceSolutionCrawler(crawled_problem_path,
                                       num_sample,
                                       save_path=save_path,
                                       max_solution_num=max_solution_num)

    # crawling solutions
    crawler.crawl()

    crawler.get_info()

if __name__ == "__main__":
    main()