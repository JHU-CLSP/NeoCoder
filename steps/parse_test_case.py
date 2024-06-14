import click
import re
import json
import os
import numpy as np

@click.command()
@click.option("--data-path", type=click.Path(exists=True), help="File Path of the crawled CodeForce problems.")
@click.option("--output-dir", type=click.Path(exists=False, file_okay=True), help="File Path of the parsed test cases.")

def main(data_path,
         output_dir):
    """Parse the test cases from the crawled CodeForce problems.
    """
    with open(data_path, 'r') as f:
        problem_set = [json.loads(line) for line in f]
    
    def parse(pattern, problem):
        match = re.search(pattern, problem['problem_statement'], re.DOTALL)
        input = match.group(1).strip().split('\n')
        if len(input[0]) == 1:
            num_test_cases = int(input.pop(0))
        else:
            # TODO: need manual correction
            return [None], [None]
        test_case_length = len(input) / num_test_cases
        if test_case_length % 1 != 0:
            # TODO: need manual correction
            return [None], [None]
        else:
            test_case_length = int(test_case_length)
        input = [item.split(' ') for item in input]
        # evenly divide the input into test cases
        input = [input[i:i + test_case_length] for i in range(0, len(input), test_case_length)]
    
        output = match.group(2).strip().split('\n')
    
        return input, output

    start = "Examples?\nInput\n"  # Replace with your start substring
    end = "\nNote"              # Replace with your end substring

    for problem in problem_set:
        pattern = f"{start}(.*)\nOutput\n(.*){end}"
        match = re.search(pattern, problem['problem_statement'], re.DOTALL)
        if match:
            input, output = parse(pattern, problem)
            if len(input) != len(output):
                # TODO: need manual correction
                problem.update({"input":input, "output":output, "error":True})
            else:
                problem.update({"input":input, "output":output})
        else:
            pattern = f"{start}(.*)\nOutput\n(.*)"
            input, output = parse(pattern, problem)
            if len(input) != len(output):
                # TODO: need manual correction
                problem.update({"input":input, "output":output, "error":True})
            else:
                problem.update({"input":input, "output":output})

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.basename(data_path)
    output_path = os.path.join(output_dir, basename[:-6] + '_test_cases.json')
    with open(output_path, 'w') as file:
        json.dump(problem_set, file, indent=4)


if __name__ == "__main__":
    main()