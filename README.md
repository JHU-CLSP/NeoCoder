# Benchmarking Language Model Creativity: A Case Study on Code Generation

This is the official code release accompanying our paper "Benchmarking Language Model Creativity: A Case Study on Code Generation". Our dataset under `datasets/CodeForce/NeoCoder` contains:
1. **NeoCoder dataset**: 199 problems with maximum of 5 temporally relevant constraints.
2. **Historical human solutions**: 30 human solutions per problem and their technique detection results (by GPT-4).
3. **Human annotated test cases**: Our manually annotated test cases for fixing certain parsing problems from crawling. 
4. **Other supporting files**: 500 crawled original codeforces problems and crawled raw test cases.

## File Structure Description

```shellscript
steps/   // callable scripts correspond to each step of denial prompting and creativity evaluation.
src/     // source code of models, evaluators, data collations, etc. 
scripts/ // bash scripts to scale up experiments.
```

## Setup
1. Setup Zenrows API for scraping: `echo "export ZENROWS_API_KEY='yourkey'" >> ~/.bashrc`
2. Setup OpenAI API for generations: `echo "export OPENAI_API_KEY='yourkey'" >> ~/.bashrc`
3. Creative environment: `conda create --name creativity python=3.9`
4. Activate environment: `conda activate creativity`
5. Setup environment: `pip install -r requirements.txt`

## Full Steps to Reproduce Our Dataset and Results

If someone wants to use only our NeoCoder to reproduce the results, please run steps [Inference](#inference-equation-1-in-the-paper) and [NeoGauge@T Calculation](#neogauget-calculation-section-4-in-the-paper).

### Prepare Dataset
1. Crawl CodeForce problems: `python steps/crawl_codeforce_problem.py --raw-data-dir datasets/CodeForce/raw/CodeForce800spreadsheet.xlsx --save-dir --num-sample --difficulty` 
2. Crawl human solutions:`python steps/crawl_codeforce_solution.py --crawled-problem-path --save-dir --max-solution-num`
3. Prepare Test Cases: `python steps/parse_test_case.py --data-path --output-dir`
4. Manually correcting test cases to match inputs and outputs. We provide our annotated results in `datasets/CodeForce/NeoCoder/test_cases_annotated.json`

### Denial Prompting (Section 3 in the paper)
1. Generate NeoCoder dataset: `python steps/generate_dp.py --problem-set-dir --model-name --num-sample --dp-rounds --output-dir`

   In our experiment, we generate NeoCoder by GPT-4 using the following script: `bash scripts/generate_dp_dataset.sh`

### Inference (Equation 1 in the paper)
1. Inference on NeoCoder dataset: `python steps/inference_dp.py --dataset-path --model-name {HF_MODEL_NAME, OPENAI_MODEL_NAME} --dp-rounds --batch-size --output-dir`

   We provide a running example in `scripts/inference_dp_dataset_llama3.slurm`

### NeoGauge@T Calculation (Section 4 in the paper)
1. Evaluate correctness: `python steps/creativity_evaluation.py --task correctness --inference-result-path --test-case-path --save-folder --model-family`

   We provide a running example in `scripts/correctness_evaluation.sh`

2. Detect Techniques: `python steps/creativity_evaluation.py --task detection --inference-result-path --human-solution-path`

   We provide a running example in `scripts/detect_techniques.sh`

3. Final NeoGauge@T Calculation: `python steps/creativity_evaluation.py --task creativity --inference-result-path --human-solution-path --save-folder`

**Note** that the `NeoCoder.json` file is originally and automatically saved with the name format of `{model_name}_diff={diff}_sample={num_sample}_dp={dp_rounds}.json`. For simplicity purposes, we manually change the name to **NeoCoder** to match the dataset name in our paper. 
