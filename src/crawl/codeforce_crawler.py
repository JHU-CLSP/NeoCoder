from overrides import overrides
import os
from bs4 import BeautifulSoup
import openpyxl
from typing import Dict, Any, Text, List
import pandas as pd
import json
import random
import time
from tqdm import tqdm
import requests
from zenrows import ZenRowsClient


from .crawler import Crawler

class CodeforceCrawler(Crawler):

    def __init__(self,
                 workbook_path: Text,
                 num_sample: int,
                 difficulty: int):
        super().__init__()

        api_key = os.environ["ZENROWS_API_KEY"]
        if api_key == None or api_key == "":
            raise Exception("ZENROWS_API_KEY not found")

        self.client = ZenRowsClient(api_key, retries=3)
        self.workbook_path = workbook_path
        self.num_sample = num_sample
        self.difficulty = difficulty

    @overrides
    def get_info(self) -> Dict[Text, Any]:
        """Get the url and other information from the workbook.
        """
        
        problem_ids, problem_links, problem_solution_links = [], [], []

        df = pd.read_excel(self.workbook_path)
        df = df[df['difficulty'] <= self.difficulty]

        problem_names = df['problem features'].tolist()
        problem_difficulties = df['difficulty'].tolist()          

        workbook = openpyxl.load_workbook(self.workbook_path)
        worksheet = workbook.active 
        for row in worksheet.iter_rows():
            if row[0].coordinate == 'A1':
                continue

            head = row[0]
            tail = row[-1]

            problem_links.append(head.hyperlink.target)
            problem_solution_links.append(tail.hyperlink.target)

            i = head.hyperlink.target.split('/')
            problem_id = i[-2] + i[-1]
            problem_ids.append(problem_id)
        
        # sanity check
        assert len(problem_names) == len(problem_links), "some problem links are missing"
        
        self.info = {
            "problem_name": problem_names[:self.num_sample],
            "problem_difficulty": problem_difficulties[:self.num_sample],
            "problem_id": problem_ids[:self.num_sample],
            "problem_link": problem_links[:self.num_sample],
            "problem_solution_link": problem_solution_links[:self.num_sample]
        }

    @overrides
    def crawl(self):
        """Crawl the Codeforce problems from the given urls.
        """
        crawl_results = []
        for url in tqdm(self.info['problem_link']):
            crawl_results.append(self._crawl(url))
        self.info['problem_statement'] = crawl_results
        
    @overrides    
    def _crawl(self, url: Text, **kwargs) -> Text:
        """Crawl the Codeforce problem from the given url.
        """
        response = self.client.get(url)
        html = response.content

        soup = BeautifulSoup(html, 'html.parser')

        # Find the div with class 'problem-statement'
        problem_statement_div = soup.find('div', class_='problem-statement')

        # Extract text from the div
        problem_statement = problem_statement_div.get_text(separator='\n', strip=True)
        return problem_statement
    
    def save_to_dir(self, 
                    path: Text):
        """Save the crawled data to the given directory.
        """
        with open(path, 'w') as f:
            for i in range(self.num_sample):
                json.dump({
                    'problem_id': self.info['problem_id'][i],
                    'problem_name': self.info['problem_name'][i],
                    'problem_difficulty': self.info['problem_difficulty'][i],
                    'problem_link': self.info['problem_link'][i],
                    'problem_solution_link': self.info['problem_solution_link'][i],
                    'problem_statement': self.info['problem_statement'][i]
                }, f)
                f.write('\n')

class CodeforceSolutionCrawler(Crawler):

    def __init__(self,
                 crawled_problem_path: Text,
                 num_sample: int,
                 save_path: Text,
                 max_solution_num: int = 30):
        super().__init__()
        # self.client = ZenRowsClient(api_key, retries=3)
        self.scraper_api_key = os.environ["SCRAPER_API_KEY"]
        if self.scraper_api_key == None or self.scraper_api_key == "":
            raise Exception("SCRAPER_API_KEY not found")

        self.submissions = {}
        self.max_solution_num = max_solution_num
        self.save_path = save_path
        with open(crawled_problem_path, 'r') as f:
            self.problem_set = [json.loads(line) for line in f.readlines()]
        self.problem_set = self.problem_set[:num_sample]

    @overrides
    def crawl(self):
        problem_ids = [problem['problem_id'] for problem in self.problem_set]
        urls = [problem['problem_solution_link'] for problem in self.problem_set]

        if os.path.exists(self.save_path):
            with open(self.save_path, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        for problem_id, url in tqdm(zip(problem_ids, urls), 'Crawling Codeforce Submissions', total=len(problem_ids)):
            if problem_id in data and len(data[problem_id]) == self.max_solution_num:
                self.submissions.update({problem_id:data[problem_id]})
                continue
            else:
                codes = self._crawl(url, total=self.max_solution_num)
                self.submissions.update({problem_id:codes})
                data[problem_id] = codes # replace the old data or add new data
                self.save_to_dir(self.save_path, data)

    @overrides
    def _crawl(self, url, **kwargs):
        """Get all submission codes for one problem
        """
        total = kwargs.get('total', -1)
        submission_urls = self.get_submission_urls(url)
        source_codes = []

        counter = 0
        for submission_url in submission_urls:
            source_code = self.get_source_code(submission_url)
            if source_code:
                source_codes.append(source_code)
                counter += 1

            if counter >= total and total != -1:
                break
        return source_codes

    def get_submission_urls(self, url):
        """Submision urls for one problem
        """
        submission_urls = []
        # latest 10 pages of submissions
        for i in range(1, 11):
            new_url = f'{url}/page/{i}?order=BY_PROGRAM_LENGTH_ASC'
            soup = self.safe_request(new_url)
            if soup:
                for link in soup.find_all('a', href=True):
                    if 'submission' in link['href']:
                        submission_url = f"https://codeforces.com{link['href']}"
                        submission_urls.append(submission_url)
            else:
                continue

        # randome shuffle the submission urls
        random.Random(42).shuffle(submission_urls)
        return submission_urls

    def get_source_code(self, url):
        """Get source code from submission url
        """
        soup = self.safe_request(url) 
        code_block = soup.find('pre') if soup else None
        if code_block:
            return code_block.text.strip()
        else:
            return None
    
    def safe_request(self, url, success_list=[200], num_retries=2, pause=60):
        """Safe request with backoff
        """
        for _ in range(num_retries):
            # response = self.client.get(url)
            response = requests.get('http://api.scraperapi.com', params={'api_key': self.scraper_api_key, 'url': url})
            if response.status_code in success_list:
                return BeautifulSoup(response.content, 'html.parser')
            else:
                print(f"Sleeping {url} for {pause} seconds")
                time.sleep(pause)
        return None
    
    @overrides
    def get_info(self):
        exception_ids = []
        check_sub = [len(i) for i in self.submissions.values()]
        for i, ID in enumerate(self.submissions):
            if len(self.submissions[ID]) == 0:
                exception_ids.append(i)
        print(f"check submissions: {check_sub}")
        print(f"total submissions {sum(check_sub)}")
        print(f"# un-crawled problems {len(exception_ids)}")
    
    def save_to_dir(self,
                    path: Text,
                    data: Dict[Text, List[Text]]):
        """Save the crawled solutions to the given json file in real time
        """
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)