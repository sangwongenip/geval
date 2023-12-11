import argparse

import openai
import json
import jsonlines
import string
import argparse
import tqdm
import time
import logger

from datasets import load_dataset
import pandas as pd
import autogen
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import autogen
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent
from termcolor import colored

from prettytable import PrettyTable
from scipy.stats import spearmanr, pearsonr, kendalltau

class MultiAgentsDebate(AssistantAgent):
    def __init__(self, n_iters=2, **kwargs):

        super().__init__(**kwargs)
        self.register_reply([Agent,None],
                            reply_func=MultiAgentsDebate._reply_user,
                            position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages  = self._oai_messages[sender]

        user_question = messages[-1]['content']

        commander = AssistantAgent(
            name="Commander",
            max_consecutive_auto_reply=1,
            system_message="Help me calculate the score, and tell other agetns think step-by-step.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        score_agent = AssistantAgent(
            name="Scoring assistant",
            max_consecutive_auto_reply=1,
            system_message="Logically think to score the following sentence.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        # user_agent = UserProxyAgent(
        #     name="Huamn admin",
        #     max_consecutive_auto_reply=0,
        #     system_message="You need to know the final score."
        # )

        critics = AssistantAgent(
            name="Critics",
            system_message="""
            Do you think this score is really accurate? If you think it's not justified, please share your opinion. 
            On the other hand, if you find the score acceptable, just say NO_ISSUES
            """,
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        # critics_2 = AssistantAgent(
        #     name="Critics",
        #     system_message="""
        #     Do you agree the response? If you think it's not justified, please share your opinion.
        #     On the other hand, if you find the score acceptable, just say NO_ISSUES
        #     """,
        #     is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
        #     llm_config=self.llm_config
        # )

        commander.initiate_chat(score_agent, message=user_question)
        time.sleep(2.5)

        for _ in range(self._n_iters):
            commander.send(message="Check if the score is justified. Task description and the following source texts are as follows: " \
                                   + '\n' + user_question + '\n And the responses from Scoring assistant as follows: ' + '\n' \
                                   + commander._oai_messages[score_agent][1]['content'],
                           recipient=critics,
                           request_reply=True)
            time.sleep(2.5)

            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your response. Please calculate the score agian!\n"
                        + feedback,
                recipient=score_agent,
                request_reply=True)
            time.sleep(2.5)

        # commander.send(
        #     message="What is the final score that was derived? Answer with just one score number.\n",
        #     recipient=score_agent,
        #     request_reply=True)

        final_score = score_agent._oai_messages[commander][-2]['content']
        return True, final_score

def set_config(model, key):
    config_list = [
        {

            'model' : model,
            'api_key' : key,

        }
    ]

    llm_config = {
        'timeout':600,
        'config_list' : config_list,
        'temperature' : 0,
        'seed': 453,
    }

    return llm_config

def set_userproxyAgent():
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    return user_proxy

import re

class MultiAgentsDebate(AssistantAgent):
    def __init__(self, n_iters=2, **kwargs):

        super().__init__(**kwargs)
        self.register_reply([Agent,None],
                            reply_func=MultiAgentsDebate._reply_user,
                            position=0)
        self._n_iters = n_iters

    def _reply_user(self, messages=None, sender=None, config=None):
        if all((messages is None, sender is None)):
            error_msg = f"Either {messages=} or {sender=} must be provided."
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if messages is None:
            messages  = self._oai_messages[sender]

        user_question = messages[-1]['content']

        commander = AssistantAgent(
            name="Commander",
            max_consecutive_auto_reply=1,
            system_message="Help me calculate the score, and tell other agetns think step-by-step.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        score_agent = AssistantAgent(
            name="Scoring assistant",
            max_consecutive_auto_reply=1,
            system_message="Logically think to score the following sentence.",
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        # user_agent = UserProxyAgent(
        #     name="Huamn admin",
        #     max_consecutive_auto_reply=0,
        #     system_message="You need to know the final score."
        # )

        critics = AssistantAgent(
            name="Critics",
            system_message="""
            Do you think this score is really accurate? If you think it's not justified, please share your opinion. 
            On the other hand, if you find the score acceptable, just say NO_ISSUES
            """,
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        # critics_2 = AssistantAgent(
        #     name="Critics",
        #     system_message="""
        #     Do you agree the response? If you think it's not justified, please share your opinion.
        #     On the other hand, if you find the score acceptable, just say NO_ISSUES
        #     """,
        #     is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
        #     llm_config=self.llm_config
        # )

        commander.initiate_chat(score_agent, message=user_question)
        time.sleep(2.5)

        for _ in range(self._n_iters):
            commander.send(message="Check if the score is justified. Task description and the following source texts are as follows: " \
                                   + '\n' + user_question + '\n And the responses from Scoring assistant as follows: ' + '\n' \
                                   + commander._oai_messages[score_agent][1]['content'],
                           recipient=critics,
                           request_reply=True)
            time.sleep(2.5)

            feedback = commander._oai_messages[critics][-1]["content"]
            if feedback.find("NO_ISSUES") >= 0:
                break
            commander.send(
                message="Here is the feedback to your response. Please calculate the score agian!\n"
                        + feedback,
                recipient=score_agent,
                request_reply=True)
            time.sleep(2.5)

        # commander.send(
        #     message="What is the final score that was derived? Answer with just one score number.\n",
        #     recipient=score_agent,
        #     request_reply=True)

        final_score = score_agent._oai_messages[commander][-2]['content']
        return True, final_score

def set_config(model, key):
    config_list = [
        {

            'model' : model,
            'api_key' : key,

        }
    ]

    llm_config = {
        'timeout':600,
        'config_list' : config_list,
        'temperature' : 0,
        'seed': 453,
    }

    return llm_config

def set_userproxyAgent():
    user_proxy = autogen.UserProxyAgent(
        name="User",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0
    )
    return user_proxy

def convert_answ(text):
    if 'no' in text:
        return 0
    else:
        return 1

def convert_real(answ):
    answ_list = []
    for a in answ:
        if convert_answ(a) == 0:
            answ_list.append(0)
        else:
            answ_list.append(1)
    if sum(answ_list) >= 2:
        return 1
    else:
        return 0

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] = pearsonr(pred_score, human_score)[0]
    result['spearman'] = spearmanr(pred_score, human_score)[0]
    result['kendalltau'] = kendalltau(pred_score, human_score)[0]

    return result

def print_correlations(result, n):
    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    if (n == 0):
        n = 1
    table.add_row(
        [round(result['pearson'] / n, 4), round(result['spearman'] / n, 4), round(result['kendalltau'] / n, 4)])
    print(table)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Consistency
    argparser.add_argument('--prompt_fp', type=str, default='prompts/qags/con_detailed.txt')
    argparser.add_argument('--save_fp', type=str, default='results/qags_con.json')

    #data url
    argparser.add_argument('--qags_fp', type=str, default='data/mturk_xsum.jsonl')
    argparser.add_argument('--key', type=str, default='sk-KEY')

    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')

    args = argparser.parse_args()

    openai.api_key = args.key

    llm_config =set_config(args.model, args.key)
    user_proxy = set_userproxyAgent()
    multiAgents = MultiAgentsDebate(
        name="Calculating score through debate",
        llm_config=llm_config
    )

    real = []
    results = []
    ignore = 0

    data = data_json = jsonlines.open(args.qags_fp)
    for d in data.iter():
        #try:
            article = d['article']
            summary = d['summary_sentences'][0]['sentence']

            #human score
            score_real = [e['response'] for e in d['summary_sentences'][0]['responses']]
            res_real = convert_real(score_real)
            real.append(res_real)

            #prediction score
            prompt = open(args.prompt_fp).read()
            prompt = prompt.replace('{{Document}}', article).replace('{{Summary}}', summary)

            user_proxy.initiate_chat(multiAgents, message=prompt)
            score_pred = user_proxy._oai_messages[multiAgents][-1]['content']

            res = convert_answ(score_pred)
            results.append(res)

        #except:
            #pass

    results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    final_results = calculate_correlation(score_pred, score_real, results)
    print(final_results)

