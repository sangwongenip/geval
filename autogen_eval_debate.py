import json
import argparse
import tqdm
import time
import logger
import jsonlines

from datasets import load_dataset
import pandas as pd
import autogen
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import openai

import autogen
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent
from termcolor import colored

class MultiAgentsDebate(AssistantAgent):
    def __init__(self, n_iters=5, **kwargs):

        super().__init__(**kwargs)
        self.register_reply([Agent,None],
                            reply_func=MultiAgentsDebate._reply_user,
                            position=0)
        self._n_iters = n_iters
        self.tiebreak_prompt = None

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

        critics = AssistantAgent(
            name="Critics",
            system_message="""
            Do you think this score is really accurate? If you think it's not justified, please share your opinion. 
            On the other hand, if you find the score acceptable, just say NO_ISSUES.
            """,
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )

        commander.initiate_chat(score_agent, message=user_question)
        time.sleep(2.5)

        communication_string = ''
        is_colab = 0

        for _ in range(self._n_iters):
            commander.send(message="Check if the score is justified. Task description and the following source texts are as follows: " \
                                   + '\n' + user_question + '\n And the responses from Scoring assistant as follows: ' + '\n' \
                                   + commander._oai_messages[score_agent][1]['content'],
                           recipient=critics,
                           request_reply=True)
            communication_string += '(Scorer)\n'
            communication_string += commander._oai_messages[score_agent][1]['content']
            communication_string += '\n'

            time.sleep(2.5)

            feedback = commander._oai_messages[critics][-1]["content"]
            communication_string += '(Critic)\n'
            communication_string += commander._oai_messages[critics][-1]["content"]
            communication_string += '\n'

            if feedback.find("NO_ISSUES") >= 0:
                is_colab += 1
                break
            commander.send(
                message="Here is the feedback to your response. Please calculate the score again!\n"
                        + feedback,
                recipient=score_agent,
                request_reply=True)
            time.sleep(2.5)

        # commander.send(
        #     message="What is the final score that was derived? Answer with just one score number.\n",
        #     recipient=score_agent,
        #     request_reply=True)

        if is_colab != 0:
            print('NO TIEBREAK!')
            final_score = score_agent._oai_messages[commander][-2]['content']
        else:
            #final_score = score_agent._oai_messages[commander][-2]['content']
            print('TIEBREAK!\n\n\n')
            tiebreaker_prompt = self.tiebreak_prompt.replace('{{Debate}}', communication_string)
            tiebreak_prompt_file = open('./tiebreak_prompts.txt', 'a')
            tiebreak_prompt_file.write(tiebreaker_prompt)
            tiebreak_prompt_file.write('\n\n###############\n\n')
            final_score = inference_tiebreaker(tiebreaker_prompt)
            tiebreak = open('./tiebreak.txt', 'a')
            tiebreak.write(final_score)
            tiebreak.write('\n')

        print('SCORED!')
        print(final_score)
        final_score = final_score

        return True, final_score 

def inference_tiebreaker(message):

    _response = openai.ChatCompletion.create(
        model=args.model,
        messages=[{"role": "system", "content": message}],
        temperature=0,
        max_tokens=10,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        n=1
    )

    all_responses = [_response['choices'][i]['message']['content'] for i in
                     range(len(_response['choices']))]

    return all_responses[0]

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

def normalize_string(s):
    s = ' '.join(s.split())
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower().strip()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # Consistency
    argparser.add_argument('--prompt_fp', type=str, default='prompts/summeval/rel_detailed.txt')
    argparser.add_argument('--prompt_tiebreak', type=str, default='prompts/summeval/rel_tiebreak.txt')
    argparser.add_argument('--aspect', type=str, default='Relevance')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_rel_detailed_openai_sangwon.json')
    argparser.add_argument('--summeval_fp', type=str, default='data/filtered_summeval.json')
    argparser.add_argument('--key', type=str, default='sk-KEY')
    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    # gpt-4-turbo: gpt-4-1106-preview
    # gpt-3.5-turbo: gpt-3.5-turbo-1106
    args = argparser.parse_args()

    openai.api_key = args.key

    summeval = json.load(open(args.summeval_fp))


    prompt = open(args.prompt_fp).read()
    tiebreak_prompt = open(args.prompt_tiebreak).read()
    aspect = args.aspect

    ct, ignore = 0, 0


    llm_config =set_config(args.model, args.key)

    user_proxy = set_userproxyAgent()
    multiAgents = MultiAgentsDebate(
        name="Calculating score through debate",
        llm_config=llm_config
    )

    results = []
    ignore = 0

    new_json = []

    example_string = ''

    for instance in tqdm.tqdm(summeval):
        source = instance['source']
        system_output = instance['system_output']
        cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)
        tiebreak_cur_prompt = tiebreak_prompt.replace('{{Document}}', source).replace('{{Summary}}', system_output)

        multiAgents.tiebreak_prompt = tiebreak_cur_prompt

        instance['prompt'] = cur_prompt
        while True:
            try:
                user_proxy.initiate_chat(multiAgents, message=cur_prompt)
                score = user_proxy._oai_messages[multiAgents][-1]['content']

                instance[aspect.lower()] = score

                new_json.append(instance)
                break

            except Exception as e:
                print(e)
                if ("limit" in str(e)):
                    time.sleep(3)
                else:
                    ignore += 1
                    print('ignored', ignore)
                    break

    print('ignored total', ignore)
    with open(args.save_fp, 'w') as f:
        json.dump(new_json, f, indent=4)
