import json
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

def normalize_string(s):
    s = ' '.join(s.split())
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower().strip()

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    # Consistency
    argparser.add_argument('--prompt_fp', type=str, default='prompts/summeval/con_detailed.txt')
    argparser.add_argument('--aspect', type=str, default='Consistency')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_con_detailed_openai.json')
    argparser.add_argument('--summeval_fp', type=str, default='data/summeval.json')
    argparser.add_argument('--key', type=str, default='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')
    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    # gpt-4-turbo: gpt-4-1106-preview	
    # gpt-3.5-turbo: gpt-3.5-turbo-1106
    args = argparser.parse_args()
    
    summeval = json.load(open(args.summeval_fp))
    
    dataset = load_dataset("mteb/summeval")
    sampled = pd.DataFrame(dataset['test'])
    sampled = sampled.sample(frac=1, random_state=1234).reset_index(drop=True)
    sampled = sampled[:10]
    
    prompt = open(args.prompt_fp).read()
    aspect = args.aspect
    
    
    llm_config =set_config(args.model, args.key)
    
    user_proxy = set_userproxyAgent()
    multiAgents = MultiAgentsDebate(
            name="Calculating score through debate",
            llm_config=llm_config
    )
    
    results = []
    ignore = 0
    for idx in tqdm.tqdm(range(sampled.shape[0])):
        sampled_idx = sampled.iloc[idx]
        source = sampled_idx['text']
        candidates_summeval4human = [x for x in summeval if x['doc_id'] == sampled_idx['id']]

        for machine_idx in range(len(sampled_idx['machine_summaries'])):
            instance_dict = {}
            instance_dict['doc_id'] = sampled_idx['id']
            instance_dict['source'] = source
            instance_dict['system_output'] = sampled_idx['machine_summaries'][machine_idx]
            cur_prompt = prompt.replace('{{Document}}', source).replace('{{Summary}}', sampled_idx['machine_summaries'][machine_idx])
            instance_dict['human_score'] = candidates_summeval4human[machine_idx]['scores'][aspect.lower()]
            # for cands in candidates_summeval4human:
            #     if normalize_string(sampled_idx['machine_summaries'][machine_idx]) == normalize_string(cands['reference']):
            #         instance_dict['human_score'] = cands['scores'][aspect]
            #         print(instance_dict['human_score'])
            try:
                user_proxy.initiate_chat(multiAgents, message=cur_prompt)
                score = user_proxy._oai_messages[multiAgents][-1]['content']
                instance_dict[aspect.lower()] = score
                
                # time.sleep(1.5)
                
                results.append(instance_dict)
                print('-'*50 + 'added result' + '-'*50)
                
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
        json.dump(results, f, indent=4)          