import json
import argparse
import tqdm
import time
import logger
import re
import random
import pandas as pd

import autogen
from autogen import AssistantAgent, Agent, UserProxyAgent, ConversableAgent

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
        
        critics = AssistantAgent(
            name="Critics",
            system_message="""
            Do you think this score is really accurate? If you think it's not justified, please share your opinion. 
            On the other hand, if you find the score acceptable, just say NO_ISSUES
            """,
            is_termination_msg=lambda x: x.get("content","").rstrip().endswith("TERMINATE"),
            llm_config=self.llm_config
        )
        
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


def load_json(data_path):
    with open(data_path) as f:
        data = json.loads(f.read())
    return data


def normalize_string(s):
    s = ' '.join(s.split())
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    return s.lower().strip()

if __name__ == '__main__':
    
    argparser = argparse.ArgumentParser()
    # Consistency
    argparser.add_argument('--prompt_fp', type=str, default='prompts/topical_chat/eng_detailed.txt')
    argparser.add_argument('--aspect', type=str, default='engagingness')
    argparser.add_argument('--save_fp', type=str, default='results/gpt4_eng_detailed_openai.json')
    argparser.add_argument('--dataset', type=str, default='data/topical_chat.json')
    argparser.add_argument('--key', type=str, default='')
    argparser.add_argument('--model', type=str, default='gpt-4-1106-preview')
    # gpt-4-turbo: gpt-4-1106-preview	
    # gpt-3.5-turbo: gpt-3.5-turbo-1106
    args = argparser.parse_args()

    prompt = open(args.prompt_fp).read()
    aspect = args.aspect
    llm_config =set_config(args.model, args.key)
    
    dataset = load_json(args.dataset)
    
    grouped_data = {}
    for entry in dataset:
        key = (entry['source'], entry['context'])
        entry['doc_id'] = key
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append(entry)

    selected_keys = random.sample(list(grouped_data.keys()), 1)

    dataset = [item for key in selected_keys for item in grouped_data[key]]
    n_data = len(dataset)
    
    cur_prompt_list = []
    for i in range(n_data):

        if aspect == 'naturalness':
            cur_prompt = prompt.replace('{{Response}}', dataset[i]['system_output'])
        
        elif aspect == 'coherence':
            cur_prompt = prompt.replace('{{Response}}', dataset[i]['system_output']).replace('{{Dialogue}}',dataset[i]['source'])

        elif aspect == 'groundedness':
            cur_prompt = prompt.replace('{{Response}}', dataset[i]['system_output']).replace('{{Dialogue}}',dataset[i]['source']).replace('{{Fact}}', dataset[i]['context'])
        
        elif aspect == 'engagingness':
            cur_prompt = prompt.replace('{{Response}}', dataset[i]['system_output']).replace('{{Dialogue}}',dataset[i]['source']).replace('{{Fact}}', dataset[i]['context'])
        
        cur_prompt_list.append(cur_prompt)
        
    
    user_proxy = set_userproxyAgent()
    multiAgents = MultiAgentsDebate(
            name="Calculating score through debate",
            llm_config=llm_config
    )
    
    results = []
    ignore = 0
    
    
    dataset = pd.DataFrame(dataset)
    
    for idx in tqdm.tqdm(range(len(dataset))):
        instance_dict = {}
        instance_dict['doc_id'] = dataset.iloc[idx]['doc_id']
        instance_dict['source'] = dataset.iloc[idx]['source']
        instance_dict['system_output'] = dataset.iloc[idx]['system_output']
        instance_dict['context'] = dataset.iloc[idx]['context']
        instance_dict['human_score'] = dataset.iloc[idx]['scores'][aspect.lower()]
        
        try:
            user_proxy.initiate_chat(multiAgents, message=cur_prompt_list[idx])
            score = user_proxy._oai_messages[multiAgents][-1]['content']
            instance_dict[aspect.lower()] = score
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