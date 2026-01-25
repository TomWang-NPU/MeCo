import os
import json
import pickle
import numpy as np
from rocobench.envs import MujocoSimEnv, EnvState
import openai
from openai import OpenAI
from datetime import datetime
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from typing import List, Tuple, Dict, Union, Optional, Any
import logging
import requests

PATH_PLAN_INSTRUCTION = """
[How to plan PATH]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach.
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects.
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)]
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
"""

client = OpenAI(api_key="", base_url="https://api.deepseek.com")



def get_chat_prompt(env: MujocoSimEnv):
    robot_names = env.get_sim_robots().keys()
    talk_order_str = ",".join([f"[{name}]" for name in robot_names])
    chat_prompt = f"""
The robots discuss to find the best strategy. They carefully analyze others' responses and use [Environment Feedback] to improve their plan.
They talk in order {talk_order_str}... Once they reach agreement, they summarize the plan by **strictly** following [Action Output Instruction] to format the output, then stop talking.
Their entire discussion and final plan are:
    """
    return chat_prompt


def get_plan_prompt(env: MujocoSimEnv):
    return """
Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Your reasoning and final plan output are:
    """

def get_HAMS_prompt(env: MujocoSimEnv):
    return """
You are a central planner. Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Use [Environment Feedback] to improve your plan. Strictly follow [Action Output Instruction] to format and output the plan.
Your reasoning and final plan output are:
    """


class HAMSPrompter:
    """
    At each round, queries LLM once for each action plan,
    query again with environment feedback if the action plan cannot be executed
    """

    def __init__(
            self,
            env: MujocoSimEnv,
            parser: LLMResponseParser,
            feedback_manager: FeedbackManager,
            comm_mode: str = "plan",  # or chat
            use_waypoints: bool = False,
            use_history: bool = True,
            max_api_queries: int = 3,
            max_calls_per_round: int = 10,
            num_replans: int = 3,
            debug_mode: bool = False,
            temperature: float = 0,
            robot_name_map: Dict[str, str] = {"panda": "Bob"},
            max_tokens: int = 1000,
            llm_source: str = "gpt-4",
    ):
        self.env = env
        self.robot_agent_names = env.get_sim_robots().keys()
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.comm_mode = comm_mode
        self.max_api_queries = max_api_queries
        self.num_replans = num_replans
        self.max_calls_per_round = max_calls_per_round
        self.robot_name_map = robot_name_map
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.temperature = temperature
        self.llm_source = llm_source
        self.max_tokens = max_tokens

        self.round_history = []  # [obs_t, action_t] but only if action_t got executed
        self.failed_plans = []  # could inherit from previous round if the final plan failed to execute in env.
        self.response_history = []  # [response_t]

    def save_state(self, save_path, fname='prompter_state.pkl'):
        state_dict = dict(
            round_history=self.round_history,
            failed_plans=self.failed_plans,
        )
        save_path = os.path.join(save_path, fname)
        with open(save_path, "wb") as f:
            pickle.dump(state_dict, f)

    def load_state(self, load_path, fname='prompter_state.pkl'):
        load_path = os.path.join(load_path, fname)
        with open(load_path, "rb") as f:
            state_dict = pickle.load(f)
        self.round_history = state_dict["round_history"]
        self.failed_plans = state_dict["failed_plans"]

    def compose_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"\n== Round#{i} ==\n{history}"
        ret += f"\n== Current Round ==\n"
        return ret

    def compose_system_prompt(
            self,
            obs_desp: str,
            plan_feedbacks: List[str] = [],
    ):

        task_desp = self.env.describe_task_context()  # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n"

        if self.use_history:
            history_desp = self.compose_round_history()
            full_prompt += history_desp + "\n"

        full_prompt += obs_desp + "\n"

        if len(self.failed_plans) > 0:
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans)
            full_prompt += execute_feedback + "\n"

        if len(plan_feedbacks) > 0:
            feedback_prompt = "Previous Plans Require Improvement:\n"
            feedback_prompt += "\n".join(plan_feedbacks) + "\n"
            full_prompt += feedback_prompt

        if self.comm_mode == "HAMS":
            comm_prompt = get_HAMS_prompt(self.env)
        else:
            raise NotImplementedError
        full_prompt += comm_prompt

        return full_prompt

    def compose_system_prompt_again(
            self,
            obs_desp: str,
            plan_feedback: str = "",
            previous_plan: str = "",
    ):

        task_desp = self.env.describe_task_context()  # should include task rules
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION

        full_prompt = f"{task_desp}\n{action_desp}\n"

        if self.use_history:
            history_desp = self.compose_round_history()
            full_prompt += history_desp + "\n"

        full_prompt += obs_desp

        comm_prompt = """
You are a central planner. Reason about the task step-by-step, and find the best strategy to coordinate the robots. Propose a plan of **exactly** one action per robot.
Your previous plan is:
            """

        full_prompt += comm_prompt

        full_prompt += previous_plan + "\n"

        full_prompt += "\n"

        full_prompt += "\n[Feedback from Local Agents]\n" + plan_feedback

        full_prompt += 'This is the feedback from local agents. If you find some errors in your previous plan, try to modify it. Otherwise, output the same plan as before. Do not explain. The output should strictly follow [Action Output Instruction] to format and output the plan. Your response:'

        return full_prompt

    def compose_system_prompt_agent(
            self,
            obs: EnvState,
            agent_name: str,
            central_plan: str,
    ) -> str:
        action_desp = self.env.get_action_prompt()
        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt_again(obs, agent_name, central_plan)

        system_prompt = f"{action_desp}\n{agent_prompt}\n"



        return system_prompt


    def prompt_one_round_agent(self, obs: EnvState, execute_for_agent, save_path: str = ""):
            plan_feedbacks = []
            agent_responses = ""
            flag = True

            for agent_name in self.robot_agent_names:
                agent_responses += f"{agent_name}:"

                system_prompt = self.compose_system_prompt_agent(
                    obs,
                    agent_name,
                    execute_for_agent
                )

                agent_prompt = f" You are {agent_name}, your response is:"
                response, usage = self.query_once(
                    system_prompt,
                    user_prompt=agent_prompt,
                )

                print("======= Agent Response ======= \n ")
                print(response)

                timestamp = datetime.now().strftime("%m%d-%H%M")
                tosave = [
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": "Planner",
                        "message": response,
                    },
                    usage,
                ]
                fname = f'{save_path}/local{agent_name}_{timestamp}.json'

                json.dump(tosave, open(fname, 'w'))

                if response == "I Agree":
                    agent_responses += " I Agree\n"

                else:
                    flag = False
                    agent_responses += f" {response}\n"

            return flag, agent_responses

    def prompt_one_round(self, obs: EnvState, save_path: str = ""):
        plan_feedbacks = []
        response_history = []
        obs_desp = self.env.describe_obs(obs)
        execute_for_agent = ''
        for i in range(self.num_replans):
            system_prompt = self.compose_system_prompt(obs_desp, plan_feedbacks)
            response, usage = self.query_once(
                system_prompt, user_prompt=""
            )  # NOTE: single_thread doesn't use user role
            response_history.append(response)

            timestamp = datetime.now().strftime("%m%d-%H%M")
            tosave = [
                {
                    "sender": "SystemPrompt",
                    "message": system_prompt,
                },
                {
                    "sender": "UserPrompt",
                    "message": "",
                },
                {
                    "sender": "Planner",
                    "message": response,
                },
                usage,
            ]
            fname = f'{save_path}/replan{i}_{timestamp}.json'

            json.dump(tosave, open(fname, 'w'))


            curr_feedback = "None"
            # try parsing
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, response)
            if not parse_succ:
                execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                curr_feedback = f"""
Parsing failed! {parsed_str}
Previous response: {execute_str}
Re-format to strictly follow [Action Output Instruction]!
                """
                plan_feedbacks.append(curr_feedback)
                ready_to_execute = False
                continue
            else:
                execute_for_agent = 'EXECUTE' + response.split('EXECUTE')[-1]
                prompt_feedback = ''
                flag, response_agent = self.prompt_one_round_agent(obs, execute_for_agent, save_path)
                if flag == False:
                    prompt_feedback += response_agent

                    system_prompt = self.compose_system_prompt_again(obs_desp, prompt_feedback, execute_for_agent)

                    response_again, usage = self.query_once(
                        system_prompt, user_prompt=""
                    )

                    print("======= Again Response ======= \n ")
                    print(response_again)

                    timestamp = datetime.now().strftime("%m%d-%H%M")
                    tosave = [
                        {
                            "sender": "SystemPrompt",
                            "message": system_prompt,
                        },
                        {
                            "sender": "UserPrompt",
                            "message": "",
                        },
                        {
                            "sender": "Planner",
                            "message": response_again,
                        },
                        usage,
                    ]
                    fname = f'{save_path}/again{i}_{timestamp}.json'

                    json.dump(tosave, open(fname, 'w'))


                    parse_succ, parsed_str, llm_plans_again = self.parser.parse(obs, response_again)
                    if not parse_succ:
                        execute_str = 'EXECUTE' + response.split('EXECUTE')[-1]
                        curr_feedback = f"""
                    Parsing failed! {parsed_str}
                    Previous response: {execute_str}
                    Re-format to strictly follow [Action Output Instruction]!
                                    """
                        plan_feedbacks.append(curr_feedback)
                        ready_to_execute = False
                        continue
                    else:
                        ready_to_execute = True
                        for j, llm_plan in enumerate(llm_plans_again):
                            ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                            if not ready_to_execute:
                                curr_feedback = env_feedback
                                break
                        if ready_to_execute:
                            plan_str = parsed_str
                            self.response_history = response_history
                            return ready_to_execute, llm_plans, plan_feedbacks, response_history

                else:
                    ready_to_execute = True
                    for j, llm_plan in enumerate(llm_plans):
                        ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)
                        if not ready_to_execute:
                            curr_feedback = env_feedback
                            break
                    if ready_to_execute:
                        plan_str = parsed_str
                        self.response_history = response_history
                        return ready_to_execute, llm_plans, plan_feedbacks, response_history

        return ready_to_execute, llm_plans, plan_feedbacks, response_history


            # else:

            #
            #
            # plan_feedbacks.append(curr_feedback)
            # tosave = [
            #     {
            #         "sender": "Feedback",
            #         "message": curr_feedback,
            #     },
            #     {
            #         "sender": "Action",
            #         "message": (response if not parse_succ else llm_plans[0].get_action_desp()),
            #     },
            # ]
            # timestamp = datetime.now().strftime("%m%d-%H%M")
            # fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            # json.dump(tosave, open(fname, 'w'))








    def query_once(self, system_prompt, user_prompt=""):
        response = None
        usage = None
        # print('======= system prompt ======= \n ', system_prompt)
        if self.debug_mode:  # query human user input
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()


        for n in range(self.max_api_queries):
            print('querying {}th time'.format(n))
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "system", "content": system_prompt + user_prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=False
                )
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

                response = response.choices[0].message.content
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except:
                print("API error, try again")
            continue
        return response, usage

    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str):
        if execute_success:
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            responses = "\n".join(self.response_history)
            self.round_history.append(
                f"[Response History]\n{responses}\n[Executed Action]\n{parsed_plan}"
            )
        else:
            self.failed_plans.append(
                parsed_plan
            )
        return

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = []
        self.response_history = []