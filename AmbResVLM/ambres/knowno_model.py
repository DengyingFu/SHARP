"""
Toyota Motor Europe NV/SA and its affiliates retain all intellectual property and proprietary rights in and to this software, 
related documentation and any modifications thereto. Any use, reproduction, disclosure or distribution of this software and 
related documentation without an express license agreement from Toyota Motor Europe NV/SA is strictly prohibited.
"""

"""Adapted from https://github.com/google-research/google-research/tree/master/language_model_uncertainty"""

import copy
import signal
import random
import numpy as np
from openai import OpenAI

client = OpenAI()

PROMPT_REAL = [
    {
        "role": "system",
        "content": "You are a robot operating in an office kitchen. You are in front of a counter. Generate four options for the following task, listing them with A, B, C and D.",
    },
    {
        "role": "user",
        "content": "On the counter, there is an orange soda, a Pepsi, and an apple. Put that drink in the top drawer.",
    },
    {
        "role": "assistant",
        "content": """
A) open the top drawer and put the orange soda in it
B) open the bottom drawer and put the Pepsi in it
C) open the bottom drawer and put the orange soda in it
D) open the top drawer and put the Pepsi in it
""",
    },
    {
        "role": "user",
        "content": "On the counter, there is an energy bar, a banana, and a microwave. Put the snack next to the microwave.",
    },
    {
        "role": "assistant",
        "content": """
A) pick up the energy bar and put it next to the microwave
B) pick up the banana and put it next to the energy bar
C) pick up the banana and put it next to the microwave
D) pick up the energy bar and put it next to the banana
""",
    },
    {
        "role": "user",
        "content": "On the counter, there is a Coke, a Sprite, and a sponge. Can you dispose of the can? It should have expired.",
    },
    {
        "role": "assistant",
        "content": """
A) pick up the sponge and put it in the landfill bin
B) pick up the Coke and put it in the recycling bin
C) pick up the Sprite and put it in the recycling bin
D) pick up the Coke and put it in the landfill bin
""",
    },
    {
        "role": "user",
        "content": "On the counter, there is a bottled water, a bag of jalapeno chips, and a bag of rice chips. I would like a bag of chips.",
    },
    {
        "role": "assistant",
        "content": """
A) pick up the bottled water
B) pick up the jalapeno chips
C) pick up the kettle chips
D) pick up the rice chips
""",
    },
    {
        "role": "user",
        "content": "On the counter, there is {scene_objects}. {task}",
    },
]

PROMPT_SIM = [
    {
        "role": "system",
        "content": "You are a robot, and you are asked to move objects to precise locations on the table. Our instructions can be ambiguous. Generate four options for the following task, listing them with A, B, C and D.",
    },
    {
        "role": "user",
        "content": "On the table there are these objects: yellow block, red block, orange block, red bowl, orange bowl, gray bowl. Place the yellow block in the red bowl.",
    },
    {
        "role": "assistant",
        "content": """
A) put the red block in the orange bowl
B) put the yellow block in the red bowl
C) put the orange block in the gray bowl
D) put the yellow block in the blue bowl
""",
    },
    {
        "role": "user",
        "content": "On the table there are these objects: yellow block, red block, orange block, red bowl, orange bowl, gray bowl. Move the red block into the bowl.",
    },
    {
        "role": "assistant",
        "content": """
A) put the red block in the orange bowl
B) put yellow block in the red bowl
C) put the red block in the gray bowl
D) put the red block in the red bowl
""",
    },
    {
        "role": "user",
        "content": "On the table there are these objects: yellow block, red block, orange block, red bowl, orange bowl, gray bowl. Place the red block in the bowl.",
    },
    {
        "role": "assistant",
        "content": """
A) put the yellow block in the orange bowl
B) put the red block in the red bowl
C) put the red block in the orange bowl
D) put the red block in the gray bowl
""",
    },
    {
        "role": "user",
        "content": "On the table there are these objects: {scene_objects}. {task}",
    },
]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# OpenAI only supports up to five tokens (logprobs argument) for getting the likelihood.
# Thus we use the logit_bias argument to force LLM only consdering the five option
# tokens: A, B, C, D, E
def lm(
    messages,
    max_tokens=256,
    temperature=0,
    logprobs=False,
    stop_seq=None,
    logit_bias={
        317: 100.0,  #  A (with space at front)
        347: 100.0,  #  B (with space at front)
        327: 100.0,  #  C (with space at front)
        360: 100.0,  #  D (with space at front)
        412: 100.0,  #  E (with space at front)
    },
    timeout_seconds=20,
):
    max_attempts = 5
    for _ in range(max_attempts):
        try:
            with timeout(seconds=timeout_seconds):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=max_tokens,
                    temperature=temperature,
                    logprobs=logprobs,
                    logit_bias=logit_bias,
                    top_logprobs=20 if logprobs else None,
                    stop=list(stop_seq) if stop_seq is not None else None,
                    messages=messages,
                )
            break
        except TimeoutError:
            print("Timeout, retrying...")
            pass
    return response, response.choices[0].message.content.strip()


def process_mc_raw(mc_raw, add_mc="an option not listed here"):
    mc_all = mc_raw.split("\n")

    mc_processed_all = []
    for mc in mc_all:
        mc = mc.strip()

        # skip nonsense
        if len(mc) < 5 or mc[0] not in ["a", "b", "c", "d", "A", "B", "C", "D", "1", "2", "3", "4"]:
            continue
        mc = mc[2:]  # remove a), b), ...
        mc = mc.strip().lower().split(".")[0]
        mc_processed_all.append(mc)
    if len(mc_processed_all) < 4:
        raise "Cannot extract four options from the raw output."

    # Check if any repeated option - use do nothing as substitue
    mc_processed_all = list(set(mc_processed_all))
    if len(mc_processed_all) < 4:
        num_need = 4 - len(mc_processed_all)
        for _ in range(num_need):
            mc_processed_all.append("do nothing")
    prefix_all = ["A) ", "B) ", "C) ", "D) "]
    if add_mc is not None:
        mc_processed_all.append(add_mc)
        prefix_all.append("E) ")
    random.shuffle(mc_processed_all)

    # get full string
    mc_prompt = ""
    for mc_ind, (prefix, mc) in enumerate(zip(prefix_all, mc_processed_all)):
        mc_prompt += prefix + mc
        if mc_ind < len(mc_processed_all) - 1:
            mc_prompt += "\n"
    add_mc_prefix = prefix_all[mc_processed_all.index(add_mc)][0]
    return mc_prompt, mc_processed_all, add_mc_prefix


def get_top_logprobs(response):
    all_token_probs = response.choices[0].logprobs.content[0].top_logprobs
    top_logprobs_full = {}
    valid_tokens = ["A", "B", "C", "D", "E"]
    for i in range(len(all_token_probs)):
        token = all_token_probs[i].token.strip()
        if token in valid_tokens and token not in top_logprobs_full:
            top_logprobs_full[token] = all_token_probs[i].logprob
    return top_logprobs_full


def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx


class KnowNo:
    def __init__(self, env: str = "real", verbose=False):
        assert env in ["sim", "real"]
        if env == "real":
            self.base_prompt = PROMPT_REAL
            self.sys_prompt = (
                "You are a robot operating in an office kitchen. You are in front of a counter."
            )
        elif env == "sim":
            self.base_prompt = PROMPT_SIM
            self.sys_prompt = "You are a robot, and you are asked to move objects to precise locations on the table."
        self.verbose = verbose
        self.qhat = 0.928
        return

    def predict_ambiguity(self, task_description: str, obj_list: list[str]) -> bool:
        instruction = task_description
        if not instruction.endswith("."):
            instruction += "."
        scene_objects = ", ".join(obj_list)
        demo_mc_gen_messages = copy.copy(self.base_prompt)
        demo_mc_gen_messages[-1]["content"] = demo_mc_gen_messages[-1]["content"].replace(
            "{task}", instruction
        )
        demo_mc_gen_messages[-1]["content"] = demo_mc_gen_messages[-1]["content"].replace(
            "{scene_objects}", scene_objects
        )

        # Generate multiple choices
        _, demo_mc_gen_raw = lm(demo_mc_gen_messages, logit_bias={})
        demo_mc_gen_raw = demo_mc_gen_raw.strip()
        demo_mc_gen_full, demo_mc_gen_all, demo_add_mc_prefix = process_mc_raw(demo_mc_gen_raw)
        if self.verbose:
            print("====== Prompt for generating possible options ======")
            for msg in demo_mc_gen_messages:
                print(msg["content"])
            print("====== Generated options ======")
            print(demo_mc_gen_full)

        # get the part of the current scenario from the previous prompt
        demo_cur_scenario_prompt = demo_mc_gen_messages[-1]

        # get new prompt
        messages = [
            {
                "role": "system",
                "content": self.sys_prompt,
            },
            {
                "role": "user",
                "content": demo_cur_scenario_prompt["content"]
                + "\n"
                + demo_mc_gen_full
                + "\nWhich option is correct? Answer with a single letter.",
            },
        ]

        # scoring
        mc_score_response, _ = lm(messages, max_tokens=1, logprobs=True)
        top_logprobs_full = get_top_logprobs(mc_score_response)
        top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        top_logprobs = [value for value in top_logprobs_full.values()]
        if self.verbose:
            print("====== Prompt for scoring options ======")
            for msg in messages:
                print(msg["content"])
            print("\n====== Raw log probabilities for each option ======")
            for token, logprob in zip(top_tokens, top_logprobs):
                print("Option:", token, "\t", "log prob:", logprob)

        # include all options with score >= 1-qhat
        mc_smx_all = temperature_scaling(top_logprobs, temperature=5)
        prediction_set = [
            token
            for token_ind, token in enumerate(top_tokens)
            if mc_smx_all[token_ind] >= 1 - self.qhat
        ]
        if self.verbose:
            print("Softmax scores:", mc_smx_all)
            print("Prediction set:", prediction_set)
            if len(prediction_set) != 1:
                print("Help needed!")
            else:
                print("No help needed!")

        return len(prediction_set) != 1
