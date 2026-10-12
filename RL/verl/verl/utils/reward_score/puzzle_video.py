import json
import re

def get_score(answer_text: str, ground_truth: str) -> float:
    answer_text = [int(x) for x in re.findall(r'\d+', answer_text)]
    ground_truth = json.loads(ground_truth)
    if len(answer_text) != len(ground_truth):
        return 0.0

    return 1.0 if answer_text==ground_truth else 0.0


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    is_format_error = False

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    try:
        acc_reward = get_score(answer_text, ground_truth)
    except:
        is_format_error = True
        acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    acc = 1.0 if acc_reward>0.99 else 0.0
    tool_reward_base = 1.0 if count_vision_1 else 0.0
    # tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = 1.0 if not is_format_error else 0.0
    score = 0.8 * acc_reward + 0.2 * format_reward
    # reward 1
    # return 0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base
    # reward 2
    return {'score': score,
            'acc': acc,
            'acc_reward': acc_reward,
            'format_reward': format_reward}

    # reward 2 
    # return 1.0 * acc_reward + 0.2 * format_reward + 1.0 * tool_reward + 0.2 * tool_reward_base
    # reward 3
    # tool_reward_alpha = 1.2 if count_vision_1 > 0 else 0.0
    # return 1.0 * acc_reward * tool_reward_alpha + 0.2 * format_reward
    # reward 4
    # extra_reward = tool_reward_base * (count_vision_1 - 1) * (1 - acc_reward)
    # return  0.8 * acc_reward + 0.2 * format_reward + 0.4 * tool_reward_base  + 0.2 * extra_reward


