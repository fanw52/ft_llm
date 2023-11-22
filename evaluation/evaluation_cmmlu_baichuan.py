import argparse
import json
import os

from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)



def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, required=True, help="model name or path"
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="val", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="ceval_output", help="output directory"
    )

    parser.add_argument("--load_in_8bit",type=bool,default=False)

    return parser.parse_args()


class CEval:

    DATA_PATH = "haonan-li/cmmlu"
    TASK2DESC = ['agronomy', 'anatomy', 'ancient_chinese', 'arts', 'astronomy', 'business_ethics',
                 'chinese_civil_service_exam', 'chinese_driving_rule', 'chinese_food_culture', 'chinese_foreign_policy',
                 'chinese_history', 'chinese_literature',
                 'chinese_teacher_qualification', 'clinical_knowledge', 'college_actuarial_science',
                 'college_education', 'college_engineering_hydrology', 'college_law', 'college_mathematics',
                 'college_medical_statistics', 'college_medicine', 'computer_science',
                 'computer_security', 'conceptual_physics', 'construction_project_management', 'economics', 'education',
                 'electrical_engineering', 'elementary_chinese', 'elementary_commonsense',
                 'elementary_information_and_technology', 'elementary_mathematics',
                 'ethnology', 'food_science', 'genetics', 'global_facts', 'high_school_biology',
                 'high_school_chemistry', 'high_school_geography', 'high_school_mathematics', 'high_school_physics',
                 'high_school_politics', 'human_sexuality',
                 'international_law', 'journalism', 'jurisprudence', 'legal_and_moral_basis', 'logical',
                 'machine_learning', 'management', 'marketing', 'marxist_theory', 'modern_chinese', 'nutrition',
                 'philosophy', 'professional_accounting', 'professional_law',
                 'professional_medicine', 'professional_psychology', 'public_relations', 'security_study', 'sociology',
                 'sports_science', 'traditional_chinese_medicine', 'virology', 'world_history', 'world_religions']

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        output_dir: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

    def run(self, shot: int, split: str):
        results, accs = {}, {}

        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"save result to {result_path}")

        # results
        acc_path = os.path.join(self.output_dir, "acc.json")
        with open(acc_path, "w") as f:
            json.dump(accs, f, indent=2)
        average_acc = sum(accs.values()) / len(accs)
        print(f"average acc: {average_acc}")

    def run_single_task(self, task_name: str, shot: int, split: str):
        dataset = load_dataset(self.DATA_PATH, task_name)
        results = []
        acc = 0
        for data in tqdm(dataset[split]):
            prompt = f"以下是选择题，请选出其中的正确答案。\n"
            if shot != 0:
                shuffled = dataset["dev"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

            logits = self.model(
                    input_ids=input_ids,
                ).logits[:,-1].flatten()

            candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in ["A", "B", "C", "D"]]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]

            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["Answer"].strip().upper(),
                    "Answer": answer,
                }
            )
            acc += answer == data["Answer"].strip().upper()
        acc /= len(dataset[split])
        return results, acc

    def build_example(self, data, with_answer: bool = True):
        question = data["Question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["Answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"


def main():
    args = parse_argument()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        load_in_8bit = args.load_in_8bit,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        add_bos_token=False,
        add_eos_token=False,
        padding_side="left",
    )
    ceval = CEval(model, tokenizer, args.output_dir)
    ceval.run(args.shot, args.split)


if __name__ == "__main__":
    main()