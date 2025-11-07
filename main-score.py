import re
import math
import os, json
import numpy as np
import unicodedata
import pandas as pd
from itertools import product
from collections import defaultdict

from datasets import load_dataset

from utils.eval import check_pattern
from utils.math_normalize import normalize_answer
from utils.math_grader import grade_answer

# =========================
# Utils
# =========================
def extract_boxed(s: str):
    start = s.find(r"\boxed{")
    if start == -1:
        return None
    i = start + len(r"\boxed{")
    depth = 1
    content = []
    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0:
                break
        content.append(s[i])
        i += 1
    return ''.join(content) 

def parse_response_mmlu(val):    
    string_tmp = val.split('<|eot_id|>')[0]
    string_tmp = string_tmp.strip('\n\t ')    
    string_tmp = string_tmp.split('answer is')[-1]
    string_tmp = check_pattern(string_tmp, r'\$\\boxed\{(.*?)\}\$')
    if 'text{' in string_tmp:
        string_tmp = check_pattern(string_tmp, r'\$\\text\{(.*?)\}\$')
    pos = string_tmp.find(r"\boxed{")
    if pos != -1:
        string_tmp = string_tmp[pos:]
        string_tmp = extract_boxed(string_tmp)

    string_tmp = string_tmp.strip('$')
    try:
        return float(string_tmp)
    except ValueError:
        # return 'PARSE_ERROR'
        return string_tmp

def parse_response_math500(val):
    # print(f'val:{val}')

    string_tmp = val.split('<|eot_id|>')[0]
    # print(f'string_tmp:{string_tmp}')

    string_tmp = string_tmp.strip('\n\t ')    
    # print(f'string_tmp:{string_tmp}')

    string_tmp = string_tmp.split('answer is')[-1]
    # print(f'string_tmp:{string_tmp}')

    string_tmp = check_pattern(string_tmp, r'\$\\boxed\{(.*?)\}\$')
    # print(f'string_tmp:{string_tmp}')

    # NOTE : 0910 기준
    # pos = string_tmp.find("$\\boxed{")
    pos = string_tmp.find(r"\boxed{")
    if pos != -1:
        string_tmp = string_tmp[pos:]
        string_tmp = extract_boxed(string_tmp)
    
    if "π" in string_tmp:
        string_tmp = string_tmp.replace("π", "\\pi")
    string_tmp = normalize_answer(string_tmp)
    # print(f'string_tmp:{string_tmp}')
    
    return string_tmp

def transform_sequence(seq, transform_type="cumsum"):
    def volatilityRescale(_vals: list):
        if len(_vals) <= 1:
            return _vals
        _val_arr = np.array(_vals)
        _val_arr = (_val_arr) / (np.std(_val_arr, ddof=1) + 1e-6)
        return _val_arr.tolist()
    if transform_type == "cumsum":
        return np.cumsum(seq, axis=1)
    elif transform_type == 'volatilityRescale':
        return list(map(volatilityRescale, seq))
    else:
        return seq

def aggregate(step_scores, method="last"):
    if isinstance(step_scores, float) and step_scores is not None:
        return step_scores
    if len(step_scores) == 0:
        return 0.0
    if method == "mean":
        return sum(step_scores) / len(step_scores)
    elif method == "sum":
        return sum(step_scores)
    elif method == "max":
        return max(step_scores)
    elif method == "min":
        return min(step_scores)
    elif method == "last":
        return step_scores[-1]
    elif method == "first":
        return step_scores[0]
    elif method == "prod":
        return math.prod(step_scores)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")


# =========================
# Load results
# =========================
def _collapse_qid_payload(data: dict):
    """MCTS 결과를 위해 신형 포맷(숫자 키: '0','1',...)을 qid 단위로 병합한다."""
    # 숫자 문자열 키만 골라 정렬
    num_keys = [k for k in data.keys() if isinstance(k, str) and k.isdigit()]
    if not num_keys:
        return None
    num_keys.sort(key=lambda x: int(x))
    collapsed = {
        "outputs": [],
        "aggregate_scores": [],
        "step_scores": []
    }

    for k in num_keys:
        items = data.get(k, {}) or {}

        for item in items:
            # aggregate_scores: 리스트면 이어붙이고, 단일값이면 append
            collapsed["outputs"].extend(item.get("outputs", []))
            collapsed["aggregate_scores"].extend(item.get("aggregate_scores", []))
            collapsed["step_scores"].extend(item.get("step_scores", []))

    return collapsed



def load_results_for_method(method, base_dir="experiments-mmlupro"):
    """단일 method 에 대한 qid별 결과 로드"""
    method_dir = os.path.join(base_dir, method)
    err_case = []
    results = {}

    method_qids = {
        'math500' : list(range(500)),
    }
    target_qids = method_qids[dataset]
    for fname in os.listdir(method_dir):
        if fname.startswith("qid") and fname.endswith(".json"):
            qid = fname.replace("qid", "").replace(".json", "")
            try:
                with open(os.path.join(method_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
            except json.decoder.JSONDecodeError as err:
                print(f'Error occurs at {fname}')
                err_case.append(int(fname.split('.')[0][3:]))

            # MCTS 용 처리
            if 'MCTS' in method:
                collapsed = _collapse_qid_payload(data)
                results[qid] = collapsed
                target_qids.pop(target_qids.index(int(qid)))
                continue
            else:
                if qid not in data.keys():
                    continue
                results[qid] = data[qid]
                target_qids.pop(target_qids.index(int(qid)))
    if err_case:
        from pprint import pprint
        pprint(sorted(err_case))

    if len(target_qids) != 0:
        print(f'Cannot find\n\t{target_qids}')
    return results


# =========================
# Single-method Evaluation
# =========================
def evaluate_method_single(llm, prm, core_method, results_dir="experiments-mmlupro", split="test", 
                           selection="BoN", aggregation="last", transform_type=None, metric="acc"):
    """
    단일 (llm, prm, core_method) 조합에 대해 평가
    """
    global max_outputs_len

    response_sep = 'assistant<|end_header_id|>'

    method_name = f"{llm}-{prm}-{core_method}"  # 실제 폴더명
    # dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    if results_dir.split('-')[-1] == 'mmlupro':
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
    elif results_dir.split('-')[-1] == 'math500':
        dataset = load_dataset("HuggingFaceH4/MATH-500")[split]
    results = load_results_for_method(method_name, results_dir)

    records = []
    LETTER2IDX = {c:i for i,c in enumerate("ABCDEFGHIJ")}
    IDX2LETTER = {i:c for c,i in LETTER2IDX.items()}

    for example_idx, example in enumerate(dataset):
        if results_dir.split('-')[-1] == 'mmlupro':
            domain = example['category']
            qid = str(example["question_id"])
            question = example["question"]
        elif results_dir.split('-')[-1] == 'math500':
            domain = None
            qid = str(example_idx)
            question = example["problem"]
        else:
            raise KeyError()

        if results_dir.split('-')[-1] == 'mmlupro':
            
            answer_index_str = example["answer"]
            answer_index = LETTER2IDX[answer_index_str]

            answer_option = example["options"][answer_index]
        elif results_dir.split('-')[-1] == 'math500':
            answer_index = None
            answer_option = example['answer']
        else:
            raise KeyError()
        
        if qid in results:
            if ('GS' in core_method) or ('Genetic' in core_method) or ('SC' in core_method):
                outputs_key = 'answer'
                outputs = results[qid].get(outputs_key, [])

                if ('Genetic' in core_method):
                    output_ids = [i for i, cast_type in enumerate(results[qid]['CaseType']) if cast_type in ["Candidate", "Candidate-Mutation"]]
                    outputs = [outputs[idx] for idx in output_ids]
            else:
                outputs_key = 'outputs'
                outputs = results[qid].get(outputs_key, [])

            if len(outputs) > max_outputs_len:
                max_outputs_len = len(outputs)
            if (len(outputs) > budget_limit) and (budget_limit != -1):
                outputs = outputs[:budget_limit]
            step_scores = results[qid].get("step_scores", [])
            step_scores_transformed = transform_sequence(step_scores, transform_type=transform_type)
            
            # ===== pass@L 모드: outputs 중 하나라도 정답이면 correct=1 =====
            try:
                if metric.lower() == "pass@l":
                    if outputs:
                        any_correct = False
                        for out in outputs:
                            resp = out.split(response_sep)[-1]
                            if results_dir.split('-')[-1] == 'mmlupro':
                                parsed = parse_response_mmlu(resp)
                                # 정답 후보(문자 인덱스 또는 옵션 텍스트)를 모두 허용
                                answer_candidates = {answer_index, answer_option}
                                if parsed in answer_candidates:
                                    any_correct = True
                                    break
                            elif results_dir.split('-')[-1] == 'math500':
                                parsed = parse_response_math500(resp)
                                if grade_answer(parsed, normalize_answer(answer_option)):
                                    any_correct = True
                                    break
                            else:
                                raise KeyError()
                        correct = any_correct
                        # pred 표기는 상태로 표시
                        pred = "AnyCorrect" if any_correct else "AllWrong"
                    else:
                        pred, correct = "Incomplete", False

                # ===== 기존 acc 모드: selection/aggregation 사용해 단일 pred =====
                else:
                    if outputs:
                        if selection == "BoN":
                            agg_scores = [aggregate(ss, method=aggregation) for ss in step_scores_transformed]
                            best_idx = max(range(len(agg_scores)), key=lambda i: agg_scores[i])
                            if results_dir.split('-')[-1] == 'mmlupro':
                                pred = parse_response_mmlu(outputs[best_idx].split(response_sep)[-1])
                            elif results_dir.split('-')[-1] == 'math500':
                                pred = parse_response_math500(outputs[best_idx].split(response_sep)[-1])
                            else:
                                raise KeyError()
                        elif selection == "MV":
                            counts = defaultdict(int)
                            for out in outputs:
                                if results_dir.split('-')[-1] == 'mmlupro':
                                    parsed = parse_response_mmlu(out.split(response_sep)[-1])
                                elif results_dir.split('-')[-1] == 'math500':
                                    parsed = parse_response_math500(out.split(response_sep)[-1])
                                else:
                                    raise KeyError()

                                counts[parsed] += 1
                            pred = max(counts, key=counts.get)

                        elif selection == "WMV":
                            weights = defaultdict(float)
                            for i, out in enumerate(outputs):
                                if results_dir.split('-')[-1] == 'mmlupro':
                                    parsed = parse_response_mmlu(out.split(response_sep)[-1])
                                elif results_dir.split('-')[-1] == 'math500':
                                    parsed = parse_response_math500(out.split(response_sep)[-1])
                                else:
                                    raise KeyError()

                                weights[parsed] += aggregate(step_scores_transformed[i], method=aggregation)
                            pred = max(weights, key=weights.get)
                        else:
                            raise ValueError(f"Unsupported selection method: {selection}")

                        if results_dir.split('-')[-1] == 'mmlupro':
                            answer_candidates = [answer_index, answer_option]
                            correct = pred in answer_candidates

                        elif results_dir.split('-')[-1] == 'math500':
                            correct = grade_answer(pred, normalize_answer(answer_option))

                    else:
                        pred, correct = None, False
            except:
                print('Error', core_method, qid)
                for _i in step_scores_transformed:
                    print(_i)
                raise AssertionError()

            records.append({
                "llm": llm,
                "prm": prm,
                "core_method": core_method,
                "method": method_name,
                "domain": domain,
                "selection": selection,
                "aggregation": aggregation,
                "qid": qid,
                "answer_index": answer_index,
                "pred": 'Success' if pred is not None else "Incomplete",
                "correct": correct
            })
        else:
            records.append({
                "llm": llm,
                "prm": prm,
                "core_method": core_method,
                "method": method_name,
                "domain": domain,
                "selection": selection,
                "aggregation": aggregation,
                "qid": qid,
                "answer_index": answer_index,
                "pred": "NOTYET",
                "correct": False,
            })            
    df = pd.DataFrame(records)
    acc = df["correct"].mean()
    return acc, df



# =========================
# Multi-method pipeline
# =========================
if __name__ == '__main__':

    seed = 12389

    dataset = 'math500'

    exp = 'BS'

    llm_models = [
        'meta-llama_Llama-3.1-8B-Instruct',
    ]

    prm_models = [
        'UW-Madison-Lee-Lab_VersaPRM',
    ]
    
    selection_methods = [
        'WMV',
    ]

    model_to_pretty_name = {
        'meta-llama_Llama-3.1-8B-Instruct' : 'Llama3-8B',
        'meta-llama_Llama-3.2-3B-Instruct' : 'Llama3-3B',
        'UW-Madison-Lee-Lab_VersaPRM' : 'VersaPRM',
        'UW-Madison-Lee-Lab_Qwen-PRM800K': 'QwenPRM',
        'RLHFlow_Llama3.1-8B-PRM-Deepseek-Data' : 'RLHFlow',
    }

    method_cfgs_chain = [
        ('chain', "SC_L10", 'min', None), 
    ]

    for llm_model, prm_model, selection_method in product(llm_models, prm_models, selection_methods):
        print(f'Processing {(llm_model, prm_model, selection_method)}')
        max_outputs_len = -1
        all_dfs = []
        acc_dict = {}
        for budget_limit in [2, 4, 8, 10]:
            # for metric in ['acc', 'Pass@L']:
            for metric in ['acc']:
                print('='*30, f'Budget (L) {budget_limit} - Metric {metric}', '='*30)
                for search_type, method, aggregation, transform_type in method_cfgs_chain:
                    try:
                        acc, df_method = evaluate_method_single(
                            llm=llm_model,
                            prm=prm_model,
                            core_method=method, 
                            selection=selection_method, 
                            aggregation=aggregation,
                            transform_type=transform_type,
                            results_dir=f"LangDec/experiments-{dataset}",
                            metric=metric,
                        )
                        df_method['Search Type'] = search_type
                        df_method['Budget(L)'] = budget_limit
                        df_method['Metric'] = metric
                        all_dfs.append(df_method)
                        acc_dict[method] = acc
                        print(f"[{method}] {metric}={acc*100:.1f}")
                    except FileNotFoundError as err:
                        print(f'{method} skipped for {err}')

        if len(all_dfs) == 0:
            continue
        df_all = pd.concat(all_dfs, ignore_index=True)
        out_file = f"{dataset}-seed{seed}-{model_to_pretty_name[llm_model]}-{model_to_pretty_name[prm_model]}-{selection_method}"
        df_all.to_excel(f'{out_file}.xlsx', index=False)
        df_all.to_csv(f'{out_file}.csv', index=False)

        df_stable = pd.read_csv(f'{out_file}-stable.csv')
        df_merge = pd.concat((df_stable, df_all), axis=0).reset_index(drop=True)
        df_merge.to_excel(f'{out_file}-merged.xlsx', index=False)
        df_merge.to_csv(f'{out_file}-merged.csv', index=False)


        print(f"저장 완료: {out_file}")
