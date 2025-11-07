import re
from utils.math_grader import grade_answer
from utils.math_normalize import normalize_answer


def check_phrase(string, phrase):
    if phrase in string:
        return string.split(phrase)[-1]
    else:
        return string


def check_pattern(string, pattern):
    match = re.search(pattern, string)
    if match:
        return match.group(1)
    else:
        return string
    

def parse_groundTruth(val, dataset):
    def parse_groundTruth_gsm8k(val):
        return float(val.split('####')[-1].strip(" ").replace(',', ''))

    def parse_groundTruth_math500(val):
        return normalize_answer(val)

    def parse_groundTruth_mmlu(val):
        return val

    def parse_groundTruth_aime2024(val):
        return val
    
    def parse_groundTruth_aqua(val):
        if val == 'A':
            return 0
        elif val == 'B':
            return 1
        elif val == 'C':
            return 2
        elif val == 'D':
            return 3
        elif val == 'E':
            return 4
    def parse_groundTruth_svamp(val):
        return float(val)

    if dataset == 'openai/gsm8k':
        return parse_groundTruth_gsm8k(val)
    elif dataset == 'HuggingFaceH4/MATH-500':
        return parse_groundTruth_math500(val)
    elif 'cais/mmlu' in dataset:
        return parse_groundTruth_mmlu(val)
    elif 'Maxwell-Jia/AIME_2024' == dataset:
        return parse_groundTruth_aime2024(val)
    elif dataset == 'deepmind/aqua_rat':
        return parse_groundTruth_aqua(val)
    elif dataset == 'deepmind/aqua_rat':
        return parse_groundTruth_aqua(val)
    elif dataset == 'ChilleD/SVAMP':
        return parse_groundTruth_svamp(val)
    else:
        raise KeyError()


def parse_response(val, dataset, **kwargs):
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
    
    # TODO Support generator format?
    def parse_response_gsm8k(string):
        string_tmp = string
        string_tmp = check_phrase(string_tmp, 'boxed{')
        string_tmp = check_phrase(string_tmp, 'The answer is')
        string_tmp = check_phrase(string_tmp, 'The final answer is')
        string_tmp = check_phrase(string_tmp, 'the final answer is')
        string_tmp = check_phrase(string_tmp, 'Therefore, the final answer')

        pos = string_tmp.find(r"\boxed{")
        if pos != -1:
            string_tmp = string_tmp[pos:]
            string_tmp = extract_boxed(string_tmp)

        pattern = r"[-+]?\d*\.?\d+"

        match = re.search(pattern, string_tmp)
        if match:
            return float(match.group())
        else:
            return None        

    def parse_response_aime2024(string):
        string_tmp = string
        string_tmp = check_phrase(string_tmp, 'boxed{')
        string_tmp = check_phrase(string_tmp, 'The answer is')
        string_tmp = check_phrase(string_tmp, 'The final answer is')
        string_tmp = check_phrase(string_tmp, 'the final answer is')
        string_tmp = check_phrase(string_tmp, 'Therefore, the final answer')

        pos = string_tmp.find(r"\boxed{")
        if pos != -1:
            string_tmp = string_tmp[pos:]
            string_tmp = extract_boxed(string_tmp)

        pattern = r"[-+]?\d*\.?\d+"

        match = re.search(pattern, string_tmp)
        if match:
            return float(match.group())
        else:
            return None        

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

        try:
            return float(string_tmp)
        except ValueError as err:
            options = kwargs.get('options', [])
            if kwargs.get('generator', False) and ('qwen' in kwargs.get('generator').lower()):
                # 1) \boxed{} 내부에서 불필요한 문구 제거
                string_tmp_qwen = re.sub(r'(?i)\b(answer(\s+is)?|the\s+answer(\s+is)?)\b', '', string_tmp).strip()

                # Qwen: {OptionNumber} {OptionText} 형태
                match = re.match(r"^\s*(\d+)\s+(.*)", string_tmp_qwen)
                if match:
                    option_num = int(match.group(1))  # 숫자 추출
                    if 0 <= option_num - 1 < len(options):  # 인덱스는 0부터 시작하므로 -1
                        return option_num - 1
                
                # 2) 텍스트만 있는 경우 → 옵션 리스트 매칭
                for i, option in enumerate(options):
                    if string_tmp_qwen.strip().lower() == option.strip().lower():
                        return i
                
                return None            
            
            # 기존 방식: 문자열 매칭
            matched_index = None
            for i, option in enumerate(options):
                if string_tmp.strip().lower() == option.strip().lower():
                    matched_index = i
                    break
            return matched_index            
    def parse_response_aqua(val):
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
        return string_tmp
        # try:
        #     return float(string_tmp)
        # except ValueError as err:
        #     options = kwargs.get('options', [])
        #     if kwargs.get('generator', False) and ('qwen' in kwargs.get('generator').lower()):
        #         # 1) \boxed{} 내부에서 불필요한 문구 제거
        #         string_tmp_qwen = re.sub(r'(?i)\b(answer(\s+is)?|the\s+answer(\s+is)?)\b', '', string_tmp).strip()

        #         # Qwen: {OptionNumber} {OptionText} 형태
        #         match = re.match(r"^\s*(\d+)\s+(.*)", string_tmp_qwen)
        #         if match:
        #             option_num = int(match.group(1))  # 숫자 추출
        #             if 0 <= option_num - 1 < len(options):  # 인덱스는 0부터 시작하므로 -1
        #                 return option_num - 1
                
        #         # 2) 텍스트만 있는 경우 → 옵션 리스트 매칭
        #         for i, option in enumerate(options):
        #             if string_tmp_qwen.strip().lower() == option.strip().lower():
        #                 return i
                
        #         return None            
            
        #     # 기존 방식: 문자열 매칭
        #     matched_index = None
        #     for i, option in enumerate(options):
        #         if string_tmp.strip().lower() == option.strip().lower():
        #             matched_index = i
        #             break
        #     return matched_index            

    def parse_response_svamp(string):
        string_tmp = string
        string_tmp = check_phrase(string_tmp, 'boxed{')
        string_tmp = check_phrase(string_tmp, 'The answer is')
        string_tmp = check_phrase(string_tmp, 'The final answer is')
        string_tmp = check_phrase(string_tmp, 'the final answer is')
        string_tmp = check_phrase(string_tmp, 'Therefore, the final answer')

        pos = string_tmp.find(r"\boxed{")
        if pos != -1:
            string_tmp = string_tmp[pos:]
            string_tmp = extract_boxed(string_tmp)

        pattern = r"[-+]?\d*\.?\d+"

        match = re.search(pattern, string_tmp)
        if match:
            return float(match.group())
        else:
            return None     

    if dataset == 'openai/gsm8k':
        return parse_response_gsm8k(val)
    elif dataset == 'Maxwell-Jia/AIME_2024':
        return parse_response_aime2024(val)
    elif dataset == 'HuggingFaceH4/MATH-500':
        return parse_response_math500(val)
    elif 'cais/mmlu' in dataset:
        return parse_response_mmlu(val)
    elif dataset == 'deepmind/aqua_rat':
        return parse_response_aqua(val)
    elif dataset == 'ChilleD/SVAMP':
        return parse_response_svamp(val)
    else:
        raise KeyError()


def get_score(pred, true, dataset):
    if dataset == 'openai/gsm8k':
        return pred == true
    elif dataset == 'Maxwell-Jia/AIME_2024':
        return pred == true
    # elif dataset == 'HuggingFaceH4/MATH-500':
    #     return grade_answer(pred, true)
    elif dataset == 'HuggingFaceH4/MATH-500':
        guess = grade_answer(pred, true)
        true = normalize_interval(true)
        unit_words = [
            r'\mbox{inches}^2', r'\mbox{cm}^2',
            ]
        for unit_word in unit_words:
            true = true.replace(unit_word, '')
        
        pred = normalize_interval(pred)
        pred = pred.replace(r'-2\lex\le7', '[-2,7]')
        guess = grade_answer(pred, true)
        return guess
    elif 'cais/mmlu' in dataset:
        return pred == true
    elif dataset == 'deepmind/aqua_rat':
        # return grade_answer(pred, true)
        return pred == true
    elif dataset == 'ChilleD/SVAMP':
        return pred == true
    else:
        raise KeyError()


def normalize_interval(s: str) -> str:
    # 공백 제거
    s = s.replace(r"x\in", "")
    # # "x∈" 같은 변수 포함 표현 제거
    # s = re.sub(r'x\in', '', s)
    return s