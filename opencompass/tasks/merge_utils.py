def pattern_in(text, pattern):
    patterns = pattern.split(".")
    texts = text.split(".")
    for i in range(len(texts)):
        for j in range(len(patterns)):
            if patterns[j] == "*":
                continue
            elif "[" in patterns[j] and "]" in patterns[j]:
                tmp_pattern = patterns[j][1:-1].split("-")
                int_range = list(range(int(tmp_pattern[0]), int(tmp_pattern[1]) + 1))
                str_range = [str(x) for x in int_range]
                if texts[i + j] in str_range:
                    continue
                else:
                    break
            else:
                if texts[i + j] == patterns[j]:
                    continue
                else:
                    break
        else:
            return True
    return False

def get_cur_id(key, prefix):
    name_list = key.split(".")
    for i in range(len(name_list)):
        if name_list[i] == prefix:
            return int(name_list[i + 1])
        
def replace_cur_id(key, cur_id, new_id, prefix):
    name_list = key.split(".")
    new_key = ""
    for i in range(len(name_list)):
        if name_list[i - 1] == prefix:
            assert name_list[i] == str(cur_id)
            new_key += str(new_id)
        else:
            new_key += name_list[i]
        if i!=len(name_list)-1:
            new_key += "."
    return new_key