from tqdm import tqdm
import os
from transformers import AutoTokenizer
from torch.utils.data.dataset import Dataset
import torch
import jsonlines
import numpy as np
import random
from datasets import load_dataset
import tree_sitter_cpp as tsc
from tree_sitter import Language, Parser




def tree_to_token_pos(root_node, lpos, node_token_pos_range):
    if (len(root_node.children) == 0 or root_node.type.find('string') != -1) and root_node.type != 'comment':
        node_token_pos_range[root_node] = (lpos, lpos + 1)
        return [{'range': (root_node.start_point, root_node.end_point), 'type':root_node.type}]
    else:
        sub_lpos = lpos
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_pos(child, sub_lpos, node_token_pos_range)
            sub_lpos = node_token_pos_range[child][1]
        node_token_pos_range[root_node] = (lpos, sub_lpos)
        return code_tokens


def pos_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def divide_code_snippets(root_node, node_token_pos_range, code_tokens, max_token_num):
    lpos, rpos = node_token_pos_range[root_node]
    if rpos - lpos <= max_token_num:
        return [code_tokens[lpos:rpos]]
    else:
        tokens = [[]]
        for child in root_node.children:
            new_tokens = divide_code_snippets(child, node_token_pos_range, code_tokens, max_token_num)
            remain_snippet_index = 0
            for snippet in new_tokens:
                if len(snippet) + len(tokens[-1]) <= max_token_num:
                    tokens[-1] = tokens[-1]  + snippet
                    remain_snippet_index += 1
                else:
                    break
            tokens += new_tokens[remain_snippet_index:]
        return tokens



#请在这里添加从这个流式数据中采集下方所需格式的数据, 以及对数据的处理, 包括限制其长度, 剪切成snippet, padding到128长度
#在这里接入tree-sitter
#未来可以在这里实现同时返回一个identifier的属性, 中间包含一个列表, 作为Identifier的mask
def GetDataFromStreaming(name, tokenizer):
    max_number=100
    #调个100方便测后面的东西
    if(name=="onlycpp"):
        CPP_LANGUAGE = Language("./build/my-languages.so","cpp")
        parser = Parser()
        parser.set_language(CPP_LANGUAGE)
        ds = load_dataset(
            "codeparrot/github-code", 
            split="train", 
            streaming=True, 
            trust_remote_code=True, 
            languages=["C++"], 
            filter_languages=True
        )
        keywords = ['if', 'else', 'while', 'for', 'primitive_type', 'using', 'break', 'continue', 'switch', 'case', 'default', 'return']
        input_ids = []
        special_words = [] 
        for step, src in enumerate(ds):
            if(step==max_number):
                break
            code = src['code']
            tree = parser.parse(
                bytes(code,"utf8")
            )
            root_node = tree.root_node

            node_token_pos_range = {}
            # 获取token对应的位置
            tokens_pos = tree_to_token_pos(root_node, 0, node_token_pos_range)
            # 获取代码行
            cpp_loc = code.split('\n')
            # 获取对应每个位置下的token
            init_code_tokens = [{'code': pos_to_code_token(x['range'], cpp_loc), 'type': x['type']} for x in tokens_pos]
            max_token_num = 128
            code_token_list = divide_code_snippets(root_node, node_token_pos_range, init_code_tokens, max_token_num)
            for code_tokens in code_token_list:
                # print(len(code_tokens))
                if (len(code_tokens) < 32):
                    continue
                input_id = []
                special_word = []  # [start, end)
                # print(len(code_tokens))
                # print(''.join([x['code'] for x in code_tokens]))
                # _ = input()
                if max([len(x['code']) for x in code_tokens], default=0) > 25:
                    continue
                for x in code_tokens:
                    index = tokenizer(x['code'], add_special_tokens=False).input_ids
                    add_keywords = (x['type'] == 'identifier' or x['type'] in keywords)
                    start_id_pos = None
                    end_id_pos = None
                    if add_keywords:
                        start_id_pos = len(input_id)
                    input_id += index
                    if add_keywords:
                        end_id_pos = len(input_id)
                        special_word.append((start_id_pos, end_id_pos))
                # if len(input_id) > 128:
                #     print("____WARNING____", len(index))
                if len(special_word) == 0:
                    continue
                input_ids.append(np.array(input_id))
                special_words.append(special_word)
        print(len(input_ids), len(special_words))
        return input_ids, special_words





def load_loop_pretrain_data(args, padding_mode, tokenizer, data_name = None):
    print("***** load " + data_name + " train src dataset*****")

    # path = os.path.join(args.data_path, data_name + '.npy')
    # input_id_list = np.load(path, allow_pickle=True)

    # # filter
    # input_id_list = np.array([input_id for input_id in input_id_list if np.count_nonzero(input_id) >= 30])

    if padding_mode == 'max_len':
        raise NotImplementedError
        dataset = Pre_dataset(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'conti_tgt':
        raise NotImplementedError
        print("using new pretrain method...")
        dataset = Pre_dataset_type2(input_id_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'mix_conti_tgt':
        # Mixture of unsupervised code generation and extended CPD
        print("using mixed two pretrain method...")
        input_id_list, special_word_list = GetDataFromStreaming(data_name, tokenizer)
        dataset = Pre_dataset_type_mix(input_id_list, special_word_list, tokenizer, mask_pro=args.mask_pro, maxlength=args.pre_max_len)
    elif padding_mode == 'block':
        print("padding block is under realization")
        pass
    else:
        return NotImplementedError

    print("example of src id lists: ", dataset[50][0])
    print("example of tgt id lists: ", dataset[50][1])
    print("total query dataset len :", len(dataset))

    return dataset
class Pre_dataset_type_mix(Dataset):
    def __init__(self, tgt_id, special_word_list, tokenizer, mask_pro=0.3, maxlength=128, mask_mode='random'):
        self.tgt_id = tgt_id
        self.special_word_list = special_word_list
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.mask_pro = mask_pro
        self.tgtpadlength = maxlength
        self.tgtmaxlength = int(maxlength * mask_pro) + 1
        self.mask_token_index = self.tokenizer.mask_token_id
        self.pad_token_index = self.tokenizer.pad_token_id
        self.all_special_token = self.tokenizer.all_special_ids
        self.double_dataset_len = self.__len__()
        self.task_order = torch.randperm(self.double_dataset_len)


    def __getitem__(self, index):
        index = self.task_order[index]
        src_input_ids = None
        tgt_input_ids = None
        task_type = None

        if index < len(self.tgt_id):
            # print(index)
            task_type = "CPD"
            special_words = self.special_word_list[index]
            tgt_example = self.tgt_id[index]
            
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",tgt_example.shape)
            
            start_pos = 0
            if(len(tgt_example)>self.maxlength):
                start_pos=np.random.randint(low=0,high=len(tgt_example)-self.maxlength+1)
                tgt_example=tgt_example[start_pos:start_pos+self.maxlength]
            # src_input_ids = tgt_example.tolist()
            tgt_input_ids = (torch.from_numpy(tgt_example)).long()
            src_input_ids = tgt_input_ids.clone()
            # id_len = torch.nonzero(src_input_ids).shape[0]
            special_word_num = len(special_words)

            # mask_span_num = int((id_len * self.mask_pro) // self.span_size) + 1
            # mask_span_len = int(id_len * self.mask_pro)
            mask_num = int(special_word_num * self.mask_pro)
            # print("mask_span_num:", mask_span_num)
            mask_pos = list(range(special_word_num))
            random.shuffle(mask_pos)
            mask_index = []
            for pos in mask_pos:
                if special_words[pos][0] >= start_pos and special_words[pos][1] - start_pos < self.maxlength:
                    mask_index += list(range(special_words[pos][0] - start_pos, special_words[pos][1] - start_pos))
                    if len(mask_index) > mask_num:
                        break

            # tgt_input_ids = src_input_ids.tolist()[mask_index:mask_index+mask_span_len]
            tgt_input_ids = src_input_ids[mask_index].tolist()

            src_input_ids[mask_index] = self.mask_token_index

            # print("mask_index:", mask_index)
            # mask_span_len
            # mask_id_mask = torch.full(src_input_ids.shape, False, dtype=torch.bool)
            retain_id_mask = torch.full(src_input_ids.shape, True, dtype=torch.bool)
            # mask_id_mask[mask_index] = True

            # del_index = mask_index.tolist()
            # del_index = list(range(mask_index + 1, mask_index + mask_span_len)) #为什么这里起点+1？
            del_index = mask_index
            del_index = torch.from_numpy(np.array(del_index))
            retain_id_mask[del_index] = False
            # src_input_ids[mask_id_mask] = self.mask_token_index
            src_input_ids = src_input_ids[retain_id_mask].tolist()
            # print("src_input_ids1:", len(src_input_ids))
            src_input_ids = src_input_ids + [self.pad_token_index] * (self.maxlength - len(src_input_ids))
            # print("src_input_ids2:", len(src_input_ids))
            tgt_input_ids = tgt_input_ids + [self.pad_token_index] * (self.tgtpadlength - len(tgt_input_ids))

            src_input_ids = torch.from_numpy(np.array(src_input_ids)).long()
            tgt_input_ids = torch.from_numpy(np.array(tgt_input_ids)).long()

            src_input_ids = src_input_ids.unsqueeze(0)
            tgt_input_ids = tgt_input_ids.unsqueeze(0)

        else:
            index -= len(self.tgt_id)
            task_type = "unsupervised_generation"
            tgt_example = self.tgt_id[index]
            
            if(len(tgt_example)>self.maxlength):
                start_pos=np.random.randint(low=0,high=len(tgt_example)-self.maxlength+1)
                tgt_example=tgt_example[start_pos:start_pos+self.maxlength]
            
            # src_input_ids = tgt_example.tolist()
            tgt_input_ids = (torch.from_numpy(tgt_example)).long()
            tgt_input_ids = tgt_input_ids.tolist()

            src_input_ids = [self.pad_token_index] * self.maxlength
            tgt_input_ids = tgt_input_ids + [self.pad_token_index] * (self.tgtpadlength - len(tgt_input_ids))

            src_input_ids = torch.from_numpy(np.array(src_input_ids)).long()
            tgt_input_ids = torch.from_numpy(np.array(tgt_input_ids)).long()

            src_input_ids = src_input_ids.unsqueeze(0)
            tgt_input_ids = tgt_input_ids.unsqueeze(0)

        return src_input_ids, tgt_input_ids, task_type

    def __len__(self):
        return len(self.tgt_id) * 2

    @classmethod
    def get_collate_fn(cls):
        def fn(features):
            src_tensor = torch.cat([feature[0] for feature in features])
            # for feature in features:
            #     print(feature[1].shape)
            tgt_tensor = torch.cat([feature[1] for feature in features])
            task_type = [feature[1] for feature in features] # List of str "CPD" or "unsupervised_generation"
            return { "src_input_ids": src_tensor, "src_attention_mask": (src_tensor != 0).long(),
                     "tgt_input_ids": tgt_tensor, "tgt_attention_mask": (tgt_tensor != 0).long(),
                      "task_type": task_type }

        return fn

if __name__ == "__main__":
    pretrain_max_len = 512