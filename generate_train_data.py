import numpy as np
from sympy import *
from vllm import LLM, SamplingParams
from statistics import mean
import random
from transformers import AutoTokenizer
import json
from tqdm import tqdm
from mcts_math.agents.utils import math_is_equiv
import argparse
from utils import ag_step,generate_numi_steps_test,extract_answer,get_model_name,get_dataset_name,eval_answer,get_prompt
import time
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--model_dir", type=str, default='')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--generate", action= "store_true")
    parser.add_argument("--get_label", action= "store_true")
    parser.add_argument("--synthetic_data_dir", type=str, default='')
    parser.add_argument("--low_level", action= "store_true")
    parser.add_argument("--use_adaptive_binary", action= "store_true")
    parser.add_argument("--sequential_search", action= "store_true")
    return parser.parse_args()


class generate_prm_data:

    def __init__(self,args)-> None:
        self.data_dir=args.data_dir
        self.save_dir=args.save_dir
        self.model_dir=args.model_dir
        self.seed=args.seed
        self.start=args.start
        self.end=args.end
        self.synthetic_data_dir=args.synthetic_data_dir
        self.low_level=args.low_level
        self.use_adaptive_binary=args.use_adaptive_binary
        self.sequential_search=args.sequential_search

    def load_data(data_dir):
        data=[]
        with open(data_dir,"r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def get_label(self):
        start_time=time.time()
        random.seed(self.seed)
        if self.sequential_search:
            mode="sequential_search"
            static_sample=True
        elif self.use_adaptive_binary:
            mode="binary_pro_search"
            static_sample=False
        else:
            mode="binary_search"
            static_sample=True
        print(f"搜索算法:{mode}\n")
        llm_dir_list=["../llama3.1-8B-Instruct","../models/qwen2-7B-Instruct"]
        gpu_memory_utilization_list=[0.5,0.45]
        llm_list=[LLM(
                    model=llm_dir_list[i],
                    tensor_parallel_size=1,
                    trust_remote_code=True,
                    gpu_memory_utilization=gpu_memory_utilization_list[i],
                    max_model_len=8192,
                    swap_space=16,
                    seed=69,
                ) for i in range(len(llm_dir_list))]
        tokenizer_list=[AutoTokenizer.from_pretrained(model_dir) for model_dir in llm_dir_list]
        synthetic_data=[]
        with open(self.synthetic_data_dir,"r") as f:
            for line in f:
                synthetic_data.append(json.loads(line))
        total_steps=0
        history=[]
        total_sampled,total_sample_tokens=0,0
        for i,data in tqdm(enumerate(synthetic_data),total=min(len(synthetic_data),self.end),desc="Main process"):
            if self.start<=i<self.end:
                question=data["question"]
                answer=data["ground_truth"]
                solutions=data["solutions"]
                step_node=ag_step(question=question,previous_steps="",ans=answer)
                if static_sample:
                    sample_num_n=48
                    sloved_list,perplexity_list,generate_tokens,_=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                else:
                    sample_num_n=72
                    sloved_list,perplexity_list,generate_tokens,generate_nodes=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                    if sum(sloved_list)<=10:
                        print(f"问题{i}跳过")
                        continue
                    sloved_list=[0 for _ in range(len(generate_nodes))]
                    perplexity_list=[0 for _ in range(len(generate_nodes))]
                    for i in range(len(generate_nodes[0])):
                        for j in range(len(generate_nodes)):
                            if generate_nodes[j][i].correct:
                                sloved_list[j]+=1
                        if sum(sloved_list)>=10:
                            sample_num_n=math.ceil(i/8)*8
                            sample_num_n=max(16,sample_num_n)
                            break
                    sloved_list,perplexity_list,generate_tokens,_=self.verify_step(step_node,llm_list,tokenizer_list,sample_num_n,history)
                    total_sampled+=sample_num_n
                    total_sample_tokens+=generate_tokens
                print(f"sampling {sample_num_n} candidates")
                for solution in tqdm(solutions,total=len(solutions),desc="Solution process"):
                    step_list=solution["solution"].split("Step")
                    step_list=step_list[1:] if step_list[0]=="" else step_list
                    final_data={"problem":question,"ground_truth_answer":answer,"total_steps":len(step_list),"llama3.1_sloved_perplexity":perplexity_list[0],"qwen2_sloved_perplexity":perplexity_list[1],"sample_num":sample_num_n}
                    total_steps+=len(step_list)
                    previous_steps=""
                    step_node_list=[]
                    final_sloved_perplexity_list=[[] for _ in range(len(step_list))]
                    for step_indx in range(len(step_list)-1):
                        previous_steps+="Step"+step_list[step_indx] if step_list[step_indx] else ""+step_list[step_indx]
                        step_node_list.append(ag_step(question=final_data["problem"],previous_steps=previous_steps,ans=final_data["ground_truth_answer"]))
                    previous_steps=""
                    if not solution["label"]:
                        first_error_step,sloved_perplexity_list,sampled,generate_tokens=self.find_error_step(final_data,0,len(step_node_list)-1,step_node_list,llm_list,tokenizer_list,sample_num_n,history,final_sloved_perplexity_list)
                        total_sampled+=sampled
                        total_sample_tokens+=generate_tokens
                        final_data["final_sloved_perplexity_list"]=sloved_perplexity_list
                    for indx,step in enumerate(step_list):
                        final_data[f"state{indx}"]="Problem:"+final_data["problem"]+"\nSelected steps:\n"+previous_steps
                        if indx==len(step_list)-1:
                            final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":solution["label"]}]
                            continue
                        previous_steps+="Step"+step if step else ""+step
                        if solution["label"]:
                            final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":solution["label"]}]
                        else:
                            if indx>=first_error_step:
                                final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":0}] if indx==first_error_step else [{"text":"Step"+step,"rating":0}]
                            else:
                                final_data[f"step{indx}"]=[{"text":"Step"+step,"rating":1}]
                    finish_time=time.time()
                    with open(self.save_dir,"a") as file:
                        file.write(json.dumps(final_data) + '\n')
                        file.write(json.dumps({"total_sampled":total_sampled,"total_sample_tokens":total_sample_tokens,"use_time":finish_time-start_time}) + '\n')
        print(f"搜索算法:{mode}, 共生成token数:{total_sample_tokens}, 共采样次数:{total_sampled}, 共用时:{finish_time-start_time}\n")


    def find_error_step(self,final_data,left,right,step_node_list,llm_list,tokenizer_list,sample_num_n,history,final_sloved_perplexity_list):
        c=0
        sloved_perplexity=final_data["llama3.1_sloved_perplexity"]+final_data["qwen2_sloved_perplexity"]
        acc_threshold=sloved_perplexity/2
        first_search=True
        total_sampled,total_sample_tokens=0,0
        if self.sequential_search:
            left=len(step_node_list)
            for i in range(len(step_node_list)):
                _,sloved_perplexity_list,generate_tokens,_=self.verify_step(step_node_list[i],llm_list,tokenizer_list,sample_num_n,history)
                total_sampled+=sample_num_n
                total_sample_tokens+=generate_tokens
                final_sloved_perplexity_list[i]=sloved_perplexity_list
                sloved=sum(sloved_perplexity_list)
                c+=1
                if sloved<acc_threshold:
                    left=i
                    print(f"\nfind error step{left},sample {c} steps\n")
                    break
            return left,final_sloved_perplexity_list,total_sampled,total_sample_tokens
        else:
            while left <= right:
                sloved_rating=min(round(5*sloved/(2*final_data["sample_num"]))*0.2,1) if not self.use_adaptive_binary else round(sloved_perplexity*5)
                mid = (right + left)//2
                if self.use_adaptive_binary and first_search and right>=4:
                    if sloved_rating<2:
                        mid-=right//4
                    elif sloved_rating>=5:
                        mid+=right//4
                _,sloved_perplexity_list,generate_tokens,_=self.verify_step(step_node_list[mid],llm_list,tokenizer_list,sample_num_n,history)
                total_sampled+=sample_num_n
                total_sample_tokens+=generate_tokens
                final_sloved_perplexity_list[mid]=sloved_perplexity_list
                c+=1
                sloved=sum(sloved_perplexity_list)
                if sloved<acc_threshold:
                    right = mid-1
                else:
                    left = mid+1
                first_search=False
            print(f"\nfind error step{left},sample {c} steps\n")
            return left,final_sloved_perplexity_list,total_sampled,total_sample_tokens


    def verify_step(self,node,llm_list,tokenizer_list,sample_num,history):
        acc_list,acc_perplexity_list=[],[],[]
        return_nodes=[]
        for i in range(len(llm_list)):
            generate_prompt = get_prompt(node,history) if len(history)>1 else None
            generate_node=generate_numi_steps_test(node, tokenizer_list[i], llm_list[i], sample_num, -10, temperature=0.8, top_p=1,ori_prompt=generate_prompt,max_tokens=4000)
            acc,acc_prob_n,prob_sum_n,generate_tokens=0,0,0,0
            for gen_node in generate_node:
                if eval_answer(gen_node.cur_step,gen_node.ans):
                    gen_node.correct=True
                    acc+=1
                    acc_prob_n+=np.exp(gen_node.step_prob/gen_node.num_step_tokens[1])
                prob_sum_n+=np.exp(gen_node.step_prob/gen_node.num_step_tokens[1])
                generate_tokens+=gen_node.num_step_tokens[1]
            acc_list.append(acc)
            acc_perplexity_list.append(acc_prob_n/prob_sum_n)
            return_nodes.append(generate_node)
        return acc_list,acc_perplexity_list,generate_tokens,return_nodes


    def chose_solution(self,node_list,chose_num):
        if len(node_list)<=chose_num:
            return node_list
        cosine_similarity_dict={}
        chose_id=[random.randint(0, len(node_list)-1)]
        for i in range(len(node_list)):
            cosine_similarity_dict[i]=[self.cosine_similarity(node_list[i],node_list[j]) for j in range(i,len(node_list))]
        for _ in range(chose_num-1):
            min_cosine_similarity,chosen=1000,chose_id[0]
            for a in range(len(node_list)):
                if a in chose_id:
                    continue
                sum_cosine_similarity=0
                for b in chose_id:
                    sum_cosine_similarity+=cosine_similarity_dict[min(b,a)][max(b,a)-min(b,a)-1]
                mean_cosine_similarity=sum_cosine_similarity/len(chose_id)
                if mean_cosine_similarity<min_cosine_similarity:
                    min_cosine_similarity=mean_cosine_similarity
                    chosen=a
            chose_id.append(chosen)
        return [node_list[id] for id in chose_id]


    def cosine_similarity(node1,node2):
        array1 = np.array(node1.cur_step_tokens)
        array2 = np.array(node2.cur_step_tokens)
        min_len = min(len(array1), len(array2))
        array1 = array1[:min_len]
        array2 = array2[:min_len]
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        cosine_similarity = dot_product / (norm1 * norm2)
        return cosine_similarity


    def generate_train_solution(self):
        random.seed(self.seed)
        history=[]
        llm = LLM(
            model=self.model_dir,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=8000,
            seed=69,
        )
        generator_name=get_model_name(self.model_dir)
        tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
        data_list=[]
        if self.low_level:
            level_dict={1:500,2:500,3:0,4:0,5:0}
        else:
            level_dict={1:0,2:0,3:500,4:1000,5:1000}
        ori_data_dict={}
        with open(self.data_dir,"r") as f:
            i=0
            for line in f:
                if 5000<i<8000:
                    item=json.loads(line)
                    if level_dict[item["level"]]:
                        data_list.append(item)
                        level_dict[item["level"]]-=1
                i+=1
        random.shuffle(data_list)
        for indx,data in tqdm(enumerate(data_list),total=len(data_list),desc="Mian process"):
            if self.start<=indx<self.end:
                if data["problem"] in ori_data_dict:
                    save_data=ori_data_dict[data["problem"]]
                    with open(self.save_dir,"a") as file:
                        file.write(json.dumps(save_data) + '\n')
                    continue
                t_solution_list,f_solution_list=[],[]
                question_node=ag_step(question=data["problem"],ans=data["answer"])
                save_data={"question":question_node.question,"ground_truth":question_node.ans,"level":data["level"],"solutions":[]}
                sloved=0                
                generate_prompt=get_prompt(question_node,history) if len(history)>1 else None
                sample_num=min(2**(data["level"]+3),64)
                solution_nodes=generate_numi_steps_test(question_node, tokenizer, llm, sample_num, -10, temperature=0.8, top_p=1,ori_prompt=generate_prompt,max_tokens=7000)
                for node in solution_nodes:
                    generate_answer=extract_answer(node.cur_step)
                    if generate_answer=="" or generate_answer=="bad_answer":
                        continue
                    if math_is_equiv(generate_answer,question_node.ans):
                        node.final_value=1
                        sloved+=1
                        history.append({"instruction":f"Instruction:{node.question}\nResponse: Let’s think step by step.\n{node.previous_steps}","response":node.cur_step})
                        t_solution_list.append(node)
                        history=history[-100:]
                    else:
                        node.final_value=0
                        f_solution_list.append(node)
                if len(t_solution_list)<1:
                    continue
                t_solution_list=self.chose_solution(t_solution_list,2)
                f_solution_list=self.chose_solution(f_solution_list,8-len(t_solution_list))
                for chose_node in t_solution_list+f_solution_list:
                    save_data["solutions"].append({"solution":chose_node.cur_step,"generator":generator_name,"label":chose_node.final_value,"extract_answer":extract_answer(chose_node.cur_step)})
                save_data[f"{generator_name}_sloved"]=sloved
                save_data[f"{generator_name}_sample_num"]=sample_num
                with open(self.save_dir,"a") as file:
                    file.write(json.dumps(save_data) + '\n')
        

if __name__=="__main__":
    args = parse_args()
    labeler=generate_prm_data(args)
    if args.generate:
        print("=========================Generating solution===========================")
        labeler.generate_train_solution()
    elif args.get_label:
        print(f"=========================Getting label================================")
        labeler.get_label()