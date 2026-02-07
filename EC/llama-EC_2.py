from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import csv
from tqdm import tqdm
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import pairwise2
from Bio.Align import PairwiseAligner
import os
import subprocess
import tempfile
import requests
from Bio import Align
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from concurrent.futures import ProcessPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import json

# 配置日志 - 同时输出到控制台和文件
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ec_prediction.log', mode='w', encoding='utf-8'),  # 保存到文件
        logging.StreamHandler()  # 同时在控制台输出
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_name, npu):
    device = f"cuda:{npu}" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        bos_token='[BOS]',
        eos_token='[EOS]'
    )
    tokenizer.pad_token = tokenizer.bos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    print("model_device:", model.device)
    return tokenizer, model

def last_digit_index(s):
    for i in range(len(s)-1, -1, -1):
        if s[i].isdigit() and (s[i-1].isdigit() or s[i-1]=='.'):
            return i
    return -1

def initi_prompt(sequence, isCoT,similar_seq,similar_EC,similar_rea):
    if isCoT == 0:
        task_definition = f"Definition: Predict the EC number which consists of four digits from the following sequence, and the output format should be EC:X.X.X.X. A few sequences have two to four EC numbers. If you are certain that there are multiple labels, your output should be separated by ';' and the format should be EC:X.X.X.X;X.X.X.X\n"
        #example=f"The maximal similarity sequence: {similar_seq}\n And the EC number of the maximal similarity sequence is {similar_EC}\n"
        task_input = f"Input: Enzyme sequence: {sequence}\nOutput:"
    elif isCoT == 1:
        task_definition = f"Definition: Predict the EC number which consists of 4 digits from the following sequence. Note: the output format should be EC:X.X.X.X. And a few sequences have two to four EC numbers. If you are certain that there are multiple labels, your output should be separated by ';' and the format should be EC:X.X.X.X;X.X.X.X\n"
        #example=f"The maximal similarity sequence which always have the same number of EC number and the same EC numbers: {similar_seq}.\n And the EC number of the maximal similarity sequence is {similar_EC}\n"
        example=f"Maximally similar sequences tend to catalyze the same type of reaction and therefore have the same EC number and the same EC number.\nThe maximum similarity sequence is: {similar_seq},\nwhich catalyzes the reaction: {similar_rea}, so the EC number of the maximum similarity sequence is: {similar_EC}\n"
        task_input = f"Input: Enzyme sequence: {sequence}.\n, Considering the similarity sequence, catalytic reaction and the biochemical environment, such as PH and temperature, let think step by step. Note: The enzyme is found mainly in prokaryotes (bacteria), thus different from similar series in the fourth level label\nOutput:"    
    return task_definition +example+ task_input
    #return task_definition + task_input

def split_into_levels(tags):
    all_levels = []
    for tag in tags:
        parts = tag.split('.')
        levels = []
        current = []
        for part in parts:
            current.append(part)
            levels.append('.'.join(current))
        all_levels.append(levels)
    return all_levels


def evaluate_levels(act, pre, k=None):
    def flatten_tags(tags_list):
        """Flatten a list of EC number strings into individual tags."""
        flattened = []
        for tags in tags_list:
            flag_1 = 0
            for tag in tags.split(';'):
                flag_1 += 1
                cleaned_tag = tag.strip()
                if cleaned_tag and flag_1 < 6:
                    flattened.append(cleaned_tag)
        return flattened

    def is_valid_ec(ec_number):
        parts = ec_number.split('.')
        return len(parts) == 4 and all(part.isdigit() for part in parts)

    def split_into_levels(ec_numbers):
        """Split EC numbers into valid ones, padding single predictions if needed."""
        valid_ecs = []
        for index, ec in enumerate(ec_numbers):
            parts = ec.split('.')
            # Handle single prediction with three parts
            if (len(ec_numbers) == 1 and len(parts) == 3 and
                all(part.isdigit() for part in parts)):
                padded_ec = '.'.join(parts + ['1'])
                valid_ecs.append(padded_ec)
            elif is_valid_ec(ec):
                valid_ecs.append(ec)
        return valid_ecs

    def generate_prefixes(ec_number):
        """Generate all hierarchical prefixes for an EC number."""
        parts = ec_number.split('.')
        prefixes = []
        current = []
        for part in parts:
            current.append(part)
            prefixes.append('.'.join(current))
        return prefixes

    def get_prefix_sets(ec_numbers, max_level=4):
        """Generate prefix sets for each hierarchical level."""
        prefix_sets = [set() for _ in range(max_level)]
        for ec in ec_numbers:
            prefixes = generate_prefixes(ec)
            for level in range(max_level):
                if level < len(prefixes):
                    prefix_sets[level].add(prefixes[level])
        return prefix_sets

    def compute_p_r_f1(act_set, pre_set):
        """Compute precision, recall, and F1 score given two sets."""
        tp = len(act_set.intersection(pre_set))
        fp = len(pre_set - act_set)
        fn = len(act_set - pre_set)
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        
        return (precision, recall, f1)

    # Flatten the input tags
    act_flat = flatten_tags(act)
    pre_flat = flatten_tags(pre)
    
    # Truncate predictions to top-k if k is specified
    if k is not None:
        pre_flat = pre_flat[:k]
    
    # Split into valid EC numbers and handle padding for single predictions
    act_valid = split_into_levels(act_flat)
    pre_valid = split_into_levels(pre_flat)
    
    # Generate prefix sets for each hierarchical level
    act_prefix_sets = get_prefix_sets(act_valid, max_level=4)
    pre_prefix_sets = get_prefix_sets(pre_valid, max_level=4)
    
    prf_values = []
    for level in range(4):
        act_set = act_prefix_sets[level]
        pre_set = pre_prefix_sets[level]
        
        # Compute metrics for this level
        p, r, f1 = compute_p_r_f1(act_set, pre_set)
        prf_values.append((p, r, f1))
    
    return prf_values

def clean_protein_sequence(sequence):
    """简单的序列清理函数"""
    return sequence.replace(" ", "").replace("\n", "").strip().upper()

def calculate_similarity_and_homology(query_sequences, search_sequences):
    """
    使用 MMseqs2 进行高效的批量序列相似性搜索
    
    参数:
    query_sequences: list of str, 查询蛋白质序列列表
    search_sequences: list of str, 目标蛋白质序列列表
    
    返回:
    list: 每个查询序列在目标序列中最大相似序列的索引值
    """
    # 检查 MMseqs2 是否安装
    try:
        subprocess.run(["mmseqs", "version"], 
                       check=True, 
                       stdout=subprocess.DEVNULL, 
                       stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("错误: MMseqs2 未安装或不在系统路径中")
        return [-1] * len(query_sequences)
    
    # 清理序列并过滤空序列
    cleaned_query = []
    valid_query_indices = []
    for i, seq in enumerate(query_sequences):
        cleaned = clean_protein_sequence(seq)
        if cleaned and len(cleaned) > 0:
            cleaned_query.append(cleaned)
            valid_query_indices.append(i)
    
    cleaned_search = [clean_protein_sequence(seq) for seq in search_sequences]
    cleaned_search = [seq for seq in cleaned_search if seq and len(seq) > 0]
    
    if not cleaned_search:
        print("错误: 目标序列列表为空")
        return [-1] * len(query_sequences)
    
    if not cleaned_query:
        print("警告: 所有查询序列均为空")
        return [-1] * len(query_sequences)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 1. 创建目标序列数据库
        search_fasta = os.path.join(tmp_dir, "search.fasta")
        with open(search_fasta, "w") as f:
            for idx, seq in enumerate(cleaned_search):
                f.write(f">seq{idx}\n{seq}\n")
        
        db_path = os.path.join(tmp_dir, "search_db")
        try:
            subprocess.run(["mmseqs", "createdb", search_fasta, db_path], 
                           check=True, 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"创建目标数据库失败: {e.stderr.decode('utf-8')}")
            return [-1] * len(query_sequences)

        # 2. 创建查询序列数据库
        query_fasta = os.path.join(tmp_dir, "query.fasta")
        with open(query_fasta, "w") as f:
            for idx, seq in enumerate(cleaned_query):
                f.write(f">query{idx}\n{seq}\n")
        
        query_db = os.path.join(tmp_dir, "query_db")
        try:
            subprocess.run(["mmseqs", "createdb", query_fasta, query_db], 
                           check=True,
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"创建查询数据库失败: {e.stderr.decode('utf-8')}")
            return [-1] * len(query_sequences)
        
        # 3. 执行搜索
        result_db = os.path.join(tmp_dir, "result_db")
        try:
            subprocess.run([
                "mmseqs", "search", 
                query_db, 
                db_path, 
                result_db, 
                os.path.join(tmp_dir, "tmp"),
                "--threads", "8",  # 使用8个线程
                "-s", "3",       # 设置灵敏度
                "--max-seqs", "1000",  # 设置最大候选序列数
                "-c", "0.3",
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"执行搜索失败: {e.stderr.decode('utf-8')}")
            return [-1] * len(query_sequences)
        
        # 4. 转换结果
        formatted_path = os.path.join(tmp_dir, "formatted.tsv")
        try:
            subprocess.run([
                "mmseqs", "convertalis", 
                query_db, 
                db_path, 
                result_db, 
                formatted_path,
                "--format-output", "query,target,pident"
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"转换结果失败: {e.stderr.decode('utf-8')}")
            return [-1] * len(query_sequences)
        
        # 5. 解析结果
        best_matches = {}
        try:
            if os.path.exists(formatted_path) and os.path.getsize(formatted_path) > 0:
                with open(formatted_path, "r") as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            try:
                                query_id = parts[0].replace("query", "")
                                target_id = parts[1].replace("seq", "")
                                identity = float(parts[2])
                                
                                q_idx = int(query_id)
                                t_idx = int(target_id)
                                
                                # 保留最佳匹配（按相似度）
                                if q_idx not in best_matches or identity >= best_matches[q_idx][0]:
                                    best_matches[q_idx] = (identity, t_idx)
                            except (ValueError, IndexError) as e:
                                print(f"解析行时出错: {line}, 错误: {e}")
                                continue
            else:
                print(f"警告: 结果文件 {formatted_path} 为空或不存在")
        except IOError as e:
            print(f"读取结果文件失败: {str(e)}")
            return [-1] * len(query_sequences)
        
        # 6. 构建最终结果
        # 初始化结果列表，全部为-1（表示失败或未匹配）
        results = [-1] * len(query_sequences)
        # 遍历有效查询索引（即非空序列的原始索引）
        for valid_idx, orig_idx in enumerate(valid_query_indices):
            if valid_idx in best_matches:
                # 将匹配到的目标序列索引赋给原始查询序列位置
                results[orig_idx] = best_matches[valid_idx][1]
            else:
                # 没有找到匹配，保留-1
                pass
        
        # 7. 输出简要统计
        found = sum(1 for r in results if r != -1)
        print(f"\n搜索结果: 成功匹配 {found}/{len(query_sequences)} 个查询序列")
        
        return results

def create_retry_session(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 504)):
    """创建带重试机制的会话"""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def get_enzyme_reactants_batch(uniprot_ids, batch_size=50, max_retries=3):
    """
    批量获取酶的催化反应物
    参数:
        uniprot_ids: UniProt ID列表
        batch_size: 每批处理的ID数量
        max_retries: 最大重试次数
    
    返回:
        dict: 映射 {uniprot_id: [reactions]} 或 {uniprot_id: []} 如果无反应物
    """
    # 创建带重试机制的会话
    session = create_retry_session(retries=3)
    
    # 结果字典
    results = {uid: [] for uid in uniprot_ids}
    
    # 分批处理
    for i in tqdm(range(0, len(uniprot_ids), batch_size), desc="批量获取反应物"):
        batch = uniprot_ids[i:i+batch_size]
        query = " OR ".join([f"accession:{uid}" for uid in batch])
        
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": query,
            "fields": "accession,cc_catalytic_activity",
            "format": "json",
            "size": len(batch)
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # 处理结果
                for result in data.get("results", []):
                    uid = result["primaryAccession"]
                    # 提取催化反应信息
                    comments = result.get("comments", [])
                    for comment in comments:
                        if comment.get("commentType") == "CATALYTIC ACTIVITY":
                            reaction = comment.get("reaction", {})
                            if "name" in reaction:
                                results[uid].append(reaction["name"])
                
                break  # 成功则跳出重试循环
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                logger.error(f"批量请求失败 (尝试 {attempt+1}/{max_retries+1}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"批量请求失败超过最大重试次数")
    
    return results

def save_prediction_details(predictions, actual_ec, uniprot_ids, output_file='prediction_details_all_8.csv'):
    """保存预测详细信息到CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Uniprot_ID', 'Actual_EC', 'Predicted_EC', 'Match'])
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_ec)):
            match = "Yes" if pred == actual else "No"
            writer.writerow([uniprot_ids[i], actual, pred, match])
    
    logger.info(f"预测详细信息已保存到 {output_file}")

def main(npu, isCoT):
    npu=1
    tokenizer, model = load_model("llama4.1-lora-finetuned", npu)
    #tokenizer, model = load_model("Meta-Llama-3.1-8B-Instruct", npu)
    #filename = "../dataset/ECtest/ECall.csv"
    filename = "../dataset/ECtest/Price-149.csv"
    #filename = "../dataset/ECtest/NEW-392.csv"
    csv_reader = csv.reader(open(filename))
    csv_reader = list(csv_reader)
    
    EC_list = []
    sequence_list = []
    uniprot_id_list = []
    for i in tqdm(range(1, len(csv_reader))):
        words = csv_reader[i][0].split('\t')
        sequence_list.append(words[2])
        EC_list.append(words[1])
        uniprot_id_list.append(words[0])
    actual = EC_list
    predict = []

    search_seq = []
    search_EC = []
    search_id_list = []
    EC_seq_num = ["10", "30", "50", "70", "100"]
    for t in EC_seq_num:
        filename = f"../EC-seqence/split{t}.csv"
        csv_reader = csv.reader(open(filename))
        next(csv_reader)
        for row in csv_reader:
            words = row[0].split('\t')
            search_seq.append(words[2])
            search_EC.append(words[1])
            search_id_list.append(words[0])
    print(f"加载了 {len(search_seq)} 条搜索序列")

    similar_list = []
    max_similar_index = []
    num_cannotgnne = 0
    
    # 构建相似序列ID列表
    similar_id_list = [] 

    # 计算相似序列索引
    max_similar_index = calculate_similarity_and_homology(sequence_list, search_seq)
    
    # 构建相似序列ID列表
    for i in range(len(max_similar_index)):
        similar_id_list.append(search_id_list[max_similar_index[i]])
    
    # 收集所有需要查询的UniProt ID（测试集和相似序列）
    all_uniprot_ids = set(similar_id_list)
    logger.info(f"开始批量获取 {len(all_uniprot_ids)} 个酶的反应物...")
    reaction_map = get_enzyme_reactants_batch(list(all_uniprot_ids))
    
    # 准备反应物列表
    #reaction = [reaction_map.get(uid, [])[0] if reaction_map.get(uid) else "None" for uid in uniprot_id_list]
    search_reaction = [reaction_map.get(sid, [])[0] if reaction_map.get(sid) else "None" for sid in similar_id_list]
    
    # 保存相似序列信息
    for i in tqdm(range(len(EC_list))):
        similar_list.append(search_EC[max_similar_index[i]])
    
    prf_values = evaluate_levels(actual, similar_list)
    for level_idx, (precision, recall, f1) in enumerate(prf_values, 1):
        print(f"Level {level_idx}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # 保存结果
    with open('similar_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Query_Index', 'Similar_Index', 'Query_EC', 'Similar_EC', 'Uniprot_id', 'Search_id', 'Search_Reaction'])
        for i in range(len(EC_list)):
            writer.writerow([
                i, 
                max_similar_index[i], 
                EC_list[i], 
                search_EC[max_similar_index[i]], 
                uniprot_id_list[i], 
                similar_id_list[i],  # 使用similar_id_list
                search_reaction[i]
            ])

    output=[]
    # 进行预测
    for i in tqdm(range(len(EC_list)), desc="预测EC号"):
        sequence = sequence_list[i]
        uid = uniprot_id_list[i]
        similar_seq = search_seq[max_similar_index[i]]
        similar_EC = search_EC[max_similar_index[i]]
        similar_rea = search_reaction[i]  # 使用索引i对应的反应物
        
        messages = initi_prompt(sequence, isCoT, similar_seq, similar_EC, similar_rea)
        input_ids = tokenizer.encode(messages, return_tensors="pt").to(model.device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(model.device)
        
        flag = 0
        response_text = ""
        while flag < 6:
            try:
                flag += 1
                generated_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.eos_token_id
                )
                response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                if "Output:" in response_text:
                    response_text = response_text.split("Output:")[1]
                    response_text_1=response_text
                    print(response_text)
                if "EC:" in response_text:
                    response_text = response_text.split("EC:")[1]
                if response_text.find("pH")!=-1:
                    response_text = response_text[:response_text.find("pH")]
                else:
                    response_text = response_text[:60]
                
                lastnum = last_digit_index(response_text)
                break
            except Exception as e:
                logger.error(f"生成失败 (尝试 {flag}/6): {str(e)}")
                continue
        
        if lastnum == -1 or flag == 6:
            num_cannotgnne += 1
            predict.append("1.1.1.1")
            output.append("1.1.1.1")   
        else:
            output.append(response_text_1) 
            predict.append(response_text[:lastnum+1].strip())
        
        # 这些日志信息会同时输出到控制台和ec_prediction.log文件
        logger.info(f"预测 {i}: 模型输出: {response_text}")
        logger.info(f"提取结果: {predict[-1]}")
        logger.info(f"实际 EC: {actual[i]}")

    # 评估预测结果
    prf_values = evaluate_levels(actual, predict)
    for level_idx, (precision, recall, f1) in enumerate(prf_values, 1):
        print(f"Level {level_idx}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    print(f'无法生成EC号的数量: {num_cannotgnne}')
    
    # 保存预测详细信息到CSV文件
    save_prediction_details(output, actual, uniprot_id_list)

if __name__ == '__main__':
    isCoT = 1
    npu = 0
    main(npu, isCoT)