### 总述
本工作是 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的参赛题目
- 本次具体选题是：2   
- 选择新模型为：Qwen-7B-Chat　


### 送分题答案（可选）
- 任务１<br>
　/root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Single node, single GPU” 部分如下命令的输出（10分）
    - python3 run.py --max_output_len=8   
    - 运行结果为：<br>
      Input: Born in north-east France, Soyer trained as a<br>Output:  chef before moving to London in the early
    - 运行说明：<br>
        这个子任务主要是熟悉trtllm把原始模型转化为FT文件，然后运行build.py文件，生成gpt2优化后引擎，运行run.py 生成输入对应的结果；主要难点在于模型下载一般受网络影响
        ，所以优先下载下来模型文件即可．<br>
- 任务2<br>
   &ensp; &ensp;/root/workspace/tensorrt_llm_july-release-v1/examples/gpt/README 里面 “Summarization using the GPT model” 部分如下命令的rouge 分数
     - python3 summarize.py --engine_dirtrt_engine/gpt2/fp16/1-gpu --test_hf  --batch_size1  --test_trt_llm  --hf_model_location=gpt2 --check_accuracy --tensorrt_llm_rouge1_threshold=14 
     <br>
    - 运行结果为：<br>
    [09/09/2023-08:42:55] [TRT-LLM] [I]  Input : ['(CNN)James Best, best known for his......contributed to this story.']<br>
[09/09/2023-08:42:55] [TRT-LLM] [I] <br>
 Reference : ['James Best, who played the sheriff on "The Dukes of Hazzard," died Monday at 88 .\n"Hazzard" ran from 1979 to 1985 and was among the most popular shows on TV .']<br>
[09/09/2023-08:42:55] [TRT-LLM] [I]<br>
 Output : [[' Best died at age 88.']]<br>
[09/09/2023-08:42:55] [TRT-LLM] [I] ---------------------------------------------------------<br>
[09/09/2023-08:43:20] [TRT-LLM] [I] TensorRT-LLM (total latency: 2.603541135787964 sec)<br>
[09/09/2023-08:43:20] [TRT-LLM] [I] TensorRT-LLM beam 0 result<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rouge1 : 15.361040799540035<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rouge2 : 3.854022269668396<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rougeL : 12.078455591738333<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rougeLsum : 13.547802733617264<br>
[09/09/2023-08:43:20] [TRT-LLM] [I] Hugging Face (total latency: 12.436068534851074 sec)<br>
[09/09/2023-08:43:20] [TRT-LLM] [I] HF beam 0 result<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rouge1 : 15.732643239575761<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rouge2 : 4.051266423605789<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rougeL : 12.611812188418664<br>
[09/09/2023-08:43:20] [TRT-LLM] [I]   rougeLsum : 14.014294213871786<br>
    -  运行说明：<br>
        这个子任务主要是利用trtllm按照和任务１相同操作获取GPT2 weights对应的加速引擎，对摘要提取任务进行推理，并且获取对应的rouge分数;<br>
        主要困难点在于summarize.py加载ccdv/cnn_dailymail数据集以及加载rouge评估，均需要从huggingface下载，所提供服务器均下载失败，<br>
        <br>
        a.首先ccdv/cnn_dailymail数据集，load_dataset函数说明<br>
          # Load from a local loading script<br>
            >>> from datasets import load_dataset<br>
            >>> ds = load_dataset('path/to/local/loading_script/loading_script.py', split='train')<br>
            ```<br>
         因此把加载数据更改为以下形式即可：当然你需要能够特殊网络下载下来相关文件cnn_dailymail.py<br>
         下载链接：https://huggingface.co/datasets/ccdv/cnn_dailymail<br>
         dataset = load_dataset(name=dataset_revision,<br>
                  revision=dataset_revision,<br>
                  path=args.dataset_path)<br>
         虽然下载速度慢，但基本上能够下载完整;第一个问题也就解决了<br><br>
        b.同理　load_metric("rouge")--->load_metric('/root/workspace/trt2023/rouge/rouge.py')　即可解决

