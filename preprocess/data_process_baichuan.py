def preprocess_function_train(examples,tokenizer, data_args,instruction_column,input_column , response_column,history_column, ):
    # 该类的出处参考链接：https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py#L39
    max_seq_length = data_args.max_length

    model_inputs = {
        "input_ids": [],
        "labels": [],
    }

    user_token_id = 195
    assistant_token_id = 196
    for i in range(len(examples[instruction_column])):
        if examples[instruction_column][i] and examples[response_column][i]:
            instruction, input, answer = examples[instruction_column][i], examples[input_column][i], \
                                         examples[response_column][i]

            if history_column is None:
                prompt = instruction + input
            else:
                prompt = ""
                history = examples[history_column][i]
                for turn_idx, (old_query, response) in enumerate(history):
                    prompt += "{}\n{}\n".format(old_query, response)
                prompt += "{}\n".format(instruction + input)

            tokenized_sources = tokenizer.encode(prompt, add_special_tokens=False)
            tokenized_targets = tokenizer.encode(answer, add_special_tokens=False)

            # 组合input
            input_ids = [user_token_id] + tokenized_sources
            labels_ids = [tokenizer.eos_token_id] + [-100] * len(tokenized_sources)

            # 组合target
            input_ids = input_ids + [assistant_token_id] + tokenized_targets + [tokenizer.eos_token_id]
            labels_ids = labels_ids + [-100] + tokenized_targets + [tokenizer.eos_token_id]

            # 截断并添加padding
            input_ids = input_ids[:max_seq_length] + [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
            labels_ids = labels_ids[:max_seq_length] + [-100] * (max_seq_length - len(labels_ids))

            # 计算损失时的输入
            '''
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()

            标签从第2个token开始，输入到倒数第二个token结束
            '''

            assert len(input_ids) == len(labels_ids)
            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels_ids)

    return model_inputs