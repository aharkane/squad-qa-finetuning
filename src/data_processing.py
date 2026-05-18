"""Preprocessing functions for extractive QA on SQuAD.

Contains two main functions:
    - prepare_features_train: tokenize + add start_positions / end_positions labels
    - prepare_features_valid: tokenize + add example_id + mask non-context offsets
"""

from src.config import MAX_LENGTH, STRIDE


def prepare_features_train(dataset, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    """Tokenize examples and compute start/end token positions of answers.
    
    Returns the tokenizer output plus 'start_positions' and 'end_positions'
    (one entry per window, with (0, 0) for windows that don't contain the answer).
    """
    
    # 1st let's get the answer, question, context separated
    questions = [q.strip() for q in dataset['question']]
    context = [c.strip() for c in dataset['context']]
    answers = [a for a in dataset['answers']]
    
    # 2nd we tokenize question and context
    tokenized_inputs = tokenizer(
        questions, context,
        max_length=max_length,
        padding="max_length",
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    # 3rd extract mappings (pop because not needed by the models)
    words_map = tokenized_inputs.pop('offset_mapping')
    windows_map = tokenized_inputs.pop('overflow_to_sample_mapping')
    
    # Labels: ONE entry per window (not per example)
    start_positions = []
    end_positions = []
    
    # 4th single loop: iterate over windows directly
    for wind_nb, expl_nb in enumerate(windows_map):
        # get the ground-truth answer for this example
        aswr_len = len(answers[expl_nb]['text'][0])
        aswr_charPos_start = answers[expl_nb]['answer_start'][0]
        aswr_charPos_end = aswr_charPos_start + aswr_len
        
        # 5th search start and end in THIS window
        local_start = None
        local_end = None
        for tok_pos in range(len(words_map[wind_nb])):
            wrd_map = words_map[wind_nb][tok_pos]
            if wrd_map[0] == aswr_charPos_start:
                local_start = tok_pos
            if wrd_map[1] == aswr_charPos_end:
                local_end = tok_pos
        
        # 6th append to labels: actual position OR (0, 0) if not in this window
        if local_start is not None and local_end is not None:
            start_positions.append(local_start)
            end_positions.append(local_end)
        else:
            start_positions.append(0)   # [CLS] = "no answer in this window"
            end_positions.append(0)
    
    tokenized_inputs['start_positions'] = start_positions
    tokenized_inputs['end_positions'] = end_positions
    return tokenized_inputs


def prepare_features_valid(dataset, tokenizer, max_length=MAX_LENGTH, stride=STRIDE):
    """Tokenize for validation: keep example_id and mask offsets outside context.
    
    Returns the tokenizer output plus 'example_id' (one per window) and
    a modified 'offset_mapping' where non-context tokens are masked to None.
    """
    
    # init
    expl_ids = []
    
    # 1st let's get the answer, question, context separated
    ids = [q.strip() for q in dataset['id']]
    contexts = [c.strip() for c in dataset['context']]
    questions = [q.strip() for q in dataset['question']]
    
    # 2nd we tokenize question and context
    tokenized_inputs = tokenizer(
        questions, contexts,
        max_length=max_length,
        padding="max_length",
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )
    
    # 3rd extract mappings (pop because not needed by the models)
    windows_map = tokenized_inputs.pop('overflow_to_sample_mapping')
    
    # 4th iterate over each window
    for wind_nb, smp_idx in enumerate(windows_map):
        seq_id = tokenized_inputs.sequence_ids(wind_nb)
        
        # mask offsets outside the context (set to None)
        tokenized_inputs['offset_mapping'][wind_nb] = [
            wrdMap if seq_id[i] == 1 else None
            for i, wrdMap in enumerate(tokenized_inputs['offset_mapping'][wind_nb])
        ]
        
        # track which SQuAD example this window belongs to
        expl_ids.append(ids[smp_idx])
    
    # 5th add example_id to the output
    tokenized_inputs['example_id'] = expl_ids
    
    return tokenized_inputs