# Tinystory
title: TinyStories: How Small Can Language Models Be and Still Speak Coherent English?  
url: https://arxiv.org/abs/2305.07759  

## Key takeaway
- novel synthesis data method: add constraints of different dimension to prompt LLM to generate a diverse dataset (word: verb, noun, adjective; sentenct; feature: good/bad ending, contain dialog, etc)
- novel evaluation method: use GPT-4 as grader, score the result's grammar, creativity, consistency and age group
- a much smaller model also can learn consistency and logic with a high quality dataset. wikipedia just contains too many words that confuses LLM to learn these feature.
- scaling rule is shown in models trained based on tinystory, larger and more complex models always better

## Design
1. idea
    ```
    LLM struggle to produce coherent and fluent text when they are small.
    Is that a model size / arch problem of a dataset problem? 

    We can answer this problem by train a small LM on a dataset which contains short stories with 2-3 paragraphs, and factual knowledge of 3-4 years old kid.
    ```
2. train dataset
    tinystory:
    ```
    prompt GPT-4 to generate short stories
    
    collected a vocabulary with 15k words, divide it into nouns, verbs and adjectives.
    constructed a list of features (eg, dialogue, plot twist, bad ending or moral value)
    
    GPT4 generation prompt: 
    - write a short story (3-5 paragraphs)
    - only uses very simple words that a 3 year old child would likely understand.
    - must have 3 random words
    - must have 1 or more features
    - remember to only use simple words!
    ```

    tinystory-instruct:
    ```
    The goal is to evaluate LLM instruction following in tinystory generation

    It's a variant of tinystory
    Add: 
    - word: words must have
    - feature: features must have
    - sentence: pick 1 random sentence
    - summary: use GPT-3.5 to generate summary
    ```

3. evaluation
    ```
    Give GPT-4 a beginning, ask it to generate a story
    Then give the tinystory begining, ask GPT-4 to grade its grammar, creativity, and consistency.

    eg,
    Please provide your general assessment about the part written by the student (the one after the *** symbol).
    Is it gramatically correct? Is it consistent with the beginning of the story? Pay special attention to whether the
    student manages to complete the sentence which is split in the middle by the separator ***

    Now, grade the student’s completion in terms of grammar, creativity, consistency with the story’s beginning and
    whether the plot makes sense. Moreover, please provide your best guess of what the age of the student might be,
    as reflected from the completion. Choose from possible age groups: A: 3 or under. B: 4-5. C: 6-7. D: 8-9. E:
    10-12. F: 13-16.
    ```

4. training
    ```
    hidden size 64,128,256,512,768,1024
    layer 1,2,4,8,12
    1m-35m weight
    <30hr on V100 GPU
    ```

## Result
- SLM can also have similar performance of consistency and coherence as LLM on story generation after training on tinystory.
- Scaling law also observed in SLM training, bigger model is always better

## Other findings
- diversity evaluation on tinystory:
    ```
    3 types of dups
    - exact memorization: use n-gram overlap, use similarity of embedding
    - simple template matching: use n-gram overlap, use rouge score between stories
    - complex template matching: hard to detect
    ```