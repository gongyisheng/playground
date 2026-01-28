from transformers import AutoTokenizer

def encode(model, text):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer([text], return_tensors="pt")

def decode(model, token_ids):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    model = "Qwen/Qwen2.5-0.5B-Instruct"
    token_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 5209, 2182, 279, 4226, 2878, 1124, 79075, 46391, 151645, 198, 151644, 872, 198, 18315, 295, 748, 77778, 10962, 220, 16, 21, 18805, 817, 1899, 13, 2932, 49677, 2326, 369, 17496, 1449, 6556, 323, 293, 2050, 54304, 1330, 369, 1059, 4780, 1449, 1899, 448, 3040, 13, 2932, 30778, 279, 26313, 518, 279, 20336, 6, 3081, 7298, 369, 400, 17, 817, 7722, 35985, 18636, 13, 2585, 1753, 304, 11192, 1558, 1340, 1281, 1449, 1899, 518, 279, 20336, 6, 3081, 30, 151645, 198, 151644, 77091, 198]
    print(decode(model, token_ids))