from torch.ao.quantization import default_fixed_qparams_range_neg1to1_observer

sent = ("Python is a high-level, interpreted programming language known for its clear syntax and readability, which makes coding faster and more efficient."
        " It is widely used in web development, data analysis, artificial intelligence, and automation due to its extensive ecosystem of libraries and supportive community.")
sent1 = sent.split(".", 1)
sent1[0] = sent1[0]+"."
for word in sent1:
    print(word)
