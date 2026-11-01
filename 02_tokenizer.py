import torch

with open("data/input.txt", "r") as f:
    text = f.read()

print(len(text))


chars = sorted(set(text))
char_to_num = {c: i for i, c in enumerate(chars)}

print(len(chars))


codificado = [char_to_num[c] for c in text]

num_para_char = {i: c for c, i in char_to_num.items()}
decodificado = "".join(num_para_char[n] for n in codificado)

data = torch.tensor(codificado)
print(data.shape)
print(data[:20])

print("".join(num_para_char[n] for n in data[:20].tolist()))