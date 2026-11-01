text = "ola mundo"

chars = sorted(set(text))
char_to_num = {c: i for i, c in enumerate(chars)}
print(char_to_num)

codificado = [char_to_num[c] for c in text]
print(codificado)

num_para_char = {i: c for c, i in char_to_num.items()}
decodificado = "".join(num_para_char[n] for n in codificado)
print(decodificado)