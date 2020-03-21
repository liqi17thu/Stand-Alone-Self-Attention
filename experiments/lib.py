import numpy as np

def get_attention(path, layer, block, head, height, width, kernel):
    with open(path, 'r') as f:
        text = f.readlines()
    
    i = 0
    length = len(text)
    # while i < length:
    #     if f'Epoch {epoch}' in text[i]:
    #         break
    #     i += 1
    while i  < length:
        if f'layer {layer}' in text[i]:
            break
        i += 1
    while i < length:
        if f'block {block}' in text[i]:
            break
        i += 1
    while i < length:
        if f'head {head}' in text[i]:
            break
        i += 1
    while i < length:
        if f'height {height} width {width}' in text[i]:
            break
        i += 1
    if i >= length:
        print("Not Found")
        return
    i += 1
    attention = []
    for k in range(kernel):
        attention.append([float(v) for v in text[i+k][:-2].split(' ')])
    return np.array(attention)