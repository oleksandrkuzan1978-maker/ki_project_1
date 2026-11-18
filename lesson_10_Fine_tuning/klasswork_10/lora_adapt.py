import numpy as np

rng = np.random.default_rng()
Q = rng.normal(0, 1, 4096*4096).reshape(4096, 4096)
print(f'Q shape: {Q.shape}')

#Q + (lora_alpha/r)BA
#B.shape -> Q.shape[0]*r
#A.shape -> r*Q.shape[1]
r = 16
lora_alpha = 64

B = rng.normal(0, 1, Q.shape[0]*r).reshape(Q.shape[0], r)
A = rng.normal(0, 1, r*Q.shape[1]).reshape(r, Q.shape[1])

result = B@A
print(f'B@A shape: {result.shape}')
lora_params = B.shape[0]*B.shape[1]*2
print(f'Lora params: {lora_params}, Q params lora params ratio: {Q.shape[0]*Q.shape[1]/lora_params:.2f}')