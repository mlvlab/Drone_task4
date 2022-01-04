from itertools import product
from tqdm import tqdm
nums = [i for i in range(7)]
cases = product(nums, nums, nums, nums, nums, nums, nums)
for i, (n1, n2, n3, n4, n5, n6, n7) in enumerate(cases):
    res = n1**2 + n2**2  + n3**2 + n4**2 + n5**2 + n6**2 + n7**2
    # res = (n1**2 + n2**2 + (n1 + n2)**2) + (n3**2 + n4**2 + (n3 + n4)**2) + (n5**2 + n6**2 + (n5 + n6)**2) + (n7**2 + n8**2 + (n7 + n8)**2) + (n9**2 + n10**2 + (n9 + n10)**2)
    if res == 311:
        print(n1, n2, n3, n4, n5, n6, n7)
    if i % 10000000 == 0:
        print(i)