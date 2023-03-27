import random
import math
dictTest = {'a': -1,'b':-1,'c':-1,'d':-1}
def test():
    len = 4
    passcount = 0  # 通过人数
    temp = int(math.log(len, 2))
    print(temp)

    for num in range(temp + 1):
        print(num)
        for k, v in dictTest.items():
            if random.randint(0, 1):
                dictTest[k] = num

            else:
                continue

    for k, v in dictTest.items():
        print(k, v)
if __name__ == '__main__':
    test()