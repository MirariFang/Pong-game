import TrainSmall

if __name__ == '__main__':
    for i in range(10):
        if i == 0:
            continue
        result = TrainSmall.train(10000, 1000, i / 10.0, 5, 10)
        print(result)
