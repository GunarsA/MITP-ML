def f(x, b) -> int:
    return (x ** b + b) * f(x, b - 1) if b else 1


def main():
    while True:
        x, b = [input("Enter number: ") for _ in range(2)]

        if not (x or b):
            break

        x, b = int(x), int(b)

        print(f(x, b))


if __name__ == '__main__':
    main()
