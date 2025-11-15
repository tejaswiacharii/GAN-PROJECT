def main():
    n = int(input().strip())
    _ = input().strip()  # read 'shuffled'
    shuffled = [input().strip() for _ in range(n)]
    _ = input().strip()  # read 'original'
    original = [input().strip() for _ in range(n)]

    # Map each instruction in original to its index
    pos = {instr: i for i, instr in enumerate(original)}

    # Convert shuffled list to indices
    idx_list = [pos[instr] for instr in shuffled]

    # Find longest contiguous increasing subsequence
    longest = 1
    current = 1
    for i in range(1, n):
        if idx_list[i] == idx_list[i - 1] + 1:
            current += 1
        else:
            current = 1
        longest = max(longest, current)

    # Minimum operations
    print(n - longest)

if __name__ == "_main_":
    main()