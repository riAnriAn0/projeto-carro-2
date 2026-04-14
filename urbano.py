def div(a , b):
    if a < b : return 0
    return 1 + div(a-b, b)

print(div(1,2))