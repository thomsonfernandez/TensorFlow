import numpy as nm


# function which return reverse of a string

def isPalindrome(s):
    return s == s[::-1]

def test(t):
    print("test "+t)

# Driver code
s = "12121"
ans = isPalindrome(s)

if ans:
    print("Yes")
    print(test("thomson"))
else:
    print("No")