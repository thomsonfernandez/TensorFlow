# Python program to find if a sentence is
# palindrome
# To check sentence is palindrome or not
def sentencePalindrome(s):
    l, h = 0, len(s) - 1

    # Lowercase string
    s = s.lower()

    # Compares character until they are equal
    while l <= h:

        # If there is another symbol in left
        # of sentence
        if not ('a' <= s[l] <= 'z'):
            l += 1

        # If there is another symbol in right
        # of sentence
        elif not ('a' <= s[h] <= 'z'):
            h -= 1

        # If characters are equal
        elif s[l] == s[h]:
            l += 1
            h -= 1

        # If characters are not equal then
        # sentence is not palindrome
        else:
            return False
    # Returns true if sentence is palindrome
    return True


# Driver program to test sentencePalindrome()
s = "Too hot to hoo1t."
if sentencePalindrome(s):
    print("Sentence is palindrome.")
else:
    print("Sentence is not palindrome.")

# This code is contributed by Sachin Bisht
